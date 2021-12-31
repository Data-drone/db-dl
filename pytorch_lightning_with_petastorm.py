# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### Setup

# COMMAND ----------

# MAGIC %pip install pytorch-lightning

# COMMAND ----------

import io
import numpy as np
from functools import partial
import datetime as dt

from PIL import Image

import torch, torchvision
from torch import nn
import torch.nn.functional as F
import torchmetrics.functional as FM
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from pyspark.sql.functions import col
from pyspark.sql.types import LongType

from petastorm import TransformSpec
from petastorm.spark import SparkDatasetConverter, make_spark_converter

print(f'Using:\n - torch: {torch.__version__}\n - torchvision: {torchvision.__version__}\n - pytorch_lightning: {pl.__version__}')

# COMMAND ----------

IS_DEV = True

DATA_DIR = '/databricks-datasets/flowers/delta'
GPU_COUNT = torch.cuda.device_count()
GPUS = min(1, GPU_COUNT)
# BATCH_SIZE = 256 if GPU_COUNT > 0 else 64
BATCH_SIZE = 64
EPOCH_COUNT = 3

WORKER_COUNT = 1
LR = 0.001
CLASS_COUNT = 5
print(f'GPU count: {GPU_COUNT}')

SAMPLE_SIZE = 100
print(f"Sample size: {SAMPLE_SIZE}")

# Set a cache directory on DBFS FUSE for intermediate data.
CACHE_DIR = "file:///dbfs/tmp/petastorm/cache"
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, CACHE_DIR)

# COMMAND ----------

# MAGIC %sh
# MAGIC #rm -r /dbfs/tmp/petastorm/cache

# COMMAND ----------

def report_duration(action, start):
  end = dt.datetime.now()
  ds = (end - start).total_seconds()
  h, rem = divmod(ds, 3600)
  m, s = divmod(rem, 60)
  if h > 0:
      run_time = '{} hours {} minutes'.format(int(h), int(m))
  elif m > 0:
      run_time = '{} minutes'.format(int(m))
  else:
      run_time = '{} seconds'.format(int(s))

  msg = f'--> {action} completed in {run_time}'
  print(msg)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Data - the flowers dataset
# MAGIC 
# MAGIC This example uses the flowers dataset from the TensorFlow team, which contains flower photos stored under five sub-directories, one per class. It is hosted under Databricks Datasets dbfs:/databricks-datasets/flower_photos for easy access.
# MAGIC 
# MAGIC The example loads the flowers table which contains the preprocessed flowers dataset using the binary file data source. It uses a small subset of the flowers dataset to reduce the running time of this notebook. When you run this notebook, you can increase the number of images used for better model accuracy.

# COMMAND ----------

def get_data(sample_size=-1):
  df = spark.read.format("delta").load(DATA_DIR).select(["content", "label"])
  if sample_size > 0:
    df = df.limit(sample_size)
  classes = list(df.select("label").distinct().toPandas()["label"])

  assert CLASS_COUNT == len(classes)

  # Add a numeric class colunmn
  def to_class_index(class_name):
    return classes.index(class_name)

  class_index = udf(to_class_index, LongType())
  df = df.withColumn("cls_id", class_index(col("label"))).drop("label")

  train_df, val_df = df.randomSplit([0.9, 0.1], seed=12345)

  # Make sure the number of partitions is at least the number of workers to benefit from distributed training
  train_df = train_df.repartition(WORKER_COUNT)
  val_df = val_df.repartition(WORKER_COUNT)

  print(f'Training dataset: {df.count()} (train: {train_df.count()}, val: {val_df.count()})')
  print(f'Labels ({CLASS_COUNT}): {classes}')
  if sample_size > 0:
    display(train_df.limit(10))
  
  return train_df, val_df

def create_spark_converters(use_sample=True):
  train_df, val_df = get_data(SAMPLE_SIZE if use_sample else -1)
  return make_spark_converter(train_df), make_spark_converter(val_df)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Dataframe and corresponding Petastorm wrapper need to be created here instead of inside the Pytorch Lightning model class. This is especially important for distributed training (Petastorm spark converter is pickleable) 

# COMMAND ----------

train_sample_converter, val_sample_converter = create_spark_converters(True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### The Model
# MAGIC 
# MAGIC The model class may look large but I deliberately piled everything into it to show how the most part of the model logic can be encapsulated into a single class
# MAGIC 
# MAGIC A callback class like this can be used for things like logging but it is easier to keep all this within the model
# MAGIC ```
# MAGIC from pytorch_lightning.callbacks import Callback
# MAGIC 
# MAGIC class LitCallback(Callback):
# MAGIC     def __init__(self):
# MAGIC         self.state = {"epochs": 0, "batches": 0}
# MAGIC     
# MAGIC     def on_epoch_start(self, trainer, pl_module):
# MAGIC         print(f"--> Epoch: {self.state['epochs']}")
# MAGIC         self.state["epochs"] += 1
# MAGIC           
# MAGIC     def on_batch_start(self, trainer, pl_module):
# MAGIC         print(f"\t- batch: {self.state['batches']}")
# MAGIC         self.state["batches"] += 1
# MAGIC ```
# MAGIC 
# MAGIC ... and used like this
# MAGIC ```
# MAGIC trainer = pl.Trainer(..., callbacks=[LitCallback()])
# MAGIC 
# MAGIC ```

# COMMAND ----------

class LitClassificationModel(pl.LightningModule):
  def __init__(self, train_converter, val_converter, class_count=CLASS_COUNT, lr=LR):
    super().__init__()
    self.lr = lr
    self.model = self.get_model(class_count, lr)
    self.train_converter = train_converter
    self.val_converter = val_converter
    self.train_dataloader_context = None
    self.state = {"epochs": 0, "batches": 0}

  def configure_optimizers(self):
    optimizer = torch.optim.SGD(self.model.classifier[1].parameters(), lr=self.lr, momentum=0.9)
    return optimizer

  def forward(self, x):
    x = self.model(x)
    return x

  def training_step(self, batch, batch_idx):
    X, y = batch['features'], batch['cls_id']
    pred = self(X)
    loss = F.cross_entropy(pred, y)
    acc = FM.accuracy(pred, y)
    print(f'\t - batch_idx: {batch_idx}, ({len(y)})')
    print(f'\t - loss: {loss}, acc: {acc}')
    return loss

  def on_epoch_start(self):
    print(f"--> Epoch: {self.state['epochs']}")
    self.state["epochs"] += 1

#     # First epoch will already have data loader
#     if self.state["epochs"] > 1:
#       # Need to explicity Enter the context, trainer does not seem to be able to work with a context
#       self.trainer.train_dataloader = self.train_dataloader()

  def on_epoch_end(self):
    print(f"<-- (epoch: {self.state['epochs'] - 1})")
    
    # Need to reset DataLoader on each epoch. ContextManager is stored in this class, not in `trainer`, destroy it. If new one is needed,
    # it will be created at the start of next epoch
    self.train_dataloader_context.__exit__(None, None, None)

  def train_dataloader(self):
    
    print(f"--> LitClassificationModel.train_dataloader")
    
    # To improve performance, do the data transformation in a TransformSpec function in petastorm instead of Spark Dataframe
    self.train_dataloader_context = self.train_converter.make_torch_dataloader(transform_spec=self.get_transform_spec(is_train=True), num_epochs=1, batch_size=BATCH_SIZE)
    return self.train_dataloader_context.__enter__()

  def get_model(self, class_count, lr):
    model = torchvision.models.mobilenet_v2(pretrained=True)

    # Freeze parameters in the feature extraction layers and replace the last layer
    for param in model.parameters():
      param.requires_grad = False

    # New modules have `requires_grad = True` by default
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, class_count)
    return model
    
  def _transform_row(self, is_train, batch):
    transformers = [transforms.Lambda(lambda x: Image.open(io.BytesIO(x)))]
    if is_train:
      transformers.extend([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
      ])
    else:
      transformers.extend([
        transforms.Resize(256),
        transforms.CenterCrop(224),
      ])
    transformers.extend([
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    trans = transforms.Compose(transformers)

    batch['features'] = batch['content'].map(lambda x: trans(x).numpy())
    batch = batch.drop(labels=['content'], axis=1)
    return batch

  def get_transform_spec(self, is_train=True):
    # The output shape of the `TransformSpec` is not automatically known by petastorm, 
    # so you need to specify the shape for new columns in `edit_fields` and specify the order of 
    # the output columns in `selected_fields`.
    return TransformSpec(partial(self._transform_row, is_train), 
                         edit_fields=[('features', np.float32, (3, 224, 224), False)], 
                         selected_fields=['features', 'cls_id'])
  
# TODO
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = F.nll_loss(logits, y)
#         preds = torch.argmax(logits, dim=1)
#         self.accuracy(preds, y)

#         # Calling self.log will surface up scalars for you in TensorBoard
#         self.log("val_loss", loss, prog_bar=True)
#         self.log("val_acc", self.accuracy, prog_bar=True)
#         return loss
      
#       def training_step(self, batch, batch_idx):
#           x, y = batch
#           y_hat = self.model(x)
#           loss = F.cross_entropy(y_hat, y)
#           pred = ...
#           return {"loss": loss, "pred": pred}

#https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/core/lightning.html#LightningModule.validation_epoch_end
#       def training_step_end(self, batch_parts):
#           # predictions from each GPU
#           predictions = batch_parts["pred"]
#           # losses from each GPU
#           losses = batch_parts["loss"]

#           gpu_0_prediction = predictions[0]
#           gpu_1_prediction = predictions[1]

#           # do something with both outputs
#           return (losses[0] + losses[1]) / 2


#       def training_epoch_end(self, training_step_outputs):
#           for out in training_step_outputs:

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### CPU Training

# COMMAND ----------

start = dt.datetime.now()

trainer = pl.Trainer(
    gpus=0,
    max_epochs=EPOCH_COUNT,
    progress_bar_refresh_rate=20,
    reload_dataloaders_every_n_epochs=1,  # Need this, otherwise process will fail trying to sample from closed DataLoader
)

lit_model = LitClassificationModel(train_sample_converter, val_sample_converter)
trainer.fit(lit_model)

report_duration("CPU training", start)

            
# TODO
#trainer.test()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Some extra feedback on what's done so far
# MAGIC 
# MAGIC ***Few surprise discoveries here***
# MAGIC 1. Must set `num_epochs=1` in the call to `make_torch_dataloader` otherwise the loader will keep on going forever as if it was set to `None`
# MAGIC     - default value is `None` (though in `https://github.com/uber/petastorm/blob/master/petastorm/reader.py` it's `1`)
# MAGIC     - from docs: ```:param num_epochs: An epoch is a single pass over all rows in the
# MAGIC             dataset. Setting `num_epochs` to `None` will result in an
# MAGIC             infinite number of epochs.```
# MAGIC     - effectively, this means setting `num_epochs=None` will result in never-ending training even if using `EPOCH_COUNT=1`
# MAGIC     - don't set it to `EPOCH_COUNT` as it will result `EPOCH_COUNT` * `batch_count` number of batches for each epoch, as opposed to expected `batch_count` (`batch_count = record_count / BATCH_SIZE`)
# MAGIC 1. Even with `num_epochs=1`, if training for multiple epochs then only a first epoch seem to be getting a correct record count with last incomplete batch size doubling for each epoch
# MAGIC     - *Example: training with BATCH_SIZE=64 and total of 92 records*
# MAGIC         - training for 1 epoch
# MAGIC             - epoch 0
# MAGIC                 - batch 0: 64 records
# MAGIC                 - batch 1: 28 records <- OK
# MAGIC         - training for 2 epochs
# MAGIC             - epoch 0
# MAGIC                 - batch 0: 64 records
# MAGIC                 - batch 1: 28 records
# MAGIC             - epoch 1 (extra 28 records)
# MAGIC                 - batch 2: 64 records
# MAGIC                 - batch 3: 56 records <-- Doubled (+28)
# MAGIC         - training for 3 epochs
# MAGIC             - *exactly like epochs 0 and 1 from training with 2 epochs*
# MAGIC             - epoch 2
# MAGIC                 - batch 4: 64 records
# MAGIC                 - batch 5: 64 records  <-- 56+8
# MAGIC                 - batch 6: 20 records  <-- +20 (spill from +28 to previous batch)
# MAGIC         - training for 4 epochs
# MAGIC             - *exactly like epochs 0, 1 and 2 from training with 3 epochs*
# MAGIC             - epoch 3
# MAGIC                 - batch 7: 64 records
# MAGIC                 - batch 8: 48 records  <-- this looks like 20+28 but where is second batch of 64 gone?    
# MAGIC     - *This pattern of adding 28 to the last batch continues for subsequent epochs if larger number of epochs is used for training*
# MAGIC         - as can be seen in training with 4 epochs, there are some occational drops of previosly used by overfilled batches
# MAGIC     - A fix: reset the dataset on every epoch, but that's not a straightforward process because `make_torch_dataloader` returns a Context Manager, so
# MAGIC         - opt 1 - ignore a reset and train with some non-expected extra data (though this will likely result in context manager not cleaning up properly), e.g. `trainer.fit(lit_model, converter_train.make_torch_dataloader(transform_spec=get_transform_spec(is_train=True), num_epochs=1, batch_size=BATCH_SIZE).__enter__())`
# MAGIC         - [IMPLEMENTED] option 2 - call `__enter__` explicitly on the Context Manager
# MAGIC           
# MAGIC This also works but no way to control resetting
# MAGIC ```
# MAGIC with converter_train.make_torch_dataloader(transform_spec=get_transform_spec(is_train=True), num_epochs=1, batch_size=BATCH_SIZE) as train_loader:
# MAGIC     trainer.fit(lit_model, train_loader)~
# MAGIC ```
# MAGIC 
# MAGIC https://github.com/uber/petastorm/blob/master/petastorm/spark/spark_dataset_converter.py

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## GPU Training

# COMMAND ----------

import horovod.torch as hvd
from pytorch_lightning.plugins.training_type.horovod import HorovodPlugin
# from sparkdl import HorovodRunner

# COMMAND ----------

def train_hvd(gpus=1):
  print(f"--> Train with Horovod using {gpus} GPUs")
  start = dt.datetime.now()

  trainer = pl.Trainer(
      gpus=gpus, 
      max_epochs=EPOCH_COUNT,
      progress_bar_refresh_rate=20,
      reload_dataloaders_every_n_epochs=1,
      strategy=HorovodPlugin()
  )

  model = LitClassificationModel(train_sample_converter, val_sample_converter)
  trainer.fit(model)

  report_duration(f"Training with {gpus} GPUs", start)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Using 1 GPU

# COMMAND ----------

train_hvd()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Using 4 GPUs

# COMMAND ----------

# This will still run on a single GPU

train_hvd(4)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Multi-GPU Training
# MAGIC 
# MAGIC Looks like simply passing number of desired GPUs to use to `gpus` paramerer of the trainer doesn't really work as expected as the trainer seem to be still using only one GPU. 
# MAGIC 
# MAGIC In order to make it work with multiple GPUs, need to use `horovod` runner
# MAGIC 
# MAGIC - https://github.com/PyTorchLightning/pytorch-lightning/blob/1fc046cde28684316323693dfca227dbd270663f/tests/models/test_horovod.py

# COMMAND ----------

from pytorch_lightning.utilities import _HOROVOD_AVAILABLE
if _HOROVOD_AVAILABLE:
  print("Horovod is available")
  import horovod
  import horovod.torch as hvd
  hvd.init()

# COMMAND ----------

def train_hvd_one():
  hvd.init()
  print(f"--> hvd.rank(): {hvd.rank()}, hvd.size(): {hvd.size()}")
  start = dt.datetime.now()

  trainer = pl.Trainer(
      gpus=1, 
      max_epochs=EPOCH_COUNT,
      progress_bar_refresh_rate=20,
      reload_dataloaders_every_n_epochs=1,
      strategy="horovod"
  )

  model = LitClassificationModel(train_sample_converter, val_sample_converter)
  trainer.fit(model)

  report_duration(f"Device {hvd.rank()}", start)


# COMMAND ----------

horovod.run(train_hvd_one, np=2)

# COMMAND ----------

# MAGIC %md
# MAGIC # TODO

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Embed the model into the PyTorch Lightning model
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC LightningModule is just a torch.nn.Module

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Appendix

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Using PyTorch Distributed Data Parallel
# MAGIC 
# MAGIC - https://pytorch.org/docs/master/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel
# MAGIC ```
# MAGIC This container parallelizes the application of the given module by
# MAGIC splitting the input across the specified devices by chunking in the batch
# MAGIC dimension. The module is replicated on each machine and each device, and
# MAGIC each such replica handles a portion of the input. During the backwards
# MAGIC pass, gradients from each node are averaged.
# MAGIC ```
# MAGIC 
# MAGIC ***This only runs on CPU in interactive mode, on GPUs (even on a single GPU) this cannot be run in interactive mode***

# COMMAND ----------

trainer = pl.Trainer(
    gpus=0, 
    accelerator='ddp', 
    max_epochs=EPOCH_COUNT,
    progress_bar_refresh_rate=20,
    reload_dataloaders_every_n_epochs=1
)

lit_model = LitModel()
trainer.fit(lit_model)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Single Node Training and Validation

# COMMAND ----------

# Check the selected model structure
#model = torchvision.models.mobilenet_v2(pretrained=True)
#model

# COMMAND ----------

def train(model, criterion, optimizer, scheduler, data_loader, device, batch_count, batch_size=BATCH_SIZE):
  model.train()  # Set model to training mode

  # statistics
  running_loss = 0.0
  running_corrects = 0

  # Iterate over the data for one epoch.
  for step in range(batch_count):
    batch = next(data_loader)
    X, y = batch['features'].to(device), batch['cls_id'].to(device)
    
    # Track history in training
    with torch.set_grad_enabled(True):
      
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward
      outputs = model(X)
      _, preds = torch.max(outputs, 1)
      loss = criterion(outputs, y)

      # backward + optimize
      loss.backward()
      optimizer.step()

    # statistics
    running_loss += loss.item() * X.size(0)
    running_corrects += torch.sum(preds == y.data)
  
  scheduler.step()

  loss = running_loss / (batch_count * batch_size)
  acc = running_corrects.double() / (batch_count * batch_size)

  print('- Train: loss: {:.4f}, acc: {:.4f}'.format(loss, acc))
  return loss, acc

def eval(model, criterion, data_loader, device, batch_count, batch_size=BATCH_SIZE, metric_agg_fn=None):
  model.eval()  # Set model to evaluate mode

  # statistics
  running_loss = 0.0
  running_corrects = 0

  # Iterate over all the validation data.
  for step in range(batch_count):
    batch = next(data_loader)
    X, y = batch['features'].to(device), batch['cls_id'].to(device)

    # Do not track history in evaluation to save memory
    with torch.set_grad_enabled(False):
      # forward
      outputs = model(X)
      _, preds = torch.max(outputs, 1)
      loss = criterion(outputs, y)

    # statistics
    running_loss += loss.item()
    running_corrects += torch.sum(preds == y.data)
  
  # Average the losses across observations for each minibatch.
  loss = running_loss / batch_count
  acc = running_corrects.double() / (batch_count * batch_size)
  
  # metric_agg_fn is used in the distributed training to aggregate the metrics on all workers
  if metric_agg_fn is not None:
    loss = metric_agg_fn(loss, 'avg_loss')
    acc = metric_agg_fn(acc, 'avg_acc')

  print('- Validation - loss: {:.4f}, acc: {:.4f}'.format(loss, acc))

  return loss, acc

# COMMAND ----------

def train_and_eval(class_count=CLASS_COUNT, lr=LR, batch_size=BATCH_SIZE):
  device = torch.device("cuda" if GPUS else "cpu")
  
  model = get_model(class_count, lr).to(device)
  criterion = torch.nn.CrossEntropyLoss()

  # Only parameters of final layer are being optimized.
  optimizer = torch.optim.SGD(model.classifier[1].parameters(), lr=lr, momentum=0.9)

  # Decay LR by a factor of 0.1 every 7 epochs
  exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
  
  with converter_train.make_torch_dataloader(transform_spec=get_transform_spec(is_train=True), 
                                             batch_size=BATCH_SIZE) as train_dl, \
       converter_val.make_torch_dataloader(transform_spec=get_transform_spec(is_train=False), 
                                           batch_size=BATCH_SIZE) as val_dl:
    
    train_data_loader = iter(train_dl)
    train_batch_count = len(converter_train) // BATCH_SIZE
    
    val_data_loader = iter(val_dl)
    val_batch_count = max(1, len(converter_val) // BATCH_SIZE)
    
    for epoch in range(EPOCHS):
      print('-' * 10)
      print('Epoch {:>3}/{}'.format(epoch + 1, EPOCHS))
      

      train_loss, train_acc = train(model, criterion, optimizer, exp_lr_scheduler, train_data_loader, device, train_batch_count) 
      val_loss, val_acc = eval(model, criterion, val_data_loader, device, val_batch_count)

  return val_loss

# COMMAND ----------

# Uncomment this to run the base local training

# loss = train_and_eval()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Playground

# COMMAND ----------

class LitModelDistributed(pl.LightningModule):
    def __init__(self, data_dir=DATA_DIR, class_count=CLASS_COUNT, lr=LR):
        super().__init__()
        self.lr = lr
        self.model = get_model(class_count, lr)
        self.train_dataloader_context = None
        self.state = {"epochs": 0, "batches": 0}
        self.base_optimiser = torch.optim.SGD(self.model.classifier[1].parameters(), lr=self.lr * hvd.size(), momentum=0.9)
        
    def configure_optimizers(self):
        return hvd.DistributedOptimizer(self.base_optimiser, named_parameters=self.model.named_parameters())
    
    def forward(self, x):
        x = self.model(x)
        return x
      
    def training_step(self, batch, batch_idx):
        X, y = batch['features'], batch['cls_id']
        pred = self(X)
        loss = F.cross_entropy(pred, y)
        acc = FM.accuracy(pred, y)
        print(f'\t - batch_idx: {batch_idx}, ({len(y)})')
        print(f'\t - loss: {loss}, acc: {acc}')
        return loss

    def on_epoch_start(self):
        print(f"--> Epoch: {self.state['epochs']}")
        self.state["epochs"] += 1
        
        # First epoch will already have data loader
        if self.state["epochs"] > 1:
          # Need to explicity Enter the context, trainer does not seem to be able to work with a context
          self.trainer.train_dataloader = self.train_dataloader()
    
    def on_epoch_end(self):
        # Need to reset DataLoader on each epoch. ContextManager is stored in this class, not in `trainer`, destroy it. If new one is needed,
        # it will be created at the start of next epoch
        self.train_dataloader_context.__exit__(None, None, None)
      
    def train_dataloader(self):
        self.train_dataloader_context = converter_train.make_torch_dataloader(transform_spec=get_transform_spec(is_train=True), num_epochs=1, cur_shard=hvd.rank(), shard_count=hvd.size(), batch_size=BATCH_SIZE)
        return self.train_dataloader_context.__enter__()


# COMMAND ----------

def train_and_evaluate_hvd(lr=LR):
  hvd.init()  # Initialize Horovod.
  
  # Horovod: pin GPU to local rank.
  if torch.cuda.is_available():
    torch.cuda.set_device(hvd.local_rank())
    device = torch.cuda.current_device()
  else:
    device = torch.device("cpu")
  
#   trainer = pl.Trainer(
#     gpus=GPUS,
#     max_epochs=EPOCH_COUNT,
#     progress_bar_refresh_rate=20,
#     reload_dataloaders_every_n_epochs=1,  # Need this, otherwise process will fail trying to sample from closed DataLoader
#   )

  model = LitModelDistributed()
  
  # Broadcast initial parameters so all workers start with the same parameters.
  hvd.broadcast_parameters(model.state_dict(), root_rank=0)
  hvd.broadcast_optimizer_state(model.base_optimiser, root_rank=0)

#   trainer.fit(lit_model)

# COMMAND ----------

# hr = HorovodRunner(np=1)   
# hr.run(train_and_evaluate_hvd)

# COMMAND ----------

# MAGIC %pip install deepspeed

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -la /tmp

# COMMAND ----------

# MAGIC %sh
# MAGIC env

# COMMAND ----------

script = """
import io
import numpy as np
from functools import partial

from PIL import Image

import torch, torchvision
from torch import nn
import torch.nn.functional as F
import torchmetrics.functional as FM
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from pyspark.sql.functions import col
from pyspark.sql.types import LongType

from petastorm import TransformSpec
from petastorm.spark import SparkDatasetConverter, make_spark_converter

from pyspark.conf import SparkConf
from pyspark.context import SparkContext



conf = SparkConf()
conf.setAppName("Databricks Shell")
conf.setMaster("local[*, 4]").setAppName("Databricks Shell")
#sc = SparkContext.getOrCreate(conf=conf)

#sc = SparkContext.getOrCreate()

from pyspark.sql import SparkSession
spark = SparkSession.builder.config(conf=conf).getOrCreate()

DATA_DIR = '/databricks-datasets/flowers/delta'
GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if GPUS else 64
EPOCH_COUNT = 3
DATA_SET_LIMIT = 100
WORKER_COUNT = 1
LR = 0.001
CLASS_COUNT = 5

df = spark.read.format("delta").load(DATA_DIR).select(["content", "label"]).limit(DATA_SET_LIMIT)
classes = list(df.select("label").distinct().toPandas()["label"])

# Add a numeric class colunmn
def to_class_index(class_name):
  return classes.index(class_name)

class_index = udf(to_class_index, LongType())
df = df.withColumn("cls_id", class_index(col("label"))).drop("label")

df_train, df_val = df.randomSplit([0.9, 0.1], seed=12345)

# Make sure the number of partitions is at least the number of workers which is required for distributed training.
df_train = df_train.repartition(WORKER_COUNT)
df_val = df_val.repartition(WORKER_COUNT)

print(f'Loaded {df.count()} records (train: {df_train.count()}, val: {df_val.count()})')
print(f'Labels ({CLASS_COUNT}): {classes}')

# Set a cache directory on DBFS FUSE for intermediate data.
CACHE_DIR = "file:///dbfs/tmp/petastorm/cache"
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, CACHE_DIR)

converter_train = make_spark_converter(df_train)
converter_val = make_spark_converter(df_val)

def get_model(class_count, lr):
  model = torchvision.models.mobilenet_v2(pretrained=True)
  
  # Freeze parameters in the feature extraction layers and replace the last layer
  for param in model.parameters():
    param.requires_grad = False

  # New modules have `requires_grad = True` by default
  model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, class_count)
  
  return model

def transform_row(is_train, batch):
  transformers = [transforms.Lambda(lambda x: Image.open(io.BytesIO(x)))]
  if is_train:
    transformers.extend([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
    ])
  else:
    transformers.extend([
      transforms.Resize(256),
      transforms.CenterCrop(224),
    ])
  transformers.extend([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])
  
  trans = transforms.Compose(transformers)
  
  batch['features'] = batch['content'].map(lambda x: trans(x).numpy())
  batch = batch.drop(labels=['content'], axis=1)
  return batch

def get_transform_spec(is_train=True):
  # The output shape of the `TransformSpec` is not automatically known by petastorm, 
  # so you need to specify the shape for new columns in `edit_fields` and specify the order of 
  # the output columns in `selected_fields`.
  return TransformSpec(partial(transform_row, is_train), 
                       edit_fields=[('features', np.float32, (3, 224, 224), False)], 
                       selected_fields=['features', 'cls_id'])

class LitModel(pl.LightningModule):
    def __init__(self, data_dir=DATA_DIR, class_count=CLASS_COUNT, lr=LR):
        super().__init__()
        self.lr = lr
        self.model = get_model(class_count, lr)
        self.train_dataloader_context = None
        self.state = {"epochs": 0, "batches": 0}
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.classifier[1].parameters(), lr=self.lr, momentum=0.9)
        return optimizer
    
    def forward(self, x):
        x = self.model(x)
        return x
      
    def training_step(self, batch, batch_idx):
        X, y = batch['features'], batch['cls_id']
        pred = self(X)
        loss = F.cross_entropy(pred, y)
        acc = FM.accuracy(pred, y)
        print(f'\t - batch_idx: {batch_idx}, ({len(y)})')
        print(f'\t - loss: {loss}, acc: {acc}')
        return loss

    def on_epoch_start(self):
        print(f"--> Epoch: {self.state['epochs']}")
        self.state["epochs"] += 1
        
        # First epoch will already have data loader
        if self.state["epochs"] > 1:
          # Need to explicity Enter the context, trainer does not seem to be able to work with a context
          self.trainer.train_dataloader = self.train_dataloader()
    
    def on_epoch_end(self):
        # Need to reset DataLoader on each epoch. ContextManager is stored in this class, not in `trainer`, destroy it. If new one is needed,
        # it will be created at the start of next epoch
        self.train_dataloader_context.__exit__(None, None, None)
      
    def train_dataloader(self):
        self.train_dataloader_context = converter_train.make_torch_dataloader(transform_spec=get_transform_spec(is_train=True), num_epochs=1, batch_size=BATCH_SIZE)
        return self.train_dataloader_context.__enter__()
      
trainer = pl.Trainer(
    gpus=GPUS, 
    max_epochs=EPOCH_COUNT,
    progress_bar_refresh_rate=20,
    reload_dataloaders_every_n_epochs=1,  # Need this, otherwise process will fail trying to sample from closed DataLoader
    strategy="deepspeed_stage_1"
)

lit_model = LitModel()
trainer.fit(lit_model)
"""
  

# COMMAND ----------

script_path = "/tmp/train.py"
with open(script_path, "w") as fout:
  fout.write(script)

# COMMAND ----------

# MAGIC %sh
# MAGIC #/databricks/python/bin/pip install pytorch_lightning

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC /databricks/python3/bin/python3 /tmp/train.py

# COMMAND ----------

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext.getOrCreate()


# COMMAND ----------

sc.getConf().get("spark.master")
# sc.getConf().get("spark.app.name")


# COMMAND ----------


