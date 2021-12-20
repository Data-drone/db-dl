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

DATA_DIR = '/databricks-datasets/flowers/delta'
GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if GPUS else 64
EPOCH_COUNT = 3
DATA_SET_LIMIT = 100
WORKER_COUNT = 1
LR = 0.001
CLASS_COUNT = 5

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Data - the flowers dataset
# MAGIC 
# MAGIC This example uses the flowers dataset from the TensorFlow team, which contains flower photos stored under five sub-directories, one per class. It is hosted under Databricks Datasets dbfs:/databricks-datasets/flower_photos for easy access.
# MAGIC 
# MAGIC The example loads the flowers table which contains the preprocessed flowers dataset using the binary file data source. It uses a small subset of the flowers dataset to reduce the running time of this notebook. When you run this notebook, you can increase the number of images used for better model accuracy.

# COMMAND ----------

df = spark.read.format("delta").load(DATA_DIR).select(["content", "label"]).limit(DATA_SET_LIMIT)
classes = list(df.select("label").distinct().toPandas()["label"])
assert CLASS_COUNT == len(classes)

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
display(df.limit(10))

# COMMAND ----------

# MAGIC %sh
# MAGIC rm -r /dbfs/tmp/petastorm/cache

# COMMAND ----------

# Set a cache directory on DBFS FUSE for intermediate data.
CACHE_DIR = "file:///dbfs/tmp/petastorm/cache"
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, CACHE_DIR)

converter_train = make_spark_converter(df_train)
converter_val = make_spark_converter(df_val)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Preprocess images
# MAGIC Before feeding the dataset into the model, you need to decode the raw image bytes and apply standard ImageNet transforms. Databricks recommends not doing this transformation on the Spark DataFrame since that substantially increases the size of the intermediate files and might decrease performance. Instead, do this transformation in a TransformSpec function in petastorm.

# COMMAND ----------

def transform_row(is_train, batch):
  """
  The input and output of this function must be pandas dataframes.
  Do data augmentation for the training dataset only.
  """
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

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Train PyTorch Classifier

# COMMAND ----------

# Check the selected model structure
model = torchvision.models.mobilenet_v2(pretrained=True)
model

# COMMAND ----------

def get_model(class_count, lr):
  model = torchvision.models.mobilenet_v2(pretrained=True)
  
  # Freeze parameters in the feature extraction layers and replace the last layer
  for param in model.parameters():
    param.requires_grad = False

  # New modules have `requires_grad = True` by default
  model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, class_count)
  
  return model

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
# MAGIC ### Wrap the model into PyTorch Lightning

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC A callback class like this can be used for things like logging but it is easier to keep all this within the model. If needed, can be as this
# MAGIC ```
# MAGIC trainer = pl.Trainer(..., callbacks=[LitCallback()])
# MAGIC ```

# COMMAND ----------

from pytorch_lightning.callbacks import Callback

class LitCallback(Callback):
    def __init__(self):
        self.state = {"epochs": 0, "batches": 0}
    
    def on_epoch_start(self, trainer, pl_module):
        print(f"--> Epoch: {self.state['epochs']}")
        self.state["epochs"] += 1
          
    def on_batch_start(self, trainer, pl_module):
        print(f"\t- batch: {self.state['batches']}")
        self.state["batches"] += 1
        

# COMMAND ----------

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
        
        # Need to explicity Enter the context, trainer does not seem to be able to work with a context
        self.trainer.train_dataloader = self.train_dataloader().__enter__()
    
    def on_epoch_end(self):
        # Need to reset DataLoader on each epoch. ContextManager is stored in this class, not in `trainer`, destroy it. If new one is needed,
        # it will be created at the start of next epoch
        self.train_dataloader_context.__exit__(None, None, None)
    
    def train_dataloader(self):
        self.train_dataloader_context = converter_train.make_torch_dataloader(transform_spec=get_transform_spec(is_train=True), num_epochs=1, batch_size=BATCH_SIZE)
        return self.train_dataloader_context

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

trainer = pl.Trainer(
    gpus=GPUS,
    max_epochs=EPOCH_COUNT,
    progress_bar_refresh_rate=20,
    reload_dataloaders_every_n_epochs=1,  # Need this, otherwise process will fail trying to sample from closed DataLoader
)

lit_model = LitModel()
trainer.fit(lit_model)

# COMMAND ----------

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

# MAGIC %md
# MAGIC 
# MAGIC ## Distributed Training with Horovod

# COMMAND ----------


