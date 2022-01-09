# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Distributed Training on Databricks Platform using Pytorch Lightning, Petastorm and Horovod
# MAGIC 
# MAGIC This notebook walks through fine tuning a series of image classifiers, starting from a single device implementation all the way through to multi-node multi-device implementation capable of handling a large-scale model training. 
# MAGIC 
# MAGIC ***Disclaimer:** Main purpose of this notebook is to show how Pytorch Lightning can be used for distributed training on Databricks Platform. It is not about training a best model so we may not necessarily follow the best practices here, e.g. we keep learning rate the same even though we use larger batches in multi-GPU training.*

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Setup
# MAGIC 
# MAGIC First part of this notebook can be executed on a single node CPU cluster. A GPU cluster will be required to run a second part; minimum GPU requirements are clearly defined for each section, some cell can be run on a single GPU single node, some will work on a multi-GPU single node cluster and the rest will require multi-node GPU cluster)
# MAGIC 
# MAGIC This notebook was developed on a Databrick Runtime 10.1 with the foolowing libraries:
# MAGIC - torch: 1.9.1+cu111
# MAGIC - torchvision: 0.10.1+cu111
# MAGIC - pytorch_lightning: 1.5.8
# MAGIC - CUDA: 11.4 (`nvidia-smi`)
# MAGIC - Horovod: 0.23.0
# MAGIC 
# MAGIC ***We recommend running this notebook in your own worspace, not the Repos, even if it was cloned from GitHub (File -> Clone and save to your workspace). We've encountered errors when running this notebook directly from the Repos.***
# MAGIC   

# COMMAND ----------

# MAGIC %pip install pytorch-lightning

# COMMAND ----------

import io
import numpy as np
from functools import partial
import datetime as dt
import logging

from PIL import Image

import torch, torchvision
from torch import nn
import torch.nn.functional as F
import torchmetrics.functional as FM
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar

from pyspark.sql.functions import col
from pyspark.sql.types import LongType

from petastorm import TransformSpec
from petastorm.spark import SparkDatasetConverter, make_spark_converter

print(f"Using:\n - torch: {torch.__version__}\n - torchvision: {torchvision.__version__}\n - pytorch_lightning: {pl.__version__}")

# COMMAND ----------

IS_DEV = True

DATA_DIR = "/databricks-datasets/flowers/delta"
GPU_COUNT = torch.cuda.device_count()
print(f"Found {GPU_COUNT if GPU_COUNT > 0 else 'no'} GPUs")

MAX_DEVICE_COUNT_TO_USE = 2

BATCH_SIZE = 64
MAX_EPOCH_COUNT = 15
STEPS_PER_EPOCH = 15

LR = 0.001
CLASS_COUNT = 5

SAMPLE_SIZE = 1000
print(f"Sample: size {SAMPLE_SIZE}")

EARLY_STOP_MIN_DELTA = 0.05
EARLY_STOP_PATIENCE = 3

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
      run_time = "{} hours {} minutes".format(int(h), int(m))
  elif m > 0:
      run_time = "{} minutes {} seconds".format(int(m), int(s))
  else:
      run_time = "{} seconds".format(int(s))

  msg = f"\n-- {action} completed in ***{run_time}*** at {end.strftime('%Y-%m-%d %H:%M:%S')}\n\n---------------------"
  print(msg)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Data - the flowers dataset
# MAGIC 
# MAGIC In this notebook we are using the flowers dataset from the TensorFlow team. Whereas the original dataset consists of flower photos stored under five sub-directories, one per class, here we are using a pre-processed dataset stored in Delta format.
# MAGIC 
# MAGIC This dataset contains ***3670*** images. To reduce the running time, we are using a smaller subset of the dataset for development and testing purposes in this notebook. 
# MAGIC 
# MAGIC ***Note:** The original/unprocced dataset can be also be used in the same ways, it is hosted under Databricks Datasets `dbfs:/databricks-datasets/flower_photos` for easy access.*

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

  train_df, val_df = df.randomSplit([0.8, 0.2], seed=12)

  # For distributed training data must have at least as many many partitions as the number of devices/processes
  train_df = train_df.repartition(MAX_DEVICE_COUNT_TO_USE)
  val_df = val_df.repartition(MAX_DEVICE_COUNT_TO_USE)

  print(f"Dataset size: {df.count()} (train: {train_df.count()}, val: {val_df.count()})")
  print(f"Labels ({CLASS_COUNT}): {classes}")
  if sample_size > 0:
    display(train_df.limit(10))
  
  return train_df, val_df

def create_spark_converters(use_sample=True):
  train_df, val_df = get_data(SAMPLE_SIZE if use_sample else -1)
  return make_spark_converter(train_df), make_spark_converter(val_df)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Dataframe and corresponding Petastorm wrappers need to be created here instead of inside the Pytorch Lightning model class. This is especially important for distributed training when model class instances will be created in other nodes where Spark context is not defined (Petastorm spark converter can be pickled) 

# COMMAND ----------

train_sample_converter, val_sample_converter = create_spark_converters(True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### The Model
# MAGIC 
# MAGIC The model class may look large but I deliberately piled everything into it to show how the most part of the model logic can be encapsulated into a single class
# MAGIC 
# MAGIC A special note about the value of parameter `num_epochs` used in `make_torch_dataloader` function. We set it to `None` (it is also a default value) to generate infinite batches of data to avoid handling the last incomplete batch. This is especially important for distributed training where we need to guarantee that the numbers of data records seen on all workers are identical per step. Given that the length of each data shard may not be identical, setting `num_epochs` to any specific number would fail to meet the guarantee and will likely result in an error.
# MAGIC 
# MAGIC Even though this is not really important for single device training, it determines the way we control epochs (training will run forever on infinite dataset which means there would be only 1 if other controls are not used), so we decided to introduce it from the beginning.
# MAGIC 
# MAGIC There is another place when this is important - the validation process. Pytorch Lightning Trainer, by default, will run a sanity validation check prior to any training, unless instructed otherwise (i.e. `num_sanity_val_steps` is set to be equal to `0`). That sanity check will initialise the validation data loader and will use those few initial batches defined by `num_sanity_val_steps`. However, because of doing so it will load the validation data loader before the first epoch but will not do so for the validation phase of the first epoch which will result in error (an attempt to read a second time from data loader which was not completed in the previous attempt). Possible workarounds is using a finite amount of epochs in `num_epochs` (e.g. `num_epochs=1` as there is no point in evaluating on repeated dataset), which is not good as it will likely result in a last batch being smaller than other batches. At the time of creating this notebook there is no way to set an equivalent of `drop_last` for the Data Loader created by `make_torch_dataloader`. The only way to work around this is to avoid doing any sanity checks (i.e. setting `num_sanity_val_steps=0`, setting it to anything else doesn't work) and using `limit_val_batches` parameter of the Trainer class.
# MAGIC 
# MAGIC A separate callback class can be used for sidecar operations like logging, etc but it is easier to keep all this within the model

# COMMAND ----------

class LitClassificationModel(pl.LightningModule):
  def __init__(self, train_converter, val_converter, class_count=CLASS_COUNT, lr=LR, logging_level=logging.INFO, device_id=0, device_count=1):
    super().__init__()
    self.lr = lr
    self.model = self.get_model(class_count, lr)
    self.train_converter = train_converter
    self.val_converter = val_converter
    self.train_dataloader_context = None
    self.val_dataloader_context = None
    self.state = {"epochs": 0}
    self.logging_level = logging_level
    self.device_id = device_id
    self.device_count = device_count

  def configure_optimizers(self):
    optimizer = torch.optim.SGD(self.model.classifier[1].parameters(), lr=self.lr, momentum=0.9)
    return optimizer

  def forward(self, x):
    x = self.model(x)
    return x
  
  def training_step(self, batch, batch_idx):
    X, y = batch["features"], batch["cls_id"]
    pred = self(X)
    loss = F.cross_entropy(pred, y)
    
    # Choosing to use step loss as a metric. Use the next (commented out) line instead if averaged epoch loss if preferred
    # self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    self.log("train_loss", loss, prog_bar=True)
    
    if self.logging_level == logging.DEBUG:
      if batch_idx == 0:
        print(f" - [{self.device_id}] training batch size: {y.shape[0]}")
      print(f" - [{self.device_id}] training batch: {batch_idx}, loss: {loss}")
      
    return loss

  def training_step_end(self, training_step_outputs):
    if self.logging_level == logging.DEBUG:
      print(f" - [{self.device_id}] training step output: {training_step_outputs.item()}")
  
  def training_epoch_end(self, training_step_outputs):
    if self.logging_level == logging.DEBUG:
      print(f" - [{self.device_id}] training epoch output: {training_step_outputs}")
    
  def on_train_epoch_start(self):
    # No need to re-load data here as `train_dataloader` will be called on each epoch
    if self.logging_level in (logging.DEBUG, logging.INFO):
      print(f"++ [{self.device_id}] Epoch: {self.state['epochs']}")
    self.state["epochs"] += 1

  def train_dataloader(self):
    return self.get_dataloader_context(is_train=True).__enter__()
    
  def validation_step(self, batch, batch_idx):
    X, y = batch["features"], batch["cls_id"]
    pred = self(X)
    loss = F.cross_entropy(pred, y)
    acc = FM.accuracy(pred, y)

    # Roll validation up to epoch level
    self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
    self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
    
    if self.logging_level == logging.DEBUG:
      print(f" - [{self.device_id}] val batch: {batch_idx}, size: {y.shape[0]}, loss: {loss}, acc: {acc}")

    return {"loss": loss, "acc": acc}

  def val_dataloader(self):
    return self.get_dataloader_context(is_train=False).__enter__()

  def on_train_end(self):
    # Close all readers as a best practice (plus actually helps avoid failure of the distributed training)
    self.train_dataloader_context.__exit__(None, None, None)
    self.val_dataloader_context.__exit__(None, None, None)
    
  def get_dataloader_context(self, is_train=True):
    if self.logging_level == logging.DEBUG:
      print(f" - [{self.device_id}] get_dataloader_context({'Train' if is_train else 'Val'})")
    
    # To improve performance, do the data transformation in a TransformSpec function in petastorm instead of Spark Dataframe
    if is_train:
      if self.train_dataloader_context:
        self.train_dataloader_context.__exit__(None, None, None)
      self.train_dataloader_context = self.train_converter.make_torch_dataloader(transform_spec=self._get_transform_spec(is_train=True), num_epochs=None,
                                                                                 cur_shard=self.device_id, shard_count=self.device_count, batch_size=BATCH_SIZE*self.device_count)
      return self.train_dataloader_context
    else:
      # https://petastorm.readthedocs.io/en/latest/_modules/petastorm/spark/spark_dataset_converter.html#SparkDatasetConverter.make_torch_dataloader
      if self.val_dataloader_context:
        self.val_dataloader_context.__exit__(None, None, None)
      self.val_dataloader_context = self.val_converter.make_torch_dataloader(transform_spec=self._get_transform_spec(is_train=False), num_epochs=None, 
                                                                             cur_shard=self.device_id, shard_count=self.device_count,  batch_size=BATCH_SIZE*self.device_count)
      return self.val_dataloader_context
  
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

    batch["features"] = batch["content"].map(lambda x: trans(x).numpy())
    batch = batch.drop(labels=["content"], axis=1)
    return batch

  def _get_transform_spec(self, is_train=True):
    # The output shape of the `TransformSpec` is not automatically known by petastorm, so we need to specify the shape for new columns in `edit_fields`
    # and specify the order of the output columns in `selected_fields`
    return TransformSpec(partial(self._transform_row, is_train), 
                         edit_fields=[("features", np.float32, (3, 224, 224), False)], 
                         selected_fields=["features", "cls_id"])


# COMMAND ----------

def train(train_converter=train_sample_converter, val_converter=val_sample_converter, gpus=0, strategy=None, device_id=0, device_count=1, logging_level=logging.INFO):
  
  start = dt.datetime.now()

  if device_id == 0:
    print(f"Train on {str(max(gpus, device_count)) + ' GPU' + ('s' if gpus > 1 or device_count > 1 else '') if gpus > 0  else 'CPU'}:")
    print(f" - max epoch count: {MAX_EPOCH_COUNT}\n - batch size: {BATCH_SIZE*device_count}\n - steps per epoch: {STEPS_PER_EPOCH}\n - sample size: {SAMPLE_SIZE}")
    print(f" - start time: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n======================\n")
  
  # Use check_on_train_epoch_end=True to evaluate at the end of each epoch
  verbose = True if device_id == 0 else False
  stopper = EarlyStopping(monitor="val_loss", min_delta=EARLY_STOP_MIN_DELTA, patience=EARLY_STOP_PATIENCE, verbose=verbose, mode='min', check_on_train_epoch_end=True)
  callbacks = [stopper]
  
  # Add checkpointing if needed
  # checkpointer = ModelCheckpoint(monitor='val_loss', mode="min", save_top_k=1, verbose=verbose)
  # callbacks.append(checkpointer)

  # You could also use an additinal progress bar but default progress reporting was sufficient. Uncomment next line if desired
  # callbacks.append(TQDMProgressBar(refresh_rate=STEPS_PER_EPOCH, process_position=0))
  
  # We could use `on_train_batch_start` to control epoch sizes as shown in the link below but it's cleaner when done here with `limit_train_batches` parameter
  # https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/core/hooks.html#ModelHooks.on_train_batch_start
  trainer = pl.Trainer(
      gpus=gpus,
      max_epochs=MAX_EPOCH_COUNT,
      limit_train_batches=STEPS_PER_EPOCH,  # this is the way to end the epoch, otherwise they will continue forever because the Train dataloader is producing infinite number of batches
      log_every_n_steps=1,
      val_check_interval=STEPS_PER_EPOCH,  # this value must be the same as `limit_train_batches`
      num_sanity_val_steps=0,  # this must be zero to prevent a Petastorm error about Data Loader not being read completely
      limit_val_batches=1,  # any value would work here but there is point in validating on repeated set of data so this should be smaller than total number of batches in Validation dataset
      reload_dataloaders_every_n_epochs=1,  # need to set this to 1
      strategy=strategy,
      callbacks=callbacks,
      default_root_dir='/tmp/lightning_logs'
  )

  model = LitClassificationModel(train_converter, val_converter, device_id=device_id, device_count=device_count, logging_level=logging_level)
  trainer.fit(model)

  if device_id == 0:
    report_duration(f"Training", start)
  
  return model.model if device_id == 0 else None

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### CPU Training

# COMMAND ----------

cpu_model = train(gpus=0)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### CPU Training Results
# MAGIC 
# MAGIC Train on CPU:
# MAGIC  - max epoch count: 15
# MAGIC  - batch size: 64
# MAGIC  - steps per epoch: 15
# MAGIC  - sample size: 1000
# MAGIC  - start time: 2022-01-07 17:39:26
# MAGIC 
# MAGIC ======================
# MAGIC 
# MAGIC Epoch: 12
# MAGIC Monitored metric val_loss did not improve in the last 3 records. Best score: 0.374. Signaling Trainer to stop.
# MAGIC 
# MAGIC -- Training completed in ***6 minutes 49 seconds*** at 2022-01-07 17:46:15
# MAGIC   
# MAGIC ----------------------

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## GPU Training

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Using 1 GPU
# MAGIC 
# MAGIC ***Note:** This section will require a cluster with a GPU. A single node cluster with a single GPU instance will suffice (e.g. `p3.2xlarge` on AWS or equivalent on other cloud providers)*

# COMMAND ----------

gpu_model = train(gpus=1)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### GPU Training Results
# MAGIC 
# MAGIC Train on 1 GPU:
# MAGIC  - max epoch count: 15
# MAGIC  - batch size: 64
# MAGIC  - steps per epoch: 15
# MAGIC  - sample size: 1000
# MAGIC  - start time: 2022-01-07 17:48:14
# MAGIC 
# MAGIC ======================
# MAGIC 
# MAGIC Epoch: 12
# MAGIC Monitored metric val_loss did not improve in the last 3 records. Best score: ***0.399***. Signaling Trainer to stop.
# MAGIC 
# MAGIC -- Training completed in ***4 minutes 34 seconds*** at 2022-01-07 17:52:48
# MAGIC 
# MAGIC ---------------------
# MAGIC 
# MAGIC ***Observations:** full training/validation cycle on GPU is faster than CPU (some runs showed up 2 times speedup), whereas similar training runs without the validation steps showed more that 3 times speedup on GPU*

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Using multiple GPUs
# MAGIC 
# MAGIC Simply passing a value of `gpus` parameter greater than 1 to a Trainer won't work, we also need to have a way to syncronise the losses between different models (each GPU would have its own model to train). This could be done by specifying a `strategy` parameter of the Trainer. According to docs, `If you request multiple GPUs or nodes without setting a mode, DDP Spawn will be automatically used.` [https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#distributed-modes]), however just giving a `gpus` value greater than 1 to the Trainer (e.g. `train(gpus=2)`) throws an error. It maybe worth checking why.
# MAGIC 
# MAGIC So we need to specify a strategy, e.g. `strategy=HorovodPlugin()`, but it produces the same loss within the same execution time as when using a single GPU, which draws a conclusion that only a single GPU gets used anyways. (this is also supported by output logs from only one process in verbose logging mode, but that could be because logging in other processes spawned for GPUs other then the 1st GPU are not shown in this process)
# MAGIC 
# MAGIC ***Note/TODO:*** strangely, it complains about interactive mode if using `strategy="horovod"` but works with `strategy=HorovodPlugin()`. Maybe using an instance of DDP plugin instead of `strategy="ddp"` will also work, worth checking.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Distributed Training
# MAGIC 
# MAGIC Since we are using PyTorch here, PyTorch DDP (Distributed Data Parallel) would be a logical choice for scaling out the training to multiple GPUs (as opposed to DP which not recommended by PyTorch), but DDP does not work in interactive mode (e.g. notebooks) so it would have to be executed as a standalone script (which in turn seems to be problematic due to missing Spark context on standalone script executed in Notebook). Furthermore, I don't know if I'll be able to use Petastorm with DDP (DDP itself does splitting and distribution of the data, I doubt it works on top of Petastom Spark Converter - something here to review: https://github.com/PyTorchLightning/pytorch-lightning/issues/5074) 
# MAGIC 
# MAGIC Then there is Horovod which works in interactive mode (e.g. otebooks), which brings the following options to explore:
# MAGIC 
# MAGIC 1. PyTorch Lightning has a `HorovodPlugin` plugin, we can use it with a `horovod` runner
# MAGIC     - https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/plugins/training_type/horovod.html
# MAGIC     - https://github.com/PyTorchLightning/pytorch-lightning/blob/1fc046cde28684316323693dfca227dbd270663f/tests/models/test_horovod.py
# MAGIC     - looks like pretty easy to run on a single node but how about multi-node - do I need to do some further configurations or will `horovod` have it automatically because it's pre-installed?
# MAGIC     - *an interesting observation, using `strategy="horovod"` without using the `horovod` runner complains about interactive mode but `strategy=HorovodPlugin()` does not*
# MAGIC 1. Use `sparkdl.HorovodRunner`
# MAGIC     - https://databricks.com/blog/2018/11/19/introducing-horovodrunner-for-distributed-deep-learning-training.html
# MAGIC     - this option will also make use of `HorovodPlugin`
# MAGIC     - `sparkdl.HorovodRunner` will "understand" a Spark cluster so no need to configure the additional cluster settings
# MAGIC 1. Use `horovod.spark.lightning` as `hvd` with `hvd.TorchEstimator` and `horovod.spark.common.backend.SparkBackend`
# MAGIC     - https://github.com/horovod/horovod/blob/master/docs/spark.rst
# MAGIC     - https://github.com/horovod/horovod/blob/master/examples/spark/pytorch/pytorch_lightning_spark_mnist.py
# MAGIC     - this is Spark "aware", no need to furher configure clusters
# MAGIC     - SparkBackend does splitting of a given dataset into multiple jobs, can I use a tiny fake dataset to do the splitting but still train using Petastorm? Do I have access to information on what device each process is running on?
# MAGIC     

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Option 1: Use Horovod
# MAGIC 
# MAGIC ***Note:** This section will require a cluster with multiple GPUs, a single node cluster with multiple GPUs instace will be sufficient (e.g. p3.8xlarge on AWS or equivalent on other cloud providers)*
# MAGIC 
# MAGIC This only works on a single machine. An attempt to train using more GPUs than available on the driver node fails, even if there are worker nodes in the cluster that have those extra GPUs. This could likely be fixed by configuring Horovod to recognise worker nodes but we are interested in using the clusters as they are without extra configuration so we'll not explore this option for multi-node GPU training.

# COMMAND ----------

from pytorch_lightning.utilities import _HOROVOD_AVAILABLE
if _HOROVOD_AVAILABLE:
  print("Horovod is available")
  import horovod
  import horovod.torch as hvd
  from pytorch_lightning.plugins.training_type.horovod import HorovodPlugin
  print(f"Horovod: {horovod.__version__}")


# COMMAND ----------

def train_hvd():
  hvd.init()
  return train(gpus=1, strategy="horovod", device_id=hvd.rank(), device_count=hvd.size())
  

# COMMAND ----------

hvd_model = horovod.run(train_hvd, np=MAX_DEVICE_COUNT_TO_USE)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 2 GPUs on Single Node Training Results
# MAGIC 
# MAGIC Train on 1 GPU:
# MAGIC  - max epoch count: 15
# MAGIC  - batch size: 128
# MAGIC  - steps per epoch: 15
# MAGIC  - sample size: 1000
# MAGIC  - start time: 2022-01-07 18:54:39
# MAGIC 
# MAGIC ======================
# MAGIC 
# MAGIC Epoch 8: 100% 16/16 [00:16<00:00,  1.05s/it, loss=0.33, v_num=0, train_loss=0.295, ***val_loss=0.337***, val_acc=0.910][1,0]
# MAGIC 
# MAGIC -- Training completed in ***3 minutes 23 seconds*** at 2022-01-07 22:20:30
# MAGIC 
# MAGIC 
# MAGIC ***Observations:*** 
# MAGIC   - we get a better training time than single GPU training
# MAGIC   - as hoped for, running on 2 GPUs gives a better loss than we can achieve using the same dataset on 1 GPU. It gets even better if we increase the number of GPUs to 4 (<= 0.3 on each GPU) so the optimiser/loss syncronisation between different Trainer instances must be working
# MAGIC   - we are getting a better progress bar reporting in this training, must be Horovod's doing

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Option 2: Use `sparkdl.HorovodRunner`
# MAGIC 
# MAGIC ***Note:** This section will require a multi-node GPU cluster, a cluster with 2 workers, each with a single GPU will be sufficient (e.g. p3.2xlarge on AWS or equivalent on other cloud providers)*
# MAGIC 
# MAGIC Here we are jumping straight into the multi-node distributed training

# COMMAND ----------

from sparkdl import HorovodRunner

# COMMAND ----------

# This code is failing when executed in a notebook in the Repos, run it in your own workspace if so (File->Clone)
hr = HorovodRunner(np=MAX_DEVICE_COUNT_TO_USE, driver_log_verbosity='all')
spark_hvd_model = hr.run(train_hvd)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 2 GPUs on 2 Nodes Training Results
# MAGIC 
# MAGIC Train on 2 GPUs:
# MAGIC  - max epoch count: 15
# MAGIC  - batch size: 128
# MAGIC  - steps per epoch: 15
# MAGIC  - sample size: 1000
# MAGIC  - start time: 2022-01-09 17:22:16
# MAGIC 
# MAGIC ======================
# MAGIC 
# MAGIC Epoch 8: 100% 16/16 [00:19<00:00,  1.20s/it, loss=0.332, v_num=0, train_loss=0.310, ***val_loss=0.356***, val_acc=0.910][1,0]
# MAGIC 
# MAGIC -- Training completed in ***3 minutes 45 seconds*** at 2022-01-09 17:26:02
# MAGIC 
# MAGIC ---------------------
# MAGIC 
# MAGIC ***Observations:*** 
# MAGIC   - training time and resulting loss are similar to training with 2 GPUs on the same node, but this time it was accross 2 nodes

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## [TODO] Option 3: Use `TorchEstimator` with `SparkBackend`
# MAGIC 
# MAGIC - https://github.com/horovod/horovod/blob/master/docs/spark.rst
# MAGIC - https://github.com/horovod/horovod/blob/976a87958d48b4359834b33564c95e808d005dab/examples/spark/pytorch/pytorch_lightning_spark_mnist.py#L194

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Clean Up

# COMMAND ----------

train_sample_converter.delete()
val_sample_converter.delete()

# COMMAND ----------

# MAGIC %md 
# MAGIC # Appendix

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Option 4: Using PyTorch Distributed Data Parallel
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

# MAGIC %md
# MAGIC 
# MAGIC #### `num_epochs` parameter of `make_torch_dataloader`
# MAGIC 
# MAGIC https://github.com/uber/petastorm/blob/master/petastorm/spark/spark_dataset_converter.py
# MAGIC 
# MAGIC If setting `num_epochs=1` and  training for multiple epochs then only a first epoch seem to be getting a correct number of records within an epoch, with last incomplete batch size doubling for each epoch
# MAGIC 
# MAGIC Example: training with BATCH_SIZE=64 and 92 records in training dataset
# MAGIC     - training for 1 epoch
# MAGIC         - epoch 0
# MAGIC             - batch 0: 64 records
# MAGIC             - batch 1: 28 records <- OK
# MAGIC     - training for 2 epochs
# MAGIC         - epoch 0
# MAGIC             - batch 0: 64 records
# MAGIC             - batch 1: 28 records
# MAGIC         - epoch 1 (extra 28 records)
# MAGIC             - batch 2: 64 records
# MAGIC             - batch 3: 56 records <-- Doubled (+28)
# MAGIC     - training for 3 epochs
# MAGIC         - *exactly like epochs 0 and 1 from training with 2 epochs*
# MAGIC         - epoch 2
# MAGIC             - batch 4: 64 records
# MAGIC             - batch 5: 64 records  <-- 56+8
# MAGIC             - batch 6: 20 records  <-- +20 (spill from +28 to previous batch)
# MAGIC     - training for 4 epochs
# MAGIC         - *exactly like epochs 0, 1 and 2 from training with 3 epochs*
# MAGIC         - epoch 3
# MAGIC             - batch 7: 64 records
# MAGIC             - batch 8: 48 records  <-- this looks like 20+28 but where is second batch of 64 gone? 
# MAGIC             
# MAGIC *This pattern of adding 28 to the last batch continues for subsequent epochs if larger number of epochs is used for training*
# MAGIC 
# MAGIC As can be seen in training with 4 epochs, there are also some occational drops of previosly used by overfilled batches
# MAGIC 
# MAGIC A fix is to reset the dataset on every epoch, but that's not a straightforward process because `make_torch_dataloader` returns a Context Manager, so cannot use the `with converter_train.make_torch_dataloader(...)` in the `train_dataloader` as it destroys the context on exit which means taking the data loader outside the lightning model class. We chose explicitly calling `__enter__` explicitly on the Context Manager within the model
# MAGIC           

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Test `mpi`
# MAGIC 
# MAGIC Add a cell after the next one if want to try with the following content and run both cells (may also need to be done outside Repos)
# MAGIC ```
# MAGIC %sh
# MAGIC 
# MAGIC mpicc -o test_mpi test_mpi.c
# MAGIC mpirun --allow-run-as-root -np 2 ./test_mpi
# MAGIC ```

# COMMAND ----------

script = """#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("Hello\\n");

    MPI_Finalize();

    return 0;
}"""

exe_path = "test_mpi.c"
with open(exe_path, "w") as fout:
  fout.write(script)

# COMMAND ----------

# MAGIC %md
