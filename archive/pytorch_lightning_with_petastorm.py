# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Distributed Training on Databricks Platform using Pytorch Lightning, Petastorm and Horovod
# MAGIC 
# MAGIC This notebook demonstrates various ways of fine-tuning an image classifier, starting from a single device model training through to multi-node multi-device implementation capable of handling a large-scale model training. 
# MAGIC 
# MAGIC ***Disclaimer:*** *The primary purpose of this notebook is to show the use of Pytorch Lightning on a Databricks Platform. It is not about training the best model, so we may not necessarily follow the best practices here, e.g. we keep a learning rate parameter the same even though we use larger batches in multi-GPU training.*

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Setup
# MAGIC 
# MAGIC First part of this notebook can be executed on a single node CPU cluster. A cluster with GPU instances will be required to run a second part, with minimum GPU requirements defined for each section. (Some cells can be run on a single node cluster with a single GPU instace, some can be run on a single node cluster with a multi-GPU instace and the rest will require a multi-node cluster with GPU instances.)
# MAGIC 
# MAGIC This notebook was developed on a Databrick Runtime **10.2** with the foolowing libraries:
# MAGIC - torch: 1.10.1+cu111
# MAGIC - torchvision: 0.11.1+cu111
# MAGIC - pytorch_lightning: 1.5.9
# MAGIC - CUDA: 11.4 (`nvidia-smi`)
# MAGIC - Horovod: 0.23.0
# MAGIC 
# MAGIC ***If you encounter errors when running the notebook from within the Repos then clone the notebook to your worspace and run it there (File -> Clone and save to your workspace).***
# MAGIC   

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC *At the time when this notebook was developed, Pythorch Lightning did not come pre-installed on Databricks and needed to be installed*

# COMMAND ----------

# MAGIC %pip install pytorch-lightning

# COMMAND ----------

import io
import numpy as np
from functools import partial
import datetime as dt
import logging

from PIL import Image

import pandas as pd

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

import pyspark.sql.types as T
from pyspark.sql.functions import col, pandas_udf, PandasUDFType

from petastorm import TransformSpec
from petastorm.spark import SparkDatasetConverter, make_spark_converter

print(f"Using:\n - torch: {torch.__version__}\n - torchvision: {torchvision.__version__}\n - pytorch_lightning: {pl.__version__}")

# COMMAND ----------

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

  msg = f"{action} completed in ***{run_time}***"
  print(msg)
  
def preprocess(img):
  image = Image.open(io.BytesIO(img))
  transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  return transform(image)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Data - the flowers dataset
# MAGIC 
# MAGIC In this notebook we are using the flowers dataset from TensorFlow. While the original dataset contains flower photos stored under five sub-directories, one per class, here we are using a pre-processed dataset stored in Delta format.
# MAGIC 
# MAGIC This dataset contains few thousand images. To reduce the running time, we are using a smaller subset of the dataset for development and testing purposes in this notebook. 
# MAGIC 
# MAGIC **Note:** *The original/unprocced dataset is available under `dbfs:/databricks-datasets/flower_photos` on Databricks and can also be used in a similar way.*

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC DataFrame and corresponding Petastorm wrappers need to be created here instead of inside the Pytorch Lightning model class. This is especially important for the distributed training when the model class instances will be created in worker nodes where Spark context is not available (Petastorm spark converter can be pickled).

# COMMAND ----------

def get_data(sample_size=-1, split=True, display_sample=True):
  df = spark.read.format("delta").load(DATA_DIR).select(["content", "label"])
  if sample_size > 0:
    df = df.limit(sample_size)
  classes = list(df.select("label").distinct().toPandas()["label"])
  print(f"Labels ({CLASS_COUNT}): {classes}")
  
  # We are using a sample, make sure we get all the classes
  assert CLASS_COUNT == len(classes)

  # Change labels to be numeric
  def to_class_index(class_name):
    return classes.index(class_name)

  class_index = udf(to_class_index, T.LongType())  # This has to be a Long type, Int won't work
  df = df.withColumn("label", class_index(col("label")))
  print(f"Dataset size: {df.count()}")

  val_df = None
  if split:
    train_df, val_df = df.randomSplit([0.8, 0.2], seed=12)

    # For distributed training data must have at least as many many partitions as the number of devices/processes
    train_df = train_df.repartition(MAX_DEVICE_COUNT_TO_USE)
    val_df = val_df.repartition(MAX_DEVICE_COUNT_TO_USE)

    print(f" - split into (train: {train_df.count()}, val: {val_df.count()})")
    df = train_df
  
  if display_sample:
    display(df.limit(10))
  
  return df, val_df

train_df, val_df = get_data(SAMPLE_SIZE)
train_sample_converter, val_sample_converter = make_spark_converter(train_df), make_spark_converter(val_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### The Model
# MAGIC 
# MAGIC The next cell is the largest cell in this notebook but we deliberately piled everything related to model training into the same class. Later in this notebook we'll show how this class can be significantly simplified by laveraging features provided by the training libraries.
# MAGIC 
# MAGIC **[TLDR]**
# MAGIC 
# MAGIC A special note about the value of parameter `num_epochs` used in `make_torch_dataloader` function. We set it to `None` (it is also a default value) to generate an infinite number of data batches to avoid handling the last, likely incomplete, batch. This is especially important for distributed training where we need to guarantee that the numbers of data records seen on all workers are identical per step. Given that the length of each data shard may not be identical, setting `num_epochs` to any specific number would fail to meet the guarantee and will likely result in an error. Even though this may not be really important for training on a single device, it determines the way we control epochs (training will run forever on infinite dataset which means there would be only 1 epoch if other means of controlling the epoch duration are not used), so we decided to introduce it here from the beginning.
# MAGIC 
# MAGIC Setting the value of `num_epochs=None` is also important for the validation process. At the time this notebook was developed, Pytorch Lightning Trainer will run a sanity validation check prior to any training, unless instructed otherwise (i.e. `num_sanity_val_steps` is set to `0`). That sanity check will initialise the validation data loader and will read the `num_sanity_val_steps` batches from it before the first training epoch. Training will not reload the validation dataset for the actual validation phase of the first epoch which will result in error (an attempt to read a second time from data loader which was not completed in the previous attempt). Possible workarounds to avoid this issue is using a finite amount of epochs in `num_epochs` (e.g. `num_epochs=1` as there is no point in evaluating on repeated dataset), which is not ideal as it will likely result in a last batch being smaller than other batches and at the time when this notebook was developed there was no way of setting an equivalent of `drop_last` for the Data Loader created by `make_torch_dataloader`. The only way we found to work around this was to avoid doing any sanity checks (i.e. setting `num_sanity_val_steps=0`, setting it to anything else doesn't work) and using `limit_val_batches` parameter of the Trainer class to avoid the infinitely running validation.
# MAGIC 
# MAGIC A separate callback class can be used for sidecar operations like logging, etc but we decided to keep evething within the model class.

# COMMAND ----------

class LitClassificationModel(pl.LightningModule):
  def __init__(self, train_converter, val_converter, class_count=CLASS_COUNT, lr=LR, logging_level=logging.INFO, 
               device_id=0, device_count=1):
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
    X, y = batch["features"], batch["label"]
    pred = self(X)
    loss = F.cross_entropy(pred, y)
    
    # Choosing to use step loss as a metric
    self.log("train_loss", loss, prog_bar=True)
    
    if self.logging_level == logging.DEBUG:
      if batch_idx == 0:
        print(f" - [{self.device_id}] training batch size: {y.shape[0]}")
      print(f" - [{self.device_id}] training batch: {batch_idx}, loss: {loss}")
      
    return loss
    
  def on_train_epoch_start(self):
    # No need to re-load data here as `train_dataloader` will be called on each epoch
    if self.logging_level in (logging.DEBUG, logging.INFO):
      print(f"++ [{self.device_id}] Epoch: {self.state['epochs']}")
    self.state["epochs"] += 1

  def train_dataloader(self):
    if self.train_dataloader_context:
        self.train_dataloader_context.__exit__(None, None, None)
    self.train_dataloader_context = self.train_converter.make_torch_dataloader(transform_spec=self._get_transform_spec(), 
                                                                               num_epochs=None,
                                                                               cur_shard=self.device_id, 
                                                                               shard_count=self.device_count, 
                                                                               batch_size=BATCH_SIZE*self.device_count)
    return self.train_dataloader_context.__enter__()
    
  def validation_step(self, batch, batch_idx):
    X, y = batch["features"], batch["label"]
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
    if self.val_dataloader_context:
        self.val_dataloader_context.__exit__(None, None, None)
    self.val_dataloader_context = self.val_converter.make_torch_dataloader(transform_spec=self._get_transform_spec(), 
                                                                           num_epochs=None, 
                                                                           cur_shard=self.device_id, 
                                                                           shard_count=self.device_count,  
                                                                           batch_size=BATCH_SIZE*self.device_count)
    return self.val_dataloader_context.__enter__()

  def on_train_end(self):
    # Close all readers (especially important for distributed training to prevent errors)
    self.train_dataloader_context.__exit__(None, None, None)
    self.val_dataloader_context.__exit__(None, None, None)
  
  def get_model(self, class_count, lr):
    model = torchvision.models.mobilenet_v2(pretrained=True)

    # Freeze parameters in the feature extraction layers and replace the last layer
    for param in model.parameters():
      param.requires_grad = False

    # New modules have `requires_grad = True` by default
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, class_count)
    return model
  
  def _transform_rows(self, batch):
    
    # To keep things simple, use the same transformation both for training and validation
    batch["features"] = batch["content"].map(lambda x: preprocess(x).numpy())
    batch = batch.drop(labels=["content"], axis=1)
    return batch

  def _get_transform_spec(self):
    return TransformSpec(self._transform_rows, 
                         edit_fields=[("features", np.float32, (3, 224, 224), False)], 
                         selected_fields=["features", "label"])


# COMMAND ----------

def train(train_converter=train_sample_converter, val_converter=val_sample_converter, gpus=0, strategy=None, 
          device_id=0, device_count=1, logging_level=logging.INFO):
  
  start = dt.datetime.now()

  if device_id == 0:
    device = str(max(gpus, device_count)) + ' GPU' + ('s' if gpus > 1 or device_count > 1 else '') if gpus > 0  else 'CPU'
    print(f"Train on {device}:")
    print(f"- max epoch count: {MAX_EPOCH_COUNT}")
    print(f"- batch size: {BATCH_SIZE*device_count}")
    print(f"- steps per epoch: {STEPS_PER_EPOCH}")
    print(f"- sample size: {SAMPLE_SIZE}")
    print("\n======================\n")
  
  # Use check_on_train_epoch_end=True to evaluate at the end of each epoch
  verbose = True if device_id == 0 else False
  stopper = EarlyStopping(monitor="val_loss", min_delta=EARLY_STOP_MIN_DELTA, patience=EARLY_STOP_PATIENCE, 
                          verbose=verbose, mode='min', check_on_train_epoch_end=True)
  callbacks = [stopper]
  
  # Uncomment the following lines to add checkpointing if needed
  # checkpointer = ModelCheckpoint(monitor='val_loss', mode="min", save_top_k=1, verbose=verbose)
  # callbacks.append(checkpointer)
  
  # You could also use an additinal progress bar but default progress reporting was sufficient. Uncomment next line if desired
  # callbacks.append(TQDMProgressBar(refresh_rate=STEPS_PER_EPOCH, process_position=0))
  
  # We could use `on_train_batch_start` to control epoch sizes as shown in the link below but it's cleaner when 
  # done here with `limit_train_batches` parameter
  # https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/core/hooks.html#ModelHooks.on_train_batch_start
  trainer = pl.Trainer(
      gpus=gpus,
      max_epochs=MAX_EPOCH_COUNT,
      limit_train_batches=STEPS_PER_EPOCH,  # this is the way to end the epoch
      log_every_n_steps=1,
      val_check_interval=STEPS_PER_EPOCH,  # this value must be the same as `limit_train_batches`
      num_sanity_val_steps=0,  # this must be zero to prevent a Petastorm error about Data Loader not being read completely
      limit_val_batches=1,  # any value would work here but there is point in validating on repeated set of data
      reload_dataloaders_every_n_epochs=1,  # need to set this to 1
      strategy=strategy,
      callbacks=callbacks,
      default_root_dir='/tmp/lightning_logs'
  )

  model = LitClassificationModel(train_converter, val_converter, device_id=device_id, device_count=device_count, logging_level=logging_level)
  trainer.fit(model)

  if device_id == 0:
    report_duration(f"Training", start)
    print("\n\n---------------------")
  
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
# MAGIC - max epoch count: 15
# MAGIC - batch size: 64
# MAGIC - steps per epoch: 15
# MAGIC - sample size: 1000
# MAGIC 
# MAGIC ======================
# MAGIC 
# MAGIC * Run 1: 
# MAGIC 
# MAGIC   - Epoch: ***6***
# MAGIC Best score: ***0.580***. Signaling Trainer to stop.
# MAGIC Training completed in ***5 minutes 54 seconds***
# MAGIC 
# MAGIC --------------------
# MAGIC 
# MAGIC * Run 2:
# MAGIC 
# MAGIC   - Epoch: ***10***
# MAGIC Best score: ***0.368***. Signaling Trainer to stop.
# MAGIC Training completed in ***8 minutes 57 seconds*** at 2022-02-08 10:29:14
# MAGIC 
# MAGIC --------------------
# MAGIC 
# MAGIC ***Observations:*** 
# MAGIC   - *The actual number of epochs used for training (and consequently the training time may vary, possibly heavily influenced by the data selection randomness in batching*
# MAGIC   - *This variation in training time seems to have a significant impact on the best score, but it may need to be tuned by setting a lower `min_delta` in the stopper*

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## GPU Training

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Using 1 GPU
# MAGIC 
# MAGIC ***Note:*** *Code in the next cell require a GPU to run. A single node cluster with a single GPU instance will suffice (e.g. `p3.2xlarge` on AWS or equivalent on other cloud providers)*

# COMMAND ----------

gpu_model = train(gpus=1)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### GPU Training Results
# MAGIC 
# MAGIC *We run multiple experiments and selected those which stopped after the same amount of epochs as the CPU training experiments run earlier*
# MAGIC 
# MAGIC Train on 1 GPU:
# MAGIC  - max epoch count: 15
# MAGIC  - batch size: 64
# MAGIC  - steps per epoch: 15
# MAGIC  - sample size: 1000
# MAGIC 
# MAGIC ======================
# MAGIC 
# MAGIC * Run 1:
# MAGIC 
# MAGIC   - Epoch: ***6***
# MAGIC Best score: ***0.529***. Signaling Trainer to stop.
# MAGIC Training completed in ***3 minutes 23 seconds***
# MAGIC 
# MAGIC ---------------------
# MAGIC 
# MAGIC * Run 2:
# MAGIC   - Epoch: ***10***
# MAGIC Best score: ***0.385***. Signaling Trainer to stop.
# MAGIC Training completed in ***3 minutes 31 seconds***
# MAGIC 
# MAGIC ---------------------
# MAGIC 
# MAGIC * Run 3
# MAGIC   - Epoch: ***8***
# MAGIC Best score: ***0.450***. Signaling Trainer to stop.
# MAGIC Training completed in ***2 minutes 23 seconds***
# MAGIC 
# MAGIC ---------------------
# MAGIC 
# MAGIC 
# MAGIC ***Observations:*** 
# MAGIC - *as expected, full training/validation cycle on GPU is faster than on CPU*
# MAGIC - *(not shown here) similar training runs but without using the validation steps were significantly faster on GPU (some showed more that 3 times speedup)*

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Using multiple GPUs
# MAGIC 
# MAGIC Simply passing a value of `gpus` parameter greater than 1 to a Trainer won't work, we also need to have a way to syncronise the losses between different models (each GPU will have its own model to train). This could be done by specifying a `strategy` parameter of the Trainer. According to docs, `If you request multiple GPUs or nodes without setting a mode, DDP Spawn will be automatically used.` [https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#distributed-modes]), however just giving a `gpus` value greater than 1 to the Trainer (e.g. `train(gpus=2)`) throws an error (we didn't investigate why).
# MAGIC 
# MAGIC So we also need to specify a strategy that works, e.g. `strategy=HorovodPlugin()`, but it produces the same loss within the same execution time as when using a single GPU, which draws a conclusion that only a single GPU gets used anyways. (this is also supported by output logs from only one process in verbose logging mode, but that could be because logging in other processes spawned for GPUs other then the 1st GPU are not shown in this process if multi-gpu training actually gets launched)
# MAGIC 
# MAGIC ***Note:*** *strangely, Trainer complains about interactive mode if using `strategy="horovod"` but works with `strategy=HorovodPlugin()`. Maybe using an instance of DDP plugin instead of `strategy="ddp"` will also work, worth checking?*

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Distributed Training
# MAGIC 
# MAGIC Since we are using PyTorch here, PyTorch DDP (Distributed Data Parallel) would be a logical choice for scaling out the training to multiple GPUs (as opposed to DP which not recommended by PyTorch), but it looks like DDP training does not work in interactive mode (e.g. in notebooks) so it would have to run as a standalone script (which in turn seems to be problematic due to missing Spark context in a standalone script executed in Notebook). Furthermore, it is unclear if we can use Petastorm with DDP (DDP itself does splitting and distribution of the data so it works with Petastom Spark Converter - something here to review: https://github.com/PyTorchLightning/pytorch-lightning/issues/5074) 
# MAGIC 
# MAGIC Then there is Horovod which works in interactive mode (e.g. in notebooks), which brings us to to the following options to explore:
# MAGIC 
# MAGIC 1. PyTorch Lightning has a `HorovodPlugin` plugin, we can use it with a `horovod` runner
# MAGIC     - https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/plugins/training_type/horovod.html
# MAGIC     - https://github.com/PyTorchLightning/pytorch-lightning/blob/1fc046cde28684316323693dfca227dbd270663f/tests/models/test_horovod.py
# MAGIC     - straightforward to run on a single node but perhaps not so for multi-node training for which we may need to do some additional Databricks cluster configurations (unless `horovod` already comes pre-configured for multi-node execution because it's pre-installed on Databricks cluster with ML runtime
# MAGIC     - *an interesting observation, using `strategy="horovod"` without using the `horovod` runner complains about interactive mode but it does not if using `strategy=HorovodPlugin()`*
# MAGIC 1. Use `sparkdl.HorovodRunner`
# MAGIC     - https://databricks.com/blog/2018/11/19/introducing-horovodrunner-for-distributed-deep-learning-training.html
# MAGIC     - this option will also use aforementioned `HorovodPlugin`
# MAGIC     - `sparkdl.HorovodRunner` will "understand" a Spark cluster so no need to make any additional cluster configurations
# MAGIC 1. Use `horovod.spark.lightning.TorchEstimator` with `horovod.spark.common.backend.SparkBackend`
# MAGIC     - https://github.com/horovod/horovod/blob/master/docs/spark.rst
# MAGIC     - https://horovod.readthedocs.io/en/stable/_modules/horovod/spark/torch/estimator.html
# MAGIC     - this is a Spark "aware" solution, no need to make extra cluster configurations
# MAGIC     - these libraries will use a Petastorm behind the scene and automatically configure the training to use Horovod to multi-device training. This should reduce the amount of code we need to write.
# MAGIC     

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Option 1: Use Horovod
# MAGIC 
# MAGIC ***Note:*** *This section will require a cluster with multiple GPUs, a single node cluster with multiple GPUs instace will be sufficient (e.g. p3.8xlarge on AWS or equivalent on other cloud providers)*
# MAGIC 
# MAGIC We are using the same training code, model and data loaders we used earlier for CPU and single GPU training.

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
  
  # `gpus` parameter here should be 1 because the parallelism is controlled by Horovod
  return train(gpus=1, strategy="horovod", device_id=hvd.rank(), device_count=hvd.size())
  

# COMMAND ----------

# This will launch a distributed training on `MAX_DEVICE_COUNT_TO_USE` devices
hvd_model = horovod.run(train_hvd, np=MAX_DEVICE_COUNT_TO_USE)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Single Node Multi-GPU Training Results
# MAGIC 
# MAGIC * Run 1:
# MAGIC 
# MAGIC ======================
# MAGIC 
# MAGIC Train on 2 GPUs:
# MAGIC  - max epoch count: 15
# MAGIC  - batch size: 128
# MAGIC  - steps per epoch: 15
# MAGIC  - sample size: 1000
# MAGIC 
# MAGIC ======================
# MAGIC 
# MAGIC ++ [0] Epoch: 8
# MAGIC 
# MAGIC ++ [1] Epoch: 8
# MAGIC 
# MAGIC 
# MAGIC Epoch 8: 100% 16/16 [00:15<00:00,  1.05it/s, loss=0.232, v_num=0, train_loss=0.212, val_loss=0.370, val_acc=0.895]
# MAGIC 
# MAGIC Epoch ***8***: 100% 16/16 [00:15<00:00,  1.03it/s, loss=0.232, v_num=0, train_loss=0.212, val_loss=***0.370***, val_acc=0.895]
# MAGIC 
# MAGIC [rank: 0] Best score: ***0.406***. Signaling Trainer to stop.
# MAGIC 
# MAGIC Training completed in ***3 minutes 20 seconds***
# MAGIC 
# MAGIC ---------------------
# MAGIC 
# MAGIC * Run 2
# MAGIC   - This run was performed on a `p3.8xlarge` single node cluster with `MAX_DEVICE_COUNT_TO_USE=4`
# MAGIC   - You will likely need to Detach & Re-Attach the nodebook to a cluster, and re-run some of the code above after changing `MAX_DEVICE_COUNT_TO_USE` value to 4
# MAGIC 
# MAGIC ======================
# MAGIC 
# MAGIC Train on 4 GPUs:
# MAGIC  - max epoch count: 15
# MAGIC  - batch size: 256
# MAGIC  - steps per epoch: 15
# MAGIC  - sample size: 1000
# MAGIC 
# MAGIC ======================
# MAGIC 
# MAGIC Epoch ***5***: 100% 16/16 [00:30<00:00,  1.91s/it, loss=0.191, v_num=9, train_loss=0.173, val_loss=***0.369***, val_acc=0.896]
# MAGIC 
# MAGIC [rank: 0] Best score: ***0.415***. Signaling Trainer to stop.
# MAGIC 
# MAGIC Training completed in ***3 minutes 21 seconds***
# MAGIC 
# MAGIC ---------------------
# MAGIC 
# MAGIC 
# MAGIC ***Observations:*** 
# MAGIC   - overall training time here is fairly similar to what we saw for a single GPU. This is expected as each model will be trained with the same amount of data/epochs
# MAGIC   - we achieved better results (lower loss) this time compared to a single GPU training (as we hoped to get). Actually, the reported score ***0.406*** is not the lowest, a lower loss of ***0.37*** was obtained in the last epoch but that model wasn't poicked because the difference in loss is lower than the `min_delta` we used (0.05). Lowering `min_delta` will result in a model with a lower final loss being picked
# MAGIC   - loss gets lower faster if we increase the number of GPUs we use. Run 2 shows the results of running the same training on 4 GPUs
# MAGIC   - we deliberately printed epoch counts from each process to show that multiple training processes were running simultaneously.
# MAGIC   - unlike the previous training runs, we got a better progress report during this training, Horovod must be doing some extra work there 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ***Note:*** Code in this section works on a single machine only. An attempt to train using more GPUs than available on the driver node fails, even if there are worker nodes in the cluster that have those extra GPUs. This could be still possibly be configuring Horovod to recognise GPUs on other nodes in the cluster but we are interested in using the clusters as they are provided without making additional configurations so we'll not explore the option of runnning multi-node GPU training using only Horovod.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Option 2: Use `sparkdl.HorovodRunner`
# MAGIC 
# MAGIC ***Note:*** *This section will require a multi-node GPU cluster, a cluster with 2 workers, each with a single GPU will be sufficient (e.g. p3.2xlarge on AWS or equivalent on other cloud providers)*
# MAGIC 
# MAGIC Here we are jumping straight into a multi-node distributed training.

# COMMAND ----------

from sparkdl import HorovodRunner

# COMMAND ----------

# Run this code from the notebook in your own workspace (e.g. File->Clone and save under Workspace) if this code fails when executed in a notebook in the Repos  
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
# MAGIC 
# MAGIC ======================
# MAGIC 
# MAGIC Epoch 8: 100% 16/16 [00:19<00:00,  1.20s/it, loss=0.332, v_num=0, train_loss=0.310, val_loss=***0.356***, val_acc=0.910]
# MAGIC 
# MAGIC Training completed in ***3 minutes 45 seconds***
# MAGIC 
# MAGIC ---------------------
# MAGIC 
# MAGIC ***Observations:*** 
# MAGIC   - training time and resulting loss are similar to training with 2 GPUs on the same node, but this time it was accross 2 nodes

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Clean Up
# MAGIC 
# MAGIC We no longer need some of the objects created earlier

# COMMAND ----------

train_sample_converter.delete()
val_sample_converter.delete()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Option 3: Use `horovod.spark.lightning.TorchEstimator` with a `SparkBackend`

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ***Note:*** *This section will require a GPU cluster. Code should work on a single node cluster with a single GPU instance as well as multi-node GPU cluster*
# MAGIC 
# MAGIC TorchEstimator accepts a `transformation_fn` parameter so technically we should be able use the same `TransformSpec` we used earlier. However, we couldn't make it work in few attempts and decided to take a different approach (though it could be less efficient than using a transformer function, it could be a better approach overall if using the same data for multiple experiments). 
# MAGIC 
# MAGIC We pre-process the training data converting images to arrays prior to training using Pandas UDF.

# COMMAND ----------

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
SAMPLE_DATA_DIR = f"/FileStore/{user}/datasets/flowers_sample/delta"
print(SAMPLE_DATA_DIR)

# COMMAND ----------

def to_array(images):
  res = []
  for img in images:
    res.append(preprocess(img).numpy().flatten())
  return pd.Series(res)

to_array_udf = pandas_udf(T.ArrayType(T.FloatType()), PandasUDFType.SCALAR)(to_array)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Convert the images to arrays in our dataset and write out as Delta
# MAGIC 
# MAGIC ***Note:*** *Run the next cell only once, subsequent runs can re-use the dataset created during the first run.*

# COMMAND ----------

df, _ = get_data(SAMPLE_SIZE, split=False)
df.withColumn("features", to_array_udf(col("content"))).drop("content").write.format("delta").save(SAMPLE_DATA_DIR)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Load the pre-processed dataset

# COMMAND ----------

processed_df = spark.read.format("delta").load(SAMPLE_DATA_DIR)
train_df, test_df = processed_df.randomSplit([0.8, 0.2], seed=12)
train_df = train_df.repartition(MAX_DEVICE_COUNT_TO_USE)
print(f"Datasets:\n - train: {train_df.count()}\n - test: {test_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### A simplified model
# MAGIC 
# MAGIC Horovod's TorchEstimator handles operations related to using Horovod and Petastorm behind the scene which means we can simplify our model.

# COMMAND ----------

class SimpleLitClassificationModel(pl.LightningModule):
  def __init__(self, class_count=CLASS_COUNT, lr=LR, logging_level=logging.INFO):
    super().__init__()
    self.lr = lr
    self.model = self.get_model(class_count, lr)
    self.state = {"epochs": 0}
    self.logging_level = logging_level

  def configure_optimizers(self):
    optimizer = torch.optim.SGD(self.model.classifier[1].parameters(), lr=self.lr, momentum=0.9)
    return optimizer

  def forward(self, x):
    
    # An input here is a one dimensional numpy array, we need to shape it into 4D
    x = x.float().reshape((-1, 3, 224, 224))
    x = self.model(x)
    return x
  
  def training_step(self, batch, batch_idx):
    X, y = batch["features"], batch["label"]
    pred = self(X)
    loss = F.cross_entropy(pred, y)
    self.log("train_loss", loss, prog_bar=True)
    
    if self.logging_level == logging.DEBUG:
      if batch_idx == 0:
        print(f" - training batch size: {y.shape[0]}")
      print(f" - training batch: {batch_idx}, loss: {loss}")
      
    return loss
    
  def on_train_epoch_start(self):
    if self.logging_level in (logging.DEBUG, logging.INFO):
      print(f"++ Epoch: {self.state['epochs']}")
    self.state["epochs"] += 1
    
  def validation_step(self, batch, batch_idx):
    X, y = batch["features"], batch["label"]
    pred = self(X)
    loss = F.cross_entropy(pred, y)
    acc = FM.accuracy(pred, y)

    # Roll validation up to epoch level
    self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
    self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
    
    if self.logging_level == logging.DEBUG:
      print(f" - val batch: {batch_idx}, size: {y.shape[0]}, loss: {loss}, acc: {acc}")

    return {"loss": loss, "acc": acc}
  
  def get_model(self, class_count, lr):
    model = torchvision.models.mobilenet_v2(pretrained=True)

    # Freeze parameters in the feature extraction layers and replace the last layer
    for param in model.parameters():
      param.requires_grad = False

    # New modules have `requires_grad = True` by default
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, class_count)
    return model

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Train the model
# MAGIC 
# MAGIC We need to add couple of extra things to train the TorchEstimator: a `SparkBackend` and a `Store`

# COMMAND ----------

import sys
from horovod.spark.lightning import TorchEstimator
from horovod.spark.lightning.estimator import MIN_PL_VERSION
from horovod.spark.common.backend import SparkBackend
from horovod.spark.common.store import Store

# COMMAND ----------

backend = SparkBackend(num_proc=1, stdout=sys.stdout, stderr=sys.stderr,  prefix_output_with_timestamp=True)
store = Store.create('file:///dbfs/tmp')
model = SimpleLitClassificationModel(logging_level=logging.DEBUG)
stopper = EarlyStopping(monitor="val_loss", min_delta=EARLY_STOP_MIN_DELTA, patience=EARLY_STOP_PATIENCE, verbose=False, mode='min')
callbacks = [stopper]

# Add model checkpointer if you want to
# callbacks.append(ModelCheckpoint(monitor='val_loss', mode="min", save_top_k=1, verbose=False))

torch_estimator = TorchEstimator(backend=backend, 
                                 store=store, 
                                 model=model,
                                 input_shapes=[[-1, 3, 224, 224]],
                                 feature_cols=['features'],
                                 label_cols=['label'],
                                 batch_size=BATCH_SIZE,
                                 epochs=MAX_EPOCH_COUNT,
                                 validation=0.1,
                                 verbose=2,
                                 callbacks=callbacks,
                                 train_steps_per_epoch=STEPS_PER_EPOCH,
                                 validation_steps_per_epoch=1,
                                 #transformation_fn=transform_spec_fn
                                )

# COMMAND ----------

start = dt.datetime.now()

print(f"Train on 4 GPUs using TorchEstimator")
print(f" - max epoch count: {MAX_EPOCH_COUNT}\n - batch size: {BATCH_SIZE}\n - steps per epoch: {STEPS_PER_EPOCH}\n - sample size: {SAMPLE_SIZE}")
print("\n======================\n")

torch_model = torch_estimator.fit(train_df)

report_duration(f"Training", start)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Evaluate the model

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Evaluate the model on the held-out test DataFrame
pred_df = torch_model.transform(test_df)

argmax = udf(lambda v: float(np.argmax(v)), returnType=T.DoubleType())
pred_df = pred_df.withColumn('label_pred', argmax(pred_df.label__output))
evaluator = MulticlassClassificationEvaluator(predictionCol='label_pred', labelCol='label', metricName='accuracy')
print('Test accuracy: ', evaluator.evaluate(pred_df))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### TorchEsimator Training Results
# MAGIC 
# MAGIC Train on a single node cluster with 4 GPUs:
# MAGIC  - max epoch count: 15
# MAGIC  - batch size: 64
# MAGIC  - steps per epoch: 15
# MAGIC  - sample size: 1000
# MAGIC 
# MAGIC ======================
# MAGIC 
# MAGIC ++ Epoch: ***8***
# MAGIC 
# MAGIC Training: Epoch 8: 100% 1616 [00:01<00:00, 12.98it/s, loss=0.399, v_num=0, train_loss=0.349, val_loss=***0.394***, val_acc=0.922]
# MAGIC 
# MAGIC Validating: Epoch 8: 100% 16/16 [00:01<00:00,  8.57it/s, loss=0.399, v_num=0, train_loss=0.349, val_loss=***0.473***, val_acc=0.859]
# MAGIC                                               
# MAGIC Training completed in ***1 minutes 26 seconds***
# MAGIC                                               
# MAGIC ---------------------
# MAGIC 
# MAGIC Test accuracy:  ***0.8855721393034826***
# MAGIC 
# MAGIC ---------------------
# MAGIC 
# MAGIC ***Observations:*** 
# MAGIC   - MLFlow experiment was automatically created and logged during the training without us doing anything (though there appear to be no artifacts logged in the experiment, only the training parameters)
# MAGIC   - though the training was run on a single node cluster with 4 GPUs, only 1 GPU was actually used for training. We'll leave it to a reader to try it out on a multi-node cluster with multiple GPUs
# MAGIC   - training took less time than we saw earlier with a single or multi-node GPUs but this could be because we used a pre-processed dataset and didn't have to do any extra data transformations during the training
# MAGIC   - despite the final epoch validation loss showing ***0.473***, the one before the last (epoch 7) had a better validation loss
# MAGIC   - accuracy of the model when tested with an unseen test data is higher than the validation accuracy of the last epoch but worse than the validation accuracy of the epoch 7. This is another indication that the model produced in epoch 7 was probably a better one
