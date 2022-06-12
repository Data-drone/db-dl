# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Main PyTorch Lightning Training notebook

# COMMAND ----------

# MAGIC %run "./Building the PyTorch Lightning Modules"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Model Hyperparameters
# MAGIC 
# MAGIC we will spec all our global training parameters here for easy alterations.
# MAGIC One way to reduce your databricks cost and also to prevent wasted cloud costs in event that a production unattended training run fails would be to transform these into databricks widgets and execute them within a databricks workflow.
# MAGIC 
# MAGIC For Widgets see: 
# MAGIC - AWS: https://docs.databricks.com/notebooks/widgets.html
# MAGIC - Azure: https://docs.microsoft.com/en-us/azure/databricks/notebooks/widgets
# MAGIC 
# MAGIC For Workflows see:
# MAGIC - AWS: https://docs.databricks.com/data-engineering/jobs/index.html
# MAGIC - Azure: https://docs.microsoft.com/en-us/azure/databricks/data-engineering/jobs/

# COMMAND ----------

MAX_EPOCH_COUNT = 15
BATCH_SIZE = 64
STEPS_PER_EPOCH = 15

SAMPLE_SIZE = 1000
default_dir = '/dbfs/Users/brian.law@databricks.com/tmp/lightning_logs'

EARLY_STOP_MIN_DELTA = 0.05
EARLY_STOP_PATIENCE = 3

NUM_DEVICES = 4

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Setup Dataset
# MAGIC 
# MAGIC Here we setup the Petastorm cache folder and load in the dataset from delta before feeding it into our petastorm converter

# COMMAND ----------

from petastorm.spark import SparkDatasetConverter, make_spark_converter
import pyspark.sql.types as T
from pyspark.sql.functions import col, pandas_udf, PandasUDFType

# COMMAND ----------

Data_Directory = '/databricks-datasets/flowers/delta'

CACHE_DIR = "file:///dbfs/tmp/petastorm/cache"
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, CACHE_DIR)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Prepare Spark Dataset
# MAGIC 
# MAGIC DataFrame and corresponding Petastorm wrappers need to be created here instead of inside the Pytorch Lightning Data Module class. 
# MAGIC This is especially important for the distributed training when the model class instances will be created in worker nodes where Spark context is not available (Petastorm spark converter can be pickled).

# COMMAND ----------

import pyspark.sql.types as T
from pyspark.sql.functions import udf

def to_class_index(classes: list, class_name:str):
    
    """
    Converts classes to a class_index so that we can create a tensor object
    """
    return classes.index(class_name)
  
  
def udf_class(class_list: list): 
    """
    Results has to be a longtype
    """
    return udf(lambda x: to_class_index(class_list, x), T.LongType())
    
    
def prepare_data(data_dir: str, num_devices: int):
    """
    This function loads the dataset and converts the label into a numeric index
    
    PyTorch Lightning suggests splitting datasets in the setup command but with petastorm we only need to do this once.
    Also spark is already distributed.
    
    """
    
    flowers_dataset = spark.read.format("delta").load(data_dir).select('content', 'label')
    
    
    flowers_dataset = flowers_dataset.repartition(num_devices*2)
    classes = list(flowers_dataset.select("label").distinct().toPandas()["label"])
    print(f'Num Classes: {len(classes)}')
    
    #class_index = udf(_to_class_index, T.LongType())  
    flowers_dataset = flowers_dataset.withColumn("label", udf_class(classes)(col("label")) )
    
    total_size = flowers_dataset.count()
    print(f"Dataset size: {total_size}")
    
    groups = flowers_dataset.groupBy('label').count()
    groups = groups.withColumn('fraction', col('count')/total_size)
    
    fractions = groups.select('label', 'fraction').toPandas()
    fractions.set_index('label')
    
    val_df = flowers_dataset.sampleBy("label", fractions=fractions.fraction.to_dict(), seed=12)
    train_df = flowers_dataset.join(val_df, flowers_dataset.content==val_df.content, "leftanti")
    print(f"Train Size = {train_df.count()} Val Size = {val_df.count()}")

    train_converter = make_spark_converter(train_df)
    val_converter = make_spark_converter(val_df)
    
    return flowers_dataset, train_converter, val_converter 


# COMMAND ----------

flowers_df, train_converter, val_converter = prepare_data(data_dir=Data_Directory, 
                                                          num_devices=NUM_DEVICES)

datamodule = FlowersDataModule(train_converter=train_converter, 
                               val_converter=val_converter)

# COMMAND ----------

# MAGIC %sh
# MAGIC # Check the GPU Status of Node
# MAGIC nvidia-smi

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Load Model

# COMMAND ----------

# We could init this in the horovod runner instead for proper logging
model = LitClassificationModel(class_count=5, learning_rate=1e-5)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Single Node Train

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Single GPU

# COMMAND ----------

train(model, datamodule, gpus=1)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Multi GPU

# COMMAND ----------

train(model, datamodule, gpus=4, strategy='dp')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Horovod Train

# COMMAND ----------

from pytorch_lightning.utilities import _HOROVOD_AVAILABLE
if _HOROVOD_AVAILABLE:
  print("Horovod is available")
  import horovod
  import horovod.torch as hvd
  from pytorch_lightning.plugins.training_type.horovod import HorovodPlugin
  print(f"Horovod: {horovod.__version__}")


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We need to move the dataloader and model instantiation into the horovod statement

# COMMAND ----------

def train_hvd():
  hvd.init()
  
  hvd_model = LitClassificationModel(class_count=5, learning_rate=1e-5, device_id=hvd.rank(), device_count=hvd.size())
  hvd_datamodule = FlowersDataModule(train_converter, val_converter, device_id=hvd.rank(), device_count=hvd.size())
  
  # `gpus` parameter here should be 1 because the parallelism is controlled by Horovod
  return train(hvd_model, hvd_datamodule, gpus=1, strategy="horovod", device_id=hvd.rank(), device_count=hvd.size())
  

# COMMAND ----------

from sparkdl import HorovodRunner

# This will launch a distributed training on np devices
hr = HorovodRunner(np=-4, driver_log_verbosity='all')

# Need to solve the issue with things and stuff

hvd_model = hr.run(train_hvd)

# COMMAND ----------


