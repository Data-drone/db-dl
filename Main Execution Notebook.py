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
# MAGIC - GCP: https://docs.gcp.databricks.com/notebooks/widgets.html
# MAGIC 
# MAGIC For Workflows see:
# MAGIC - AWS: https://docs.databricks.com/data-engineering/jobs/index.html
# MAGIC - Azure: https://docs.microsoft.com/en-us/azure/databricks/data-engineering/jobs/
# MAGIC - GCP: https://docs.gcp.databricks.com/data-engineering/index.html

# COMMAND ----------

MAX_EPOCH_COUNT = 15
BATCH_SIZE = 64
STEPS_PER_EPOCH = 15

SAMPLE_SIZE = 1000

# When using databricks repos, it is not possible to write into working directories
# specifying a dbfs default dir helps to avoid this
default_dir = '/dbfs/Users/brian.law@databricks.com/tmp/lightning'

EARLY_STOP_MIN_DELTA = 0.05
EARLY_STOP_PATIENCE = 3

NUM_DEVICES = 8

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Setup Mlflow Parameters
# MAGIC 
# MAGIC Mlflow is designed to only work with one python process. As we scale up our training job into multiple python processes we do not want to log the same model multiple times. Hence some additional boilerplate is required to manage this

# COMMAND ----------

import os

## Specifying the mlflow host server and access token 
# We put them to a variable to feed into horovod later on
db_host = "https://e2-demo-tokyo.cloud.databricks.com/"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# We put them into these environment variables as this is where mlflow will look by default
os.environ['DATABRICKS_HOST'] = db_host
os.environ['DATABRICKS_TOKEN'] = db_token


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
    
    #### the following code is to make sure to sample each class by the proportion that it appears in the original dataset
    # Spark doesn't include a Stratified Sampling class hence we have to calculate what fraction each class is.
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

train(model, datamodule, gpus=1, default_dir=default_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Multi GPU

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Horovod Single Node

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
  
  # mlflow workaround to ensure that horovod subprocesses can find and connect to mlflow
  mlflow.set_tracking_uri("databricks")
  os.environ['DATABRICKS_HOST'] = db_host
  os.environ['DATABRICKS_TOKEN'] = db_token
  
  # we hard code the experiment ID to make sure we log to the right experiment
  ## worker nodes may not see the experiment id of the notebook
  mlflow_experiment_id = 'b5fedd94c5524daab5f574ee7e132f1f'

  
  hvd_model = LitClassificationModel(class_count=5, learning_rate=1e-5, device_id=hvd.rank(), device_count=hvd.size())
  hvd_datamodule = FlowersDataModule(train_converter, val_converter, device_id=hvd.rank(), device_count=hvd.size())
  
  # `gpus` parameter here should be 1 because the parallelism is controlled by Horovod
  return train(hvd_model, hvd_datamodule, gpus=1, device_id=hvd.rank(), device_count=hvd.size(), 
               mlflow_experiment_id=mlflow_experiment_id,
              default_dir=default_dir)
  

# COMMAND ----------

from sparkdl import HorovodRunner

# This will launch a distributed training on np devices
hr = HorovodRunner(np=-4, driver_log_verbosity='all')
hvd_model = hr.run(train_hvd)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Multi-Node with Horovod 

# COMMAND ----------

from sparkdl import HorovodRunner

# This will launch a distributed training on np devices
hr = HorovodRunner(np=8, driver_log_verbosity='all')
hvd_model = hr.run(train_hvd)

# COMMAND ----------


