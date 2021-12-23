# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### Setup

# COMMAND ----------

import os

from pyspark.sql.types import *
import pyspark.sql.functions as F

# COMMAND ----------

SRC_DIR = "dbfs:/databricks-datasets/flower_photos/tulips"
SRC_FILE = "100930342_92e8746431_n.jpg"  #"2431737309_1468526f8b.jpg"
SRC_FILE_PATH = os.path.join(SRC_DIR, SRC_FILE)

DST_FILE_PATH = "dbfs:/FileStore/nul/result.jpg"

# COMMAND ----------

def save_image_to(dst_path, simulate_slow=True):
  def save_image(row):
    import time
    
#     dst = os.path.join(dir_path, os.path.split(row.path)[1])
    dst = dst_path
    with open(dst, 'wb') as fout:
      fout.write(row.content)
    
    if simulate_slow:
      # Give some delay to simulate slow arrival of images
      time.sleep(0.1)
    
  return save_image

# COMMAND ----------

# Read the file as binary as it preserves the original file content as bytes (loading it as `format('image')` seems to either base64 encode it or have it in CV2 format)

df_images = spark.read.format("binaryFile").load(SRC_DIR).withColumn("length", F.length(F.col("content")))
df_images.printSchema()
display(df_images)

# COMMAND ----------

dst_dir = os.path.split(DST_FILE_PATH)[0].replace("dbfs:", "/dbfs")
if not os.path.exists(dst_dir):
  os.makedirs(dst_dir, exists_ok=True)
dst_file_path = DST_FILE_PATH.replace("dbfs:", "/dbfs")

df_images.foreach(save_image_to(dst_file_path))

# COMMAND ----------

img_display_path = DST_FILE_PATH.replace('dbfs:/FileStore', '/files')
html = f"""
<!DOCTYPE html>
<head>
  <meta charset="utf-8">
  <!-- <meta http-equiv="refresh" content="10" > -->
</head>
<body>
<img src="{img_display_path}" id="img" onLoad="setTimeout( () => {{ document.getElementById('img').src='{img_display_path}' + '?' + new Date().getMilliseconds() }}, 100)" />
</body>
</html>
 """
displayHTML(html)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # EXTRA

# COMMAND ----------

static_df = spark.read.format("binaryFile").option("pathGlobFilter", "*.jpg").option("recursiveFileLookup", "true").load(DATA_DIR).select("content").limit(1)

# COMMAND ----------

binay_schema = StructType([StructField("path", StringType(), False),StructField("modificationTime", TimestampType(), False),StructField("length", LongType(), False),StructField("content", BinaryType(), True)])
image_schema = StructType([
  StructField(
    "image", 
    StructType([
      StructField("origin", StringType(), True),
      StructField("height", IntegerType(), True),
      StructField("width", IntegerType(), True),
      StructField("nChannels", IntegerType(), True),
      StructField("mode", IntegerType(), True),
      StructField("data", BinaryType(), True)]), 
    True)
])

# COMMAND ----------

df = spark.read.format("image").option("pathGlobFilter", "*.jpg").option("recursiveFileLookup", "true").schema(image_schema).load(DATA_DIR)
df.schema

# COMMAND ----------

# df = spark.read.format("binaryFile").schema(image_schema).option("recursiveFileLookup", "true").load(DATA_DIR).limit(10)
df = spark.read.format("image").option("pathGlobFilter", "*.jpg").option("recursiveFileLookup", "true").schema(image_schema).load(DATA_DIR).select("image.data")
# df = spark.read.format("binaryFile").option("pathGlobFilter", "*.jpg").option("recursiveFileLookup", "true").load(DATA_DIR).select("content")
display(df.foreach())

# COMMAND ----------

df = spark.readStream.format("delta").load('/databricks-datasets/flowers/delta').select(["content"]).limit(10)
display(df)
