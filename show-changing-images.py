# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### Using Databricks Platform Realtime Computer Vision

# COMMAND ----------

# MAGIC %pip install opencv-python

# COMMAND ----------

import os

import cv2
import numpy as np

import pyspark.sql.types as T
import pyspark.sql.functions as F

# COMMAND ----------

SRC_DIR = "dbfs:/databricks-datasets/flower_photos/tulips"
SRC_FILE = "100930342_92e8746431_n.jpg"  #"2431737309_1468526f8b.jpg"
SRC_FILE_PATH = os.path.join(SRC_DIR, SRC_FILE)

DST_FILE_PATH = "dbfs:/FileStore/nul/result.jpg"

dst_dir_local = os.path.split(DST_FILE_PATH)[0].replace("dbfs:", "/dbfs")
if not os.path.exists(dst_dir_local):
  os.makedirs(dst_dir_local, exists_ok=True)
dst_file_path_local = DST_FILE_PATH.replace("dbfs:", "/dbfs")

# COMMAND ----------

def save_image_to(dst_path, is_binary=True, simulate_slow=True):
  def save_binary(row):
    import time
    
    # dst = os.path.join(dir_path, os.path.split(row.path)[1])
    dst = dst_path
    with open(dst, 'wb') as fout:
      fout.write(row.content)
    
    if simulate_slow:
      # Give some delay to simulate slow arrival of images
      time.sleep(0.1)

  def save_image(row):
    import time
    import cv2
    import numpy as np

    img_arr = np.reshape(np.frombuffer(row.image.data, dtype=np.uint8), (row.image.height, row.image.width, -1))
    cv2.imwrite(dst_path, img_arr)
    
    if simulate_slow:
      # Give some delay to simulate slow arrival of images
      time.sleep(0.1)
    
  return save_binary if is_binary else save_image

def get_type(content):
  return type(content)

get_type_udf = spark.udf.register("get_type_udf", get_type, T.StringType())

# COMMAND ----------

# Read the file as 'format("binaryFile")' because it preserves the original file content as bytes (loading it as `format('image')` seems to convert it in CV2 format)
df_images = spark.read.format("binaryFile").load(SRC_DIR).withColumn("length", F.length(F.col("content"))).withColumn("type", get_type_udf("content"))
df_images.printSchema()
display(df_images)

# COMMAND ----------

df_images.foreach(save_image_to(dst_file_path_local))

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

# MAGIC %md
# MAGIC 
# MAGIC ## Use `image` format
# MAGIC 
# MAGIC So what's stored in `image.data` if using `.format('image')`? Turns out it's a numpy/cv2 array in BGR format

# COMMAND ----------

# Read the file as `format('image')` - type of the content is the same here but length is different, maybe because it is in CV2 format
images = (spark.read.format("image").load(SRC_DIR)
  .withColumn("length", F.length(F.col("image.data")))
  .withColumn("type", get_type_udf("image.data")).limit(2)
  .withColumn("width", F.col("image.width"))
  .withColumn("height", F.col("image.height")))
images.printSchema()
display(images)

# COMMAND ----------

import matplotlib.pyplot as plt

image_row = images.collect()
image_data = image_row[0].image.data
img_arr = np.reshape(np.frombuffer(image_data, dtype=np.uint8), (441, 500, -1))
img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

plt.imshow(img_arr)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### So we can achieve the same as above using

# COMMAND ----------

images.limit(1).foreach(save_image_to(dst_file_path_local, is_binary=False))

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
