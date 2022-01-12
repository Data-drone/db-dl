# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Training Object Detection Pytorch Lightning Model on Databricks Platform
# MAGIC 
# MAGIC This notebook walks through fine tuning an Object Detection model using Pytorch Lightning. 
# MAGIC 
# MAGIC - https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# MAGIC - https://github.com/pytorch/vision/tree/v0.3.0/references/detection

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Setup
# MAGIC 
# MAGIC *Running this notebook on CPU can be slow, a GPU cluster is recommended.* 
# MAGIC 
# MAGIC To make it easlier to clone this notebook we kept it as much self-contained as possible, and brought some helper functions from https://github.com/pytorch/vision/blob/main/references/detection (with few minor modifications).
# MAGIC 
# MAGIC This notebook was developed on a Databrick Runtime 10.2 with the foolowing libraries:
# MAGIC - torch: 1.10.1+cu111
# MAGIC - torchvision: 0.11.1+cu111
# MAGIC - pytorch_lightning: 1.5.8
# MAGIC - CUDA: 11.4 (`nvidia-smi`)
# MAGIC - pycocotools: 2.0.4

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Install extra libraries
# MAGIC 
# MAGIC Following libraries are not available by default on Databricks Platform at the time this notebook was developed and need to be installed:
# MAGIC - Pytorch Lightning
# MAGIC - pycocotools

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC pip install pytorch-lightning
# MAGIC pip install pycocotools

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC 
# MAGIC import io
# MAGIC import os
# MAGIC import numpy as np
# MAGIC from functools import partial
# MAGIC import datetime as dt
# MAGIC import logging
# MAGIC import random
# MAGIC import copy
# MAGIC from contextlib import redirect_stdout
# MAGIC from collections import defaultdict, deque
# MAGIC 
# MAGIC from PIL import Image
# MAGIC import matplotlib.pyplot as plt
# MAGIC import matplotlib.patches as patches
# MAGIC 
# MAGIC import torch, torchvision
# MAGIC from torch import nn
# MAGIC import torch.nn.functional as F
# MAGIC import torchmetrics.functional as FM
# MAGIC from torchmetrics import Accuracy
# MAGIC from torch.utils.data import DataLoader, random_split
# MAGIC from torchvision import transforms
# MAGIC from torchvision.transforms import functional as TVF
# MAGIC from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# MAGIC 
# MAGIC import pytorch_lightning as pl
# MAGIC from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# MAGIC from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
# MAGIC from pytorch_lightning.callbacks import TQDMProgressBar
# MAGIC 
# MAGIC from pyspark.sql.functions import col
# MAGIC from pyspark.sql.types import LongType
# MAGIC 
# MAGIC from petastorm import TransformSpec
# MAGIC from petastorm.spark import SparkDatasetConverter, make_spark_converter
# MAGIC 
# MAGIC from pycocotools.coco import COCO
# MAGIC import pycocotools.mask as mask_util
# MAGIC from pycocotools.cocoeval import COCOeval
# MAGIC 
# MAGIC # import torch.distributed as dist
# MAGIC 
# MAGIC print(f"Using:\n - torch: {torch.__version__}\n - torchvision: {torchvision.__version__}\n - pytorch_lightning: {pl.__version__}")

# COMMAND ----------

DBFS_DATA_DIR = "dbfs:/user/nul/PennFudanPed"
LOCAL_DATA_DIR = DBFS_DATA_DIR.replace("dbfs:", "/dbfs")

GPU_COUNT = torch.cuda.device_count()
print(f"Found {GPU_COUNT if GPU_COUNT > 0 else 'no'} GPUs")

MAX_DEVICE_COUNT_TO_USE = 2

BATCH_SIZE = 4
MAX_EPOCH_COUNT = 15
STEPS_PER_EPOCH = 15

LR = 0.001
CLASS_COUNT = 1  # There is only one class here, not counting the background

EARLY_STOP_MIN_DELTA = 0.02
EARLY_STOP_PATIENCE = 3

TEST_IMAGE_COUNT = 2

# Set a cache directory on DBFS FUSE for intermediate data.
CACHE_DIR = "file:///dbfs/tmp/petastorm/cache"
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, CACHE_DIR)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Generic helper functions

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
  

def mask_2_bboxes(mask_arr):
  # Do not converted the mask to RGB because each color corresponds to a different instance
  # with 0 being background
  obj_ids = np.unique(mask_arr)
  
  # Ignore the first id, it is the background
  obj_ids = obj_ids[1:]

  # Split the color-encoded mask into a set of binary masks
  masks = mask_arr == obj_ids[:, None, None]

  # Get bounding box coordinates for each mask
  obj_count = len(masks)
  boxes = []
  for i in range(obj_count):
      pos = np.where(masks[i])
      xmin = np.min(pos[1])
      xmax = np.max(pos[1])
      ymin = np.min(pos[0])
      ymax = np.max(pos[0])
      boxes.append([xmin, ymin, xmax, ymax])

  return boxes

def show_image_with_bboxes(image, bboxes):
  fig, ax = plt.subplots()  # or just ax = plt.gca()
   
  plt.imshow(image)
  for (x1, y1, x2, y2) in bboxes:
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
  # Altenatively, to draw and save
#   fig = plt.figure()
#   ax = fig2.add_subplot(123, aspect='equal')
#   ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none'))
#   fig.savefig('rect2.png', dpi=90, bbox_inches='tight')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Data - the PennFudan dataset
# MAGIC 
# MAGIC This dataset contains ***170*** images of various sizes (see the samples below)
# MAGIC 
# MAGIC Download the dataset, unzip and store in DBFS to do this once to save time for future runs. 
# MAGIC 
# MAGIC ***Uncomment the code in the next cell, run and comment it out again***

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC #wget -P /tmp/ https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip 
# MAGIC #unzip /tmp/PennFudanPed.zip -d /dbfs/user/nul
# MAGIC 
# MAGIC #ls -la /dbfs/user/nul/PennFudanPed/PNGImages/*.png | wc -l
# MAGIC #ls -la /dbfs/user/nul/PennFudanPed/PedMasks/*.png | wc -l

# COMMAND ----------

display(dbutils.fs.ls(DBFS_DATA_DIR))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Display some data samples
# MAGIC 
# MAGIC Few images (amount controlled by `TEST_IMAGE_COUNT`) will be set aside for testing the model and will not be used for training, show those here

# COMMAND ----------

img_dir = os.path.join(LOCAL_DATA_DIR, "PNGImages")
mask_dir = os.path.join(LOCAL_DATA_DIR, "PedMasks")
test_image_paths = []
for img_name in list(sorted(os.listdir(img_dir)))[:TEST_IMAGE_COUNT]:
  test_img_path = os.path.join(img_dir, img_name)
  test_image_paths.append(test_img_path)
  test_img = Image.open(test_img_path)
  print(f"Image: {img_name} ({test_img.size})")
  plt.figure()
  plt.imshow(test_img)
  
  # Show the mask. Each mask instance has a different color, from zero to N. For better visualisation we can add a color palette to the mask
  test_mask_path = os.path.join(mask_dir, img_name.replace('.', '_mask.'))
  test_mask_img = Image.open(test_mask_path)
  print(f"- mask size: ({test_mask_img.size})")
  test_mask_img.putpalette([
    0, 0, 0, # black background
    255, 0, 0,
    255, 255, 0,
    255, 153, 0
  ])
  
  # Add bounding boxes to mask image
  show_image_with_bboxes(test_mask_img, mask_2_bboxes(np.array(test_mask_img)))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Create a custom dataset
# MAGIC 
# MAGIC Dataset format used for Object Detection models are not as straightforward as the one used for Image Classificaion. Here we are construcing a dataset from images and using a basic DataLoader. More advanced DataLoaders, like Petastorm's Apache Spark Dataframe converter, can also be used instead. For an example see [pytorch_lightning_with_petastorm](https://github.com/r3stl355/db-dl/blob/main/pytorch_lightning_with_petastorm.py)

# COMMAND ----------

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        
        # Load all image files, sorting them to ensure that they are aligned. Exclude images set aside for model testing
        self.img_dir = os.path.join(root, "PNGImages")
        self.mask_dir = os.path.join(root, "PedMasks")
        self.imgs = list(sorted(os.listdir(self.img_dir)))[TEST_IMAGE_COUNT:]
        self.masks = list(sorted(os.listdir(self.mask_dir)))[TEST_IMAGE_COUNT:]

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        
        # Do not converted the mask to RGB because each color corresponds to a different instance
        # with 0 being background
        boxes = mask_2_bboxes(np.array(Image.open(mask_path)))

        # Convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # There is only one class
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)   #suppose all instances are not crowd

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
      
# Some data transform helpers specific to object detection
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = TVF.to_tensor(image)
        return image, target
      
def get_transforms(is_train):
    transformers = [ToTensor()]
    if is_train:
        transformers.append(RandomHorizontalFlip(0.5))
    return Compose(transformers)
  
def collate_fn(batch):
    # `batch` in here is a list of (image, target) tuples, need to convert that to tuple of two lists
    return tuple(zip(*batch))

# COMMAND ----------

# Split the dataset into training and validation sets, make sure validation set does not use transforms used for train set
dataset = PennFudanDataset(LOCAL_DATA_DIR, get_transforms(is_train=True))
data_count = len(dataset)
val_split = int(data_count * 0.2)
indices = torch.randperm(len(dataset)).tolist()
train_dataset = torch.utils.data.Subset(dataset, indices[:-val_split])
val_dataset = torch.utils.data.Subset(PennFudanDataset(LOCAL_DATA_DIR, get_transforms(is_train=False)), indices[-val_split:])

print(f"Dataset size: {data_count} (train: {len(train_dataset)}, val: {len(val_dataset)})")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Create DataLoaders

# COMMAND ----------

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### The Model
# MAGIC 
# MAGIC Metrics used for Object Detection (or segmentation) model validation are more complex compared to Classification models, e.g. Accuracy is not sufficient to measure OD model performance. Metric used here Intersection over Union (IoU), computed following the COCO metric for intersection over union. Again, we used implementation from https://github.com/pytorch/vision/blob/main/references/detection for metric calculation (simplified for bounding box evaluation only)

# COMMAND ----------

class CocoBboxEvaluator:
    def __init__(self, coco_gt):
        self.coco_gt = copy.deepcopy(coco_gt)
        self.iou_types = ["bbox"]
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            with redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = self._evaluate(coco_eval)
            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            eval_imgs = np.concatenate(self.eval_imgs[iou_type], 2)
            self._create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, eval_imgs)

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        return self.prepare_for_coco_detection(predictions)

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = self._convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results


    def _convert_to_xywh(self, boxes):
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

    def _evaluate(self, imgs):
      with redirect_stdout(io.StringIO()):
          imgs.evaluate()
      return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))


    def _merge(self, img_ids, eval_imgs):
        all_img_ids = self._all_gather(img_ids)
        all_eval_imgs = self._all_gather(eval_imgs)

        merged_img_ids = []
        for p in all_img_ids:
            merged_img_ids.extend(p)

        merged_eval_imgs = []
        for p in all_eval_imgs:
            merged_eval_imgs.append(p)

        merged_img_ids = np.array(merged_img_ids)
        merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

        # keep only unique (and in sorted order) images
        merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
        merged_eval_imgs = merged_eval_imgs[..., idx]

        return merged_img_ids, merged_eval_imgs


    def _create_common_coco_eval(self, coco_eval, img_ids, eval_imgs):
        img_ids, eval_imgs = self._merge(img_ids, eval_imgs)
        img_ids = list(img_ids)
        eval_imgs = list(eval_imgs.flatten())

        coco_eval.evalImgs = eval_imgs
        coco_eval.params.imgIds = img_ids
        coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

    def _all_gather(self, data):
      
      # Implement this if distributed training is used, e.g. Horovod
      raise NotImplementedError
      #      """
      #     Run all_gather on arbitrary picklable data (not necessarily tensors)
      #     Args:
      #         data: any picklable object
      #     Returns:
      #         list[data]: list of data gathered from each rank
      #     """
      #     world_size = get_world_size()
      #     if world_size == 1:
      #         return [data]
      #     data_list = [None] * world_size
      #     torch.distributed.all_gather_object(data_list, data)
      #     return data_list
    
# This will create a COCO dataset from the iteratable dataset we used for Object Detection training (each Dataset element is a tuple of Image array and Targets dictionary)
def convert_to_coco(source_dataset):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(source_dataset)):
        img, targets = source_dataset[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"]
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### The actual model
# MAGIC 
# MAGIC We used a pre-trained torchvision Object Detection model wrapped into the PyTorch Lightning model 

# COMMAND ----------

class LitDetectionModel(pl.LightningModule):
  def __init__(self, lr=LR, logging_level=logging.INFO, class_count=CLASS_COUNT, device_id=0, device_count=1):
      super().__init__()
      self.lr = lr
      self.model = self.get_model(class_count, lr)
      self.state = {"epochs": 0}
      self.logging_level = logging_level
      self.device_id = device_id
      self.device_count = device_count

      # TODO: find a right place to implement this, in `validation_dataloader`?
      # COCO validation dataset has to be created only once and can be re-used in multiple epochs
      self.coco = convert_to_coco(val_data_loader.dataset)
      self.coco_evaluator = None
      
      if self.logging_level == logging.DEBUG:
        print(f"--> [{self.device_id}] Model initilised")

  def train_dataloader(self):
      return train_data_loader

  def val_dataloader(self):
    if self.logging_level == logging.DEBUG:
        print(f" - [{self.device_id}] val_dataloader")
      
    # Unlike COCO dataset, COCO evaluator needs to be created for every evaluation cycle
    if self.coco_evaluator:
        del self.coco_evaluator.coco_gt
        del self.coco_evaluator
    self.coco_evaluator = CocoBboxEvaluator(self.coco)
    return val_data_loader
        
  def configure_optimizers(self):
      optimizer = torch.optim.SGD(self.model.roi_heads.box_predictor.parameters(), lr=self.lr, momentum=0.9)
      return optimizer

  def forward(self, images, targets):
      return self.model(images, targets)
  
  def on_train_epoch_start(self):
      if self.logging_level in (logging.DEBUG, logging.INFO):
        print(f"++ [{self.device_id}] Epoch: {self.state['epochs']}")
      self.state["epochs"] += 1
    
  def training_step(self, batch, batch_idx):
    
      # The model will take a list of images and and a list of target dictionaries
      images, targets = batch
      images = list(image for image in images)
      targets = [{k: v for k, v in t.items()} for t in targets]  # `targets` here is a tuple, not a list

      # Output of training call is dict of losses
      output = self(images, targets)
      loss = sum(loss for loss in output.values())
      self.log("train_loss", loss, prog_bar=True)

      if self.logging_level == logging.DEBUG:
          print(f" - [{self.device_id}] training batch: {batch_idx}{'' if batch_idx > 0 else ' (batch size: ' + str(len(images)) + ')'}, loss: {loss}")

      return loss

  def validation_step(self, batch, batch_idx):
      images, targets = batch
      images = list(image for image in images)
      targets = [{k: v for k, v in t.items()} for t in targets]
      outputs = self(images, targets)

      # Unlike in training step, output of validation is a list of predictions
      res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
      self.coco_evaluator.update(res)

      if self.logging_level == logging.DEBUG:
          print(f" - [{self.device_id}] validation, batch: {batch_idx}{'' if batch_idx > 0 else ' (batch size: ' + str(len(images)) + ')'}")

      # Nothing to log or return here
      return {}
  
  def validation_epoch_end(self, outputs):
    
      if self.device_count > 0:
        self.coco_evaluator.synchronize_between_processes()
      
      self.coco_evaluator.accumulate()
      self.coco_evaluator.summarize()
      
      # Use first stats as a final validation score
      score = self.coco_evaluator.coco_eval["bbox"].stats[0]
      self.log("val_score", score, prog_bar=True)

      if self.logging_level == logging.DEBUG:
          print(f" - [{self.device_id}] validation score: {score}")

      return {'val_score': score}
    
  def get_model(self, class_count, lr):
    
      # Use a pre-trained model from `torchvision`
      model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

      # Freeze parameters in the feature extraction layers
      for param in model.parameters():
          param.requires_grad = False

      # Replace the classifier with a new one, add one more class for background
      in_features = model.roi_heads.box_predictor.cls_score.in_features
      model.roi_heads.box_predictor = FastRCNNPredictor(in_features, class_count+1)

      return model

# COMMAND ----------

def train(gpus=0, strategy=None, device_id=0, device_count=1, logging_level=logging.INFO):
  
  start = dt.datetime.now()

  if device_id == 0:
    print(f"Train on {str(max(gpus, device_count)) + ' GPU' + ('s' if gpus > 1 or device_count > 1 else '') if gpus > 0  else 'CPU'}:")
    print(f" - max epoch count: {MAX_EPOCH_COUNT}\n - batch size: {BATCH_SIZE*device_count}\n - steps per epoch: {STEPS_PER_EPOCH}")
    print(f" - start time: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n======================\n")

  # Early stopping will monitor a validation score
  verbose = True if device_id == 0 else False
  stopper = EarlyStopping(monitor="val_score", min_delta=EARLY_STOP_MIN_DELTA, patience=EARLY_STOP_PATIENCE, verbose=verbose, mode='max', check_on_train_epoch_end=True)
  callbacks = [stopper]
  
  trainer = pl.Trainer(
      gpus=gpus,
      max_epochs=MAX_EPOCH_COUNT,
      log_every_n_steps=1,
      num_sanity_val_steps=0,  # don't do sanity validations, can cause issues if Petastorm infinite batch dataset is used
      reload_dataloaders_every_n_epochs=1,  # set this to 1, could be important for some cases, e.g. if Petastorm is used
      callbacks=callbacks,
      strategy=strategy,
      default_root_dir='/tmp/lightning_logs'
  )

  model = LitDetectionModel(device_id=device_id, device_count=device_count, logging_level=logging.INFO)
  trainer.fit(model)

  if device_id == 0:
    report_duration(f"Training", start)
  
  return model.model if device_id == 0 else None

# COMMAND ----------

# Training on CPU is too slow, uncomment and run if GPUs are are not available
# cpu_model = train()

# COMMAND ----------

gpu_model = train(gpus=1)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Training Results
# MAGIC 
# MAGIC Train on 1 GPU:
# MAGIC  - max epoch count: 15
# MAGIC  - batch size: 4
# MAGIC  - steps per epoch: 15
# MAGIC  - start time: 2022-01-12 14:10:16
# MAGIC 
# MAGIC ======================
# MAGIC 
# MAGIC ++ [0] Epoch: 9
# MAGIC 
# MAGIC IoU metric: bbox
# MAGIC - Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.712
# MAGIC - ...
# MAGIC 
# MAGIC Monitored metric val_score did not improve in the last 3 records. Best score: ***0.713***. Signaling Trainer to stop.
# MAGIC 
# MAGIC -- Training completed in ***1 minutes 46 seconds*** at 2022-01-12 14:12:03
# MAGIC 
# MAGIC ***Observations:*** - tuning this model on GPU is really fast compared to CPU

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Test the model
# MAGIC 
# MAGIC Use the images we set aside for testing
# MAGIC 
# MAGIC ***Observations:***
# MAGIC - pretty good detections with high confidence
# MAGIC - second test image has two correct detections of overlapping objects whereas the original dataset shows only 1 object (as seen in test image display done earlier)

# COMMAND ----------

threshold = 0.5

gpu_model.eval()
if GPU_COUNT > 0:
  gpu_model.to("cuda")
  
test_images = []
for img_path in test_image_paths:
  test_images.append(Image.open(img_path).convert("RGB"))

test_tensors = [TVF.to_tensor(img) for img in test_images]
if GPU_COUNT > 0:
  test_tensors = [t.to("cuda") for t in test_tensors]

outputs = gpu_model(test_tensors)

for i, test_image in enumerate(test_images):
    boxes = outputs[i]['boxes'].data.cpu().numpy()
    scores = outputs[i]['scores'].data.cpu().numpy()

    filter = scores >= threshold
    boxes = boxes[filter].astype(np.int32)
    scores = scores[filter]
    print(f"Image: {os.path.split(test_image_paths[i])[1]}, boxes: {boxes}, scores {scores}")
    show_image_with_bboxes(test_image, boxes)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## TODO
# MAGIC 
# MAGIC - Add MLFlow model logging
# MAGIC - Add an example of using a model in Apache Spark UDF

# COMMAND ----------


