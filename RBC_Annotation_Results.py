import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
import torch
from torchvision import datasets, models
from torch.utils.data import DataLoader
import copy
import math
import re
import cv2
import albumentations as A  # our data augmentation library
# remove warnings (optional)
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm # progress bar
from pycocotools.coco import COCO
# Now, we will define our transforms
from albumentations.pytorch import ToTensorV2


# User parameters
SAVE_NAME      = "./Models/OSRS_Mining-0.model"
USE_CHECKPOINT = True
IMAGE_SIZE     = int(re.findall(r'\d+', SAVE_NAME)[-1] ) # Row and column size 
DATASET_PATH   = "./Dataset/"



def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec) ) )



# Starting stopwatch to see how long process takes
start_time = time.time()

torch.cuda.empty_cache()


dataset_path = DATASET_PATH



#load classes
coco = COCO(os.path.join(DATASET_PATH, "test", "_annotations.coco.json"))
categories = coco.cats
data = coco.dataset
annotations = data['annotations']
n_classes = len(categories.keys())
categories

# Only care about id/key/label 2 (RBC)

previous_image_id = 9999
count_rbc = 0
count_rbc_list = []

for id_index, info in enumerate(annotations):
    image_id = info['image_id']
    category_id = info['category_id']
    
    if image_id != previous_image_id:
        count_rbc_list.append(count_rbc)
        count_rbc = 0
    
    if category_id == 2:
        count_rbc += 1
    
    previous_image_id = image_id







print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)
