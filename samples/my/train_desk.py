#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import desk
#get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR,"logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/mask_rcnn_desks_0050.h5")
# Download COCO trained weights from Releases if needed

# In[2]:


config = desk.DeskConfig()
config.display()


# In[3]:


dataset_t = desk.DeskDataset()
dataset_t.load_desk(subset='m')
dataset_t.prepare()

dataset_v = desk.DeskDataset()
dataset_v.load_desk(subset='val')
dataset_v.prepare()


# In[4]:


from mrcnn import visualize
from mrcnn.visualize import display_images
image_id = 0

image = dataset_t.load_image(image_id)
mask, class_ids = dataset_t.load_mask(image_id)
#visualize.display_top_masks(image, mask, class_ids, dataset_t.class_names)

image = dataset_v.load_image(image_id)
mask, class_ids = dataset_v.load_mask(image_id)
#visualize.display_top_masks(image, mask, class_ids, dataset_v.class_names)


# In[5]:


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,model_dir=MODEL_DIR)


# In[6]:


model.load_weights(COCO_MODEL_PATH, by_name=True,
                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                    "mrcnn_bbox", "mrcnn_mask"])


# In[ ]:


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_t, dataset_v, 
            learning_rate=config.LEARNING_RATE, 
            epochs=5, 
            layers='heads',
            #layers='4+'
            )


# In[ ]:





# In[ ]:




