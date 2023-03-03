# %% [markdown]
# # How to Load Nanodet in PyTorch
# 
# This article is an introductory tutorial to infer nanodet with PyTorch.
# 
# **Note**, we suppose this notebook is in the root directory of nonadet!
# 
# ## Install nanodet first
# 
# For us to begin with, PyTorch should be installed. TorchVision is also required since we will be using it as our model zoo.
# 
# A quick solution is to install via pip
# 
# ```shell
# pip install torch==1.7.1
# pip install torchvision==0.8.2
# ```
# 
# or please refer to official site https://pytorch.org/get-started/locally/
# 
# PyTorch versions should be backwards compatible but should be used with the proper TorchVision version.
# 
# And then don't forget to install other dependencies.
# 
# ```shell
# pip install -r requirements.txt
# ```
# 
# Next enter the key part, let's install `nanodet`!
# 
# ```shell
# python setup.py develop
# ```
# 
# ## Set Environmental Parameters

# %%
import os
import cv2
import torch
import json

from tqdm import tqdm

# %%
from nanodet.util import cfg, load_config, Logger

from nanodet.util import overlay_bbox_cv
from pycocotools.coco import COCO

from demo.demo import Predictor

from nanodet.evaluator import build_evaluator
from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset

from nanodet.util import convert_avg_params, gather_results, mkdir

import torch.distributed as dist

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = torch.device('cuda')

# %%
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# %% [markdown]
# ## Set Model Configuration and Logger


# 
# %%
ROOTPATH = '/home/ubuntu/Workspace/thinhplg-dev/domain_adaptation/od/nanodet_v2'

config_path_DA = '/home/ubuntu/Workspace/thinhplg-dev/domain_adaptation/od/nanodet_v2/config/DA/Inhouse-a2d2.yml'
load_config(cfg, config_path_DA)

config_path = '/home/ubuntu/Workspace/thinhplg-dev/domain_adaptation/od/nanodet_v2/config/inhouse-nanodet-plus-m_320_darknet-fisheye-newloss.yml'
load_config(cfg, config_path)



# model_path = os.path.join(ROOTPATH,cfgDA.save_dir,'model_best/model_best.ckpt')
# save_path = os.path.join(ROOTPATH,'results',cfgDA.save_dir)

# if not os.path.exists(save_path):
#     os.makedirs(save_path)

# valDataFolder = cfgDA.data.val.img_path
# valDataPath = cfgDA.data.val.ann_path

# valcoco_annotations = COCO(annotation_file=valDataPath)
# imagesList = dict(valcoco_annotations.imgs)

# # print(imagesList[:2])

# logger = Logger(-1, use_tensorboard=False)
# predictor = Predictor(cfgDA, model_path, logger, device=device)




# # # %%
# print('RUN INFER')
# imshow_scale=1.0
# count = 2
# for image in tqdm(imagesList.values()):
#     # for i in range(count):
#     file_name = image['file_name']
#     image_path=os.path.join(valDataFolder,file_name)
#     meta, res = predictor.inference(image_path)
#     result = overlay_bbox_cv(meta['raw_img'][0], res[0], cfgDA.class_names, score_thresh=0.35)
#     out_path = os.path.join(save_path,os.path.basename(image_path))
#     # print("Done: {}".format(out_path))

#     cv2.imwrite(out_path,cv2.resize(result, None, fx=imshow_scale, fy=imshow_scale))