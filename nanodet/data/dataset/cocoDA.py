# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
import random
from .baseDA import BaseDatasetDA


class CocoDatasetDA(BaseDatasetDA):
    def get_data_info(self, ann_path,ann_target_path):
        """
        Load basic information of dataset such as image path, label and so on.
        :param ann_path: coco json file path
        :return: image info:
        [{'license': 2,
          'file_name': '000000000139.jpg',
          'coco_url': 'http://images.cocodataset.org/val2017/000000000139.jpg',
          'height': 426,
          'width': 640,
          'date_captured': '2013-11-21 01:34:01',
          'flickr_url':
              'http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg',
          'id': 139},
         ...
        ]
        """
        self.coco_api = COCO(ann_path)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.class_names = [cat["name"] for cat in self.cats]
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)

        self.coco_api_target = COCO(ann_target_path)
        self.img_target_ids = sorted(self.coco_api_target.imgs.keys())
        img_target_info = self.coco_api_target.loadImgs(self.img_target_ids)
        return img_info, img_target_info

    def get_per_img_info(self, idx, DA = False):
        if DA:
            len_data_target_info= len(self.data_target_info)
            if idx>=len_data_target_info:
                img_info = self.data_target_info[abs(len_data_target_info-idx)]
            else:
                img_info = self.data_target_info[idx]
        else:
            img_info = self.data_info[idx]
        file_name = img_info["file_name"]
        height = img_info["height"]
        width = img_info["width"]
        id = img_info["id"]
        if not isinstance(id, int):
            raise TypeError("Image id must be int.")
        info = {"file_name": file_name, "height": height, "width": width, "id": id}
        return info

    def get_img_annotation(self, idx):
        """
        load per image annotation
        :param idx: index in dataloader
        :return: annotation dict
        """
        img_id = self.img_ids[idx]
        ann_ids = self.coco_api.getAnnIds([img_id])
        anns = self.coco_api.loadAnns(ann_ids)
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        if self.use_instance_mask:
            gt_masks = []
        if self.use_keypoint:
            gt_keypoints = []
        for ann in anns:
            if ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get("iscrowd", False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann["category_id"]])
                if self.use_instance_mask:
                    gt_masks.append(self.coco_api.annToMask(ann))
                if self.use_keypoint:
                    gt_keypoints.append(ann["keypoints"])
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        annotation = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore
        )
        if self.use_instance_mask:
            annotation["masks"] = gt_masks
        if self.use_keypoint:
            if gt_keypoints:
                annotation["keypoints"] = np.array(gt_keypoints, dtype=np.float32)
            else:
                annotation["keypoints"] = np.zeros((0, 51), dtype=np.float32)
        return annotation

    @staticmethod
    def copy_paste_syndata(img,syndata_path):
        file = random.choice(os.listdir(syndata_path))
        img_sys = cv2.imread(syndata_path+file,-1)
        img_new2 = img_sys[:,:,-1]
        img_new2 = cv2.resize(img_new2,(156,156),interpolation=cv2.INTER_NEAREST)
        img_new2 = np.stack([img_new2,img_new2,img_new2],-1)
        img2 = img_sys[:,:,:-1]
        img2 = cv2.resize(img2,(156,156))

        h,w,c = img.shape
        h2,w2,c2 = img2.shape
        img_new = np.zeros((h,w,c),dtype=np.uint8)
        #img_new2 = np.zeros((h2,w2,c2),dtype=np.uint8)
        img_new3 = np.zeros((h,w,c),dtype=np.uint8)
        img_new3.shape
        x_start = random.choice(range(img.shape[1]-img2.shape[1]))
        y_start = random.choice(range(400,img.shape[0]-img2.shape[0]-100))
        box = np.array([x_start,y_start,x_start+w2,y_start+h2])

        img_new[y_start:y_start+img2.shape[0],x_start:x_start+img2.shape[1]] = img2
        img_new3[y_start:y_start+img2.shape[0],x_start:x_start+img2.shape[1]] = img_new2

        result = cv2.bitwise_and(src1=img_new, src2=img_new3)
        i = result > 0  # pixels to replace
        img[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug
        return img,box


    # def get_train_data(self, idx):
    def get_train_data(self, idx, is_train=False):
        """
        Load image and annotation
        :param idx:
        :return: meta-data (a dict containing image, annotation and other information)
        """
        img_info = self.get_per_img_info(idx)

        img_target_info = self.get_per_img_info(idx, DA=True)
        file_name = img_info["file_name"]
        file_target_name = img_target_info['file_name']
        # if "woodscape" in file_name:
        #     name = file_name.split("woodscape_")[-1] + ".png"
        #     file_name = f"woodscape/{name}"
        # else:
        #     fname = "_".join(file_name.split("_")[:-1])
        #     file_name = fname + '/images/' + file_name
        image_path = os.path.join(self.img_path, file_name)
        image_target_path = os.path.join(self.img_target_path, file_target_name)

        img = cv2.imread(image_path)
        img_target = cv2.imread(image_target_path)
        # if self.resizecrop:
        #     img = cv2.resize(img, tuple(self.image_size))
        #     img_target = cv2.resize(img_target, tuple(self.image_size))
            
        #     pix_crop = [int((self.image_size[0] - self.input_size[0])/2), int(self.image_size[1] - self.input_size[1])]
        #     #opencv read HWC while size is defined as WH
        #     # img = img[pix_crop[0]:self.image_size[0]-pix_crop[0], pix_crop[1]:, :]
        #     img = img[ pix_crop[1]:, pix_crop[0]:self.image_size[0]-pix_crop[0], :]
        #     img_target = img_target[ pix_crop[1]:, pix_crop[0]:self.image_size[0]-pix_crop[0], :]

        if img is None:
            print("image {} read failed.".format(image_path))
            # raise FileNotFoundError("Cant load image! Please check image path!")
            return None
        if img_target is None:
            print("image {} read failed.".format(image_target_path))

        ann = self.get_img_annotation(idx)

        # if is_train and random.random()>=0.5 and img.shape[0] == 800:
        #     img,box = self.copy_paste_syndata(img,'/home/ubuntu/Workspace/datasets/od/synthesized_dog_samples/')
        #     ann['bboxes'] = np.concatenate((ann['bboxes'],np.expand_dims(box,0)),0)
        #     ann['labels'] = np.concatenate((ann['labels'],np.array([3])),0)

        meta = dict(
            img=img, img_info=img_info, gt_bboxes=ann["bboxes"], gt_labels=ann["labels"],
            img_target=img_target, img_target_info = img_target_info
        )
        if self.use_instance_mask:
            meta["gt_masks"] = ann["masks"]
        if self.use_keypoint:
            meta["gt_keypoints"] = ann["keypoints"]

        input_size = self.input_size
        if self.multi_scale:
            input_size = self.get_random_size(self.multi_scale, input_size)

        meta = self.pipeline(self, meta, input_size)

        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1))
        meta["img_target"] = torch.from_numpy(meta["img_target"].transpose(2, 0, 1))
        return meta

    def get_val_data(self, idx):
        """
        Currently no difference from get_train_data.
        Not support TTA(testing time augmentation) yet.
        :param idx:
        :return:
        """
        # TODO: support TTA
        return self.get_train_data(idx)
