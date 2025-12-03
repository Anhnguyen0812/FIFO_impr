"""
Dataset classes with proper preprocessing for SegFormer models.
Uses RGB format with correct normalization parameters.
"""
import os
import os.path as osp
import numpy as np
import torch
from torch.utils import data
from PIL import Image

# SegFormer normalization parameters (ImageNet stats used by OpenMMLab)
SEGFORMER_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
SEGFORMER_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


class SegformerFoggyZurichDataSet(data.Dataset):
    """Foggy Zurich dataset with SegFormer preprocessing."""
    
    def __init__(self, root, list_path, max_iters=None, crop_size=(1152, 648)):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.mean = SEGFORMER_MEAN
        self.std = SEGFORMER_STD
        
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(19)))

        for name in self.img_ids:
            img_file = osp.join(self.root, "foggy_zurich/Foggy_Zurich/%s" % (name))
            label_file = osp.join(self.root, "foggy_zurich/Foggy_Zurich/%s" % ("gt_labelTrainIds/"+name[4:]))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        datafiles = self.files[index]
      
        # Load image in RGB format (PIL default)
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        label = np.array(label, dtype=np.float32)
        name = datafiles["name"]

        # Resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        image = np.asarray(image, np.float32)  # Keep RGB format

        # Process label
        label_img = Image.fromarray(label.astype(np.uint8))
        lbl = np.array(label_img.resize((self.crop_size[0], self.crop_size[1]), Image.NEAREST), dtype=np.float32)
        label = lbl.astype(int)

        size = image.shape
        
        # SegFormer normalization: (RGB - mean) / std
        # Keep RGB format (no [:, :, ::-1])
        image = (image - self.mean) / self.std
        image = image.transpose((2, 0, 1))  # HWC -> CHW

        return image.copy(), label.copy(), np.array(size), name


class SegformerFoggyDrivingDataSet(data.Dataset):
    """Foggy Driving dataset with SegFormer preprocessing."""
    
    def __init__(self, root, list_path, max_iters=None, scale=None):
        self.root = root
        self.list_path = list_path
        self.mean = SEGFORMER_MEAN
        self.std = SEGFORMER_STD
        self.scale = scale
        
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.ignore_index = 255

        for name in self.img_ids:
            img_file = osp.join(self.root, name)
            self.files.append({
                "img": img_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        datafiles = self.files[index]
        
        # Load image in RGB format
        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]

        # Get original size
        w, h = image.size
        
        # Apply scale if specified
        if self.scale is not None:
            new_w = int(w * self.scale)
            new_h = int(h * self.scale)
            image = image.resize((new_w, new_h), Image.BICUBIC)
        
        image = np.asarray(image, np.float32)  # Keep RGB format
        
        size = [image.shape[0], image.shape[1]]
        
        # SegFormer normalization: (RGB - mean) / std
        image = (image - self.mean) / self.std
        image = image.transpose((2, 0, 1))  # HWC -> CHW

        return image.copy(), size, name


class SegformerCityscapesDataSet(data.Dataset):
    """Cityscapes dataset with SegFormer preprocessing."""
    
    def __init__(self, root, list_path, max_iters=None, crop_size=(2048, 1024), 
                 scale=False, mirror=False, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = SEGFORMER_MEAN
        self.std = SEGFORMER_STD
        self.is_mirror = mirror
        self.set = set
        
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        
        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/leftImg8bit/%s/%s" % (self.set, name))
            self.files.append({
                "img": img_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        # Load image in RGB format
        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]

        # Get size
        w, h = image.size
        
        # Resize to crop_size
        if isinstance(self.crop_size, tuple):
            if isinstance(self.crop_size[0], float):
                new_w = int(self.crop_size[0])
                new_h = int(self.crop_size[1])
            else:
                new_w, new_h = self.crop_size
            image = image.resize((new_w, new_h), Image.BICUBIC)

        image = np.asarray(image, np.float32)  # Keep RGB format
        size = image.shape
        
        # SegFormer normalization: (RGB - mean) / std
        image = (image - self.mean) / self.std
        image = image.transpose((2, 0, 1))  # HWC -> CHW

        return image.copy(), np.array(size), name
