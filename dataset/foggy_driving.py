import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
from os.path import join
import json
import scipy.misc as m

class foggydrivingDataSet(data.Dataset):
    colors = [  
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    def __init__(self, root, list_path, max_iters=None, mean=(104.00698793, 116.66876762, 122.67891434), 
                 scale=None, use_segformer_norm=False, norm_cfg=None):
        """
        Args:
            use_segformer_norm: If True, use SegFormer normalization (RGB + mean/std normalize)
                               If False, use ResNet normalization (BGR + subtract mean)
            norm_cfg: Dict with 'mean', 'std', 'to_rgb' for SegFormer normalization
        """
        self.root = root
        self.list_path = list_path
        self.mean = mean
        self.scale = scale
        self.use_segformer_norm = use_segformer_norm
        self.norm_cfg = norm_cfg
        if use_segformer_norm and norm_cfg is None:
            # Default SegFormer normalization
            self.norm_cfg = dict(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True
            )
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

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

        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]
        
        if self.scale != 1:
            w, h = image.size
            new_size = (int(w*self.scale), int(h*self.scale))
            image = image.resize(new_size, Image.BICUBIC)
        image = np.asarray(image, np.float32)

        size = image.shape
        
        # Apply normalization based on model type
        if self.use_segformer_norm:
            # SegFormer: Keep RGB, normalize with mean and std
            if self.norm_cfg['to_rgb']:
                # Already RGB, no need to convert
                pass
            else:
                image = image[:, :, ::-1]  # Convert to BGR if needed
            
            # Normalize: (image - mean) / std
            mean = np.array(self.norm_cfg['mean'], dtype=np.float32)
            std = np.array(self.norm_cfg['std'], dtype=np.float32)
            image = (image - mean) / std
        else:
            # ResNet: Convert to BGR, subtract mean only
            image = image[:, :, ::-1]  # change to BGR
            image -= self.mean
        
        image = image.transpose((2, 0, 1))

        return image.copy(), np.array(size), name
    
    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

if __name__ == '__main__':
    dst = foggydrivingDataSet("/root/data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
