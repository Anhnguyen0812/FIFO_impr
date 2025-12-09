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

class cityscapesDataSet(data.Dataset):

    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), 
                 scale=True, mirror=True, ignore_label=255, set='val', 
                 use_segformer_norm=False, norm_cfg=None):
        """
        Args:
            use_segformer_norm: If True, use SegFormer normalization (RGB + mean/std normalize)
                               If False, use ResNet normalization (BGR + subtract mean)
            norm_cfg: Dict with 'mean', 'std', 'to_rgb' for SegFormer normalization
        """
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
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
        self.set = set
        for name in self.img_ids:
            img_file = osp.join(self.root, "Cityscapes/leftImg8bit/%s/%s" % (self.set, name))
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

        # resize
        w, h = image.size

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


if __name__ == '__main__':
    dst = GTA5DataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
