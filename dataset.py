import open3d as o3d
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import random
#from utils import *

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]

class ModelNet(data.Dataset):
    def __init__(self, split = "train", npoints = 8192, classes=[]):
        assert split in ["train", "test", "val"]
        self.split = split
        if split == "train" or split == "val":
            self.dataset_path = './data/modelnet10/point_clouds/train/pcd'
        elif split == "test":
            self.dataset_path = './data/modelnet10/point_clouds/test/pcd'

        self.npoints = npoints
        self.train = split == "train"

        self.model_list = os.listdir(self.dataset_path)
        if len(classes) > 0: # Only use specific classes specified in classes list
            self.model_list = [x for x in self.model_list for cls in classes if cls in x ]

        if split == "train":
            self.model_list = [x for x in self.model_list if int(x.split("_")[1]) >= 20]
        elif split == "val":
            self.model_list = [x for x in self.model_list if int(x.split("_")[1]) < 20]

        random.shuffle(self.model_list)
        self.len = len(self.model_list * 50)

    def __getitem__(self, index):
        model_id = self.model_list[index // 50]
        scan_id = index % 50
        def read_pcd(filename):
            pcd = o3d.io.read_point_cloud(filename)
            return torch.from_numpy(np.array(pcd.points)).float()
        partial = read_pcd(os.path.join(self.dataset_path, model_id,f'{scan_id}.pcd'))
        full_name = model_id.split("_")[0] + "_" + self.split + "_" + model_id
        complete = read_pcd(os.path.join("./data/modelnet10/point_clouds/complete/", '%s.pcd' % full_name))
        return model_id, resample_pcd(partial, 5000), resample_pcd(complete, self.npoints)

    def __len__(self):
        return self.len


class ShapeNet(data.Dataset): 
    def __init__(self, train = True, npoints = 81921):
        if train:
            self.list_path = './data/train.list'
        else:
            self.list_path = './data/val.list'
        self.npoints = npoints
        self.train = train

        with open(os.path.join(self.list_path)) as file:
            self.model_list = [line.strip().replace('/', '_') for line in file]
        random.shuffle(self.model_list)
        self.len = len(self.model_list * 50)

    def __getitem__(self, index):
        model_id = self.model_list[index // 50]
        scan_id = index % 50
        def read_pcd(filename):
            pcd = o3d.io.read_point_cloud(filename)
            return torch.from_numpy(np.array(pcd.points)).float()
        if self.train:
            partial = read_pcd(os.path.join("./data/train/", model_id + '_%d_denoised.pcd' % scan_id))
        else:
            partial = read_pcd(os.path.join("./data/val/", model_id + '_%d_denoised.pcd' % scan_id))
        complete = read_pcd(os.path.join("./data/complete/", '%s.pcd' % model_id))       
        return model_id, resample_pcd(partial, 5000), resample_pcd(complete, self.npoints)

    def __len__(self):
        return self.len