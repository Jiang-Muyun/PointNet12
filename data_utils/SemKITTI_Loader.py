import os
import cv2
import json
import yaml
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import threading
import multiprocessing
from PIL import Image

from .kitti_utils import Semantic_KITTI_Utils
from .redis_utils import Mat_Redis_Utils

def pcd_jitter(pcd, sigma=0.01, clip=0.05):
    N, C = pcd.shape
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip).astype(pcd.dtype)
    jittered_data += pcd
    return jittered_data

def pcd_normalize(pcd):
    pcd = pcd.copy()
    pcd[:,0] = pcd[:,0] / 70
    pcd[:,1] = pcd[:,1] / 70
    pcd[:,2] = pcd[:,2] / 3
    pcd[:,3] = (pcd[:,3] - 0.5)*2
    pcd = np.clip(pcd,-1,1)
    return pcd

def pcd_unnormalize(pcd):
    pcd = pcd.copy()
    pcd[:,0] = pcd[:,0] * 70
    pcd[:,1] = pcd[:,1] * 70
    pcd[:,2] = pcd[:,2] * 3
    pcd[:,3] = pcd[:,3] / 2 + 0.5
    return pcd


class SemKITTI_Loader(Dataset):
    def __init__(self, root, npoints, train = True, subset = 'all'):
        self.root = root
        self.train = train
        self.npoints = npoints
        self.np_redis = Mat_Redis_Utils()
        self.utils = Semantic_KITTI_Utils(root,subset)

        part_length = {'00': 4540,'01':1100,'02':4660,'03':800,'04':270,'05':2760,
            '06':1100,'07':1100,'08':4070,'09':1590,'10':1200}

        self.keys = []
        alias = subset[0]
        
        if self.train:
            for part in ['00','01','02','03','04','05','06','07','09','10']:
                length = part_length[part]
                for index in range(0,length,2):
                    self.keys.append('%s/%s/%06d'%(alias, part, index))
        else:
            for part in ['08']:
                length = part_length[part]
                for index in range(0,length):
                    self.keys.append('%s/%s/%06d'%(alias, part, index))

    def __len__(self):
            return len(self.keys)

    def get_data(self, key):
        if not self.np_redis.exists(key):
            alias, part, index = key.split('/')
            point_cloud, label = self.utils.get(part, int(index))
            
            to_store = np.concatenate((point_cloud, label.reshape((-1,1)).astype(np.float32)),axis=1)
            self.np_redis.set(key, to_store)
        else:
            data = self.np_redis.get(key)
            point_cloud = data[:,:4]
            label = data[:,4].astype(np.int32)
        
        # Unnormalized Point cloud
        return point_cloud, label

    def __getitem__(self, index):
        point_cloud, label = self.get_data(self.keys[index])
        pcd = pcd_normalize(point_cloud)
        if self.train:
            pcd = pcd_jitter(pcd)

        length = pcd.shape[0]
        # if length == self.npoints:
        #     pass
        # elif length > self.npoints:
        #     start_idx = np.random.randint(0, length - self.npoints)
        #     end_idx = start_idx + self.npoints
        #     pcd = pcd[start_idx:end_idx]
        #     label = label[start_idx:end_idx]
        # else:
        #     rows_short = self.npoints - length
        #     pcd = np.concatenate((pcd,pcd[0:rows_short]),axis=0)
        #     label = np.concatenate((label,label[0:rows_short]),axis=0)

        choice = np.random.choice(length, self.npoints, replace=True)
        pcd = pcd[choice]
        label = label[choice]

        return pcd, label


