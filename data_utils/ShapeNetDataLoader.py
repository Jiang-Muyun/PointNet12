# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
import gc
from tqdm import tqdm
import h5py
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')
import sys
from .augmentation import rotate_point_cloud, jitter_point_cloud, point_cloud_normalize

class PartNormalDataset(Dataset):
    def __init__(self, root, cache = {}, npoints=2500, split='train', normalize=True, data_augmentation=False):
        self.npoints = npoints
        self.root = root
        self.category = {}
        self.normalize = normalize
        self.cache = cache
        self.data_augmentation = data_augmentation

        self.wordnet_id_to_category = {}
        with open(os.path.join(self.root, 'synsetoffset2category.txt'), 'r') as f:
            for line in f:
                line = line.strip().split()
                self.category[line[0]] = line[1]
                self.wordnet_id_to_category[line[1]] = line[0]

        fn_split = os.path.join(self.root, 'train_test_split')
        with open(os.path.join(fn_split,'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(fn_split,'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(fn_split,'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
            
        self.meta = {}
        for item in self.category:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.category[item])
            fns = sorted(os.listdir(dir_point))

            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                raise ValueError('Unknown split: %s. Exiting..' % (split))

            for fn in fns:
                self.meta[item].append(os.path.join(dir_point, fn))

        self.datapath = []
        for item in self.category:
            for fn in self.meta[item]:
                self.datapath.append(fn)

        self.classes = dict(zip(self.category, range(len(self.category))))
        # print('classes',self.classes.keys())

        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

    def __getitem__(self, index):
        fn_full = self.datapath[index]
        parts = fn_full.split('/')
        wordnet_id = parts[-2]
        category = self.wordnet_id_to_category[wordnet_id]
        cls_id = np.array([self.classes[category]]).astype(np.int32)
        token = parts[-1].split('.')[0]
        h5_index = '%s_%s'%(wordnet_id,token)

        if h5_index in self.cache.keys():
            data = self.cache[h5_index]
            pointcloud = data[:, 0:3]
            normal = data[:, 3:6]
            seg = data[:, -1].astype(np.int32)
        else:
            print('Error: cache miss',h5_index)
            data = np.loadtxt(fn_full).astype(np.float32)

        if self.normalize:
            pointcloud = point_cloud_normalize(pointcloud)
            
        if self.data_augmentation:
            pointcloud = np.expand_dims(pointcloud,axis=0)
            pointcloud = rotate_point_cloud(pointcloud)
            pointcloud = jitter_point_cloud(pointcloud).astype(np.float32)
            pointcloud = np.squeeze(pointcloud, axis=0)

        # resample
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        pointcloud = pointcloud[choice, :]
        seg = seg[choice]
        normal = normal[choice, :]
        return pointcloud, cls_id, seg, normal

    def __len__(self):
        return len(self.datapath)