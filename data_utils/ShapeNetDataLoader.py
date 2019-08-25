# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')
import sys
sys.path.append('.')
from colors import *

def build_cache(root):
    cache = {}
    fp_category = os.path.join(root, 'synsetoffset2category.txt')
    category = {}
    with open(fp_category, 'r') as f:
        for line in f:
            line = line.strip().split()
            category[line[0]] = line[1]
    print_err(category)

    print_info('Building cache...')
    for item in category.keys():
        dir_point = os.path.join(root, category[item])
        fns = os.listdir(dir_point)
        print_kv('item', item)
        for fn in tqdm(fns):
            token = fn.split('.')[0]
            fn_full = os.path.join(dir_point, fn)
            cache[token] = np.loadtxt(fn_full).astype(np.float32)
            break
    
    print_info('Saving cache...')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

class PartNormalDataset(Dataset):
    def __init__(self, root, npoints=2500, split='train', normalize=True, jitter=False):
        # self.root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normalize = normalize
        self.jitter = jitter

        with open(self.catfile, 'r') as f:
            for line in f:
                line = line.strip().split()
                self.cat[line[0]] = line[1]
        # {'Airplane': '02691156', 'Bag': '02773838', 'Cap': '02954340', ...}

        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
            
        self.meta = {}
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
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

            # print(os.path.basename(fns))
            for fn in fns:
                self.meta[item].append(os.path.join(dir_point, fn))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = dict(zip(self.cat, range(len(self.cat))))
        print_kv('classes',self.classes)

        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
        self.cache = {}




    def __getitem__(self, index):
        print_err('cached',len(self.cache.keys()), index)
        if index in self.cache:
            point_set, normal, seg, classi = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            classi = self.classes[cat]
            classi = np.array([classi]).astype(np.int32)
            print_kv('fn',fn)
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:, 0:3]
            normal = data[:, 3:6]
            seg = data[:, -1].astype(np.int32)
            self.cache[index] = (point_set, normal, seg, classi)
        
        if self.normalize:
            point_set = pc_normalize(point_set)
            
        if self.jitter:
            jitter_point_cloud(point_set)
        
        choice = np.random.choice(len(seg), self.npoints, replace=True)

        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        normal = normal[choice, :]
        return point_set,classi, seg, normal


    def __len__(self):
        return len(self.datapath)

# class PartNormalDataset(Dataset):
#     def __init__(self, root, npoints=2500, split='train', normalize=True, jitter=False):
#         # self.root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
#         self.npoints = npoints
#         self.root = root
#         self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
#         self.cat = {}
#         self.normalize = normalize
#         self.jitter = jitter

#         with open(self.catfile, 'r') as f:
#             for line in f:
#                 ls = line.strip().split()
#                 self.cat[ls[0]] = ls[1]
#         self.cat = {k: v for k, v in self.cat.items()}
#         # print_kv('self.cat', self.cat)
#         # {'Airplane': '02691156', 'Bag': '02773838', 'Cap': '02954340', ...}

#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
#             train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
#             val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
#             test_ids = set([str(d.split('/')[2]) for d in json.load(f)])

#         self.meta = {}
#         for item in self.cat:
#             # print('category', item)
#             self.meta[item] = []
#             dir_point = os.path.join(self.root, self.cat[item])
#             fns = sorted(os.listdir(dir_point))

#             if split == 'trainval':
#                 fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
#             elif split == 'train':
#                 fns = [fn for fn in fns if fn[0:-4] in train_ids]
#             elif split == 'val':
#                 fns = [fn for fn in fns if fn[0:-4] in val_ids]
#             elif split == 'test':
#                 fns = [fn for fn in fns if fn[0:-4] in test_ids]
#             else:
#                 raise ValueError('Unknown split: %s. Exiting..' % (split))

#             # print(os.path.basename(fns))
#             # print(fns)
#             for fn in fns:
#                 self.meta[item].append(os.path.join(dir_point, fn))

#         self.datapath = []
#         for item in self.cat:
#             for fn in self.meta[item]:
#                 self.datapath.append((item, fn))

#         self.classes = dict(zip(self.cat, range(len(self.cat))))
#         print_kv('classes',self.classes)

#         self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
#                             'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
#                             'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
#                             'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
#                             'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
#         self.cache = {}

#     def __getitem__(self, index):
#         print_err('cached',len(self.cache.keys()), index)
#         if index in self.cache:
#             point_set, normal, seg, classi = self.cache[index]
#         else:
#             fn = self.datapath[index]
#             cat = self.datapath[index][0]
#             classi = self.classes[cat]
#             classi = np.array([classi]).astype(np.int32)
#             data = np.loadtxt(fn[1]).astype(np.float32)
#             print_kv('data',data.shape)
#             point_set = data[:, 0:3]
#             normal = data[:, 3:6]
#             seg = data[:, -1].astype(np.int32)
#             self.cache[index] = (point_set, normal, seg, classi)
        
#         if self.normalize:
#             point_set = pc_normalize(point_set)
            
#         if self.jitter:
#             jitter_point_cloud(point_set)
        
#         choice = np.random.choice(len(seg), self.npoints, replace=True)

#         # resample
#         point_set = point_set[choice, :]
#         seg = seg[choice]
#         normal = normal[choice, :]
#         return point_set,classi, seg, normal


#     def __len__(self):
#         return len(self.datapath)