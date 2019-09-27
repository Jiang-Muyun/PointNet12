import os
import json
import numpy as np
import random
from tqdm import tqdm
import h5py
from torch.utils.data import Dataset

def pcd_jitter(pcd, sigma=0.01, clip=0.05):
    N, C = pcd.shape
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip).astype(pcd.dtype)
    jittered_data += pcd
    return jittered_data

def pcd_normalize(pcd):
    pcd[:,0] = pcd[:,0] / 70
    pcd[:,1] = pcd[:,1] / 70
    pcd[:,2] = pcd[:,2] / 3
    pcd[:,3] = (pcd[:,3] - 0.5)/2
    pcd = np.clip(pcd,-1,1)
    return pcd

class_names = [
    'unlabelled',     # 0
    'car',            # 1
    'bicycle',        # 2
    'motorcycle',     # 3
    'truck',          # 4
    'other-vehicle',  # 5
    'person',         # 6
    'bicyclist',      # 7
    'motorcyclist',   # 8
    'road',           # 9
    'parking',        # 10
    'sidewalk',       # 11
    'other-ground',   # 12
    'building',       # 13
    'fence',          # 14
    'vegetation',     # 15
    'trunk',          # 16
    'terrain',        # 17
    'pole',           # 18
    'traffic-sign'    # 19
]

sem_kitti_slim_mapping = {
    'unlabelled':   'unlabelled', # 0
    'car':          'vehicle',    # 1
    'bicycle':      'vehicle',    # 2
    'motorcycle':   'vehicle',    # 3
    'truck':        'vehicle',    # 4
    'other-vehicle':'vehicle',    # 5
    'person':       'human',      # 6
    'bicyclist':    'human',      # 7
    'motorcyclist': 'human',      # 8
    'road':         'ground',     # 9
    'parking':      'ground',     # 10
    'sidewalk':     'ground',     # 11
    'other-ground': 'ground',     # 12
    'building':     'structure',  # 13
    'fence':        'structure',  # 14
    'vegetation':   'nature',     # 15
    'trunk':        'nature',     # 16
    'terrain':      'ground',     # 17
    'pole':         'structure',  # 18
    'traffic-sign': 'structure'   # 19
}
num_classes = 6
slim_class_names = ['unlabelled', 'vehicle', 'human', 'ground', 'structure', 'nature']
colors = [[0, 0, 0],[245, 150, 100],[30, 30, 255],[255, 0, 255],[0, 200, 255],[0, 175, 0]]
slim_colors = np.array(colors,np.uint8)
slim_colors_bgr = np.array([list(reversed(c)) for c in colors],np.uint8)
index_to_name = {i:name for i,name in enumerate(slim_class_names)}
name_to_index = {name:i for i,name in enumerate(slim_class_names)}
mapping_list = [slim_class_names.index(sem_kitti_slim_mapping[name]) for name in class_names]
mapping_pcd_img = np.array(mapping_list,dtype=np.int32)


def load_data(root, train = False, selected = None):
    def process_data(fp,key):
        data = fp[key+'/pt'][()].astype(np.float32)
        label = fp[key+'/label'][()].astype(np.uint8)
        return data, mapping_pcd_img[label]

    part_length = {'00': 4540,'01':1100,'02':4660,'03':800,'04':270,'05':2760,'06':1100,'07':1100,'08':4070,'09':1590,'10':1200}
    if selected is None:
        selected = part_length.keys()
    
    fp = h5py.File(root,'r')
    train_data, train_label, test_data, test_label= [],[],[],[]

    for part in tqdm(selected, dynamic_ncols=True):
        length = part_length[part]

        for index in range(length):
            key = '%s/%06d'%(part, index)

            if index < length * 0.3:
                data,label = process_data(fp, key)
                test_data.append(data)
                test_label.append(label)
            
            if train and index >= length * 0.3:
                data,label = process_data(fp, key)
                train_data.append(data)
                train_label.append(label)
            
    fp.close()
    return train_data, train_label, test_data, test_label

class SemKITTIDataLoader(Dataset):
    def __init__(self, data, labels, npoints = 6000, data_augmentation = False):
        self.data = data
        self.labels = labels
        self.npoints = npoints
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pcd = self.data[index]
        label = self.labels[index]

        pcd = pcd_normalize(pcd)

        if self.data_augmentation:
            pcd = pcd_jitter(pcd)

        length = pcd.shape[0]
        if length == self.npoints:
            pass
        elif length > self.npoints:
            start_idx = np.random.randint(0, length - self.npoints)
            end_idx = start_idx + self.npoints
            pcd = pcd[start_idx:end_idx]
            label = label[start_idx:end_idx]
        else:
            rows_short = self.npoints - length
            pcd = np.concatenate((pcd,pcd[0:rows_short]),axis=0)
            label = np.concatenate((label,label[0:rows_short]),axis=0)
        return pcd, label


def print_distro(labels):
    count = [0] * num_classes
    total = 0
    for label in labels:
        total += label.shape[0]
        for i in range(num_classes):
            count[i] += (label == i).sum()
    print((np.array(count)/total*100).astype(np.int))

if __name__ == "__main__":
    train_data, train_label, test_data, test_label = load_data('experiment/pts_sem_voxel_0.10.h5',True)
    print_distro(train_label)
    print_distro(test_label)