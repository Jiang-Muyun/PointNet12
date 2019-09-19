import os
import json
import warnings
import numpy as np
import gc
from tqdm import tqdm
import h5py
from torch.utils.data import Dataset
from .augmentation import rotate_point_cloud, jitter_point_cloud, point_cloud_normalize

def load_parts(fn_h5,parts):
    fp = h5py.File(fn_h5,'r')
    length = {'00': 4540,'01':1100,'02':4660,'03':800,'04':270,'05':2760,'06':1100,'07':1100,'08':4070,'09':1590,'10':1200}
    data = []
    labels = []
    for part in parts:
        for index in range(length[part]):
            key = '%s/%06d'%(part, index)
            tmp = fp[key][()]
            data.append(tmp[:,:3].astype(np.float32))
            labels.append(tmp[:,3].astype(np.int32))
        print('Load part %s: %s'%(part, len(data)))
    fp.close()
    return data, labels


def load_data(root, train = False):
    test_data, test_label = load_parts(root, ['08','09','10'])
    # test_data, test_label = load_parts(root, ['04'])
    if train:
        train_data, train_label = load_parts(root, ['00','01','02','03','04','05','06','07'])
        # train_data, train_label = load_parts(root, ['03'])
        return train_data, train_label, test_data, test_label
    else:
        return test_data, test_label
    

class SemKITTIDataLoader(Dataset):
    def __init__(self, data, labels, npoints = 5000, normalize=True ,data_augmentation = False):
        self.data = data
        self.labels = labels
        self.npoints = npoints
        self.normalize = normalize
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pcd = self.data[index]
        label = self.labels[index]

        if self.normalize:
            pcd = point_cloud_normalize(pcd)

        if self.data_augmentation:
            pcd = np.expand_dims(pcd,axis=0)
            pcd = rotate_point_cloud(pcd)
            pcd = jitter_point_cloud(pcd).astype(np.float32)
            pcd = np.squeeze(pcd, axis=0)

        choice = np.random.choice(pcd.shape[0], self.npoints, replace=True)
        pcd = pcd[choice]
        label = label[choice]

        return pcd, label