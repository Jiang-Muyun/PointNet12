import os
import json
import warnings
import numpy as np
import gc
from tqdm import tqdm
import h5py
from torch.utils.data import Dataset


def load_data(root):
    length = {
        '00': 4540,'01':1100,'02':4660,'03':800,'04':270,'05':2760,
        '06':1100,'07':1100,'08':4070,'09':1590,'10':1200
    }
    fp = h5py.File(root,'r')
    train_keys = []
    test_keys = []
    for part in ['00','01','02','03','04','05','06','07','08','09','10']:
        ids = list(fp[part].keys())
        keys = ['%s/%s'%(part,x) for x in ids]

        if len(keys) != length[part]:
            print('Inconsistent sequence %s:%d should be %d' % (part, len(keys), length[part]))
            # print(keys[0],keys[-1])
        # if part in ['00','01','02','03','04','05','06']:
        #     train_keys.extend(keys)
        # else:
        #     test_keys.extend(keys)


if __name__ == '__main__':
    load_data('/media/james/Ubuntu_Data/git/point_cloud/Open3D-Semantic-KITTI-Vis/tmp/sem_kitti.h5')