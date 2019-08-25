# *_*coding:utf-8 *_*
import os
from torch.utils.data import Dataset
import numpy as np
import h5py
import sys
sys.path.append('.')
from colors import *

classes = ['ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter']
class2label = {cls: i for i,cls in enumerate(classes)}

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    print_debug(filename)
    return load_h5(filename)

def recognize_all_data(root, test_area = 5):
    ALL_FILES = getDataFiles(os.path.join(root, 'all_files.txt'))
    room_filelist = [line.rstrip() for line in open(os.path.join(root, 'room_filelist.txt'))]
    data_batch_list = []
    label_batch_list = []
    for h5_filename in ALL_FILES:
        fn = os.path.join(os.path.abspath(os.path.join(root,'..')),h5_filename)
        data_batch, label_batch = loadDataFile(fn)
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)
    data_batches = np.concatenate(data_batch_list, 0)
    label_batches = np.concatenate(label_batch_list, 0)

    test_area = 'Area_' + str(test_area)
    train_idxs = []
    test_idxs = []
    for i, room_name in enumerate(room_filelist):
        if test_area in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)

    train_data = data_batches[train_idxs, ...]
    train_label = label_batches[train_idxs]
    test_data = data_batches[test_idxs, ...]
    test_label = label_batches[test_idxs]
    return train_data,train_label,test_data,test_label


class S3DISDataLoader(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
