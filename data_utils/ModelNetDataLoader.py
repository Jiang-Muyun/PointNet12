import numpy as np
import warnings
import h5py
from torch.utils.data import Dataset
import sys
from .augmentation import rotate_point_cloud, jitter_point_cloud, point_cloud_normalize

class_names = ['airplane','bathtub','bed','bench','bookshelf','bottle',
                'bowl','car','chair','cone','cup','curtain','desk','door',
                'dresser','flower_pot','glass_box','guitar','keyboard','lamp',
                'laptop','mantel','monitor','night_stand','person','piano',
                'plant','radio','range_hood','sink','sofa','stairs','stool',
                'table','tent','toilet','tv_stand','vase','wardrobe','xbox']

def load_h5(h5_filename):
    f = h5py.File(h5_filename,'r')
    data = f['data'][:]
    label = f['label'][:]
    seg = []
    return (data, label, seg)

def load_data(path,train=True,classification = True):
    if train:
        data_train0, label_train0, Seglabel_train0  = load_h5(path + 'ply_data_train0.h5')
        data_train1, label_train1, Seglabel_train1 = load_h5(path + 'ply_data_train1.h5')
        data_train2, label_train2, Seglabel_train2 = load_h5(path + 'ply_data_train2.h5')
        data_train3, label_train3, Seglabel_train3 = load_h5(path + 'ply_data_train3.h5')
        data_train4, label_train4, Seglabel_train4 = load_h5(path + 'ply_data_train4.h5')
        train_data = np.concatenate([data_train0,data_train1,data_train2,data_train3,data_train4])
        train_label = np.concatenate([label_train0,label_train1,label_train2,label_train3,label_train4])
        train_Seglabel = np.concatenate([Seglabel_train0,Seglabel_train1,Seglabel_train2,Seglabel_train3,Seglabel_train4])

    data_test0, label_test0, Seglabel_test0 = load_h5(path + 'ply_data_test0.h5')
    data_test1, label_test1, Seglabel_test1 = load_h5(path + 'ply_data_test1.h5')
    test_data = np.concatenate([data_test0,data_test1])
    test_label = np.concatenate([label_test0,label_test1])

    test_Seglabel = np.concatenate([Seglabel_test0,Seglabel_test1])
    if train:
        if classification:
            return train_data, train_label, test_data, test_label
        else:
            return train_data, train_Seglabel, test_data, test_Seglabel
    else:
        if classification:
            return test_data, test_label
        else:
            return test_data, test_Seglabel


class ModelNetDataLoader(Dataset):
    def __init__(self, data, labels, augmentation = False):
        self.data = data
        self.labels = labels
        self.augmentation = augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pointcloud = self.data[index]
        label = self.labels[index]

        if self.data_augmentation:
            pcd = np.expand_dims(pcd,axis=0)
            pcd = rotate_point_cloud(pcd)
            pcd = jitter_point_cloud(pcd).astype(np.float32)
            pcd = np.squeeze(pcd, axis=0)

        return pointcloud, label