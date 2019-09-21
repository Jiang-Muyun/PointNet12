import os
import json
import numpy as np
from tqdm import tqdm
import h5py
from torch.utils.data import Dataset
import sys
sys.path.append('.')
from data_utils.augmentation import rotate_point_cloud, jitter_point_cloud, point_cloud_normalize

# kitti_class_names = [
#     'unlabelled',    # 1
#     'road',          # 2
#     'sidewalk',      # 3
#     'building',      # 4
#     'wall',          # 5
#     'fence',         # 6
#     'pole',          # 7
#     'traffic_light', # 8
#     'traffic_sign',  # 9
#     'vegetation',    # 10
#     'terrain',       # 11
#     'sky',           # 12
#     'person',        # 13
#     'rider',         # 14
#     'car',           # 15
#     'truck',         # 16
#     'bus',           # 17
#     'train',         # 18
#     'motorcycle',    # 19
#     'bicycle'        # 20
# ]

# kitti_mapping = {
#     'unlabelled':    'unlabelled', # 0
#     'road':          'ground',     # 1
#     'sidewalk':      'ground',     # 2
#     'building':      'structure',  # 3
#     'wall':          'structure',  # 4
#     'fence':         'structure',  # 5
#     'pole':          'structure',  # 6
#     'traffic_light': 'structure',  # 7
#     'traffic_sign':  'structure',  # 8
#     'vegetation':    'nature',     # 9
#     'terrain':       'nature',     # 10
#     'sky':           'nature',     # 11
#     'person':        'human',      # 12
#     'rider':         'human',      # 13
#     'car':           'vehicle',    # 14
#     'truck':         'vehicle',    # 15
#     'bus':           'vehicle',    # 16
#     'train':         'vehicle',    # 17
#     'motorcycle':    'vehicle',    # 18
#     'bicycle':       'vehicle'     # 19
# }

num_classes = 20
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

mapping = {
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
reduced_class_names = ['unlabelled', 'vehicle', 'human', 'ground', 'structure', 'nature']
label_id_to_name = {i:cat for i,cat in enumerate(reduced_class_names)}
mapping_list = [reduced_class_names.index(mapping[name]) for name in class_names]
mapping = np.array(mapping_list,dtype=np.int32)

def process_data(fp,key):
    data = fp[key+'/pt'][()].astype(np.float32)
    label = mapping[fp[key+'/label'][()].astype(np.uint8)]
    # label = tmp[:,3].astype(np.uint8)
    # mark_removed = label != 3
    # data = data[mark_removed]
    # label = label[mark_removed]
    return data, label

def load_data(root, train = False):
    part_length = {'00': 4540,'01':1100,'02':4660,'03':800,'04':270,'05':2760,'06':1100,'07':1100,'08':4070,'09':1590,'10':1200}
    # part_length = {'04':270}
    fp = h5py.File(root,'r')
    train_data, train_label, test_data, test_label= [],[],[],[]

    for part in part_length.keys():
        length = part_length[part]

        for index in tqdm(range(length), desc=' > Loading %s: %s'%(part, length)):
            key = '%s/%06d'%(part, index)

            if index < length * 0.4:
                data,label = process_data(fp, key)
                test_data.append(data)
                test_label.append(label)
            
            if train and index >= length * 0.4:
                data,label = process_data(fp, key)
                train_data.append(data)
                train_label.append(label)
            
    fp.close()
    return train_data, train_label, test_data, test_label


class SemKITTIDataLoader(Dataset):
    def __init__(self, data, labels, npoints = 6000, normalize=True ,data_augmentation = False):
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
            pcd[:,0] = pcd[:,0] / 70
            pcd[:,1] = pcd[:,1] / 70
            pcd[:,2] = pcd[:,2] / 3
            pcd[:,3] = (pcd[:,3] - 0.5)/2
            pcd = np.clip(pcd,-1,1)

        if self.data_augmentation:
            pcd = np.expand_dims(pcd,axis=0)
            # pcd = rotate_point_cloud(pcd)
            pcd = jitter_point_cloud(pcd).astype(np.float32)
            pcd = np.squeeze(pcd, axis=0)
        
        choice = np.random.choice(pcd.shape[0], self.npoints, replace=True)
        pcd = pcd[choice]
        label = label[choice]
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