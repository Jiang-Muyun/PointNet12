import os
import json
import yaml
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset
import threading
import multiprocessing 

def pcd_jitter(pcd, sigma=0.01, clip=0.05):
    N, C = pcd.shape
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip).astype(pcd.dtype)
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

num_classes = 20
index_to_name = {i:name for i,name in enumerate(class_names)}
name_to_index = {name:i for i,name in enumerate(class_names)}

import redis

def toRedis(r, key, array):
    assert len(array.shape) == 1
    r.set(key, array.tobytes())
    return

def fromRedis(r, key):
   encoded = r.get(key)
   if encoded is None:
       return None
   array = np.frombuffer(encoded, dtype=np.float32, offset=0)
   return array

class Full_SemKITTILoader(Dataset):
    def __init__(self, root, npoints, train = True):
        self.root = root
        self.train = train
        self.npoints = npoints
        self.r = redis.Redis(host='127.0.0.1', port=6379, db=0)

        part_length = {'00': 4540,'01':1100,'02':4660,'03':800,'04':270,'05':2760,'06':1100,'07':1100,'08':4070,'09':1590,'10':1200}

        self.keys = []
        for part in part_length.keys():
            length = part_length[part]

            for part_index in range(length):
                if self.train:
                    if part_index > length * 0.3:
                        self.keys.append('%s/%06d'%(part, part_index))
                else:
                    if part_index < length * 0.3:
                        self.keys.append('%s/%06d'%(part, part_index))

        sem_cfg = yaml.load(open('config/semantic-kitti.yaml','r'), Loader=yaml.SafeLoader)
        self.class_names = sem_cfg['labels']
        self.learning_map = sem_cfg['learning_map']

    def learning_mapping(self,sem_label):
        # Note: Here the 19 classs are different from the original KITTI 19 classes
        sem_label_learn = [self.learning_map[x] for x in sem_label]
        sem_label_learn = np.array(sem_label_learn).astype(np.int32)
        return sem_label_learn

    def __len__(self):
            return len(self.keys)

    def get_data(self, key):
        data = fromRedis(self.r, key)
        if data is None:
            part, part_index = key.split('/')
            fn_velo = os.path.join(self.root, '%s/velodyne/%s.bin'%(part, part_index))
            fn_label = os.path.join(self.root, '%s/labels/%s.label'%(part, part_index))
            assert os.path.exists(fn_velo), 'Broken dataset %s' % (fn_velo)
            assert os.path.exists(fn_label), 'Broken dataset %s' % (fn_label)

            points = np.fromfile(fn_velo, dtype=np.float32).reshape(-1, 4)
            label = np.fromfile(fn_label, dtype=np.int32).reshape((-1))
        
            pcd = pcd_normalize(points)

            if label.shape[0] == points.shape[0]:
                sem_label = label & 0xFFFF  # semantic label in lower half
                inst_label = label >> 16  # instance id in upper half
                assert((sem_label + (inst_label << 16) == label).all()) # sanity check
            else:
                print("Points shape: ", points.shape)
                print("Label shape: ", label.shape)
                raise ValueError("Scan and Label don't contain same number of points")
            
            label = self.learning_mapping(sem_label)
            to_store = np.concatenate((pcd, label.reshape((-1,1)).astype(np.float32)),axis=1).reshape((-1,))
            toRedis(self.r, key, to_store)
            
            print('add', key)
            
        else:
            data = data.reshape((-1,5))
            pcd = data[:,:4]
            label = data[:,4].astype(np.int32)

        return pcd, label

    def __getitem__(self, index):
        pcd, label = self.get_data(self.keys[index])
        if self.train:
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

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    root = os.environ['KITTI_ROOT']

    dataset = Full_SemKITTILoader(root, 7000, train=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=6)
    
    test_dataset = Full_SemKITTILoader(root, 13000, train=False)
    testdataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=6)

    for i in range(2):
        for data in tqdm(dataloader, 0,total=len(dataloader), smoothing=0.9, dynamic_ncols=True):
            pass
        for data in testdataloader:
            pass