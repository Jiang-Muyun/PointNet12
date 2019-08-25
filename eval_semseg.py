import argparse
import os
import torch
import time
import datetime
import numpy as np
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader
from collections import defaultdict
from torch.autograd import Variable
from data_utils.S3DISDataLoader import S3DISDataLoader, recognize_all_data,class2label
import torch.nn.functional as F
from pathlib import Path
from utils import test_semseg, select_avaliable, mkdir
from colors import *
from tqdm import tqdm
from model.pointnet2 import PointNet2SemSeg
from model.pointnet import PointNetSeg, feature_transform_reguliarzer

seg_classes = class2label
seg_label_to_cat = {}
for i,cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def parse_args():
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--model_name', type=str, default='pointnet2', help='pointnet or pointnet2')
    parser.add_argument('--batchsize', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer')
    return parser.parse_args()

def main(args):
    experiment_dir = mkdir('./experiment/')
    checkpoints_dir = mkdir('./experiment/semseg/%s/'%(args.model_name))

    dataset_root = select_avaliable([
        '/media/james/MyPassport/James/dataset/ShapeNet/indoor3d_sem_seg_hdf5_data/',
        '/home/james/dataset/ShapeNet/indoor3d_sem_seg_hdf5_data/'
    ])

    print('Load data...')
    train_data, train_label, test_data, test_label = recognize_all_data(dataset_root, test_area = 5)

    dataset = S3DISDataLoader(train_data,train_label)
    dataloader = DataLoader(dataset, batch_size=args.batchsize,shuffle=True, num_workers=int(args.workers))
    
    test_dataset = S3DISDataLoader(test_data,test_label)
    testdataloader = DataLoader(test_dataset, batch_size=args.batchsize,shuffle=True, num_workers=int(args.workers))

    num_classes = 13
    if args.model_name == 'pointnet2':
        model = PointNet2SemSeg(num_classes) 
    else:
        model = PointNetSeg(num_classes,feature_transform=True,semseg = True)

    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain))
        print('load model %s'%args.pretrain)
    else:
        print('Training from scratch')

    pretrain = args.pretrain
    init_epoch = int(pretrain[-14:-11]) if args.pretrain is not None else 0

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)
