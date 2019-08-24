import argparse
import os
import time
import datetime
import torch
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data_utils.ModelNetDataLoader import ModelNetDataLoader, load_data
from pathlib import Path
from tqdm import tqdm
from utils import test, save_checkpoint, select_avaliable, mkdir
from colors import *
from model.pointnet2 import PointNet2ClsMsg
from model.pointnet import PointNetCls, feature_transform_reguliarzer
from torchsummary import summary

def parse_args():
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--model_name', default='pointnet2', help='pointnet or pointnet2')
    parser.add_argument('--batchsize', type=int, default=24, help='batch size in training')
    parser.add_argument('--epoch',  default=100, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--multi_gpu', type=str, default=None, help='whether use multi gpu training')
    parser.add_argument('--train_metric', type=str, default=False, help='whether evaluate on training dataset')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate of learning rate')
    parser.add_argument('--rotation',  default=None, help='range of training rotation')
    parser.add_argument('--feature_transform', default=False, help="use feature transform in pointnet")
    return parser.parse_args()

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    num_class = 40
    if args.model_name == 'pointnet':
        model = PointNetCls(num_class,args.feature_transform).cuda()
    else:
        model = PointNet2ClsMsg().cuda()

    print(model)

if __name__ == '__main__':
    args = parse_args()
    main(args)
