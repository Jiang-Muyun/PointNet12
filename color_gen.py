import open3d
import argparse
import os
import time
import json
import h5py
import datetime
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
from matplotlib import pyplot as plt
import my_log as log

from model.pointnet import PointNetColorGen, feature_transform_reguliarzer
from model.pointnet2 import PointNet2SemSeg
from model.utils import load_pointnet

from utils import mkdir, select_avaliable
from data_utils.kitti_utils import Semantic_KITTI_Utils
from data_utils.SemKITTI_Loader import ColorGeneratorLoader

KITTI_ROOT = os.environ['KITTI_ROOT']

def parse_args(notebook = False):
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--model_name', type=str, default='pointnet', choices=('pointnet', 'pointnet2'))
    parser.add_argument('--mode', default='train', help='train or eval')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--map', type=str, default='learning', choices=('slim', 'learning'))
    parser.add_argument('--subset', type=str, default='inview', choices=('inview', 'all'))
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--pretrain', type=str, default=None, help='whether use pretrain model')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer')
    parser.add_argument('--augment', default=False, action='store_true', help="Enable data augmentation")
    if notebook:
        return parser.parse_args([])
    else:
        return parser.parse_args()

def calc_decay(init_lr, epoch):
    return init_lr * 1/(1 + 0.03*epoch)


def train(args):
    experiment_dir = mkdir('experiment/')
    checkpoints_dir = mkdir('experiment/color_gen/%s/'%(args.model_name))
    
    kitti_utils = Semantic_KITTI_Utils(KITTI_ROOT, subset=args.subset, map_type = args.map)
    num_classes = 3

    dataset = ColorGeneratorLoader(KITTI_ROOT, 8000, train=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    
    test_dataset = ColorGeneratorLoader(KITTI_ROOT, 24000, train=False)
    testdataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    if args.model_name == 'pointnet':
        model = PointNetColorGen(num_classes, input_dims = 4, feature_transform=True)
    else:
        model = PointNet2SemSeg(num_classes, feature_dims = 1)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4)

    torch.backends.cudnn.benchmark = True
    model = torch.nn.DataParallel(model)
    model.cuda()
    log.debug('Using gpu:',args.gpu)
    
    if args.pretrain is not None:
        log.debug('Use pretrain model...')
        model.load_state_dict(torch.load(args.pretrain))
        init_epoch = int(args.pretrain[:-4].split('-')[-1])
        log.info('Restart training', epoch=init_epoch)
    else:
        log.msg('Training from scratch')
        init_epoch = 0

    best_acc = 0
    best_miou = 0

    for epoch in range(init_epoch,args.epoch):
        lr = calc_decay(args.learning_rate, epoch)
        log.info(job='ColorGen'+args.map, model=args.model_name,gpu=args.gpu, epoch=epoch, lr=lr)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        for points, target in tqdm(dataloader, total=len(dataloader), smoothing=0.9, dynamic_ncols=True):
            points = points.float().transpose(2, 1).cuda()
            target = target.float().cuda()

            optimizer.zero_grad()
            model = model.train()
            
            if args.model_name == 'pointnet':
                pred, trans_feat = model(points)
                trans_loss = feature_transform_reguliarzer(trans_feat) * 0.001
            else:
                pred = model(points)

            loss1 = torch.mean((pred - target) **2)            
            # print(loss1.detach().cpu().item(), trans_loss.detach().cpu().item())

            loss = loss1 + trans_loss
            loss.backward()
            optimizer.step()
        
        #log.debug('clear cuda cache')
        #torch.cuda.empty_cache()

        model.eval()
        best_MSE = 10.0
        buf = []

        for points, target in tqdm(testdataloader, total=len(testdataloader), smoothing=0.9, dynamic_ncols=True):
            points = points.float().transpose(2, 1).cuda()
            target = target.float().cuda()
            
            with torch.no_grad():
                if args.model_name == 'pointnet':
                    pred, trans_feat = model(points)
                else:
                    pred = model(points)

                buf.append(torch.mean((pred - target) **2).detach().cpu().item())

        MSE = np.mean(buf)
        print(MSE, best_MSE)
        save_model = False
        if MSE < best_MSE:
            best_MSE = MSE
            save_model = True
        
        if save_model:
            fn_pth = 'ColorGen-%s-%.5f-%04d.pth' % (args.model_name, best_MSE, epoch)
            log.info('Save model...',fn = fn_pth)
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, fn_pth))
        else:
            log.info('No need to save model')

        log.warn('Curr',MSE=MSE)
        log.warn('Best',MSE=best_MSE)


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.mode == "train":
        train(args)
