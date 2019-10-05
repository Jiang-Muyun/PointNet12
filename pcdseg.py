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

from model.pointnet import PointNetSeg, feature_transform_reguliarzer
from model.pointnet2 import PointNet2SemSeg
from model.utils import load_pointnet

from utils import mkdir, select_avaliable
from data_utils.SemKITTI_Loader import SemKITTI_Loader
from data_utils.kitti_utils import Semantic_KITTI_Utils

KITTI_ROOT = os.environ['KITTI_ROOT']

def parse_args(notebook = False):
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('mode', default='train', choices=('train', 'eval'))
    #parser.add_argument('--model_name', type=str, default='pointnet', choices=('pointnet', 'pointnet2'))
    parser.add_argument('--pn2', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--subset', type=str, default='inview', choices=('inview', 'all'))
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--pretrain', type=str, default=None, help='whether use pretrain model')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer')
    parser.add_argument('--augment', default=False, action='store_true', help="Enable data augmentation")
    if notebook:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    
    if args.pn2 == False:
        args.model_name = 'pointnet'
    else:
        args.model_name = 'pointnet2'
    return args

def calc_decay(init_lr, epoch):
    return init_lr * 1/(1 + 0.03*epoch)

def test_kitti_semseg(model, loader, model_name, num_classes, class_names):
    ious = np.zeros((num_classes,), dtype = np.float32)
    count = np.zeros((num_classes,), dtype = np.uint32)
    count[0] = 1
    accuracy = []
    
    for points, target in tqdm(loader, total=len(loader), smoothing=0.9, dynamic_ncols=True):
        batch_size, num_point, _ = points.size()
        points = points.float().transpose(2, 1).cuda()
        target = target.long().cuda()

        with torch.no_grad():
            if model_name == 'pointnet':
                pred, _ = model(points)
            else:
                pred = model(points)

            pred_choice = pred.argmax(-1)
            target = target.squeeze(-1)

            for class_id in range(num_classes):
                I = torch.sum((pred_choice == class_id) & (target == class_id)).cpu().item()
                U = torch.sum((pred_choice == class_id) | (target == class_id)).cpu().item()
                iou = 1 if U == 0 else I/U
                ious[class_id] += iou
                count[class_id] += 1

            correct = (pred_choice == target).sum().cpu().item()
            accuracy.append(correct/ (batch_size * num_point)) 

    categorical_iou = ious / count
    df = pd.DataFrame(categorical_iou, columns=['mIOU'], index=class_names)
    df = df.sort_values(by='mIOU', ascending=False)

    log.info('categorical mIOU')
    log.msg(df)

    acc = np.mean(accuracy)
    miou = np.mean(categorical_iou[1:])
    return acc, miou

def train(args):
    experiment_dir = mkdir('experiment/')
    checkpoints_dir = mkdir('experiment/%s/'%(args.model_name))
    
    kitti_utils = Semantic_KITTI_Utils(KITTI_ROOT, subset=args.subset)
    class_names = kitti_utils.class_names
    num_classes = kitti_utils.num_classes

    dataset = SemKITTI_Loader(KITTI_ROOT, 8000, train=True, subset=args.subset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    
    test_dataset = SemKITTI_Loader(KITTI_ROOT, 24000, train=False, subset=args.subset)
    testdataloader = DataLoader(test_dataset, batch_size=int(args.batch_size/2), shuffle=False, num_workers=args.workers)

    if args.model_name == 'pointnet':
        model = PointNetSeg(num_classes, input_dims = 4, feature_transform=True)
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
        log.info(subset=args.subset, model=args.model_name, gpu=args.gpu, epoch=epoch, lr=lr)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        for points, target in tqdm(dataloader, total=len(dataloader), smoothing=0.9, dynamic_ncols=True):
            points = points.float().transpose(2, 1).cuda()
            target = target.long().cuda()

            optimizer.zero_grad()
            model = model.train()
            
            if args.model_name == 'pointnet':
                pred, trans_feat = model(points)
            else:
                pred = model(points)

            pred = pred.contiguous().view(-1, num_classes)
            target = target.view(-1, 1)[:, 0]
            loss = F.nll_loss(pred, target)

            if args.model_name == 'pointnet':
                loss += feature_transform_reguliarzer(trans_feat) * 0.001

            loss.backward()
            optimizer.step()
        
        log.debug('clear cuda cache')
        torch.cuda.empty_cache()

        acc, miou = test_kitti_semseg(model.eval(), testdataloader,args.model_name,num_classes,class_names)

        save_model = False
        if acc > best_acc:
            best_acc = acc
        
        if miou > best_miou:
            best_miou = miou
            save_model = True
        
        if save_model:
            fn_pth = '%s-%s-%.5f-%04d.pth' % (args.model_name, args.subset, best_miou, epoch)
            log.info('Save model...',fn = fn_pth)
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, fn_pth))
        else:
            log.info('No need to save model')

        log.warn('Curr',accuracy=acc, mIOU=miou)
        log.warn('Best',accuracy=best_acc, mIOU=best_miou)

def evaluate(args):
    kitti_utils = Semantic_KITTI_Utils(KITTI_ROOT, subset=args.subset)
    class_names = kitti_utils.class_names
    num_classes = kitti_utils.num_classes

    test_dataset = SemKITTI_Loader(KITTI_ROOT, 24000, train=False, subset=args.subset)
    testdataloader = DataLoader(test_dataset, batch_size=int(args.batch_size/2), shuffle=False, num_workers=args.workers)

    model = load_pointnet(args.model_name, kitti_utils.num_classes, args.pretrain)

    acc, miou = test_kitti_semseg(model.eval(), testdataloader,args.model_name,num_classes,class_names)

    log.info('Curr', accuracy=acc, mIOU=miou)

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.mode == "train":
        train(args)
    if args.mode == "eval":
        evaluate(args)
