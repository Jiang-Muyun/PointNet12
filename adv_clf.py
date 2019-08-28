import open3d
import argparse
import os
import time
import h5py
import datetime
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data_utils.ModelNetDataLoader import ModelNetDataLoader, load_data, class_names
from pathlib import Path
from tqdm import tqdm
from utils import test, save_checkpoint, select_avaliable, mkdir
import log
from model.pointnet2 import PointNet2ClsMsg
from model.pointnet import PointNetCls, feature_transform_reguliarzer
import colorsys
from clf import parse_args, root

def vis(args):
    test_data, test_label = load_data(root, train = False)
    testDataset = ModelNetDataLoader(test_data, test_label)
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=1, shuffle=False)

    log.debug('Building Model',args.model_name)
    if args.model_name == 'pointnet':
        num_class = 40
        model = PointNetCls(num_class,args.feature_transform).cuda()  
    else:
        model = PointNet2ClsMsg().cuda()

    if args.pretrain is None:
        log.err('No pretrain model')
        return

    log.debug('Loading pretrain model...')
    checkpoint = torch.load(args.pretrain)
    model.load_state_dict(checkpoint)
    model.eval()

    log.info('Attacking', batch_size = args.batch_size)

    for points, gt in testDataLoader:
        gt = gt[:, 0].long()
        points = points.transpose(2, 1)
        points, gt = points.cuda(), gt.cuda()
        points.requires_grad = True
        pred, _ = model(points)
        pred_choice = pred.data.max(1)[1]

        loss = F.nll_loss(pred, gt)
        model.zero_grad()
        loss.backward()
        points_grad = points.grad.data
        
        log.info(points_grad=points_grad.shape,
            min=points_grad.cpu().detach().numpy().min(),
            l1=np.abs(points_grad.cpu().detach().numpy()).mean())

        geometries = []
        
        num = 20
        for i,eps in enumerate(np.linspace(0,0.1,num=num)):
            perturbed_data = points + eps * points_grad.sign()
            output, _ = model(perturbed_data)
            adv_chocie = output.data.max(1)[1]
            
            adv_np = perturbed_data[0].transpose(1, 0).cpu().detach().numpy()
            adv_cloud = open3d.geometry.PointCloud()
            adv_cloud.points = open3d.utility.Vector3dVector(adv_np)
            adv_cloud.paint_uniform_color(colorsys.hsv_to_rgb(0.5*(i/num), 1, 1))
            geometries.append(adv_cloud)

        vis = open3d.visualization.Visualizer()
        vis.create_window(args.model_name, height=800, width=800, left=200, top=0)
        opt = vis.get_render_option().background_color = np.asarray([0, 0, 0])
        for geo in geometries:
            vis.add_geometry(geo)
        vis.run()
        vis.destroy_window()

def adv(args):
    test_data, test_label = load_data(root, train = False)
    testDataset = ModelNetDataLoader(test_data, test_label)
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=args.batch_size, shuffle=False)

    log.debug('Building Model',args.model_name)
    if args.model_name == 'pointnet':
        num_class = 40
        model = PointNetCls(num_class,args.feature_transform).cuda()  
    else:
        model = PointNet2ClsMsg().cuda()

    if args.pretrain is None:
        log.err('No pretrain model')
        return

    log.debug('Loading pretrain model...')
    checkpoint = torch.load(args.pretrain)
    model.load_state_dict(checkpoint)
    model.eval()

    log.info('Attacking', batch_size = args.batch_size)

    num = 20
    for eps in np.linspace(0,0.1,num=num):
        succ, total = 0,0
        for points, gt in testDataLoader:
            gt = gt[:, 0].long().cuda()
            points = points.transpose(2, 1).cuda()
            points.requires_grad = True
            pred, _ = model(points)
            pred_choice = pred.data.max(1)[1]

            loss = F.nll_loss(pred, gt)
            model.zero_grad()
            loss.backward()
            points_grad = points.grad.data
            perturbed_data = points + eps * points_grad.sign()
            output, _ = model(perturbed_data)
            adv_chocie = output.data.max(1)[1]
            
            result = gt == adv_chocie
            succ += result.sum().cpu().detach().numpy()
            total += len(result)

        succ_rate = succ/total * 100
        log.info(eps='%.5f'%(eps),accuracy='%.5f%%'%(succ_rate))

def plot():
    pass
    # pointnet fgsm
    # eps: 0.00000 accuracy: 89.70827% 
    # eps: 0.00526 accuracy: 78.16045% 
    # eps: 0.01053 accuracy: 70.38088% 
    # eps: 0.01579 accuracy: 67.26094% 
    # eps: 0.02105 accuracy: 65.84279% 
    # eps: 0.02632 accuracy: 65.68071% 
    # eps: 0.03158 accuracy: 65.43760% 
    # eps: 0.03684 accuracy: 65.31605% 
    # eps: 0.04211 accuracy: 65.31605% 
    # eps: 0.04737 accuracy: 65.23501% 
    # eps: 0.05263 accuracy: 64.99190% 
    # eps: 0.05789 accuracy: 64.54619% 
    # eps: 0.06316 accuracy: 63.57374% 
    # eps: 0.06842 accuracy: 62.88493% 
    # eps: 0.07368 accuracy: 61.58833% 
    # eps: 0.07895 accuracy: 60.53485% 
    # eps: 0.08421 accuracy: 59.31929% 
    # eps: 0.08947 accuracy: 56.96921% 
    # eps: 0.09474 accuracy: 54.25446% 
    # eps: 0.10000 accuracy: 51.62075%

    # eps: 0.00000 accuracy: 91.57212% 
    # eps: 0.00526 accuracy: 84.35981% 
    # eps: 0.01053 accuracy: 79.74068% 
    # eps: 0.01579 accuracy: 77.14749% 
    # eps: 0.02105 accuracy: 74.83793% 
    # eps: 0.02632 accuracy: 72.60940% 
    # eps: 0.03158 accuracy: 68.59806% 
    # eps: 0.03684 accuracy: 64.58671% 
    # eps: 0.04211 accuracy: 62.76337% 
    # eps: 0.04737 accuracy: 58.95462% 
    # eps: 0.05263 accuracy: 56.56402% 
    # eps: 0.05789 accuracy: 52.71475% 
    # eps: 0.06316 accuracy: 50.00000% 
    # eps: 0.06842 accuracy: 45.86710% 
    # eps: 0.07368 accuracy: 42.01783% 
    # eps: 0.07895 accuracy: 37.56078% 
    # eps: 0.08421 accuracy: 33.79254% 
    # eps: 0.08947 accuracy: 29.25446% 
    # eps: 0.09474 accuracy: 26.37763% 
    # eps: 0.10000 accuracy: 22.77147% 

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.mode == "vis":
        vis(args)
    if args.mode == "adv":
        adv(args)