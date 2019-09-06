import open3d
import argparse
import sys
sys.path.append('.')
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
from adv.chamfer import chamfer_distance_without_batch, chamfer_distance_with_batch

args = parse_args([])
args.pretrain = 'experiment/weights/clf-pointnet-0.89730-0076.pth'
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
    exit()

log.debug('Loading pretrain model...')
checkpoint = torch.load(args.pretrain)
model.load_state_dict(checkpoint)
model.eval()
print('Done')

num = 20
for eps in np.linspace(0,0.1,num=num):
    succ, total = 0,0
    for pts_ch_last, gt in testDataLoader:
        pts_ch_last = pts_ch_last.cuda()
        gt = gt[:, 0].long().cuda()
        
        points = pts_ch_last.transpose(2, 1)
        points.requires_grad = True

        pred, _ = model(points)
        loss = F.nll_loss(pred, gt)
        model.zero_grad()
        loss.backward()
        points_grad = points.grad.data
        pts_adv = points + eps * points_grad.sign()
        pts_adv_ch_last = pts_adv.transpose(2, 1)

        # print(chamfer_distance_with_batch(pts_adv_ch_last, pts_ch_last,False))
        
        output, _ = model(pts_adv)
        adv_chocie = output.data.max(1)[1]

        result = gt == adv_chocie
        succ += result.sum().cpu().detach().numpy()
        total += len(result)

    log.info(eps='%.5f'%(eps),accuracy='%.5f%%'%(succ/total * 100))


def vis(args):
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

        vis = open3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(args.model_name, height=800, width=800, left=200, top=0)
        opt = vis.get_render_option().background_color = np.asarray([0, 0, 0])
        for geo in geometries:
            vis.add_geometry(geo)
        vis.register_key_callback(32, lambda vis: exit())
        vis.run()
        vis.destroy_window()


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.mode == "vis":
        vis(args)