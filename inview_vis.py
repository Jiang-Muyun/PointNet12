import open3d
import argparse
import os
import time
import json
import h5py
import datetime
import cv2
import yaml
import colorsys
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

from kitti_semseg import parse_args
from kitti_base import calib_velo2cam,calib_cam2cam

from data_utils.SemKITTIDataLoader import SemKITTIDataLoader, load_data
from data_utils.Full_SemKITTIDataLoader import pcd_normalize, Semantic_KITTI_Utils

KITTI_ROOT = os.environ['KITTI_ROOT']
kitti_utils = Semantic_KITTI_Utils(KITTI_ROOT, where='all', map_type = 'learning')
num_classes = kitti_utils.num_classes
class_names = kitti_utils.class_names
index_to_name = kitti_utils.index_to_name
colors = kitti_utils.colors
colors_bgr = kitti_utils.colors_bgr

class Window_Manager():
    def __init__(self):
        self.param = open3d.io.read_pinhole_camera_parameters('config/ego_view.json')
        self.vis = open3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=800, height=800, left=100)
        self.vis.register_key_callback(32, lambda vis: exit())
        self.vis.get_render_option().load_from_json('config/render_option.json')
        self.pcd = open3d.geometry.PointCloud()
    
    def update(self, pts_3d, colors):
        self.pcd.points = open3d.utility.Vector3dVector(pts_3d)
        self.pcd.colors = open3d.utility.Vector3dVector(colors/255)
        self.vis.remove_geometry(self.pcd)
        self.vis.add_geometry(self.pcd)
        self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.param)
        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()

    def capture_screen(self,fn):
        self.vis.capture_screen_image(fn, False)

part = '03'

def export_video():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    font = cv2.FONT_HERSHEY_SIMPLEX
    out = cv2.VideoWriter('experiment/pn_compare.avi',fourcc, 15.0, (int(1600*0.8),int(740*0.8)))

    # mkdir('experiment/imgs/%s/'%(args.model_name))
    # vis_handle.capture_screen('experiment/imgs/%s/%d_3d.png'%(args.model_name,i))
    # cv2.imwrite('experiment/imgs/%s/%d_sem.png'%(args.model_name, i), img_semantic)

    for index in range(100, 320):
        pn_3d = cv2.imread('experiment/imgs/pointnet/%d_3d.png' % (index))
        pn_sem = cv2.imread('experiment/imgs/pointnet/%d_sem.png' % (index))
        pn2_3d = cv2.imread('experiment/imgs/pointnet2/%d_3d.png' % (index))
        pn2_sem = cv2.imread('experiment/imgs/pointnet2/%d_sem.png' % (index))

        pn_3d = pn_3d[160:650]
        pn2_3d = pn2_3d[160:650]

        pn_sem = cv2.resize(pn_sem, (800, 250))
        pn2_sem = cv2.resize(pn2_sem, (800, 250))

        pn = np.vstack((pn_3d, pn_sem))
        pn2 = np.vstack((pn2_3d, pn2_sem))

        cv2.putText(pn, 'PointNet', (20, 100), font,1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pn, 'PointNet', (20, 520), font,1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pn2, 'PointNet2', (20, 100), font,1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pn2, 'PointNet2', (20, 520), font,1, (255, 255, 255), 2, cv2.LINE_AA)

        merge = np.hstack((pn, pn2))
        class_names = ['unlabelled', 'vehicle', 'human', 'ground', 'structure', 'nature']
        colors = [[255, 255, 255],[245, 150, 100],[30, 30, 255],[255, 0, 255],[0, 200, 255],[0, 175, 0]]
        for i,(name,c) in enumerate(zip(class_names, colors)):
            cv2.putText(merge, name, (200 + i * 200, 50), font,1, [c[2],c[1],c[0]], 2, cv2.LINE_AA)

        cv2.line(merge,(0,70),(1600,70),(255,255,255),2)
        cv2.line(merge,(800,70),(800,1300),(255,255,255),2)

        merge = cv2.resize(merge,(0,0),fx=0.8,fy=0.8)
        # cv2.imshow('merge', merge)
        # if 32 == waitKey(1):
        #     break
        out.write(merge)

        print(index)
    out.release()

def vis(args):
    vis_handle = Window_Manager()

    # _,_,test_data, test_label = load_data(args.h5, train = False, selected = [part])
    # test_dataset = SemKITTIDataLoader(test_data, test_label)
    # testdataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    test_dataset = Full_SemKITTILoader(KITTI_ROOT, 15000, train=False, where='inview', map_type = 'slim')
    testdataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    log.msg('Building Model', args.model_name)
    if args.model_name == 'pointnet':
        model = PointNetSeg(num_classes, input_dims = 4, feature_transform=True)
    else:
        model = PointNet2SemSeg(num_classes, feature_dims = 1)

    torch.backends.cudnn.benchmark = True
    model = torch.nn.DataParallel(model)
    log.msg('Using gpu:',args.gpu)

    if args.model_name == 'pointnet':
        args.pretrain = 'checkpoints/kitti_semseg-pointnet-0.51023-0052.pth'
    else:
        args.pretrain = 'checkpoints/kitti_semseg-pointnet2-0.56290-0009.pth'

    assert args.pretrain is not None,'No pretrain model'
    checkpoint = torch.load(args.pretrain)
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()

    for index in range(100, kitti_utils.get_max_index(part)):
        pcd, label = kitti_utils.get(part, index)
        pcd = pcd_normalize(pcd)

        points = torch.from_numpy(pcd).unsqueeze(0)
        points = points.transpose(2, 1).cuda()

        with torch.no_grad():
            if args.model_name == 'pointnet':
                pred, _ = model(points)
            else:
                pred = model(points)
            pred_choice = pred.data.max(-1)[1].cpu().squeeze_(0).numpy()
            sem_label = pred_choice

        print(index, pred_choice.shape)
        
        pts_3d = test_data[index][:,:3]
        pts_2d = handle.project_3d_to_2d(pts_3d)

        vis_handle.update(pts_3d, colors[pred_choice])
        sem_img = handle.draw_2d_points(pts_2d, colors_bgr[pred_choice])

        cv2.imshow('semantic', sem_img)
        if 32 == cv2.waitKey(1):
            break

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    vis(args)