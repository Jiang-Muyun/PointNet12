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
from data_utils.SemKITTIDataLoader import num_classes, label_id_to_name, reduced_class_names, reduced_colors

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

class Semantic_KITTI_Slim():
    def __init__(self, root):
        self.root = root
        self.init()

    def set_part(self, part='00'):
        length = {
            '00': 4540,'01':1100,'02':4660,'03':800,'04':270,'05':2760,
            '06':1100,'07':1100,'08':4070,'09':1590,'10':1200
        }
        assert part in length.keys(), 'Only %s are supported' %(length.keys())
        self.sequence_root = os.path.join(self.root, 'sequences/%s/'%(part))
        self.index = 0
        self.max_index = length[part]
        return self.max_index
    
    def get_max_index(self):
        return self.max_index

    def init(self):
        self.R, self.T = calib_velo2cam('config/calib_velo_to_cam.txt')
        self.P = calib_cam2cam('config/calib_cam_to_cam.txt' ,mode="02")
        self.RT = np.concatenate((self.R, self.T), axis=1)

        self.sem_cfg = yaml.load(open('config/semantic-kitti.yaml','r'), Loader=yaml.SafeLoader)
        self.class_names = self.sem_cfg['labels']

    def load(self,index = None):
        """  Load the frame, point cloud and semantic labels from file """

        self.index = index
        if self.index == self.max_index:
            print('End of sequence')
            return False

        fn_frame = os.path.join(self.sequence_root, 'image_2/%06d.png' % (self.index))
        self.frame = cv2.imread(fn_frame)
        return True

    def project_3d_to_2d(self, pts_3d):
        assert pts_3d.shape[1] == 3, pts_3d.shape
        pts_3d = pts_3d.copy()
        
        # Create a [N,1] array
        one_mat = np.ones((pts_3d.shape[0], 1),dtype=np.float32)

        # Concat and change shape from [N,3] to [N,4] to [4,N]
        xyz_v = np.concatenate((pts_3d, one_mat), axis=1).T

        # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c)
        for i in range(xyz_v.shape[1]):
            xyz_v[:3, i] = np.matmul(self.RT, xyz_v[:, i])

        xyz_c = xyz_v[:3]

        # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y)
        for i in range(xyz_c.shape[1]):
            xyz_c[:, i] = np.matmul(self.P, xyz_c[:, i])

        # normalize image(pixel) coordinates(x,y)
        xy_i = xyz_c / xyz_c[2]

        # get pixels location
        pts_2d = xy_i[:2].T
        return pts_2d
    
    def draw_2d_points(self, pts_2d, colors):
        """ draw 2d points in camera image """
        assert pts_2d.shape[1] == 2, pts_2d.shape

        image = self.frame.copy()
        pts = pts_2d.astype(np.int32).tolist()

        for (x,y),c in zip(pts, colors.tolist()):
            cv2.circle(image, (x, y), 2, [c[2],c[1],c[0]], -1)

        return image

part = '03'
KITTI_ROOT = '/media/james/Ubuntu_Data/dataset/KITTI/odometry/dataset/'

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
        reduced_class_names = ['unlabelled', 'vehicle', 'human', 'ground', 'structure', 'nature']
        reduced_colors = [[255, 255, 255],[245, 150, 100],[30, 30, 255],[255, 0, 255],[0, 200, 255],[0, 175, 0]]
        for i,(name,c) in enumerate(zip(reduced_class_names, reduced_colors)):
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
    handle = Semantic_KITTI_Slim(root = KITTI_ROOT)
    handle.set_part(part)
    vis_handle = Window_Manager()

    _,_,test_data, test_label = load_data(args.h5, train = False, selected = [part])
    test_dataset = SemKITTIDataLoader(test_data, test_label)
    testdataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    log.msg('Building Model', args.model_name)
    if args.model_name == 'pointnet':
        model = PointNetSeg(num_classes, input_dims = 4, feature_transform=True)
    else:
        model = PointNet2SemSeg(num_classes, feature_dims = 1)

    torch.backends.cudnn.benchmark = True
    model = torch.nn.DataParallel(model)
    log.msg('Using gpu:',args.gpu)

    assert args.pretrain is not None,'No pretrain model'
    checkpoint = torch.load(args.pretrain)
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()

    for i in range(100, len(test_data)):
        handle.load(i)

        points = torch.from_numpy(test_data[i]).unsqueeze(0)
        points = points.transpose(2, 1).cuda()
        points[:,0] = points[:,0] / 70
        points[:,1] = points[:,1] / 70
        points[:,2] = points[:,2] / 3
        points[:,3] = (points[:,3] - 0.5)/2
        with torch.no_grad():
            if args.model_name == 'pointnet':
                pred, _ = model(points)
            else:
                pred = model(points)
            pred_choice = pred.data.max(-1)[1].cpu().squeeze_(0).numpy()
            sem_label = pred_choice

        print(i, pred_choice.shape)
        
        pts_3d = test_data[i][:,:3]
        pts_2d = handle.project_3d_to_2d(pts_3d)

        colors = reduced_colors[pred_choice]
        vis_handle.update(pts_3d, colors)

        img_semantic = handle.draw_2d_points(pts_2d, colors)

        cv2.imshow('semantic', img_semantic)
        if 32 == cv2.waitKey(1):
            break

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.model_name == 'pointnet':
        args.pretrain = 'experiment/weights/kitti_semseg-pointnet-0.51023-0052.pth'
    else:
        args.pretrain = 'experiment/weights/kitti_semseg-pointnet2-0.56290-0009.pth'
    vis(args)