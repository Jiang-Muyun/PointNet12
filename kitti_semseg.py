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

def mkdir(fn):
    os.makedirs(fn, exist_ok=True)
    return fn

def select_avaliable(fn_list):
    selected = None
    for fn in fn_list:
        if os.path.exists(fn):
            selected = fn
            break
    if selected is None:
        log.err(log.yellow("Could not find dataset from"), fn_list)
    else:
        return selected

def parse_args(notebook = False):
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--model_name', type=str, default='pointnet', help='pointnet or pointnet2')
    parser.add_argument('--mode', default='train', help='train or eval')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--workers', type=int, default=6, help='number of data loading workers')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--pretrain', type=str, default=None, help='whether use pretrain model')
    parser.add_argument('--h5', type=str, default = 'experiment/data/pts_sem_voxel_0.10.h5', help='pts h5 file')
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

def calc_categorical_iou(pred, target, num_classes ,iou_tabel):
    choice = pred.max(-1)[1]
    target.squeeze_(-1)
    for cat in range(num_classes):
        I = torch.sum((choice == cat) & (target == cat)).float()
        U = torch.sum((choice == cat) | (target == cat)).float()
        if U == 0:
            iou = 1
        else:
            iou = (I / U).cpu().item()
        iou_tabel[cat,0] += iou
        iou_tabel[cat,1] += 1
    return iou_tabel

def test_kitti_semseg(model, loader, catdict, model_name, num_classes):
    iou_tabel = np.zeros((len(catdict),3))
    metrics = {'accuracy':[]}
    
    for points, target in tqdm(loader, total=len(loader), smoothing=0.9, dynamic_ncols=True):
        batch_size, num_point, _ = points.size()
        points, target = points.float(), target.long()
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        with torch.no_grad():
            if model_name == 'pointnet':
                pred, _ = model(points)
            else:
                pred = model(points)

        iou_tabel = calc_categorical_iou(pred,target,num_classes,iou_tabel)
        
        target.squeeze_(-1)
        pred_choice = pred.data.max(-1)[1]
        correct = (pred_choice == target.data).sum().cpu().item()
        metrics['accuracy'].append(correct/ (batch_size * num_point))

    iou_tabel[:,2] = iou_tabel[:,0] /iou_tabel[:,1]
    metrics['accuracy'] = np.mean(metrics['accuracy'])
    metrics['iou'] = np.mean(iou_tabel[:, 2])

    iou_tabel = pd.DataFrame(iou_tabel,columns=['iou','count','mean_iou'])
    iou_tabel['Category_IOU'] = [catdict[i] for i in range(len(catdict)) ]
    cat_iou = iou_tabel.groupby('Category_IOU')['mean_iou'].mean()

    return metrics, cat_iou

def train(args):
    experiment_dir = mkdir('experiment/')
    checkpoints_dir = mkdir('experiment/kitti_semseg/%s/'%(args.model_name))
    train_data, train_label, test_data, test_label = load_data(args.h5, train = True)

    dataset = SemKITTIDataLoader(train_data, train_label, npoints = 5000, data_augmentation = args.augment)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    
    test_dataset = SemKITTIDataLoader(test_data, test_label, npoints = 13072)
    testdataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
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

    history = {'loss':[]}
    best_acc = 0
    best_meaniou = 0

    for epoch in range(init_epoch,args.epoch):
        lr = calc_decay(args.learning_rate, epoch)
        log.info(job='kitti_semseg',model=args.model_name,gpu=args.gpu, epoch=epoch, lr=lr)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        for i, data in tqdm(enumerate(dataloader, 0),total=len(dataloader), smoothing=0.9, dynamic_ncols=True):
            points, target = data
            points, target = points.float(), target.long()
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
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

            history['loss'].append(loss.cpu().data.numpy())
            loss.backward()
            optimizer.step()
        
        log.debug('clear cuda cache')
        torch.cuda.empty_cache()

        test_metrics, cat_mean_iou = test_kitti_semseg(
            model.eval(), 
            testdataloader,
            label_id_to_name,
            args.model_name,
            num_classes = num_classes,
        )
        mean_iou = np.mean(cat_mean_iou)

        save_model = False
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        
        if mean_iou > best_meaniou:
            best_meaniou = mean_iou
            save_model = True
        
        if save_model:
            fn_pth = 'kitti_semseg-%s-%.5f-%04d.pth' % (args.model_name, best_meaniou, epoch)
            log.info('Save model...',fn = fn_pth)
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, fn_pth))
            log.msg(cat_mean_iou)
        else:
            log.info('No need to save model')
            log.msg(cat_mean_iou)

        log.warn('Curr',accuracy=test_metrics['accuracy'], meanIOU=mean_iou)
        log.warn('Best',accuracy=best_acc, meanIOU=best_meaniou)

def evaluate(args):
    if args.model_name == 'pointnet':
        args.pretrain = 'experiment/weights/kitti_semseg-pointnet-0.51178-0053.pth'
    else:
        args.pretrain = 'experiment/weights/kitti_semseg-pointnet2-0.59957-0023.pth'

    _,_,test_data, test_label = load_data(args.h5, train = False)

    test_dataset = SemKITTIDataLoader(test_data, test_label, npoints = 13072)
    testdataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    log.debug('Building Model', args.model_name)
    if args.model_name == 'pointnet':
        model = PointNetSeg(num_classes, input_dims = 4, feature_transform=True)
    else:
        model = PointNet2SemSeg(num_classes, feature_dims = 1)

    torch.backends.cudnn.benchmark = True
    model = torch.nn.DataParallel(model)
    log.debug('Using gpu:',args.gpu)

    if args.pretrain is None:
        log.err('No pretrain model')
        return

    log.debug('Loading pretrain model...')
    checkpoint = torch.load(args.pretrain)
    model.load_state_dict(checkpoint)
    model.cuda()

    test_metrics, cat_mean_iou = test_kitti_semseg(
        model.eval(), 
        testdataloader,
        label_id_to_name,
        args.model_name,
        num_classes = num_classes,
    )
    
    mean_iou = np.mean(cat_mean_iou)
    log.warn(cat_mean_iou)
    log.info('Curr', accuracy=test_metrics['accuracy'], meanIOU=mean_iou)

from kitti_base import PointCloud_Vis, Semantic_KITTI_Utils

def vis(args):
    part = '03'
    KITTI_ROOT = '/media/james/Ubuntu_Data/dataset/KITTI/odometry/dataset/'
    cfg_data = json.load(open('config/ego_view.json'))
    handle = Semantic_KITTI_Utils(root = KITTI_ROOT)
    handle.set_filter(cfg_data['h_fov'], cfg_data['v_fov'])
    handle.set_part(part)
    vis_handle = Window_Manager()

    args = parse_args()
    if args.model_name == 'pointnet':
        args.pretrain = 'experiment/weights/kitti_semseg-pointnet-0.53106-0053.pth'
    else:
        args.pretrain = 'experiment/weights/kitti_semseg-pointnet2-0.59957-0023.pth'
    _,_,test_data, test_label = load_data(args.h5, train = False, selected = [part])
    test_dataset = SemKITTIDataLoader(test_data, test_label)
    testdataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    log.debug('Building Model', args.model_name)
    if args.model_name == 'pointnet':
        model = PointNetSeg(num_classes, input_dims = 4, feature_transform=True)
    else:
        model = PointNet2SemSeg(num_classes, feature_dims = 1)

    torch.backends.cudnn.benchmark = True
    model = torch.nn.DataParallel(model)
    log.debug('Using gpu:',args.gpu)

    if args.pretrain is None:
        log.err('No pretrain model')

    log.debug('Loading pretrain model...')
    checkpoint = torch.load(args.pretrain)
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()

    for i in range(100, len(test_data)):
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

        pts_3d = test_data[i][:,:3].copy()
        colors = reduced_colors[pred_choice]
        vis_handle.update(pts_3d, colors)
        mkdir('experiment/imgs/%s/'%(args.model_name))
        vis_handle.capture_screen('experiment/imgs/%s/%d_3d.png'%(args.model_name,i))

        handle.load(i)
        pts_2d, color = handle.project_3d_to_2d(test_data[i][:,:3])

        img_semantic = handle.frame.copy()
        pts = pts_2d.astype(np.int32).tolist()

        for (x,y),c in zip(pts, colors.tolist()):
            cv2.circle(img_semantic, (x, y), 2, [c[2],c[1],c[0]], -1)

        cv2.imshow('semantic', img_semantic)
        cv2.imwrite('experiment/imgs/%s/%d_sem.png'%(args.model_name, i), img_semantic)

        print(i, pred_choice.shape)

        if 32 == cv2.waitKey(1):
            break

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.mode == "train":
        train(args)
    if args.mode == "eval":
        evaluate(args)
    if args.mode == "vis":
        vis(args)