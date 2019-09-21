import open3d
import argparse
import os
import torch
import time
import h5py
import datetime
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader
from collections import defaultdict
from torch.autograd import Variable
from data_utils.S3DISDataLoader import S3DISDataLoader, recognize_all_data,class2label
import torch.nn.functional as F
from pathlib import Path
from utils import test_semseg, select_avaliable, mkdir, auto_complete
import my_log as log
from tqdm import tqdm
from model.pointnet2 import PointNet2SemSeg
from model.pointnet import PointNetSeg, feature_transform_reguliarzer

seg_classes = class2label
seg_label_to_cat = {}
for i,cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def parse_args():
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--model_name', type=str, default='pointnet', help='pointnet or pointnet2')
    parser.add_argument('--mode', default='train', help='train or eval')
    parser.add_argument('--batch_size', type=int, default=0, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--epoch', type=int, default=200, help='number of epochs for training')
    parser.add_argument('--pretrain', type=str, default=None, help='whether use pretrain model')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer')
    parser.add_argument('--augment', default=False, action='store_true', help="Enable data augmentation")
    return parser.parse_args()

def _load(load_train = True):
    dataset_tmp = 'experiment/indoor3d_sem_seg_hdf5_data.h5'
    if not os.path.exists(dataset_tmp):
        log.info('Loading data...')
        root = select_avaliable([
            '/media/james/HDD/James_Least/Large_Dataset/ShapeNet/indoor3d_sem_seg_hdf5_data/',
            '/media/james/Ubuntu_Data/dataset/ShapeNet/indoor3d_sem_seg_hdf5_data/',
            '/media/james/MyPassport/James/dataset/ShapeNet/indoor3d_sem_seg_hdf5_data/',
            '/home/james/dataset/ShapeNet/indoor3d_sem_seg_hdf5_data/'
        ])
        train_data, train_label, test_data, test_label = recognize_all_data(root, test_area = 5)
        fp_h5 = h5py.File(dataset_tmp,"w")
        fp_h5.create_dataset('train_data', data = train_data)
        fp_h5.create_dataset('train_label', data = train_label)
        fp_h5.create_dataset('test_data', data = test_data)
        fp_h5.create_dataset('test_label', data = test_label)
    else:
        log.info('Loading from h5...')
        fp_h5 = h5py.File(dataset_tmp, 'r')
        if load_train:
            train_data = fp_h5.get('train_data')[()]
            train_label = fp_h5.get('train_label')[()]
        test_data = fp_h5.get('test_data')[()]
        test_label = fp_h5.get('test_label')[()]
    
    if load_train:
        log.info(train_data=train_data.shape, train_label=train_label.shape)
        log.info(test_data=test_data.shape, test_label=test_label.shape)
        return train_data, train_label, test_data, test_label
    else:
        log.info(test_data=test_data.shape, test_label=test_label.shape)
        return test_data, test_label

def train(args):
    experiment_dir = mkdir('./experiment/')
    checkpoints_dir = mkdir('./experiment/semseg/%s/'%(args.model_name))
    train_data, train_label, test_data, test_label = _load()

    dataset = S3DISDataLoader(train_data, train_label, data_augmentation = args.augment)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,shuffle=True, num_workers=args.workers)
    
    test_dataset = S3DISDataLoader(test_data, test_label)
    testdataloader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=True, num_workers=args.workers)

    num_classes = 13
    if args.model_name == 'pointnet':
        model = PointNetSeg(num_classes,feature_transform=True,semseg = True)
    else:
        model = PointNet2SemSeg(num_classes) 

    device_ids = [int(x) for x in args.gpu.split(',')]
    torch.backends.cudnn.benchmark = True
    model.cuda(device_ids[0])
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    log.debug('Using DataParallel:',device_ids)

    if args.pretrain is not None:
        log.debug('Use pretrain model...')
        model.load_state_dict(torch.load(args.pretrain))
        init_epoch = int(args.pretrain[:-4].split('-')[-1])
        log.debug('start epoch from', init_epoch)
    else:
        log.debug('Training from scratch')
        init_epoch = 0

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate)
            
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    LEARNING_RATE_CLIP = 1e-5

    history = defaultdict(lambda: list())
    best_acc = 0
    best_meaniou = 0

    for epoch in range(init_epoch,args.epoch):
        scheduler.step()
        lr = max(optimizer.param_groups[0]['lr'],LEARNING_RATE_CLIP)

        log.info(job='semseg',model=args.model_name,gpu=args.gpu,epoch='%d/%s' % (epoch, args.epoch),lr=lr)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        for i, data in tqdm(enumerate(dataloader, 0),total=len(dataloader),smoothing=0.9):
            points, target = data
            points, target = Variable(points.float()), Variable(target.long())
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

        test_metrics, cat_mean_iou = test_semseg(
            model.eval(), 
            testdataloader, 
            seg_label_to_cat,
            num_classes = num_classes,
            pointnet2 = args.model_name == 'pointnet2'
        )
        mean_iou = np.mean(cat_mean_iou)

        save_model = False
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        
        if mean_iou > best_meaniou:
            best_meaniou = mean_iou
            save_model = True
        
        if save_model:
            fn_pth = 'semseg-%s-%.5f-%04d.pth' % (args.model_name, best_meaniou, epoch)
            log.info('Save model...',fn = fn_pth)
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, fn_pth))
            log.warn(cat_mean_iou)
        else:
            log.info('No need to save model')
            log.warn(cat_mean_iou)

        log.warn('Curr',accuracy=test_metrics['accuracy'], meanIOU=mean_iou)
        log.warn('Best',accuracy=best_acc, meanIOU=best_meaniou)


def evaluate(args):
    test_data, test_label = _load(load_train = False)
    test_dataset = S3DISDataLoader(test_data,test_label)
    testdataloader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=True, num_workers=args.workers)

    log.debug('Building Model', args.model_name)
    num_classes = 13
    if args.model_name == 'pointnet2':
        model = PointNet2SemSeg(num_classes) 
    else:
        model = PointNetSeg(num_classes,feature_transform=True,semseg = True)

    if args.pretrain is None:
        log.err('No pretrain model')
        return

    log.debug('Loading pretrain model...')
    checkpoint = torch.load(args.pretrain)
    model.load_state_dict(checkpoint)
    model.cuda()

    test_metrics, cat_mean_iou = test_semseg(
        model.eval(), 
        testdataloader, 
        seg_label_to_cat,
        num_classes = num_classes,
        pointnet2 = args.model_name == 'pointnet2'
    )
    mean_iou = np.mean(cat_mean_iou)
    log.info(Test_accuracy=test_metrics['accuracy'], Test_meanIOU=mean_iou)

def vis(args):
    test_data, test_label = _load(load_train = False)
    test_dataset = S3DISDataLoader(test_data,test_label)
    testdataloader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False, num_workers=args.workers)

    log.debug('Building Model', args.model_name)
    num_classes = 13
    if args.model_name == 'pointnet2':
        model = PointNet2SemSeg(num_classes) 
    else:
        model = PointNetSeg(num_classes,feature_transform=True,semseg = True)

    if args.pretrain is None:
        log.err('No pretrain model')
        return

    log.debug('Loading pretrain model...')
    checkpoint = torch.load(args.pretrain)
    model.load_state_dict(checkpoint)
    model.cuda().eval()
    cmap = plt.cm.get_cmap("hsv", 13)
    cmap = np.array([cmap(i) for i in range(13)])[:, :3]

    for batch_id, (points, target) in enumerate(testdataloader):
        log.info('Press space to exit','press Q for next frame')
        batchsize, num_point, _ = points.size()
        points, target = Variable(points.float()), Variable(target.long())
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        if args.model_name == 'pointnet2':
            pred = model(points)
        else:
            pred, _ = model(points)

        points = points[:, :3, :].transpose(-1, 1)
        pred_choice = pred.data.max(-1)[1]

        for idx in range(batchsize):
            pt, gt, pred = points[idx], target[idx], pred_choice[idx]
            gt_color = cmap[gt.cpu().numpy() - 1, :]
            pred_color = cmap[pred.cpu().numpy() - 1, :]

            point_cloud = open3d.geometry.PointCloud()
            point_cloud.points = open3d.utility.Vector3dVector(pt.cpu().numpy())
            point_cloud.colors = open3d.Vector3dVector(gt_color)

            vis = open3d.visualization.VisualizerWithKeyCallback()
            vis.create_window()
            vis.get_render_option().background_color = np.asarray([0, 0, 0])
            vis.add_geometry(point_cloud)

            vis.register_key_callback(32, lambda vis: exit())
            vis.run()
            vis.destroy_window()

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.mode == "train":
        train(args)
    if args.mode == "eval":
        evaluate(args)
    if args.mode == "vis":
        vis(args)
