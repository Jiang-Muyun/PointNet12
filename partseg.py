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
from utils import to_categorical
from collections import defaultdict
from torch.autograd import Variable
from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch.nn.functional as F
from pathlib import Path
from utils import test_partseg, select_avaliable, mkdir, auto_complete
from utils import Tick,Tock
import log
from tqdm import tqdm
from model.pointnet2 import PointNet2PartSegMsg_one_hot
from model.pointnet import PointNetDenseCls,PointNetLoss

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def parse_args():
    parser = argparse.ArgumentParser('PointNet2')
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
    return auto_complete(parser.parse_args(),'partseg')

def _load(root):
    fn_cache = 'experiment/shapenetcore_partanno_segmentation_benchmark_v0_normal.h5'
    if not os.path.exists(fn_cache):
        log.debug('Indexing Files...')
        fns_full = []
        fp_h5 = h5py.File(fn_cache,"w")

        for line in open(os.path.join(root, 'synsetoffset2category.txt'), 'r'):
            name,wordnet_id = line.strip().split()
            pt_folder = os.path.join(root, wordnet_id)
            log.info('Building',name, wordnet_id)
            for fn in tqdm(os.listdir(pt_folder)):
                token = fn.split('.')[0]
                fn_full = os.path.join(pt_folder, fn)
                data = np.loadtxt(fn_full).astype(np.float32)

                h5_index = '%s_%s'%(wordnet_id,token)
                fp_h5.create_dataset(h5_index, data = data)

        log.debug('Building cache...')
        fp_h5.close()

    log.debug('Loading from cache...')
    fp_h5 = h5py.File(fn_cache, 'r')
    cache = {}
    for token in fp_h5.keys():
        cache[token] = fp_h5.get(token)[()]
    return cache

root = select_avaliable([
    '/media/james/HDD/James_Least/Large_Dataset/ShapeNet/shapenetcore_partanno_segmentation_benchmark_v0_normal/',
    '/media/james/Ubuntu_Data/dataset/ShapeNet/shapenetcore_partanno_segmentation_benchmark_v0_normal/',
    '/media/james/MyPassport/James/dataset/ShapeNet/shapenetcore_partanno_segmentation_benchmark_v0_normal/',
    '/home/james/dataset/ShapeNet/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
])

def train(args):
    experiment_dir = mkdir('./experiment/')
    checkpoints_dir = mkdir('./experiment/partseg/%s/'%(args.model_name))
    cache = _load(root)

    norm = True if args.model_name == 'pointnet' else False
    train_ds = PartNormalDataset(root,cache,npoints=2048, split='trainval', data_augmentation = args.augment)
    dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))
    
    test_ds = PartNormalDataset(root,cache,npoints=2048, split='test')
    testdataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))
    
    log.info("The number of training data is:",len(train_ds))
    log.info("The number of test data is:", len(test_ds))

    num_classes = 16
    num_part = 50

    if args.model_name == 'pointnet':
        model = PointNetDenseCls(cat_num=num_classes,part_num=num_part)
    else:
        model = PointNet2PartSegMsg_one_hot(num_part)

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

    device_ids = [int(x) for x in args.gpu.split(',')]
    if len(device_ids) >= 2:
        torch.backends.cudnn.benchmark = True
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        log.debug('Using multi GPU:',device_ids)
    else:
        model.cuda()
        log.debug('Using single GPU:',device_ids)

    criterion = PointNetLoss()
    LEARNING_RATE_CLIP = 1e-5

    history = defaultdict(lambda: list())
    best_acc = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    def feature_transform_reguliarzer(trans):
        d = trans.size()[1]
        I = torch.eye(d)[None, :, :]
        if trans.is_cuda:
            I = I.cuda()
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
        return loss

    def PointNet_Loss(labels_pred, label, seg_pred, seg, trans_feat):
        mat_diff_loss_scale = 0.001
        weight = 1
        seg_loss = F.nll_loss(seg_pred, seg)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        label_loss = F.nll_loss(labels_pred, label)
        loss = weight * seg_loss + (1-weight) * label_loss + mat_diff_loss * mat_diff_loss_scale
        return loss, seg_loss, label_loss

    for epoch in range(init_epoch,args.epoch):
        scheduler.step()
        lr = max(optimizer.param_groups[0]['lr'],LEARNING_RATE_CLIP)
        log.debug(job='partseg',model=args.model_name,gpu=args.gpu,epoch='%d/%s' % (epoch, args.epoch),lr=lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for i, data in tqdm(enumerate(dataloader, 0),total=len(dataloader),smoothing=0.9):
            points, label, target, norm_plt = data
            points, label, target = Variable(points.float()),Variable(label.long()),  Variable(target.long())
            points = points.transpose(2, 1)
            norm_plt = norm_plt.transpose(2, 1)
            points, label, target,norm_plt = points.cuda(),label.squeeze().cuda(), target.cuda(), norm_plt.cuda()
            optimizer.zero_grad()
            model = model.train()

            if args.model_name == 'pointnet':
                labels_pred, seg_pred, trans_feat = model(points, to_categorical(label, 16))
                seg_pred = seg_pred.contiguous().view(-1, num_part)
                target = target.view(-1, 1)[:, 0]
                # loss, seg_loss, label_loss = criterion(labels_pred, label, seg_pred, target, trans_feat)
                loss, seg_loss, label_loss = PointNet_Loss(labels_pred, label, seg_pred, target, trans_feat)
            else:
                seg_pred = model(points, norm_plt, to_categorical(label, 16))
                seg_pred = seg_pred.contiguous().view(-1, num_part)
                target = target.view(-1, 1)[:, 0]
                loss = F.nll_loss(seg_pred, target)

            history['loss'].append(loss.cpu().data.numpy())
            loss.backward()
            optimizer.step()
            
        log.debug('clear cuda cache')
        torch.cuda.empty_cache()

        forpointnet2 = args.model_name == 'pointnet2'
        test_metrics, test_hist_acc, cat_mean_iou = test_partseg(model.eval(), testdataloader, seg_label_to_cat,50,forpointnet2)

        log.info('Test Accuracy', '%.5f' % test_metrics['accuracy'])
        log.info('Class avg mIOU:', '%.5f' % test_metrics['class_avg_iou'])
        log.info('Inctance avg mIOU:', '%.5f' % test_metrics['inctance_avg_iou'])

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            fn_pth = 'partseg-%s-%.5f-%04d.pth' % (args.model_name, best_acc, epoch)
            log.info('Save model...',fn_pth)
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, fn_pth))
            log.info(cat_mean_iou)

        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']

        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']

        log.info('Best accuracy:', '%.5f'%(best_acc))
        log.info('Best class avg mIOU:', '%.5f'%(best_class_avg_iou))
        log.info('Best inctance avg mIOU:', '%.5f'%(best_inctance_avg_iou))

def evaluate(args):
    cache = _load(root)
    norm = True if args.model_name == 'pointnet' else False
    test_ds = PartNormalDataset(root, cache, npoints=2048, split='test')
    testdataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))
    log.info("The number of test data is:", len(test_ds))

    log.info('Building Model', args.model_name)
    num_classes = 16
    num_part = 50
    if args.model_name == 'pointnet2':
        model = PointNet2PartSegMsg_one_hot(num_part) 
    else:
        model = PointNetDenseCls(cat_num=num_classes,part_num=num_part)

    if args.pretrain is None:
        log.err('No pretrain model')
        return

    log.info('Loading pretrain model...')
    checkpoint = torch.load(args.pretrain)
    model.load_state_dict(checkpoint)
    model.cuda()

    log.info('Testing pretrain model...')
    forpointnet2 = args.model_name == 'pointnet2'
    test_metrics, test_hist_acc, cat_mean_iou = test_partseg(model.eval(), testdataloader, seg_label_to_cat, num_part, forpointnet2)

    log.info('test_hist_acc',len(test_hist_acc))
    log.info(cat_mean_iou)
    log.info('Test Accuracy','%.5f' % test_metrics['accuracy'])
    log.info('Class avg mIOU:','%.5f' % test_metrics['class_avg_iou'])
    log.info('Inctance avg mIOU:','%.5f' % test_metrics['inctance_avg_iou'])

def vis(args):
    cache = _load(root)
    norm = True if args.model_name == 'pointnet' else False
    test_ds = PartNormalDataset(root, cache, npoints=2048, split='test')
    testdataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))
    log.info("The number of test data is:", len(test_ds))

    log.info('Building Model', args.model_name)
    num_classes = 16
    num_part = 50
    if args.model_name == 'pointnet':
        model = PointNetDenseCls(cat_num=num_classes,part_num=num_part)
    else:
        model = PointNet2PartSegMsg_one_hot(num_part) 

    if args.pretrain is None:
        log.err('No pretrain model')
        return

    log.info('Loading pretrain model...')
    checkpoint = torch.load(args.pretrain)
    model.load_state_dict(checkpoint)
    model.cuda()
    log.info('Press space to exit, press Q for next frame')
    for batch_id, (points, label, target, norm_plt) in enumerate(testdataloader):
        batchsize, num_point, _= points.size()
        points, label, target, norm_plt = Variable(points.float()),Variable(label.long()), Variable(target.long()),Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(), label.squeeze().cuda(), target.cuda(), norm_plt.cuda()
        if args.model_name == 'pointnet':
            labels_pred, seg_pred, _  = model(points,to_categorical(label,16))
        else:
            seg_pred = model(points, norm_plt, to_categorical(label, 16))
        pred_choice = seg_pred.max(-1)[1]
        log.info(seg_pred=seg_pred.shape, pred_choice=pred_choice.shape)
        log.info(seg_pred=seg_pred.shape, pred_choice=pred_choice.shape)

        cmap_plt = plt.cm.get_cmap("hsv", num_part)
        cmap_list = [cmap_plt(i)[:3] for i in range(num_part)]
        np.random.shuffle(cmap_list)
        cmap = np.array(cmap_list)

        #log.info('points',points.shape,'label',label.shape,'target',target.shape,'norm_plt',norm_plt.shape)  
        for idx in range(batchsize):
            pt, gt, pred = points[idx].transpose(1, 0), target[idx], pred_choice[idx].transpose(-1, 0)
            # log.info('pt',pt.size(),'gt',gt.size(),'pred',pred.shape)

            gt_color = cmap[gt.cpu().numpy() - 1, :]
            pred_color = cmap[pred.cpu().numpy() - 1, :]

            point_cloud = open3d.geometry.PointCloud()
            point_cloud.points = open3d.Vector3dVector(pt.cpu().numpy())
            point_cloud.colors = open3d.Vector3dVector(pred_color)

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