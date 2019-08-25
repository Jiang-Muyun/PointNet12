import argparse
import os
import torch
import time
import datetime
import numpy as np
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader
from collections import defaultdict
from torch.autograd import Variable
from data_utils.S3DISDataLoader import S3DISDataLoader, recognize_all_data,class2label
import torch.nn.functional as F
from pathlib import Path
from utils import test_semseg, select_avaliable, mkdir
from colors import *
from tqdm import tqdm
from model.pointnet2 import PointNet2SemSeg
from model.pointnet import PointNetSeg, feature_transform_reguliarzer

seg_classes = class2label
seg_label_to_cat = {}
for i,cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def parse_args():
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--model_name', type=str, default='pointnet2', help='pointnet or pointnet2')
    parser.add_argument('--batchsize', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer')

    return parser.parse_args()

def main(args):
    experiment_dir = mkdir('./experiment/')
    checkpoints_dir = mkdir('./experiment/semseg/%s/'%(args.model_name))

    dataset_root = select_avaliable([
        '/media/james/MyPassport/James/dataset/ShapeNet/indoor3d_sem_seg_hdf5_data/',
        '/home/james/dataset/ShapeNet/indoor3d_sem_seg_hdf5_data/'
    ])

    print('Load data...')
    train_data, train_label, test_data, test_label = recognize_all_data(dataset_root, test_area = 5)

    dataset = S3DISDataLoader(train_data,train_label)
    dataloader = DataLoader(dataset, batch_size=args.batchsize,shuffle=True, num_workers=int(args.workers))
    
    test_dataset = S3DISDataLoader(test_data,test_label)
    testdataloader = DataLoader(test_dataset, batch_size=args.batchsize,shuffle=True, num_workers=int(args.workers))

    num_classes = 13
    if args.model_name == 'pointnet2':
        model = PointNet2SemSeg(num_classes) 
    else:
        model = PointNetSeg(num_classes,feature_transform=True,semseg = True)

    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain))
        print('load model %s'%args.pretrain)
    else:
        print('Training from scratch')

    pretrain = args.pretrain
    init_epoch = int(pretrain[-14:-11]) if args.pretrain is not None else 0

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

    device_ids = [int(x) for x in args.gpu.split(',')]
    if len(device_ids) >= 2:
        torch.backends.cudnn.benchmark = True
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        print_info('Using multi GPU:',device_ids)
    else:
        model.cuda()
        print_info('Using single GPU:',device_ids)

    history = defaultdict(lambda: list())
    best_acc = 0
    best_meaniou = 0

    for epoch in range(init_epoch,args.epoch):
        scheduler.step()
        lr = max(optimizer.param_groups[0]['lr'],LEARNING_RATE_CLIP)

        print(green('semseg'),
            yellow('model:'), blue(args.model_name),
            yellow('gpu:'), blue(args.gpu),
            yellow('epoch:'), blue('%d/%s' % (epoch, args.epoch)),
            yellow('lr:'), blue(lr)
        )
        
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
                pred = model(points[:,:3,:],points[:,3:,:])

            pred = pred.contiguous().view(-1, num_classes)
            target = target.view(-1, 1)[:, 0]
            loss = F.nll_loss(pred, target)

            if args.model_name == 'pointnet':
                loss += feature_transform_reguliarzer(trans_feat) * 0.001

            history['loss'].append(loss.cpu().data.numpy())
            loss.backward()
            optimizer.step()
        
        print_debug('clear cuda cache')
        torch.cuda.empty_cache()
        time.sleep(0.05)

        test_metrics, test_hist_acc, cat_mean_iou = test_semseg(
            model.eval(), 
            testdataloader, 
            seg_label_to_cat,
            num_classes = num_classes,
            pointnet2 = args.model_name == 'pointnet2'
        )
        mean_iou = np.mean(cat_mean_iou)

        print_kv('Test accuracy','%.5f' % (test_metrics['accuracy']))
        print_kv('Test meanIOU','%.5f' % (mean_iou))

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            fn_pth = 'semseg-%s-%.5f-%04d.pth' % (args.model_name, best_acc, epoch)
            print_kv('Save model...',fn_pth)            
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, fn_pth))
            print_info(cat_mean_iou)
        
        if mean_iou > best_meaniou:
            best_meaniou = mean_iou

        print_kv('Best accuracy:' , '%.5f' % (best_acc))
        print_kv('Best meanIOU:','%.5f' % (best_meaniou))

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)
