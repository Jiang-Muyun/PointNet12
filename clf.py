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
from utils import test, save_checkpoint, select_avaliable, mkdir, auto_complete
from utils import Tick,Tock
import log
from model.pointnet2 import PointNet2ClsMsg
from model.pointnet import PointNetCls, feature_transform_reguliarzer

def parse_args():
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--model_name', default='pointnet', help='pointnet or pointnet2')
    parser.add_argument('--mode', default='train', help='train or eval')
    parser.add_argument('--batch_size', type=int, default=0, help='batch size in training')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--train_metric', type=str, default=False, help='whether evaluate on training dataset')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--pretrain', type=str, default=None, help='whether use pretrain model')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate of learning rate')
    parser.add_argument('--feature_transform', default=False, help="use feature transform in pointnet")
    parser.add_argument('--augment', default=False, action='store_true', help="Enable data augmentation")
    return auto_complete(parser.parse_args(),'clf')

root = select_avaliable([
    '/media/james/Ubuntu_Data/dataset/ShapeNet/modelnet40_ply_hdf5_2048/',
    '/media/james/MyPassport/James/dataset/ShapeNet/modelnet40_ply_hdf5_2048/',
    '/home/james/dataset/ShapeNet/modelnet40_ply_hdf5_2048/',
    '/media/james/HDD/James_Least/Large_Dataset/ShapeNet/modelnet40_ply_hdf5_2048/'
])

def train(args):
    experiment_dir = mkdir('./experiment/')
    checkpoints_dir = mkdir('./experiment/clf/%s/'%(args.model_name))
    train_data, train_label, test_data, test_label = load_data(root)

    trainDataset = ModelNetDataLoader(train_data, train_label, data_augmentation = args.augment)
    trainDataLoader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True)

    testDataset = ModelNetDataLoader(test_data, test_label)
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=args.batch_size, shuffle=False)

    log.info('Building Model',args.model_name)
    if args.model_name == 'pointnet':
        num_class = 40
        model = PointNetCls(num_class,args.feature_transform).cuda()  
    else:
        model = PointNet2ClsMsg().cuda()

    if args.pretrain is not None:
        log.info('Use pretrain model...')
        model.load_state_dict(torch.load(args.pretrain))
        init_epoch = int(args.pretrain[:-4].split('-')[-1])
        log.info('start epoch from', init_epoch)
    else:
        log.info('Training from scratch')
        init_epoch = 0

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate,
                                    betas=(0.9, 0.999),eps=1e-08,weight_decay=args.decay_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    LEARNING_RATE_CLIP = 1e-5

    device_ids = [int(x) for x in args.gpu.split(',')]
    if len(device_ids) >= 2:
        torch.backends.cudnn.benchmark = True
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        log.info('Using multi GPU:',device_ids)
    else:
        model.cuda()
        log.info('Using single GPU:',device_ids)

    global_epoch = 0
    global_step = 0
    best_tst_accuracy = 0.0

    log.info('Start training...')
    for epoch in range(init_epoch,args.epoch):
        scheduler.step()
        lr = max(optimizer.param_groups[0]['lr'],LEARNING_RATE_CLIP)

        log.debug(job='clf',model=args.model_name,gpu=args.gpu,epoch='%d/%s' % (epoch, args.epoch),lr=lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            model = model.train()
            pred, trans_feat = model(points)
            # log.info('points',points.shape,'pred',pred.shape)
            loss = F.nll_loss(pred, target.long())
            if args.feature_transform and args.model_name == 'pointnet':
                loss += feature_transform_reguliarzer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            global_step += 1
        
        log.debug('clear cuda cache')
        torch.cuda.empty_cache()

        if args.train_metric:
            train_acc = test(model.eval(), trainDataLoader)
            log.info('Train Accuracy', train_acc)

        acc = test(model, testDataLoader)
        log.info(loss='%.5f' % (loss.data))
        log.info(Test_Accuracy='%.5f' % acc)

        if acc >= best_tst_accuracy:
            best_tst_accuracy = acc
            fn_pth = 'clf-%s-%.5f-%04d.pth'%(args.model_name, acc, epoch)
            log.debug('Saving model....', fn_pth)
            torch.save(model.state_dict(), os.path.join(checkpoints_dir,fn_pth))
        global_epoch += 1

    log.info(Best_Accuracy = best_tst_accuracy)
    log.info('End of training...')

def evaluate(args):
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

    acc = test(model.eval(), testDataLoader)
    log.debug(Test_Accurac='%.5f' % (acc))

def vis(args):
    test_data, test_label = load_data(root, train = False)
    log.info(test_data=test_data.shape,test_label=test_label.shape)

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

    log.info('Press space to exit, press Q for next frame')
    
    for idx in range(test_data.shape[0]):
        point_np = test_data[idx:idx+1]
        gt = test_label[idx][0]

        points = torch.from_numpy(point_np)
        points = points.transpose(2, 1).cuda()

        pred, trans_feat = model(points)
        pred_choice = pred.data.max(1)[1]
        log.info(gt=class_names[gt], pred_choice=class_names[pred_choice.cpu().numpy().item()])

        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(point_np[0])

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
