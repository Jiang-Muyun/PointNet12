import argparse
import os
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from data_utils.ModelNetDataLoader import ModelNetDataLoader, load_data
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from utils import test, save_checkpoint, select_avaliable
from colors import *
from model.pointnet2 import PointNet2ClsMsg
from model.pointnet import PointNetCls, feature_transform_reguliarzer

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batchsize', type=int, default=24, help='batch size in training')
    parser.add_argument('--epoch',  default=100, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--train_metric', type=str, default=False, help='whether evaluate on training dataset')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer for training')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate of learning rate')
    parser.add_argument('--rotation',  default=None, help='range of training rotation')
    parser.add_argument('--model_name', default='pointnet2', help='range of training rotation')
    parser.add_argument('--feature_transform', default=False, help="use feature transform in pointnet")
    return parser.parse_args()

def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    root = select_avaliable([
        '/media/james/MyPassport/James/dataset/ShapeNet/modelnet40_ply_hdf5_2048/',
        '/home/james/dataset/ShapeNet/modelnet40_ply_hdf5_2048/'
    ])

    if args.rotation is not None:
        ROTATION = (int(args.rotation[0:2]),int(args.rotation[3:5]))
    else:
        ROTATION = None

    '''CREATE DIR'''
    experiment_dir = './experiment/'
    os.makedirs(experiment_dir,exist_ok=True)

    checkpoints_dir = './experiment/clf/%s/'%(args.model_name)
    os.makedirs(checkpoints_dir, exist_ok=True)

    '''DATA LOADING'''
    print_info('Load dataset ...')
    train_data, train_label, test_data, test_label = load_data(root, classification=True)
    print_kv("The number of training data is:",train_data.shape[0])
    print_kv("The number of test data is:", test_data.shape[0])
    trainDataset = ModelNetDataLoader(train_data, train_label, rotation=ROTATION)

    if ROTATION is not None:
        print_kv('The range of training rotation is',ROTATION)
    
    testDataset = ModelNetDataLoader(test_data, test_label, rotation=ROTATION)
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=args.batchsize, shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=args.batchsize, shuffle=False)

    '''MODEL LOADING'''
    num_class = 40
    model = PointNetCls(num_class,args.feature_transform).cuda() if args.model_name == 'pointnet' else PointNet2ClsMsg().cuda()
    if args.pretrain is not None:
        print_info('Use pretrain model...')
        checkpoint = torch.load(args.pretrain)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print_info('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    global_epoch = 0
    global_step = 0
    best_tst_accuracy = 0.0

    '''TRANING'''
    print('Start training...')
    for epoch in range(start_epoch,args.epoch):
        print_kv('train_clf',args.model_name)
        print_kv('gpu',args.gpu)
        print_kv('Epoch','%d/%s'%(epoch, args.epoch))

        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            model = model.train()
            pred, trans_feat = model(points)
            loss = F.nll_loss(pred, target.long())
            if args.feature_transform and args.model_name == 'pointnet':
                loss += feature_transform_reguliarzer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            global_step += 1

        train_acc = test(model.eval(), trainDataLoader) if args.train_metric else None
        acc = test(model, testDataLoader)

        print_kv('loss', loss.data)
        print_kv('Test Accuracy', acc)
        if args.train_metric:
            print_kv('Train Accuracy', train_acc)

        if acc >= best_tst_accuracy:
            best_tst_accuracy = acc
            fn_pth = 'clf-%s-%.5f-%04d.pth'%(args.model_name, acc, epoch)
            print_kv('Saving model....', fn_pth)
            torch.save(model.state_dict(), os.path.join(checkpoints_dir,fn_pth))
        global_epoch += 1

        torch.cuda.empty_cache()
    print_kv('Best Accuracy', best_tst_accuracy)
    print_info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
