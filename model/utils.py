import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .pointnet import PointNetSeg, feature_transform_reguliarzer
from .pointnet2 import PointNet2SemSeg

def load_pointnet(model_name, num_classes, fn_pth):
    if model_name == 'pointnet':
        model = PointNetSeg(num_classes, input_dims = 4, feature_transform=True)
    else:
        model = PointNet2SemSeg(num_classes, feature_dims = 1)

    torch.backends.cudnn.benchmark = True
    model = torch.nn.DataParallel(model)

    assert fn_pth is not None,'No pretrain model'
    checkpoint = torch.load(fn_pth)
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()
    return model