import os
import h5py
from tqdm import tqdm
import time
import numpy as np
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torch.nn.functional as F

root = '/media/james/MyPassport/James/dataset/Localization/Pittsburgh/'
model = models.segmentation.deeplabv3_resnet101(pretrained=1).cuda().eval()

transform = T.Compose([# T.Resize(256),T.CenterCrop(224),
                    T.Resize((224, 224)),
                    T.ToTensor(), 
                    T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]
                    )

def demo():
    img = Image.open('./data/voc/test1.jpg')
    transform = T.Compose([T.Resize(256),T.CenterCrop(224),T.ToTensor(), 
                    T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    img_batch = transform(img).cuda().unsqueeze(0)

    with torch.no_grad():
        logits = model(img_batch)['out']
        inter = F.interpolate(logits, (50, 40), mode='bilinear', align_corners=False)
    print('inter',inter.size())
    print(inter.cpu().numpy()[0].sum(0))
    pred = inter.max(axis = 1)[1]
    print(pred.shape)
    plt.imshow(pred.cpu().numpy()[0])
    plt.show()

def process_list(fn_list, fn_h5):
    fns = open(os.path.join(root, fn_list), 'r').readlines() #[:10]
    fp_h5 = h5py.File(fn_h5,"a")
    try:
        for key in tqdm(fns, total=len(fns), smoothing=0.9):
            key = key.strip()
            if not key in fp_h5.keys():
                filename = os.path.join(root, key)
                img = Image.open(filename)
                img_batch = transform(img).cuda().unsqueeze(0)
                with torch.no_grad():
                    logits = model(img_batch)['out']
                    inter = F.interpolate(logits, (40, 50), mode='bilinear', align_corners=False)
                    logits_np = inter.squeeze(0).cpu().numpy()
                fp_h5[key] = logits_np
            else:
                print('skip', key)
    except Exception as err:
        print(err)
    except KeyboardInterrupt:
        fp_h5.close()

if __name__ == '__main__':
    process_list('pitts30k_train.txt','pitts30k_train.h5')
    # process_list('pitts250k_train.txt','pitts250k_train.h5')