import time
import numpy as np
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torch.nn.functional as F

model = models.segmentation.fcn_resnet101(pretrained=True).cuda().eval()
model = models.segmentation.deeplabv3_resnet101(pretrained=1).cuda().eval()

label_colors = np.array([(0, 0, 0),
            (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
            (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
            (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
            (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
 
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