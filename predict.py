#predict.py

import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet
import matplotlib.pyplot as plt
import numpy as np

#预处理
transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
net.load_state_dict(torch.load('Lenet.pth'))

im = Image.open('IandCat.jpg')      #测试飞机
#im = Image.open('猫.jpg')         #测试猫
#im = Image.open('轮船.jpg')       #测试狗
#im = Image.open('狗。jpg’)       #测试猫，可能出现错误
im1 = transform(im)  # [C, H, W]
im = torch.unsqueeze(im1, dim=0)  # [N, C, H, W]

#显示图像
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
imshow(im1)


with torch.no_grad():
    outputs = net(im)
    predict = torch.max(outputs, dim=1)[1].data.numpy()
    predict1 = torch.softmax(outputs, dim=1)
    print(predict1)
    print(classes[int(predict)])

