# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:31:53 2020

@author: user
"""

import PIL
PIL.PILLOW_VERSION = PIL.__version__

import numpy as np
import torch
#import torch.nn as nn
#import torch.optim as optim
#import torch.autograd as autograd
#import torch.nn.functional as F
from torchvision import datasets, transforms
from model import Net

import matplotlib.pyplot as plt
#from torch.utils.data import Dataset
#from torchvision.utils import make_grid
import copy
from sklearn.metrics.pairwise import cosine_similarity
#from scipy.stats import spearmanr

def imshow(img, filename="test.png"):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(filename)

device = 'cuda'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset_test = datasets.CIFAR10(root='./data', train=False, 
                                download=True, transform=transform)
dataset_tmp = copy.deepcopy(dataset_test)
classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
class_order = [8, 0, 9, 1, 7, 4, 5, 3, 2, 6]
sorted_classes = [classes[i] for i in class_order]

num_classes = len(classes)
num_each_class = 10
num_samples = int(num_classes*num_each_class)

print("Defining model...")
model = Net().to(device)
model_path = 'model.pth'
model.load_state_dict(torch.load(model_path))

y_data = np.zeros((num_samples, 10))
h8_data = np.zeros((num_samples, 256))
h2_data = np.zeros((num_samples, 64*8*8))
#h7_data = np.zeros((num_samples, 512))

model.eval()
for i in range(num_classes):
    dataset_tmp = copy.deepcopy(dataset_test)

    # https://discuss.pytorch.org/t/how-to-use-one-class-of-number-in-mnist/26276/8
    idx_bool = torch.tensor(dataset_tmp.targets) == class_order[i]
    #idx = np.where(idx_bool)[0]

    dataset_tmp.targets = torch.tensor(dataset_tmp.targets)
    dataset_tmp.data = dataset_tmp.data[idx_bool.numpy().astype(np.bool)]
    dataset_tmp.targets = dataset_tmp.targets[idx_bool]
    
    dataloader_tmp = torch.utils.data.DataLoader(dataset_tmp, batch_size=num_each_class, shuffle=True)
    
    # show images
    dataiter = iter(dataloader_tmp)
    images, labels = dataiter.next()
    
    #imshow(make_grid(images), filename="test"+str(i)+".png")
    
    images = images.to(device)
    y, h8, h2 = model(images)
    begin_idx = i*num_each_class
    #y_data[begin_idx:begin_idx+num_each_class] = y.cpu().detach().numpy()
    y_data[begin_idx:begin_idx+num_each_class] = torch.sigmoid(y).cpu().detach().numpy()
    h8_data[begin_idx:begin_idx+num_each_class] = h8.cpu().detach().numpy()
    h2_data[begin_idx:begin_idx+num_each_class] = torch.flatten(h2, start_dim=1).cpu().detach().numpy()
    #h7_data[begin_idx:begin_idx+num_each_class] = h7.cpu().detach().numpy()
    #_, predicted = torch.max(y.data, 1)
    #print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(num_each_class)))
    
y_rdm = 1 - cosine_similarity(y_data)
h8_rdm = 1 - cosine_similarity(h8_data)
h2_rdm = 1 - cosine_similarity(h2_data)
#y_rdm = 1 - spearmanr(y_data).correlation
#y_rdm = 1 - spearmanr(y_data)

# https://stackoverflow.com/questions/36274802/putting-some-text-to-a-python-plot
def annotate_axes(x1,y1,x2,y2,x3,y3,text, arrowstyle='<->', rotation=0):                       
    plt.annotate('', xy=(x1, y1),xytext=(x2,y2),             #draws an arrow from one set of coordinates to the other
                 arrowprops=dict(arrowstyle=arrowstyle),              #sets style of arrow
                 annotation_clip=False)                          #This enables the arrow to be outside of the plot

    plt.annotate(text,xy=(0,0),xytext=(x3,y3),               #Adds another annotation for the text
                 annotation_clip=False, rotation=rotation)
    
plt.figure(figsize=(15,7))
plt.subplot(1,3,1)
plt.title("conv2")
plt.imshow(h2_rdm)
annotate_axes(0,110,40,110,12,114,'inanimate', '|-|')
annotate_axes(39.5,110,100,110,62,114,'animate', '|-|')
annotate_axes(-15,0,-15,40,-19,15,'inanimate', '|-|', 90)
annotate_axes(-15,39.5,-15,100,-19,65,'animate', '|-|', 90)
plt.xticks([i for i in range(5, 100, 10)], sorted_classes)
plt.yticks([i for i in range(5, 100, 10)], sorted_classes)
plt.colorbar(orientation="horizontal")

plt.subplot(1,3,2)
plt.title("fc2")
plt.imshow(h8_rdm)
annotate_axes(0,110,40,110,12,114,'inanimate', '|-|')
annotate_axes(39.5,110,100,110,62,114,'animate', '|-|')
plt.xticks([i for i in range(5, 100, 10)], sorted_classes)
plt.yticks([])
plt.colorbar(orientation="horizontal")

plt.subplot(1,3,3)
plt.title("fc3")
plt.imshow(y_rdm)
annotate_axes(0,110,40,110,12,114,'inanimate', '|-|')
annotate_axes(39.5,110,100,110,62,114,'animate', '|-|')
plt.xticks([i for i in range(5, 100, 10)], sorted_classes)
plt.yticks([])
plt.colorbar(orientation="horizontal")
plt.tight_layout()
plt.savefig("rdm.png", bbox_inches='tight')