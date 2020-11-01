# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:33:03 2020

@author: user
"""

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1) # 16 x 16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 8 x 8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 4 x 4
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1) # 2 x 2
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # 1 x 1
        self.fc1 = nn.Linear(256 * 1 * 1, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.conv4(h3))
        h5 = F.relu(self.conv5(h4))
        h6 = h5.view(-1, 256 * 1 * 1)
        h7 = F.relu(self.fc1(h6))
        h7 = self.dropout(h7)
        h8 = F.relu(self.fc2(h7))
        h8 = self.dropout(h8)
        y = self.fc3(h8)
        return y, h8, h2
