# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:31:53 2020

@author: user
"""

import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model import Net
from tqdm import tqdm
import torch.nn as nn

import matplotlib.pyplot as plt

def main():
    print("Preparing datasets...")
    
    batch_size = 32
    device = 'cuda'
    n_epochs = 20
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_trainval = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    
    n_samples = len(dataset_trainval) 
    train_size = int(len(dataset_trainval) * 0.9) 
    val_size = n_samples - train_size
    print("train_size: ", train_size, ", val_size: ", val_size)
    
    # shuffle and split
    dataset_train, dataset_valid = torch.utils.data.random_split(dataset_trainval, [train_size, val_size])

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        shuffle=True
    )
    
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        shuffle=False
    )
            
    print("Defining model...")
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    
    train_loss = np.zeros(n_epochs)
    val_loss = np.zeros(n_epochs)
    val_acc = np.zeros(n_epochs)
    
    print("Start　training...")
    
    torch.backends.cudnn.benchmark = True
    for epoch in tqdm(range(n_epochs)):
        losses = []
        count = 0

        model.train()

        for x, t in tqdm(dataloader_train):
    
            x = x.to(device)
            t = t.to(device)
    
            model.zero_grad()
    
            y, _, _ = model(x)
            loss = criterion(y, t)
            loss.backward()
            optimizer.step()
    
            losses.append(loss.cpu().detach().numpy())

            count += 1        
            
        losses_val = []
        model.eval()

        correct = 0
        total = 0        
        for x, t in tqdm(dataloader_valid):
    
            x = x.to(device)
            t = t.to(device)
    
            y, _, _ = model(x)
            loss = criterion(y, t)
            losses_val.append(loss.cpu().detach().numpy())
        
            _, predicted = torch.max(y.data, 1)
            total += t.size(0)
            correct += (predicted == t).sum().item()

        val_acc[epoch] = 100 * float(correct/total)
        train_loss[epoch] = np.average(losses)
        val_loss[epoch] = np.average(losses_val)
        
        tqdm.write('EPOCH:%d, Train Loss:%lf, Valid Loss:%lf, Accuracy:%lf' %
                   (epoch+1, train_loss[epoch], val_loss[epoch], val_acc[epoch]))
        
    
    # loss
    print("Saving loss curve...")
    plt.figure(figsize=(5, 4))
    plt.title("Loss")
    plt.plot(np.arange(n_epochs), train_loss, label="train")
    plt.plot(np.arange(n_epochs), val_loss, label="val")
    plt.legend()
    plt.savefig("loss_curve.png")
    
    print("Saving model...")
    model_path = 'model.pth'
    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    main()
