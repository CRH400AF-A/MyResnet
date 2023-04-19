from ResNet_cifar10 import BasicBlock, Bottleneck, MyResNet

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import matplotlib.pyplot as plt
import numpy as np

# GPU
def Device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

# load data
def Load_Data():
    NUM_TRAIN = 49000

    transform_train = T.Compose([
                    T.RandomCrop(size=32,padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])

    transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])

    cifar10_train = dset.CIFAR10('./datasets', train=True, download=True, transform=transform_train)
    loader_train = DataLoader(cifar10_train, batch_size=256, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

    cifar10_val = dset.CIFAR10('./datasets', train=True, download=True, transform=transform)
    loader_val = DataLoader(cifar10_val, batch_size=256, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

    cifar10_test = dset.CIFAR10('./datasets', train=False, download=True, transform=transform)
    loader_test = DataLoader(cifar10_test, batch_size=256)

    return loader_train, loader_val, loader_test

# Train
def Train(model, optimizer, epochs = 1):
    criterion = nn.CrossEntropyLoss().to(device)
    model = model.to(device = device)
    model.train() # setup for BN
    for e in range(epochs):
        print('Epoch(%d)' % e)
        if(e==30):
          for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        for t,(x,y) in enumerate(loader_train):
            x = x.to(device, dtype = torch.float32)
            y = y.to(device, dtype = torch.long)

            scores = model(x)
            optimizer.zero_grad()
            loss = criterion(scores, y)
            loss.backward()
            optimizer.step()
            
            correct = 0
            total = 0
            _, predict = torch.max(scores, 1)
            correct += (predict == y).sum()
            total += predict.size(0)
            accuracy_train = (correct/total).cpu().detach().numpy()
            accuracy_train_history.append(accuracy_train)

            if t % 5 == 0:
                with open('output.txt', 'a') as f:
                    print('Epoch(%d)' % e, 'Loss: %.4f' % loss, file=f)
                loss_val = loss.cpu().detach().numpy()
                loss_history.append(loss_val)
                Check_Accuracy(loader_val, model)

# Accuracy
def Check_Accuracy(loader, model):
    global best_accuracy
    global best_model
    correct = 0
    total = 0
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device, dtype = torch.float32)
            y = y.to(device, dtype = torch.long)

            scores = model(x)
            _, predict = torch.max(scores, 1)
            correct += (predict == y).sum()
            total += predict.size(0)

        if loader.dataset.train:    
            accuracy = (correct / total).cpu().detach().numpy()
            accuracy_val_history.append(accuracy)
            with open('output.txt', 'a') as f:
                print("Accuracy on Validation Set: ", '{:.2%}'.format(accuracy), file=f)
            if(accuracy > best_accuracy):
                best_accuracy = accuracy
                best_model = model
        else:   
            accuracy = (correct / total).cpu().detach().numpy()
            print("Accuracy on Test Set: ", '{:.2%}'.format(accuracy))

# Draw
def Draw(train_picture):
    plt.figure(figsize=(16, 4))
    plt.subplot(131)
    plt.xlabel('n')
    plt.ylabel('loss')
    n = np.arange(1, len(loss_history) + 1)
    plt.plot(n, loss_history, color = 'g')

    plt.subplot(132)
    plt.xlabel('n')
    plt.ylabel('val_accuracy')
    m = np.arange(1, len(accuracy_val_history) + 1)
    plt.plot(m, accuracy_val_history, color = 'r')

    plt.subplot(133)
    plt.xlabel('n')
    plt.ylabel('train_accuracy')
    k = np.arange(1, len(accuracy_train_history) + 1) / 5
    plt.plot(k, accuracy_train_history, color = 'b')

    plt.savefig(train_picture)
    

# main
if __name__ == '__main__':
    loss_history = []
    accuracy_train_history = []
    accuracy_val_history = []

    best_accuracy = 0
    best_model = None

    device = Device()
    loader_train, loader_val, loader_test = Load_Data()

    learning_rate = 1e-3
    model = MyResNet(Bottleneck, [3,4,6,3], 10)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
    #                     momentum=0.9, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    Train(model, optimizer, epochs = 120)
    Draw('train_acc.png')
    
    Check_Accuracy(loader_test, best_model)