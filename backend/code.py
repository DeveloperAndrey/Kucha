import glob
import os
import time
import json
import pandas as pd
import numpy as np
import torch

from torch import nn, optim, cuda
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from ignite.engine import Engine
from ignite import metrics# import Accuracy, Recall, Precision

from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

from statistics import mean

class PileFinder(nn.Module): 

    def __init__(self, activation=nn.ReLU, init=None):
        super(PileFinder, self).__init__()        
        self.conv_layers = nn.Sequential(
            # Layer 1
            #nn.BatchNorm2d(num_features=1), 
            nn.Conv2d(1, 50, kernel_size=7, stride=2, bias=False), # 9*9*50 = 4050
            activation(),
            nn.Conv2d(50, 75, kernel_size=5, stride=1, bias=False),
            nn.MaxPool2d(kernel_size=2),
            activation(),

              # Layer 2
              nn.BatchNorm2d(num_features=75), 
              nn.Conv2d(75, 100, kernel_size=5, stride=1, bias=False), # 7*7*50*75 = 3675*50 = 183 750
              activation(),
              nn.Conv2d(100, 100, kernel_size=3, stride=1, bias=False),
              nn.MaxPool2d(kernel_size=2),
              activation(),
          
            # Layer 3
            #nn.BatchNorm2d(num_features=75), 
            nn.Conv2d(100, 125, kernel_size=3,  stride=1, bias=False), # 5*5*75*100 = 2500*75 = 187 500
            activation(),
            nn.Conv2d(125, 125, kernel_size=3,  stride=1, bias=False), 
            nn.MaxPool2d(kernel_size=2), 
            activation(),
            
              # Layer 4
              nn.BatchNorm2d(num_features=125), 
              nn.Conv2d(125, 150, kernel_size=3, stride=1, bias=False), # 5*5*100*100 = 2500*100 = 250 000
              activation(),
              nn.Conv2d(150, 150, kernel_size=3, stride=1, bias=False), 
              nn.MaxPool2d(kernel_size=2),
              activation(),
          
            # Layer 5
            #nn.BatchNorm2d(num_features=75), 
            nn.Conv2d(150, 175, kernel_size=5,  stride=1, bias=False), # 5*5*75*100 = 2500*75 = 187 500
            activation(),
            nn.Conv2d(175, 200, kernel_size=3,  stride=1, bias=False), 
            nn.MaxPool2d(kernel_size=2), 
            activation(),
            
              # Layer 6
              nn.BatchNorm2d(num_features=200), 
              nn.Conv2d(200, 250, kernel_size=3, stride=1, bias=False), # 5*5*100*100 = 2500*100 = 250 000
              activation(),
            
        )
        self.drop_out = nn.Dropout(p=0.025)
        
        self.fc_layers = nn.Sequential(
            nn.BatchNorm1d(1750),
            nn.Linear(1750, 500, bias=False), activation(), # 900*300 = 270 000
            nn.Linear(500, 100, bias=False), activation(), # 300*100 = 30 000
            nn.BatchNorm1d(100),
            nn.Linear(100, 25, bias=False), activation(), # 100*25 = 2 500
            nn.Linear(25, 2, bias=False), activation(), # 25 * 2 = 50
        )
        
        self.softmax = nn.Softmax(dim=1)
        
        if train_on_gpu:
            self.conv_layers = self.conv_layers.cuda()
            #self.drop_out = self.drop_out.cuda()
            self.fc_layers = self.fc_layers.cuda()
            #self.softmax = self.softmax.cuda()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc_layers(x)
        x = self.softmax(x)
        return x
    
    def init_weights(init):
        if np.isnan(init):
            print('Init skipped')
            return        
        init(self.conv_layers)
        init(self.fc_layers.weight)

def get_epoch_statistics(stat):
    stat_series = pd.Series(stat)
    return {
        'mean': stat_series.mean(),
        'std': stat_series.std(),
        'max': stat_series.max()
    }

def print_epoch_statistics(stats, epoch, epoch_accuracy, epoch_recall):
    if not isinstance(stats, dict):
        stats = get_epoch_statistics(stats)
    print('Epoch {} stats: Average Loss: {:.4f}  Std: {:.4f}  Max: {:.4f}'.format(
             epoch, stats['mean'], stats['std'], stats['max'] ))
    print('Metrics on Train: Acc={:.4f}; Recall=[{:.4f}, {:.4f}]'.format(
             epoch_accuracy, epoch_recall[0], epoch_recall[1]))
    print('')

def test_network(net):
    #print('Testing...')
    accuracy, precision, recall = metrics.Accuracy(), metrics.Precision(), metrics.Recall()
    net.eval()
    with torch.no_grad():
        for butch_num, dataset in enumerate(test_dataloader):
            data, target = dataset[0], dataset[1]
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            net_out = net(data)
            accuracy.update((net_out, target))
            precision.update((net_out, target))
            recall.update((net_out, target))
    net.train()
    res = {'Accuracy': accuracy.compute(), 'Precision': precision.compute(), 'Recall': recall.compute()}
    res['F1_score'] = 2.0 * res['Precision']*res['Recall'] / (res['Precision']+res['Recall']) 
    return res

def print_test_report(network, metrics_dict=None):
    if not isinstance(metrics_dict, dict):
        metrics_dict = test_network(network)
    print('Metrics on Test:')
    for metric_name, metric_val in metrics_dict.items():
        print('_', metric_name, ': ', metric_val, sep='')

def save_model(net, filename):
    path = './trained/{}.pt'.format(filename)
    torch.save(net.state_dict(), path)
    
def load_model(net, filename):
    path = './trained/{}.pt'.format(filename)
    net.load_state_dict(torch.load(path))

def save_statistics(stat, filename='statistics.json'):
    with open(filename, 'w') as f:
        json.dump(stat, f)

def load_statistics(filename='statistics.json'):
    with open(filename) as f:
        return json.load(f)


net = PileFinder(activation=nn.ELU, init=nn.init.kaiming_uniform_)

load_model(net, 'n1_good')

pil_img_to_torch = transforms.Compose([transforms.PILToTensor()])


# USAGE USAGE USAGE

image_tensor = pil_img_to_torch(PIL_IMG) # PIL_IMG - Pil изображение

net.eval()
with torch.no_grad():
    net_out = net(image_tensor)

