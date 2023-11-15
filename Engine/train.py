
"""
docs

"""

import os

from Config.config import Paths
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from Utils.eval_utils import binary_accuracy
from Utils.eval_utils import contrastive_accuracy






def train(model:nn.Module, dataset:DataLoader, paths:Paths, criterion:nn.Module, opt:Optimizer, dev:str="cpu"):
    """
    apply the training procedure to the model
    """

    model.train()
    model.to(dev)
    train_acc = 0.0
    train_loss = 0.0
    num_batchs = len(dataset)

    l1 = nn.L1Loss()

    for X1, X2, Y in dataset:
        out1, out2 = model(X1, X2)
        loss = criterion(Y, out1, out2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss += loss.item()
        train_acc += contrastive_accuracy(prd1=out1, prd2=out2, labels=Y, margin=2)
    
    return dict(acc=train_acc/num_batchs, loss=train_loss/num_batchs)
        
        

def valid(model:nn.Module, dataset:DataLoader, paths:Paths, criterion:nn.Module, dev:str="cpu"):
    """
    apply the training procedure to the model
    """

    model.eval()
    model.to(dev)
    valid_acc = 0.0
    valid_loss = 0.0
    num_batchs = len(dataset)

    for X1, X2, Y in dataset:
        out = model(X1, X2)
        loss = criterion(Y, out)
        valid_loss += loss.item()
        valid_acc += binary_accuracy(gt=Y, prd=out)
    
    return dict(acc=valid_acc/num_batchs, loss=valid_loss/num_batchs)
