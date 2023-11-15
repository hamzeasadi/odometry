"""
docs

"""


import os

import torch
from torch.nn import functional as F


def binary_accuracy(prd:torch.tensor, gt:torch.tensor, threshold:float=0.5):
    """
    calculate accuracy for binary classifier

    """
    prd.squeeze_()
    gt.squeeze_()

    y_hat = F.sigmoid(prd)
    prd = y_hat>threshold
    prd = prd.float()

    acc = torch.sum(prd==gt)/len(prd)
    return acc


def contrastive_accuracy(labels:torch.tensor, prd1:torch.tensor, prd2:torch.tensor, margin:float):
    dist = torch.nn.functional.pairwise_distance(x1=prd1, x2=prd2)
    lbls = dist<margin
    lbls = lbls.float()
    acc_arr = lbls==labels
    acc = torch.mean(acc_arr.float())
    return acc
