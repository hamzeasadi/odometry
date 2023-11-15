"""
docs

"""


import os

import torch
from torch import nn as nn
from torch.nn import BCEWithLogitsLoss
from Utils.gutils import XRepresentation
from Config.config import Paths



class LivenessLoss(nn.Module):
    """
    custom loss
    """

    def __init__(self, config_name:str=None, paths:Paths=None, margin:float=2.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        xpepresentation = XRepresentation()
        if paths is not None:
            self.paths = paths

        if config_name is not None:
            self.loss_config = xpepresentation.json2cls(os.path.join(paths.config, config_name))
        
        self.margin = margin
        
    def forward(self, labels, prd1, prd2):
        dist = torch.nn.functional.pairwise_distance(x1=prd1, x2=prd2)
        loss = (1 - labels) * torch.pow(dist, 2) \
        + (labels) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        
        return torch.mean(loss)
        
        