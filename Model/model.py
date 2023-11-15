"""
docs
"""


import os

import torch
from torch import nn as nn
from torch.nn import BCEWithLogitsLoss

from Utils.gutils import XRepresentation
from Config.config import Paths

class LstmBase(nn.Module):
    """
    docs
    """

    def __init__(self, config_name:str, paths:Paths, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        xpepresentation = XRepresentation()
        self.model_config = xpepresentation.json2cls(os.path.join(paths.config, config_name))

        self.lstm = nn.LSTM(
            input_size=self.model_config.lstm.input_size, 
            hidden_size=self.model_config.lstm.hidden_size, 
            num_layers=self.model_config.lstm.num_layer, 
            batch_first=self.model_config.lstm.batch_first, 
            bidirectional=self.model_config.lstm.bidirectional, 
            dropout=self.model_config.lstm.drop_out)
        self.act = nn.ReLU()

    
    def forward(self, x):
        output, (final_hidden_state, final_cell_state) = self.lstm(x)
        return self.act(output)[:, -1, :]








class Conv2dBase(nn.Module):
    """
    a conv base model
    """

    def __init__(self, config_name:str, paths:Paths, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        xpepresentation = XRepresentation()
        self.model_config = xpepresentation.json2cls(os.path.join(paths.config, config_name))

        self.layer0 = nn.Sequential(
                                    nn.Conv2d(in_channels=self.model_config.conv2d.inch,out_channels=self.model_config.conv2d.outch, 
                                                        kernel_size=self.model_config.conv2d.ks,stride=self.model_config.conv2d.stride,
                                                        padding=1),
                                    nn.BatchNorm2d(num_features=self.model_config.conv2d.outch),nn.ReLU())
        
        self.layer1 = nn.Sequential(
                                    nn.Conv2d(self.model_config.conv2d.outch,self.model_config.conv2d.outch*2, 
                                              kernel_size=(3, 6), stride=1,padding=1), 
                                    nn.BatchNorm2d(self.model_config.conv2d.outch*2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=[2, 4], stride=2))
        
        self.layer2 = nn.Sequential(
                                    nn.Conv2d(self.model_config.conv2d.outch*2,self.model_config.conv2d.outch*3, 
                                              kernel_size=(2, 4), stride=1,padding=1), 
                                    nn.BatchNorm2d(self.model_config.conv2d.outch*3),
                                    nn.ReLU(), nn.AvgPool2d(kernel_size=3, stride=2))
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        """
        docs
        """

        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        
        return self.flatten(out)







class LivenessModel(nn.Module):
    """
    head of classifer
    """
    def __init__(self, config_name:str, paths:Paths, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_model = Conv2dBase(config_name=config_name, paths=paths)

        self.fc1 = nn.Linear(
            in_features=self.base_model.model_config.conv2d.outch*3*2, 
            out_features=self.base_model.model_config.head.hidden_size
            )

        self.drp_out = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(
            in_features=self.base_model.model_config.head.hidden_size,
            out_features=1
            )
        self.act = nn.ReLU()
    
    def forward(self, x1, x2):
        out1 = self.base_model(x1)
        out2 = self.base_model(x2)

        out = torch.hstack((out1, out2))
        out = self.act(self.fc1(out))
        out = self.drp_out(out)
        logits = self.fc2(out)

        return logits



class LinearModel(nn.Module):
    """
    docs
    """

    def __init__(self, feature_size:int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden = int(2*feature_size/3)
        logit_size = int(1*hidden//2)

        self.base = nn.Sequential(
                    nn.Linear(in_features=feature_size, out_features=hidden), nn.ReLU(),
                    nn.Linear(in_features=hidden, out_features=logit_size)
        )

    def forward(self, x1, x2):
        out1 = self.base(x1)
        out2 = self.base(x2)

        return out1, out2







def main():
    """
    docs
    """
    
    # base_model = LivenessModel(config_name="model_config.json", paths=Paths())

    base_model = LinearModel(feature_size=15)

    x = torch.randn(size=(2, 15))
    
    out1, out2 = base_model(x, x)

    print(out1)
    print(out2)
    print(out1.shape)






if __name__ == "__main__":
    main()

