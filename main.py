"""
docs
"""

import os
from pprint import pprint

from torch.optim import Adam

from Config.config import Paths
from Model.model import LinearModel
from Dataset.dataset import CreateLoader
from Loss.loss import LivenessLoss
from Engine.train import (
                            train,
                            valid
                            )
from Utils.eval_utils import contrastive_accuracy


def main():
    """
    docs
    """
    paths = Paths()
    data_loader = CreateLoader(config_name="loader_config.json", paths=paths)

    train_loader = data_loader.train_loader()
    valid_loader = data_loader.validation_loader()

    # model = LivenessModel(config_name="model_config.json", paths=paths)

    model = LinearModel(feature_size=15)



    criterion = LivenessLoss(config_name=None, paths=None, margin=2)

    optimizer = Adam(params=model.parameters(), lr=3e-4)


    # for i in range(100):
    #     tloss = 0
    #     num_batches = len(train_loader)
    #     for x, y, l in train_loader:
    #         out1, out2 = model(x, y)
    #         loss = criterion(l, out1, out2)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         tloss += loss.item()
    #         contrastive_accuracy(labels=l, prd1=out1, prd2=out2, margin=0.0002)
    #         break
    #     # print(f"epoch={i} loss={tloss/num_batches}")
    #     break



    for i in range(100):

        train_result = train(model=model, dataset=train_loader, paths=paths, criterion=criterion, opt=optimizer)
        # valid_result = valid(model=model, dataset=train_loader, paths=paths, criterion=criterion)

        print(train_result)
    #     print(valid_result)








if __name__ == "__main__":
    main()
