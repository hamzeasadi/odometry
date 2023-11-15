"""
dataset generation for the kitti liveness verification 
"""

import os
from glob import glob
import pickle

import torch
from torch.utils.data import (
                            Dataset,
                            DataLoader
                            )
import numpy as np

from Config.config import Paths
from Utils.gutils import XRepresentation



class KittiLiveness(Dataset):
    """
    dataset
    """

    def __init__(self, datatype:str, paths:Paths, attach_type:str, normalize:bool=False) -> None:
        super().__init__()
        self.paths = paths

        self.dataset_root = os.path.join(paths.dataset, "kitti", attach_type, datatype)

        self.samples = self.load_sample_paths()
        self.dataset_size = len(self.samples)


    def load_sample_paths(self):
        normal_path = os.path.join(self.dataset_root, "normal")
        attack_path = os.path.join(self.dataset_root, "attack")

        attack_samples_path = glob(os.path.join(attack_path, "*.pkl"))
        normal_samples_path = glob(os.path.join(normal_path, "*.pkl"))

        sampes = []
        for path in attack_samples_path:
            pair = (path, 0)
            sampes.append(pair)

        for path in normal_samples_path:
            pair = (path, 1)
            sampes.append(pair)

        return sampes
    

    def unpack_pickle(self, data_path:str):
        with open(data_path, "rb") as data_file:
            data = pickle.load(data_file)
        
        return data
    

    def form_sample(self, data_path:str):
        data = self.unpack_pickle(data_path=data_path)
        
        return np.array(data['gt']).astype(np.float32)[:15], np.array(data['prd']).astype(np.float32)[:15]
    

    def __len__(self):
        """
        docs
        """
        return self.dataset_size


    def __getitem__(self, index):
        """
        docs
        """
        sample, label = self.samples[index]
        gt, prd = self.form_sample(sample)

        gt_out = torch.from_numpy(gt)
        # gt_out.unsqueeze_(dim=0)
        prd_out = torch.from_numpy(prd)
        # prd_out.unsqueeze_(dim=0)
        label = torch.tensor(label, dtype=torch.float32)

        return gt_out, prd_out, label



class CreateLoader:
    """
    create a data loader for train, validation, test

    """
    def __init__(self, paths:Paths, config_name:str) -> None:
        self.paths = paths
        xpepresentation = XRepresentation()
        self.loader_config = xpepresentation.json2cls(os.path.join(paths.config, config_name))

    
    def validation_loader(self):
        """
        docs

        """
        dataset = KittiLiveness(datatype="validation", paths=self.paths, attach_type="delete")
        loader = DataLoader(dataset=dataset, 
                            batch_size=self.loader_config.valid.batch_size,
                            shuffle=self.loader_config.valid.shuffle)
        
        return loader

    
    def train_loader(self):
        """
        docs

        """
        dataset = KittiLiveness(datatype="train", paths=self.paths, attach_type="delete")
        loader = DataLoader(dataset=dataset, 
                            batch_size=self.loader_config.train.batch_size,
                            shuffle=self.loader_config.train.shuffle)
        
        return loader
    

    def test_loader(self):
        """
        docs
        
        """

        raise NotImplemented
        




def main():
    """
    paths
    """

    dataset = KittiLiveness(datatype="train", paths=Paths(), attach_type="delete")


    gt, prd, y = dataset[1]

    print(gt)
    print(prd.shape)
    print(y)




if __name__ == "__main__":
    main()
