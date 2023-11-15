"""
specify the paths and general informatin and configuration about the project

"""

import os
from dataclasses import dataclass


@dataclass
class Paths:
    """
    specifiy determined paths of folders, sub-packages,
    result folders etc

    """

    root:str = os.path.dirname(os.path.dirname(__file__))
    data:str = os.path.join(root, "data")
    dataset:str = os.path.join(data, "dataset")
    model:str = os.path.join(data, "model")
    result:str = os.path.join(data, "result")

    config:str = os.path.join(data, "config")

    @staticmethod
    def crtdir(path):
        """
        docs
        
        """

        if not os.path.exists(path):
            os.makedirs(path)


@dataclass
class KitiPaths:
    """
    docs

    """

    dataset:str = "/home/hasadi/project/Dataset/kitti/dataset"
    sequences:str = os.path.join(dataset, "sequences")
    poses:str = os.path.join(dataset, "poses")




class Device:
    """
    device to be used for training

    """

    def __init__(self) -> None:
        self.cpu = "cpu"
        self.gpu = "cuda"








if __name__ == "__main__":
    """
    docs
    """

    paths = Paths()
    paths_dict = paths.__dict__
    for path_name in paths_dict:
        paths.crtdir(paths_dict[path_name])