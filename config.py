"""
general paths
"""

import os
from typing import NamedTuple


class Paths(NamedTuple):
    """
    paths
    """
    project_root:str = os.path.dirname(__file__)
    data_path:str = os.path.join(project_root, "data")
    dataset_path:str = os.path.join(data_path, "dataset")
    model_path:str = os.path.join(data_path, "model")
    result_path:str = os.path.join(data_path, "result")
    other_path:str = os.path.join(data_path, "other")

    @staticmethod
    def crtdir(path:str):
        if(not os.path.exists(path)):
            os.makedirs(path)

    def initPath(self):
        for path in Paths():
            self.crtdir(path)





if __name__ == '__main__':
    print(__file__)
    paths = Paths()
    paths.initPath()