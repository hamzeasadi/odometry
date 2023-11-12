"""
a class to delete one or more fram from a series of frame

"""

import os
from typing import Dict
from typing import Optional

from PIL import Image

from Config.config import Paths
from Config.config import Device
from Utils.gutils import XRepresentation





class FrameDelet:
    """
    docs

    """
    def __init__(self, delete_config:str, seq_info:Dict, paths:Paths) -> None:
        self.seq_info = seq_info
        xpresentation = XRepresentation()
        self.delete_config = xpresentation.json2dict(os.path.join(paths.config, delete_config))

    def get_samples(self, num_samples:Optional[int]=None):
        pass

                



print(12345)


if __name__ == "__main__":
    print(__file__)
