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
        self.delete_config = xpresentation.json2cls(os.path.join(paths.config, delete_config))


    def get_sample(self, intial_point:int, start_point:int, sample_size:int, num_drops:int, last_drop:bool):
        """
        return one sample from a sequence
        args:
            intial_point: intial frame to calculate othe frames motion based on
            start_point: first frame of sample
            sample_size: the size of sample (#frames for evaluation)
            last_drop: if true drop the last frame as manipulation
        returns:
            a sequence of motions [(i, j), (i, j+1), (i, j+2), ...]
        """

        

                



print(12345)


if __name__ == "__main__":
    print(__file__)
