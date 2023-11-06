"""
docs
"""

import os

from Config.config import Paths
from DataPrepration.frame_deleting import FrameDelet







if __name__ == "__main__":
    paths = Paths()
    Paths.crtdir(paths.model)
    Paths.crtdir(paths.result)