"""
create a motion estimation abstract class
"""

import os
from typing import Any
from abc import ABCMeta
from abc import abstractmethod
from typing import Optional
from typing import Dict

import numpy as np





class MotionQueuAbstract(metaclass=ABCMeta):
    """
    a custome data structure to keep track of data
    it is more like a cirlcular queue, but we return the index at each time as well
    """

    @abstractmethod
    def pop(self):
        """
        takes the first element from the front of circular queue
        and increase the pointer by one
        """


    @abstractmethod
    def push(self, element:Any):
        """
        take the new element and add it to the end of the
        circular q
        args:
           element: an element to be pushed to the circular queue 
        """

    @abstractmethod
    def peek(self):
        """
        same as push without changing the pointer value
        """

    @abstractmethod
    def reset(self):
        """
        reset the cqueue
        """
    

    @abstractmethod
    def set_front(self, index):
        """
        set the front pointer in a specific position
        of the queue buffer
        """

    @abstractmethod
    def get_value(self, index):
        """
        get value in a specific position
        of the queue buffer
        """




class MotionData:
    """
    motion data with its fields
    """




class MotionAbstract(metaclass=ABCMeta):
    """
    a class that estimate the motion between two frame
    """

    @abstractmethod
    def creat_detector(self):
        """
        based on specified configuration it will
        create an detector to calculate featuers from the image
        """

    @abstractmethod
    def create_matcher(self):
        """
        based on specified configuration it will
        create a feature matcher to calculate matching pairs of two feature set
        """

    @abstractmethod
    def get_euler_angles(self):
        """
        calculate euler angles from rotation matrix
        """

    def calc_motion(self):
        """
        calculate the motion between two images
        """




class MotionEstimationAbstract(metaclass=ABCMeta):
    """
    calculate the motion between two frame
    
    """

    @abstractmethod
    def init_feature_extractor(self, config_name:Optional[str:Dict]=None):
        """
        intialize an featuer extractor for motion estimation pipeline
        args:
            config_name: the configuration for the feature extractor
        """


    # @abstractmethod
    # def init_matcher(self, config_name:Optional[str:Dict]=None):
    #     """
    #     intialize an featuer matcher for motion estimation pipeline
    #     args:
    #         config_name: the configuration for the feature matcher
    #     """

    # @abstractmethod
    # def get_featuers(self, img:np.ndarray):
    #     """
    #     gt the featuers from input image
    #     args:
    #         img: an image in numpy format 
    #     """
    
    # @abstractmethod
    # def get_match(self, feat0, feat1):
    #     """
    #     calculate the feature matches between two set of featuers
    #     args:
    #         feat0: featuers descriptors of the first image
    #         feat1: featuers descriptors of the second image
        
    #     return:
    #         matching points
    #     """


    # def get_motion(self, img0:np.ndarray, img1:np.ndarray):
    #     """
    #     calculate the motion between two frames
    #     args:
    #         img0: first frame
    #         img1: second frame
        
    #     return:
    #         motion matrix
    #     """







if __name__ == "__main__":
    print(__file__)

