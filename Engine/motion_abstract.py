"""
create a motion estimation abstract class
"""

import os
from typing import Any
from abc import ABCMeta
from abc import abstractmethod





# class MotionData:
#     points:np.






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







if __name__ == "__main__":
    print(__file__)
