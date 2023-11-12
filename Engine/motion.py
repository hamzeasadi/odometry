"""

motion module for motion estimation and management
"""


import os
from typing import Any, Dict, Optional
# from collections import deque
import cv2


from Engine.motion_abstract import (
                                    MotionQueuAbstract,
                                    MotionEstimationAbstract
                                    )
from Config.config import Paths
from Utils.gutils import XRepresentation




class MotionQueue(MotionQueuAbstract):
    """
    docstring
    
    """
    def __init__(self, mq_size:int) -> None:
        super().__init__()
        self.mq_size = mq_size
        self.cqueue = ["" for _ in range(self.mq_size)]
        self.front = -1
        self.rear = -1


    def push(self, element:Any):
        if self.front == -1 and self.rear == -1:
            self.front = 0
            self.rear = 0
            self.cqueue[0] = element
        else:
            self.rear = (self.rear + 1)%self.mq_size
            self.cqueue[self.rear] = element
            if self.rear == self.front:
                self.front = (self.front + 1)%self.mq_size


    def pop(self):
        if self.front<0:
            raise IOError("Queue is empty!!!")

        output = self.cqueue[self.front]

        if self.front == self.rear:
            self.reset()
        else:
            self.front = (self.front + 1)%self.mq_size

        return output



    def peek(self):
        if self.front<0:
            raise IOError("Queue is empty!!!")

        output = self.cqueue[self.front]
        return output


    def reset(self):
        self.front = -1
        self.rear = -1


    def set_front(self, index):
        if index >= self.mq_size:
            return False

        total_front = index + self.front
        new_front =  total_front % self.mq_size

        if self.rear > self.front:

            if total_front<self.rear:
                self.front = new_front
            else:
                return False
        else:
            if total_front>self.mq_size:
                if new_front < self.rear:
                    self.front = new_front
                else:
                    return False
            else:
                self.front = new_front


    def get_value(self, index):
        if index >= self.mq_size:
            return False

        total_front = index + self.front
        new_front =  total_front % self.mq_size

        if self.rear > self.front:

            if total_front<self.rear:
                return self.cqueue[new_front]
            return False
        else:
            if total_front>self.mq_size:
                if new_front < self.rear:
                    return self.cqueue[new_front]
                return False
            else:
                return self.cqueue[new_front]

    def get_occupied_size(self):
        """
        calulate th occupide size
        """

        if self.rear>self.front:
            return self.rear - self.front
        else:
            return self.mq_size - self.front + self.rear




class MotionEstimation(MotionEstimationAbstract):

    def __init__(self, motion_estimate_config:str, paths:Paths):
        self.paths = paths
        self.detector = self.init_feature_extractor()


    def init_feature_extractor(self, config_name: Any | None = None):
        
        sift = cv2.SIFT_create()

        return sift


    









def main():
    """docs"""

    img_path = "/home/hasadi/project/Dataset/kitti/dataset/sequences/00/image_0/000000.png"
    
    # motion_estimate = MotionEstimation(motion_estimate_config="motion_estimation_config.json", paths=Paths())
    
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # print(img.shape)




 








if __name__ == "__main__":
    main()

