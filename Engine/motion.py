"""

motion module for motion estimation and management
"""


import os
from typing import Any, Dict
# from collections import deque
import cv2
import numpy as np


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
        self.matcher = self.init_matcher()



    def init_feature_extractor(self, config_name: Dict = None):
        
        detector = cv2.SIFT_create()
        # detector = cv2.ORB_create(3000)
        return detector
    

    def init_matcher(self, config_name: str | Dict = None):
        FLANN_INDEX_LSH = 6
        FLANN_INDEX_KDTREE = 1
        # index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        return flann
    

    def get_featuers(self, img: np.ndarray):
        key, des = self.detector.detectAndCompute(img, None)
        return dict(key=key, des=des)
    

    def get_match(self, feat0, feat1):

        if len(feat0['key']) > 6 and len(feat1['key']) > 6:
            matches = self.matcher.knnMatch(feat0['des'], feat1['des'], k=2)
            if(len(matches)<10):
                return None, None

            good_matches = []
            try:
                for m, n in matches:
                    if m.distance < 0.9 * n.distance:
                        good_matches.append(m)
            except ValueError:
                return None, None
            
            if  len(good_matches)<8:
                return None, None

            q1 = np.float32([feat0['key'][m.queryIdx].pt for m in good_matches])
            q2 = np.float32([feat1['key'][m.trainIdx].pt for m in good_matches])
        
            return q1, q2
        else:
            return None, None


    def get_motion(self, img0: np.ndarray, img1: np.ndarray, K:np.ndarray=np.eye(3)):
        feat0 = self.get_featuers(img=img0)
        feat1 = self.get_featuers(img=img1)
        q1, q2 = self.get_match(feat0=feat0, feat1=feat1)
        
        if q1 is None:
            return None
        
        E, mask = cv2.findEssentialMat(q1, q2, K)
        points, R, t, mask = cv2.recoverPose(E, q1, q2)
        # print(E)
        # if E is not None:
        #     points, R, t, mask = cv2.recoverPose(E, q1, q2)
        # else:
        #     return None

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3:4] = t

        return T











def main():
    """docs"""

    img0_path = "/home/hasadi/project/Datasets/kitti/dataset/sequences/00/image_0/000000.png"
    img1_path = "/home/hasadi/project/Datasets/kitti/dataset/sequences/00/image_0/000001.png"
    
    motion_estimate = MotionEstimation(motion_estimate_config="motion_estimation_config.json", paths=Paths())
    
    img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

    motion_estimate.get_motion(img0=img0, img1=img1)



    # print(ky.__dict__)




 








if __name__ == "__main__":
    main()

