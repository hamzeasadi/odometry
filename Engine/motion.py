"""

motion module for motion estimation and management
"""


import os
from typing import Any
# from collections import deque

from Engine.motion_abstract import MotionQueuAbstract
# from Engine.motion_abstract import MotionData
# from Engine.motion_abstract import MotionQueuAbstract





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













def main():
    """docs"""
    print(__file__)

    mq = MotionQueue(mq_size=10)

    for i in range(11):
        mq.push((i+1)*2)
        print(f"front={mq.front} rear={mq.rear} size={mq.get_occupied_size()} value = {mq.get_value(i-1)}")


 








if __name__ == "__main__":
    main()
