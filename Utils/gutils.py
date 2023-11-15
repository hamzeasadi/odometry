"""
general utility functions for the project

"""

import os
import csv
import json
from typing import Dict
from typing import Any
import math

import numpy as np
import pandas as pd










class Dict2class:
    """
    convert dictionary to python class attributes

    """
    def __init__(self, data:Dict) -> None:
        """
        docs

        """
        self.data = data
        for ky, val in data.items():
            setattr(self, ky, self._xch(val))

    def _xch(self, data_item:Any):
        """
        managing nested dictionary: if the data_item is a dictionary return its class 
        otherwise return it without a change

        args:
            data_item: an input value that could be an dictionary or value

        return:
            value or a class data
        """
        if isinstance(data_item, dict):
            return Dict2class(data_item)
        return data_item






class XRepresentation:
    """
    exchange the data representation:
    json ---> dict
    json <---> cls
    dict <---> cls

    """

    def json2dict(self, json_path:str):
        """docs"""
        with open(json_path, "r", encoding="utf-8") as json_file:
            dict_file = json.load(json_file)
        return dict_file

    def json2cls(self, json_path:str):
        """docs"""
        dict_file = self.json2dict(json_path=json_path)
        return Dict2class(data=dict_file)

    def dict2cls(self, data:Dict):
        """docs"""
        return Dict2class(data=data)





def rotation_euler(rotation:np.ndarray):
    """
    calculate euler angles based on rotaion matrix
    args:
        rotation: a 3x3 rotational matrix
    
    returns:
        angles: list of angles e.g [theta, phi, psi]

    """
    r31 = rotation[2, 0]
    r32 = rotation[2, 1]
    r33 = rotation[2, 2]
    r21 = rotation[1, 0]
    r11 = rotation[0, 0]
    r12 = rotation[0, 1]
    r13 = rotation[0, 2]

    if r31!=-1 and r31!=1:
        theta1 = math.asin(r31)
        theta2 = np.pi - theta1

        psi1 = math.atan2(r32/math.cos(theta1), r33/math.cos(theta1))
        psi2 = math.atan2(r32/math.cos(theta2), r33/math.cos(theta2))

        phi1 = math.atan2(r21/math.cos(theta1), r11/math.cos(theta1))
        phi2 = math.atan2(r21/math.cos(theta2), r11/math.cos(theta2))

    else:
        phi1 = phi2 = 0
        if(r31==-1):
            theta1 = np.pi/2
            psi1 = phi1 + math.atan2(r12, r13)
        else:
            theta2 = -np.pi/2
            psi2 = -phi2 + math.atan2(-r12, -r13)

    angles = np.array([[theta1, phi1, psi1], [theta2, phi2, psi2]])
    return angles




def form_transform(motion_matrix:np.ndarray):
    """
    takes a 3x4 transformation matrix and turn it to 4x4 matrix
    for ease of further calculation
    args:
        motion_matrix: a 3x4 transformation matrix
    returns
        a 4x4 motion matrix
    """
    temp_mtx = np.eye(4)
    temp_mtx[:3, :] = motion_matrix

    return temp_mtx


if __name__ == "__main__":
    print(__file__)

    x = dict(a=123, d="hamzeh", c=dict(z=345, f=12))
    
    xcls = Dict2class(x)

    for a in xcls:
        print(a)

  

    


