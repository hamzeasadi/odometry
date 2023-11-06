"""
general utility functions for the project

"""

import os
import csv
import json
from typing import Dict
from typing import Any
import pandas as pd




class Dict2class:
    """
    convert dictionary to python class attributes

    """
    def __init__(self, data:Dict) -> None:
        """
        docs

        """

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
        with open(json_path, "r") as json_file:
            dict_file = json.load(json_file)
        return dict_file
    
    def json2cls(self, json_path:str):
        dict_file = self.json2dict(json_path=json_path)
        return Dict2class(data=dict_file)

    def dict2cls(self, data:Dict):
        return Dict2class(data=data)





if __name__ == "__main__":
    print(__file__)

    x = 5
    # y =  "{:0>{}}".format(x, 2)
    y = f"{x:0>2}"
    print(y)

