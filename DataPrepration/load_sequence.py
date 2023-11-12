"""
read the sequece of data

"""
import os
from glob import glob
from typing import Dict
from abc import ABCMeta
from abc import abstractmethod
import numpy as np

from Config.config import KitiPaths
from Utils.gutils import Dict2class









class LoadData(metaclass=ABCMeta):
    """
    load a sequence of a dataset with all information

    """

    @abstractmethod
    def load_seq(self, seq_num:int):
        """
        load the full information of the sequence e.g(img 
        sequence, time sequence, pose sequence, etc)

        """

    @abstractmethod
    def load_img_seq(self, seq_num:int):
        """
        load imgs of the sequence

        """

    @abstractmethod
    def load_pose_seq(self, seq_num:int):
        """
        load poses of the sequence 

        """

    @abstractmethod
    def load_time_seq(self, seq_num:int):
        """
        load timestamp of the sequence

        """

    @abstractmethod
    def load_calib_seq_cam(self, seq_num:int):
        """
        load the calibration information of the camera recorded the sequence
        """



class KittiLoad(LoadData):
    """
    load the sequences of the dataset (e.g kitti dataset)
    the structure of paths and folder structure should be similar for any other dataset
    """
    def __init__(self, dataset_path:KitiPaths, camera:str, imgext:str="png") -> None:
        super().__init__()
        self.imgext = imgext
        self.dataset_path = dataset_path
        if camera=="left":
            self.img_seq = "image_0"
        elif camera=="right":
            self.img_seq = "image_1"
        else:
            self.img_seq = camera


    def get_sequence_info(self, seq_num:int)->Dict:
        """
        general meta data about the sequence
        """
        seq_name = f"{seq_num:0>2}"
        sequenc_path = os.path.join(self.dataset_path.sequences, seq_name)
        seq_img_path = os.path.join(sequenc_path, self.img_seq)
        seq_calib_path = os.path.join(sequenc_path, "calib.txt")
        seq_timestamp_path = os.path.join(sequenc_path, "times.txt")
        seq_gt_poses = os.path.join(self.dataset_path.dataset, "poses", f"{seq_name}.txt")

        return dict(seq_img_path=seq_img_path, seq_calib_path=seq_calib_path, 
                    seq_timestamp_path=seq_timestamp_path, seq_gt_poses=seq_gt_poses)




    def load_img_seq(self, seq_num: int):
        seq_info = Dict2class(self.get_sequence_info(seq_num=seq_num))
        # pylint: disable=no-member
        img_paths = glob(pathname=os.path.join(seq_info.seq_img_path, f"*.{self.imgext}"))
        img_paths = sorted(list(img_paths))
        return img_paths

    def load_pose_seq(self, seq_num: int):
        seq_info = Dict2class(self.get_sequence_info(seq_num=seq_num))
        poses = []
        # pylint: disable=no-member
        with open(seq_info.seq_gt_poses, encoding="utf-8") as poses_file:
            poses_lines = poses_file.readlines()
            for line in poses_lines:
                matrix_elements = line.strip().split(" ")
                if len(matrix_elements)!=12:
                    raise Exception(f"the current pose doesnt match with template: {matrix_elements}")

                mtx = [float(element) for element in matrix_elements]
                mtx = np.reshape(np.array(mtx), newshape=(3, 4))
                poses.append(mtx)

        return poses


    def load_time_seq(self, seq_num: int):
        seq_info = Dict2class(self.get_sequence_info(seq_num=seq_num))
        time_stamps = []
        # pylint: disable=no-member
        with open(seq_info.seq_timestamp_path, encoding="utf-8") as timestamp_file:
            timestamp_lines = timestamp_file.readlines()
            for line in timestamp_lines:
                time_stamp = float(line.strip())
                time_stamps.append(time_stamp)

        return time_stamps


    def load_calib_seq_cam(self, seq_num: int):
        seq_info = Dict2class(self.get_sequence_info(seq_num=seq_num))
        # pylint: disable=no-member
        with open(seq_info.seq_calib_path, encoding="utf-8") as calib_file:
            calib_lines = calib_file.readlines()
            calib_line = calib_lines[0]
            calib_elements = calib_line.strip().split(" ")[1:]
            calib_ =  [float(element) for element in calib_elements]
            calib_mtx = np.reshape(np.array(calib_), newshape=(3, 4))

        return calib_mtx


    def load_seq(self, seq_num: int):
        img_seq = self.load_img_seq(seq_num=seq_num)
        pose_seq = self.load_pose_seq(seq_num=seq_num)
        time_seq = self.load_time_seq(seq_num=seq_num)
        calib = self.load_calib_seq_cam(seq_num=seq_num)

        return Dict2class(dict(img_seq=img_seq, pose_seq=pose_seq, time_seq=time_seq, calib=calib))





def main():
    """
    Entry point
    """
    print(__file__)
    kitti_loader = KittiLoad(dataset_path=KitiPaths(), camera="left")

    tss = kitti_loader.load_calib_seq_cam(seq_num=0)
    print(tss)

if __name__ == "__main__":
    main()



