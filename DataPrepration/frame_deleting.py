"""
a class to delete one or more fram from a series of frame

"""

import os
from typing import Dict
from typing import Optional
import pickle

import cv2
import numpy as np
from matplotlib import pyplot as plt

from Config.config import (
                            Paths,
                            KitiPaths
                        )
from Utils.gutils import (
                            XRepresentation,
                            rotation_euler,
                            form_transform
                        )
from Engine.motion import MotionEstimation
from DataPrepration.load_sequence import KittiLoad





class FrameDelet:
    """
    docs

    """
    def __init__(self, delete_config:str, seq_info:Dict, paths:Paths, motion_estimate:MotionEstimation) -> None:
        self.seq_info = seq_info
        # xpresentation = XRepresentation()
        self.paths = paths
        self.motion_estimate = motion_estimate
        # self.delete_config = xpresentation.json2cls(os.path.join(paths.config, delete_config))


    def get_sample(self, first_frame:int, second_frame:int, num_drops:int, last_drop:bool):
        """
        return one sample from a sequence
        args:
            first_frame: intial frame to calculate othe frames motion based on
            second_frame: first frame of sample
            num_drops: number of frames to add to the second frame for estimate motion
            last_drop: if true drop the last frame as manipulation
        returns:
            a sequence of motions [(i, j), (i, j+1), (i, j+2), ...]
        """

        if not last_drop:
            return None

        calib = self.seq_info.calib

        first_img_path = self.seq_info.img_seq[first_frame]
        second_img_path = self.seq_info.img_seq[second_frame + num_drops]

        img0 = cv2.imread(first_img_path, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(second_img_path, cv2.IMREAD_GRAYSCALE)

        transf = self.motion_estimate.get_motion(img0=img0, img1=img1, K=calib[:3, :3])
        if transf is None:
            return None
        
        delta_t = self.seq_info.time_seq[second_frame] - self.seq_info.time_seq[first_frame] 

        intial_pose = form_transform(self.seq_info.pose_seq[first_frame])
        T = np.matmul(intial_pose, np.linalg.inv(transf))
        # T = transf

        current_pose = form_transform(self.seq_info.pose_seq[second_frame-1])

        # T_gt = np.matmul(current_pose, np.linalg.inv(intial_pose))   

        # current_pose = T_gt

        R0 = current_pose[:3, :3]
        R0_euler = rotation_euler(rotation=R0)
        t0 = current_pose[:3, 3]

        R_estimate = T[:3, :3]
        R_estimate_euler = rotation_euler(rotation=R_estimate)
        t_estimate = T[:3, 3]

        gt_motion = np.hstack((R0.flatten(), R0_euler[0], R0_euler[1], t0, np.array([delta_t])))
        prediction_motion = np.hstack((R_estimate.flatten(), R_estimate_euler[0], R_estimate_euler[1], t_estimate, np.array([delta_t])))

        return dict(gt=gt_motion, prd=prediction_motion)


    def gen_samples(self, num_samples:int, datatype:str):
        """
        generate num of samples

        args:
            num_samples: specify number of samples that need to be generated for delete attack
              and same for normal condition
            sample_size: sample size
            datatype: train or validation
        """
        
        num_available_frames = len(self.seq_info.time_seq)

        first_frames = np.random.choice(np.arange(10, num_available_frames-130), size=num_samples, replace=True)
        second_frames = np.random.choice(np.arange(1, 5), size=num_samples, replace=True)
        num_drops = np.random.choice(np.arange(10, 20), size=num_samples, replace=True)


        # generate attacked samples
  
        for i, first_idx in enumerate(first_frames):
            sample = []
            save_path = os.path.join(self.paths.dataset, "kitti", "delete", datatype, "attack")
            self.paths.crtdir(save_path)

            second_idx = first_idx + second_frames[i]
            num_drop = num_drops[i]
            result = self.get_sample(first_frame=first_idx, second_frame=second_idx, num_drops=num_drop, last_drop=True)
 
            with open(os.path.join(save_path, f"sample_{i}.pkl"), "wb") as sample_file:
                pickle.dump(result, sample_file)

        # normal
        for i, first_idx in enumerate(first_frames):
            sample = []
            save_path = os.path.join(self.paths.dataset, "kitti", "delete", datatype, "normal")
            self.paths.crtdir(save_path)

            second_idx = first_idx + second_frames[i]
            num_drop = num_drops[i]
            result = self.get_sample(first_frame=first_idx, second_frame=second_idx, num_drops=0, last_drop=True)
 
            with open(os.path.join(save_path, f"sample_{i}.pkl"), "wb") as sample_file:
                pickle.dump(result, sample_file)






                



def main():
    """
    entry point
    """

    paths = Paths()

    kiti_load = KittiLoad(dataset_path=KitiPaths(), camera='left', imgext="png")

    motion_estimate = MotionEstimation(motion_estimate_config=None, paths=Paths())
        
    frame_delete = FrameDelet(delete_config="delete_config.json",
                                seq_info=kiti_load.load_seq(seq_num=9), 
                                paths=Paths(), 
                                motion_estimate=motion_estimate
                                )
    

    frame_delete.gen_samples(num_samples=1000, datatype="validation")


    # firsts = np.random.choice(np.arange(2000), size=20, replace=False)
    # seconds = np.random.choice(np.arange(1, 4), size=20, replace=True)
    # drops = np.random.choice(np.arange(10, 20), size=20, replace=True)

    # for i, idx in enumerate(firsts):
    #     sec_idx = idx + seconds[i]
    #     num_drop = drops[i]
        
    #     output_ok = frame_delete.get_sample(first_frame=idx, second_frame=sec_idx, num_drops=0, last_drop=True)
    #     output_nook = frame_delete.get_sample(first_frame=idx, second_frame=sec_idx, num_drops=num_drop, last_drop=True)
        
    #     fig, axs = plt.subplots(nrows=2, ncols=2, figsize=[24, 12])

    #     axs[0, 0].plot(output_ok['gt'][:15], label=f"gt-{idx}-{sec_idx}-{sec_idx-idx}-{0}-ok")
    #     axs[0, 0].plot(output_ok['prd'][:15], label=f"prd-{idx}-{sec_idx}-{sec_idx-idx}-{0}-ok")
    #     axs[0,0].legend()

    #     axs[0, 1].plot(output_nook['gt'][:15], label=f"gt-{idx}-{sec_idx}-{sec_idx-idx}-{num_drop}-nook")
    #     axs[0, 1].plot(output_nook['prd'][:15], label=f"prd-{idx}-{sec_idx}-{sec_idx-idx}-{num_drop}-nook")
    #     axs[0,1].legend()

    #     axs[1, 0].plot(output_ok['gt'][:15], label=f"gt-ok")
    #     axs[1, 0].plot(output_nook['gt'][:15], label=f"gt-nook")
    #     axs[1,0].legend()

    #     axs[1, 1].plot(output_ok['prd'][:15], label=f"prd-ok")
    #     axs[1, 1].plot(output_nook['prd'][:15], label=f"prd-nook")
    #     axs[1,1].legend()
 
        
    #     # plt.legend()
    #     plt.savefig(os.path.join(paths.result, f"cmp-{i}.png"))
    #     plt.close()









if __name__ == "__main__":
    
    main()
