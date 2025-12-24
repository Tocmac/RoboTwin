import sys

sys.path.append("./policy/ACT/")

import os
import h5py
import numpy as np
import pickle
import cv2
import argparse
import pdb
import json
from tqdm import tqdm


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper = root["/joint_action/left_gripper"][()]
        right_gripper = root["/joint_action/right_gripper"][()]
        print(left_gripper.shape)
        print(right_gripper.shape)

    return left_gripper, right_gripper

if __name__ == "__main__":
    
    for i in tqdm(range(200)):
        source_dir = "/home/ps/wx_ws/code/2512_keyframes/RoboTwin/data/blocks_ranking_rgb/aloha_clean_500/data/"
        target_dir = "/home/ps/wx_ws/code/2512_keyframes/RoboTwin/policy/ACT/processed_data/sim-blocks_ranking_rgb/aloha_clean_500-200/"
        dataset_path = os.path.join(source_dir, f"episode{i}.hdf5")
        target_path = os.path.join(target_dir, f"episode_{i}.hdf5")
    
        left_gripper, right_gripper = load_hdf5(dataset_path)
        
        step = 0
        keyframe = 0
        keyframe_list = []
        start_frame = 0
        
        for i in range(left_gripper.shape[0] - 1):
            # 抓红
            if (step == 0) and (left_gripper[i] == 0.0 or right_gripper[i] == 0.0):
                step = 1
                keyframe = 1
                start_frame = i
            elif (step == 1) and (i - start_frame > 20):
                step = 2
                keyframe = 0
            # 放红
            elif (step == 2) and (left_gripper[i] >= 0.5 and right_gripper[i] >= 0.5):
                step = 3
                keyframe = 2
                start_frame = i
            elif (step == 3) and (i - start_frame > 20):
                step = 4
                keyframe = 0
            # 抓绿
            elif (step == 4) and (left_gripper[i] == 0.0 or right_gripper[i] == 0.0):
                step = 5
                keyframe = 3
                start_frame = i
            elif (step == 5) and (i - start_frame > 20):
                step = 6
                keyframe = 0
            # 放绿
            elif (step == 6) and (left_gripper[i] >= 0.5 and right_gripper[i] >= 0.5):
                step = 7
                keyframe = 4
                start_frame = i
            elif (step == 7) and (i - start_frame > 20):
                step = 8
                keyframe = 0
            # 抓蓝
            elif (step == 8) and (left_gripper[i] == 0.0 or right_gripper[i] == 0.0):
                step = 9
                keyframe = 5
                start_frame = i
            elif (step == 9) and (i - start_frame > 20):
                step = 10
                keyframe = 0
            # 放蓝
            elif (step == 10) and (left_gripper[i] >= 0.5 and right_gripper[i] >= 0.5):
                step = 11
                keyframe = 6
                start_frame = i
            elif (step == 11) and (i - start_frame > 20):
                step = 12
                keyframe = 0

            keyframe_list.append(keyframe)
            
        # keyframe_array = np.array(keyframe_list)
        # print(keyframe_array.shape)
        # print(keyframe_array)
        

        with h5py.File(target_path, "a") as f:
            f.create_dataset("keyframe", data=np.array(keyframe_list))
    