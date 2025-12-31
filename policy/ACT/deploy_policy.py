import sys
import numpy as np
import torch
import os
import pickle
import cv2
import time  # Add import for timestamp
import h5py  # Add import for HDF5
from datetime import datetime  # Add import for datetime formatting
from .act_policy import ACT
import copy
from argparse import Namespace

def encode_obs(observation):
    head_cam = cv2.resize(observation["observation"]["head_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
    left_cam = cv2.resize(observation["observation"]["left_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
    right_cam = cv2.resize(observation["observation"]["right_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
    head_cam = np.moveaxis(head_cam, -1, 0) / 255.0
    left_cam = np.moveaxis(left_cam, -1, 0) / 255.0
    right_cam = np.moveaxis(right_cam, -1, 0) / 255.0
    qpos = (observation["joint_action"]["left_arm"] + [observation["joint_action"]["left_gripper"]] +
            observation["joint_action"]["right_arm"] + [observation["joint_action"]["right_gripper"]])
    return {
        "head_cam": head_cam,
        "left_cam": left_cam,
        "right_cam": right_cam,
        "qpos": qpos,
    }


# 新增辅助函数：处理单帧数据，使其符合数据集格式 (HWC, uint8)
def process_frame_for_saving(observation):
    # 提取 qpos (14,)
    qpos = (observation["joint_action"]["left_arm"] + [observation["joint_action"]["left_gripper"]] +
            observation["joint_action"]["right_arm"] + [observation["joint_action"]["right_gripper"]])
    
    # 提取图像，Resize 到 640x480，保持 uint8 (0-255)
    # 注意：这里直接操作 Raw observation，不需要归一化
    def process_img(img):
        img_resized = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
        return img_resized.astype(np.uint8) # 确保是 uint8

    return {
        "qpos": np.array(qpos, dtype=np.float32),
        "cam_high": process_img(observation["observation"]["head_camera"]["rgb"]),
        "cam_left_wrist": process_img(observation["observation"]["left_camera"]["rgb"]),
        "cam_right_wrist": process_img(observation["observation"]["right_camera"]["rgb"])
    }


def get_model(usr_args):
    return ACT(usr_args, Namespace(**usr_args))


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    # instruction = TASK_ENV.get_instruction()  

    # Get action from model
    actions = model.get_action(obs)
    
    # save 
    collected_steps = []
    step_data = process_frame_for_saving(observation)
    step_data['action'] = actions[0] # action 已经是 (14,) 的 numpy array 或 tensor
    collected_steps.append(step_data)
    
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
    return observation, collected_steps


def reset_model(model):
    # Reset temporal aggregation state if enabled
    if model.temporal_agg:
        model.all_time_actions = torch.zeros([
            model.max_timesteps,
            model.max_timesteps + model.num_queries,
            model.state_dim,
        ]).to(model.device)
        model.t = 0
        print("Reset temporal aggregation state")
    else:
        model.t = 0
