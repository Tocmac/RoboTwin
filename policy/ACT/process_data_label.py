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


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        left_gripper_endpose, left_endpose = (
            root["/endpose/left_gripper"][()],
            root["/endpose/left_endpose"][()],
        )
        right_gripper_endpose, right_endpose = (
            root["/endpose/right_gripper"][()],
            root["/endpose/right_endpose"][()],
        )
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict, left_gripper_endpose, left_endpose, right_gripper_endpose, right_endpose


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def data_transform(path, episode_num, save_path):
    begin = 0
    floders = os.listdir(path)
    assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(episode_num):
        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict, left_gripper_endpose_all, left_endpose_all, right_gripper_endpose_all, right_endpose_all = (load_hdf5(
            os.path.join(path, f"episode{i}.hdf5")))
        qpos = []
        qpos_endpose = []
        
        qpos_cal = []

        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []

        last_state = None
        
        # === 1. 创建该 episode 的目录结构 ===
        episode_dir = os.path.join(save_path, f"{i:03d}")
        os.makedirs(name=episode_dir, exist_ok=True)
        # 创建图片子文件夹
        cam_names = ["cam_high", "cam_right_wrist", "cam_left_wrist"]
        # 对应原始数据中的 key
        raw_cam_keys = ["head_camera", "right_camera", "left_camera"]
        for cam_name in cam_names:
            os.makedirs(os.path.join(episode_dir, cam_name), exist_ok=True)
        
        for j in range(0, left_gripper_all.shape[0]):

            left_gripper, left_arm, right_gripper, right_arm, left_gripper_endpose, left_endpose, right_gripper_endpose, right_endpose = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
                left_gripper_endpose_all[j],
                left_endpose_all[j],
                right_gripper_endpose_all[j],
                right_endpose_all[j],
            )

            if j != left_gripper_all.shape[0] - 1:   
                state = np.concatenate(([j], [left_gripper], [right_gripper], left_arm,  right_arm), axis=0)  # joint
                endpose_state = np.concatenate(([j], [left_gripper_endpose], [right_gripper_endpose], left_endpose, right_endpose), axis=0)

                # state = state.astype(np.float32)
                qpos.append(np.round(state, 4))
                # endpose_state = endpose_state.astype(np.float32)
                qpos_endpose.append(np.round(endpose_state, 4))

                camera_high_bits = image_dict["head_camera"][j]
                camera_high = cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_high_bgr = cv2.cvtColor(camera_high, cv2.COLOR_RGB2BGR)
                camera_high_resized = cv2.resize(camera_high_bgr, (640, 480))
                # cam_high.append(camera_high_resized)
                camera_high_path = os.path.join(episode_dir, "cam_high", f"{j:06d}.jpg")
                cv2.imwrite(camera_high_path, camera_high_resized)

                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_right_wrist_bgr = cv2.cvtColor(camera_right_wrist, cv2.COLOR_RGB2BGR)
                camera_right_wrist_resized = cv2.resize(camera_right_wrist_bgr, (640, 480))
                # cam_right_wrist.append(camera_right_wrist_resized)
                camera_right_wrist_path = os.path.join(episode_dir, "cam_right_wrist", f"{j:06d}.jpg")
                cv2.imwrite(camera_right_wrist_path, camera_right_wrist_resized)

                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_left_wrist_bgr = cv2.cvtColor(camera_left_wrist, cv2.COLOR_RGB2BGR)
                camera_left_wrist_resized = cv2.resize(camera_left_wrist_bgr, (640, 480))
                # cam_left_wrist.append(camera_left_wrist_resized)
                camera_left_wrist_path = os.path.join(episode_dir, "cam_left_wrist", f"{j:06d}.jpg")
                cv2.imwrite(camera_left_wrist_path, camera_left_wrist_resized)

            if j != 0:
                action = state
                actions.append(action)
                left_arm_dim.append(left_arm.shape[0])
                right_arm_dim.append(right_arm.shape[0])
                
                left_endpose_xyz = left_endpose[0:3]
                right_endpose_xyz = right_endpose[0:3]
                
                left_before_endpose_xyz = left_endpose_all[j - 1][0:3]
                right_before_endpose_xyz = right_endpose_all[j - 1][0:3]

                left_diff = np.linalg.norm(left_before_endpose_xyz - left_endpose_xyz)
                right_diff = np.linalg.norm(right_before_endpose_xyz - right_endpose_xyz)
                
                endpose_cal_state = np.concatenate(([j], [left_gripper_endpose], [right_gripper_endpose], [np.round(left_diff,6)], [np.round(right_diff, 6)]), axis=0)
                qpos_cal.append(endpose_cal_state)

        #
        np.save(os.path.join(episode_dir, "qpos.npy"), np.array(qpos))
        np.save(os.path.join(episode_dir, "action.npy"), np.array(actions))
        np.save(os.path.join(episode_dir, "qpos_endpose.npy"), np.array(qpos_endpose))
        np.save(os.path.join(episode_dir, "qpos_endpose_cal.npy"), np.array(qpos_cal))

        # hdf5path = os.path.join(save_path, f"episode_{i}.hdf5")

        # with h5py.File(hdf5path, "w") as f:
        #     f.create_dataset("action", data=np.array(actions))
        #     obs = f.create_group("observations")
        #     obs.create_dataset("qpos", data=np.array(qpos))
        #     obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
        #     obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
        #     image = obs.create_group("images")
        #     # cam_high_enc, len_high = images_encoding(cam_high)
        #     # cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
        #     # cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)
        #     image.create_dataset("cam_high", data=np.stack(cam_high), dtype=np.uint8)
        #     image.create_dataset("cam_right_wrist", data=np.stack(cam_right_wrist), dtype=np.uint8)
        #     image.create_dataset("cam_left_wrist", data=np.stack(cam_left_wrist), dtype=np.uint8)

        begin += 1
        print(f"proccess {i} success!")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "--task_name",
        default= "blocks_ranking_rgb",
        type=str,
        help="The name of the task (e.g., adjust_bottle)",
    )
    parser.add_argument("--task_config", default="aloha_clean_500", type=str)
    parser.add_argument("--expert_data_num", default=10, type=int)

    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num

    begin = 0
    begin = data_transform(
        path=os.path.join("../../data", task_name, task_config, 'data'),
        # os.path.join("../../data/dataset_ori", task_name, task_config, 'data'),
        episode_num=expert_data_num,
        save_path=f"processed_data/sim-{task_name}/{task_config}-vis-img-endpose",
        # save_path=f"processed_data/sim-{task_name}/{task_config}-{expert_data_num}",
    )

    # SIM_TASK_CONFIGS_PATH = "./SIM_TASK_CONFIGS.json"

    # try:
    #     with open(SIM_TASK_CONFIGS_PATH, "r") as f:
    #         SIM_TASK_CONFIGS = json.load(f)
    # except Exception:
    #     SIM_TASK_CONFIGS = {}

    # SIM_TASK_CONFIGS[f"sim-{task_name}-{task_config}-{expert_data_num}"] = {
    #     "dataset_dir": f"./processed_data/sim-{task_name}/{task_config}-{expert_data_num}",
    #     "num_episodes": expert_data_num,
    #     "episode_len": 1000,
    #     "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"],
    # }

    # with open(SIM_TASK_CONFIGS_PATH, "w") as f:
    #     json.dump(SIM_TASK_CONFIGS, f, indent=4)
