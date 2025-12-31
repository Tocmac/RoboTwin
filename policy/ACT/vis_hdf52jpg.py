import h5py
import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm  # 如果没有安装，请运行 pip install tqdm

def extract_images_from_hdf5(hdf5_path, output_dir=None):
    # 如果没指定输出目录，默认在 hdf5 文件同级目录下创建一个同名文件夹
    if output_dir is None:
        file_name = os.path.splitext(os.path.basename(hdf5_path))[0]
        output_dir = os.path.join(os.path.dirname(hdf5_path), f"{file_name}_images")

    if not os.path.exists(hdf5_path):
        print(f"Error: File {hdf5_path} does not exist.")
        return

    print(f"Processing: {hdf5_path}")
    print(f"Output Directory: {output_dir}")

    try:
        with h5py.File(hdf5_path, 'r') as f:
            # 确认 HDF5 结构中包含图像组
            if 'observations/images' not in f:
                print("Error: Structure 'observations/images' not found in HDF5.")
                return

            # 定义你需要提取的相机名称
            camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']

            for cam_name in camera_names:
                dataset_path = f'observations/images/{cam_name}'
                
                if dataset_path not in f:
                    print(f"Warning: {cam_name} not found in file, skipping.")
                    continue

                # 读取图像数据到内存
                # 注意：如果显存/内存不足，可以不一次性读取，改为在循环中 f[path][i] 读取
                print(f"Loading {cam_name} data...")
                images_data = f[dataset_path][:] 
                total_frames = images_data.shape[0]

                # 创建对应的子文件夹
                cam_output_dir = os.path.join(output_dir, cam_name)
                os.makedirs(cam_output_dir, exist_ok=True)

                print(f"Saving {total_frames} images to {cam_output_dir} ...")

                # 使用 tqdm 显示进度条
                for i in tqdm(range(total_frames)):
                    img = images_data[i]

                    # 【重要】色彩空间转换
                    # 仿真器/HDF5 通常存储的是 RGB
                    # OpenCV imwrite 需要 BGR
                    # 如果发现导出的图片颜色不对（比如蓝色变红色），请注释或开启下面这行
                    if img.shape[-1] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    # 构造文件名：时间步.jpg
                    save_path = os.path.join(cam_output_dir, f"{i}.jpg")
                    
                    # 保存图片
                    cv2.imwrite(save_path, img)

        print(f"\nDone! All images extracted to {output_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images from RoboTwin/ACT HDF5 file.")
    parser.add_argument("--hdf5_file", type=str, default="/home/ps/wx_ws/code/2512_keyframes/RoboTwin/policy/ACT/processed_data/sim-blocks_ranking_rgb/aloha_clean_500-200/episode_0.hdf5", help="Path to the .hdf5 file")
    parser.add_argument("--output", type=str, default=None, help="Directory to save images (optional)")
    
    args = parser.parse_args()
    
    extract_images_from_hdf5(args.hdf5_file, args.output)