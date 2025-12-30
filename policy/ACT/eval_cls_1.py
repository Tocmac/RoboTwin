import os
import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import time

# ================= 1. 模型定义 (需与 train_cls.py 保持完全一致) =================

class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int):
        super().__init__()
        return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        return self.body(tensor)

def build_backbone(name='resnet18', train_backbone=True):
    norm_layer = FrozenBatchNorm2d
    backbone = getattr(torchvision.models, name)(
        replace_stride_with_dilation=[False, False, False],
        pretrained=False, # Eval模式不需要下载预训练权重，因为会加载checkpoint
        norm_layer=norm_layer
    )
    num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
    return BackboneBase(backbone, num_channels)

class ACTStyleClassifier(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.backbone = build_backbone('resnet18', train_backbone=False)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)
        
        self.spatial_proj = nn.Sequential(
            nn.Linear(hidden_dim * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.num_cameras = 3
        self.num_frames = 3
        self.feature_dim_per_cam = hidden_dim
        
        total_visual_dim = self.num_frames * self.num_cameras * self.feature_dim_per_cam
        total_qpos_dim = self.num_frames * 14
        
        self.classifier = nn.Sequential(
            nn.Linear(total_visual_dim + total_qpos_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, images, qpos):
        b, t, n_cam, c, h, w = images.shape
        images_flat_time = images.view(b * t, n_cam, c, h, w)
        all_cam_features = []
        
        for cam_id in range(n_cam):
            features_dict = self.backbone(images_flat_time[:, cam_id])
            features = features_dict['0']
            projected = self.input_proj(features)
            flat_features = projected.flatten(1) 
            cam_feature = self.spatial_proj(flat_features) 
            all_cam_features.append(cam_feature)
        
        visual_features = torch.cat(all_cam_features, dim=1)
        visual_features = visual_features.view(b, t, -1)
        visual_flat = visual_features.view(b, -1)
        combined = torch.cat([visual_flat, qpos], dim=1)
        logits = self.classifier(combined)
        return logits

# ================= 2. 数据处理与统计量计算 =================

def get_norm_stats(dataset_dir, num_episodes):
    """重新计算统计量，必须与 Training 阶段完全一致"""
    all_qpos_data = []
    # 只为了获取 stats，不需要遍历整个 dataset，快速扫描即可
    print(f"Recalculating stats from top {num_episodes} files...")
    
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        if not os.path.exists(dataset_path):
            continue 
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]
        all_qpos_data.append(torch.from_numpy(qpos))

    max_qpos_len = max(q.size(0) for q in all_qpos_data)
    padded_qpos = []
    for qpos in all_qpos_data:
        current_len = qpos.size(0)
        if current_len < max_qpos_len:
            pad = qpos[-1:].repeat(max_qpos_len - current_len, 1)
            qpos = torch.cat([qpos, pad], dim=0)
        padded_qpos.append(qpos)

    all_qpos_data = torch.stack(padded_qpos)
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    return qpos_mean.numpy().squeeze(), qpos_std.numpy().squeeze()

class RobotKeyframeDataset(Dataset):
    def __init__(self, data_dir, qpos_stats, history_frames=[0, 4, 8], is_train=False):
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, '*.hdf5')))
        # Eval 模式下，通常我们看验证集 (后20%)
        # 如果你想 Eval 整个文件夹，可以修改这里
        if is_train:
            self.file_paths = self.file_paths[:int(len(self.file_paths)*0.8)]
        else:
            self.file_paths = self.file_paths[int(len(self.file_paths)*0.8):]
            
        self.history_frames = history_frames
        self.camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
        self.index_map = []
        self.qpos_mean, self.qpos_std = qpos_stats['mean'], qpos_stats['std']
        
        print(f"Scanning {len(self.file_paths)} files for Evaluation...")
        for f_idx, f_path in enumerate(self.file_paths):
            with h5py.File(f_path, 'r') as f:
                length = f['action'].shape[0]
                for i in range(length):
                    self.index_map.append((f_idx, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, t = self.index_map[idx]
        file_path = self.file_paths[file_idx]
        
        frames_qpos = []
        frames_images = []
        
        with h5py.File(file_path, 'r') as f:
            label = float(f['keyframe'][t])
            label = 1.0 if label > 0.5 else 0.0
            
            for delta in self.history_frames:
                curr_t = max(0, t - delta)
                qpos = f['observations/qpos'][curr_t]
                qpos = (qpos - self.qpos_mean) / self.qpos_std
                frames_qpos.append(torch.from_numpy(qpos).float())
                
                image_dict = {}
                image_dict['cam_high'] = f['observations/images/cam_high'][curr_t]
                image_dict['cam_left_wrist'] = f['observations/images/cam_left_wrist'][curr_t]
                image_dict['cam_right_wrist'] = f['observations/images/cam_right_wrist'][curr_t]
                
                all_cam_images = []
                for cam_name in self.camera_names:
                    img = torch.from_numpy(image_dict[cam_name])
                    img = img.permute(2, 0, 1).float() / 255.0
                    img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
                    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    img = normalize(img)
                    all_cam_images.append(img)
                
                timestep_imgs = torch.stack(all_cam_images, axis=0)
                frames_images.append(timestep_imgs)

        visual_input = torch.stack(frames_images, dim=0)
        qpos_input = torch.stack(frames_qpos, dim=0).view(-1)
        
        return visual_input, qpos_input, torch.tensor([label], dtype=torch.float32)

# ================= 3. 核心功能实现 =================

def count_parameters(model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model Info] Total Parameters: {total_params / 1e6:.2f}M")
    print(f"[Model Info] Trainable Parameters: {trainable_params / 1e6:.2f}M\n")
    return total_params

def evaluate_dataset(model, dataloader, device):
    """
    功能 1: 评估整个数据集，输出 Acc 和 混淆矩阵
    """
    model.eval()
    y_true = []
    y_pred = []
    
    print("Running evaluation on validation set...")
    with torch.no_grad():
        for imgs, qpos, labels in dataloader:
            imgs, qpos = imgs.to(device), qpos.to(device)
            logits = model(imgs, qpos)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).cpu().numpy().flatten()
            labels = labels.cpu().numpy().flatten()
            
            y_true.extend(labels)
            y_pred.extend(preds)
            
    # 计算指标
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*30)
    print(f"Evaluation Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("-" * 30)
    print("Confusion Matrix:")
    print(f"TN: {cm[0,0]} | FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]} | TP: {cm[1,1]}")
    print("="*30 + "\n")
    
    return cm

def inference_single_file_ori(model, file_path, qpos_stats, device):
    """
    功能 2: 单个文件推理，模拟 sliding window
    """
    model.eval()
    results = []
    history_frames = [0, 4, 8]
    camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    
    # 提取统计量
    q_mean = qpos_stats['mean']
    q_std = qpos_stats['std']

    print(f"Inferencing single file: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        length = f['action'].shape[0]
        # 预读取所有数据以加快速度 (假设显存/内存够用)
        # 如果内存不够，请移回循环内读取
        all_qpos = f['observations/qpos'][:]
        all_imgs = {}
        for cam in camera_names:
            all_imgs[cam] = f[f'observations/images/{cam}'][:]
            
        start_time = time.time()
        
        # 逐帧推理
        for t in range(length):
            frames_qpos = []
            frames_images = []
            
            for delta in history_frames:
                curr_t = max(0, t - delta)
                
                # 1. 处理 QPos
                qpos = all_qpos[curr_t]
                qpos = (qpos - q_mean) / q_std
                frames_qpos.append(torch.from_numpy(qpos).float())
                
                # 2. 处理图像
                all_cam_images = []
                for cam_name in camera_names:
                    raw_img = all_imgs[cam_name][curr_t]
                    
                    # 预处理 pipeline (To Tensor -> /255 -> Resize -> Normalize)
                    img = torch.from_numpy(raw_img).permute(2, 0, 1).float() / 255.0
                    img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
                    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    img = normalize(img)
                    all_cam_images.append(img)
                
                timestep_imgs = torch.stack(all_cam_images, axis=0)
                frames_images.append(timestep_imgs)
            
            # 堆叠并增加 Batch 维度 (Batch=1)
            visual_input = torch.stack(frames_images, dim=0).unsqueeze(0).to(device) # (1, 3, 3, C, H, W)
            qpos_input = torch.stack(frames_qpos, dim=0).view(-1).unsqueeze(0).to(device) # (1, 42)
            
            # 推理
            with torch.no_grad():
                logits = model(visual_input, qpos_input)
                prob = torch.sigmoid(logits).item()
                pred = 1 if prob > 0.5 else 0
                results.append(pred)
            
            if (t+1) % 50 == 0:
                print(f"Processed frame {t+1}/{length}...")

        fps = length / (time.time() - start_time)
        print(f"Inference finished. Average FPS: {fps:.2f}")

    return results

import cv2  # 记得导入 cv2

def inference_single_file(model, file_path, qpos_stats, device, save_video=True):
    """
    单文件推理 + 视频可视化生成
    """
    model.eval()
    results = []
    
    # 配置参数
    history_frames = [0, 4, 8]
    camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    
    # 统计量
    q_mean = qpos_stats['mean']
    q_std = qpos_stats['std']

    print(f"Inferencing single file: {file_path}")
    
    # 视频保存初始化
    video_writer = None
    if save_video:
        video_name = os.path.basename(file_path).replace('.hdf5', '_eval.mp4')
        # 假设原图尺寸为 480x640，2x2 布局后总尺寸为 960x1280
        # 如果你的图片尺寸不同，请相应调整
        video_h, video_w = 480 * 2, 640 * 2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 或者 'XVID'
        video_writer = cv2.VideoWriter(video_name, fourcc, 30.0, (video_w, video_h))
        print(f"Video will be saved to: {video_name}")

    with h5py.File(file_path, 'r') as f:
        length = f['action'].shape[0]
        
        # 读取 GT Label 序列 (如果有的话，用于对比)
        # 如果 hdf5 里有 keyframe 真值，我们可以读出来显示
        gt_labels = f['keyframe'][:] if 'keyframe' in f else np.zeros(length)
        
        # 预读取所有数据
        all_qpos = f['observations/qpos'][:]
        all_imgs = {}
        for cam in camera_names:
            all_imgs[cam] = f[f'observations/images/{cam}'][:]
            
        start_time = time.time()
        
        # 逐帧推理
        for t in range(length):
            frames_qpos = []
            frames_images = []
            
            # --- 1. 数据准备 (给模型用) ---
            for delta in history_frames:
                curr_t = max(0, t - delta)
                
                # QPos
                qpos = all_qpos[curr_t]
                qpos = (qpos - q_mean) / q_std
                frames_qpos.append(torch.from_numpy(qpos).float())
                
                # Images
                all_cam_tensor_imgs = []
                for cam_name in camera_names:
                    raw_img = all_imgs[cam_name][curr_t]
                    # 模型输入预处理 pipeline
                    img = torch.from_numpy(raw_img).permute(2, 0, 1).float() / 255.0
                    img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
                    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    img = normalize(img)
                    all_cam_tensor_imgs.append(img)
                
                timestep_imgs = torch.stack(all_cam_tensor_imgs, axis=0)
                frames_images.append(timestep_imgs)
            
            # Stack & Batch
            visual_input = torch.stack(frames_images, dim=0).unsqueeze(0).to(device)
            qpos_input = torch.stack(frames_qpos, dim=0).view(-1).unsqueeze(0).to(device)
            
            # --- 2. 模型推理 ---
            with torch.no_grad():
                logits = model(visual_input, qpos_input)
                prob = torch.sigmoid(logits).item()
                pred = 1 if prob > 0.5 else 0
                results.append(pred)
            
            # --- 3. 视频生成逻辑 ---
            if save_video:
                # 获取当前帧的原始图像 (OpenCV 需要 BGR 格式，hdf5 通常是 RGB)
                # 假设 all_imgs 是 RGB，cv2 需要转 BGR
                img_high = cv2.cvtColor(all_imgs['cam_high'][t], cv2.COLOR_RGB2BGR)
                img_left = cv2.cvtColor(all_imgs['cam_left_wrist'][t], cv2.COLOR_RGB2BGR)
                img_right = cv2.cvtColor(all_imgs['cam_right_wrist'][t], cv2.COLOR_RGB2BGR)
                
                h, w, _ = img_high.shape # 假设 480, 640
                
                # 创建大画布 (2*h, 2*w)
                canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
                
                # 布局：
                # (0,0) High   | (0,1) Info
                # (1,0) Left   | (1,1) Right
                
                # 贴图
                canvas[0:h, 0:w] = img_high              # Top-Left
                canvas[h:2*h, 0:w] = img_left            # Bottom-Left
                canvas[h:2*h, w:2*w] = img_right         # Bottom-Right
                
                # 绘制信息面板 (Top-Right)
                # 背景已经是全黑了，直接写字
                info_start_x = w + 50
                info_start_y = 100
                line_height = 80
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3
                
                # 1. Frame Count
                cv2.putText(canvas, f"Frame: {t}/{length}", (info_start_x, info_start_y), 
                            font, font_scale, (255, 255, 255), thickness)
                
                # 2. GT (Green)
                gt_val = int(gt_labels[t])
                gt_color = (0, 255, 0) # Green in BGR
                cv2.putText(canvas, f"GT: {gt_val}", (info_start_x, info_start_y + line_height), 
                            font, font_scale, gt_color, thickness)
                
                # 3. Pred (Orange)
                pred_color = (0, 165, 255) # Orange in BGR (BBGGRR -> 0, 165, 255)
                cv2.putText(canvas, f"Pred: {pred} ({prob:.2f})", (info_start_x, info_start_y + line_height * 2), 
                            font, font_scale, pred_color, thickness)

                # 4. Result Status
                status_text = "CORRECT" if gt_val == pred else "WRONG"
                status_color = (0, 255, 0) if gt_val == pred else (0, 0, 255)
                cv2.putText(canvas, status_text, (info_start_x, info_start_y + line_height * 3), 
                            font, font_scale, status_color, thickness)

                video_writer.write(canvas)
            
            if (t+1) % 50 == 0:
                print(f"Processed frame {t+1}/{length}...")

        fps = length / (time.time() - start_time)
        print(f"Inference finished. Average FPS: {fps:.2f}")
        
        if video_writer is not None:
            video_writer.release()
            print("Video saved successfully.")

    return results

# ================= 4. 主入口 =================

def main():
    # 配置路径
    data_dir = '/home/ps/wx_ws/code/2512_keyframes/RoboTwin/policy/ACT/processed_data/sim-blocks_ranking_rgb/aloha_clean_500-200'
    ckpt_path = '/home/ps/wx_ws/code/2512_keyframes/RoboTwin/policy/ACT/cls_ckpt/test_1342/act_style_model_ep0.pth' # 修改为你最好的 checkpoint
    
    # 待测试的单个文件路径 (用于功能2)
    # 随便找一个验证集里的文件
    test_file_path = os.path.join(data_dir, 'episode_190.hdf5') 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ----------------------------------------------------
    # 1. 必须重新计算统计量 (因为训练时没保存)
    # ----------------------------------------------------
    all_files = sorted(glob.glob(os.path.join(data_dir, '*.hdf5')))
    train_files_len = int(len(all_files) * 0.8)
    print("Recalculating Norm Stats (Crucial)...")
    qpos_mean, qpos_std = get_norm_stats(data_dir, train_files_len)
    stats = {'mean': qpos_mean, 'std': qpos_std}
    
    # ----------------------------------------------------
    # 2. 加载模型
    # ----------------------------------------------------
    print(f"Loading model from {ckpt_path}...")
    model = ACTStyleClassifier(hidden_dim=256)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 功能 1: 输出参数量
    count_parameters(model)

    # ----------------------------------------------------
    # 功能 1: 评估整个验证集 (带混淆矩阵)
    # ----------------------------------------------------
    # print("\n[Mode 1] Evaluating Validation Dataset...")
    # val_dataset = RobotKeyframeDataset(data_dir, stats, is_train=False)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # evaluate_dataset(model, val_loader, device)
    
    # ----------------------------------------------------
    # 功能 2: 单文件推理
    # ----------------------------------------------------
    if os.path.exists(test_file_path):
        print("\n[Mode 2] Single File Inference...")
        # pred_list = inference_single_file(model, test_file_path, stats, device)
        pred_list = inference_single_file(model, test_file_path, stats, device, save_video=True)
        
        print(f"\nResult Length: {len(pred_list)}")
        print(f"Preds: {pred_list}")
        # 你可以在这里加入保存 list 的代码，例如保存为 json 或 npy
        # np.save('preds.npy', pred_list)
    else:
        print(f"Test file not found: {test_file_path}")

if __name__ == "__main__":
    main()