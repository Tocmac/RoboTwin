import os
import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import time
import cv2
from tqdm import tqdm

# ================= 1. 模型定义 (保持不变) =================

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
        pretrained=False, 
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

# ================= 2. 辅助功能 =================

def count_parameters(model):
    """输出模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model Info] Total Parameters: {total_params / 1e6:.2f}M")
    return total_params

def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
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

def evaluate_dataset_ori(model, dataloader, device):
    """评估整个数据集"""
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
            
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*30)
    print(f"Dataset Evaluation Results:")
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

def evaluate_dataset(model, dataloader, device):
    """评估整个数据集 (增加进度条和FPS)"""
    model.eval()
    y_true = []
    y_pred = []
    
    # [新增] 1. 初始化时间统计
    start_time = time.time()
    total_samples = 0
    
    print("Running evaluation on validation set...")
    
    with torch.no_grad():
        # [新增] 2. 使用 tqdm 包装 dataloader 实现进度条
        # desc: 进度条前缀, unit: 单位
        pbar = tqdm(dataloader, desc="Evaluating", unit="batch")
        
        for imgs, qpos, labels in pbar:
            imgs, qpos = imgs.to(device), qpos.to(device)
            
            # 记录样本数量
            batch_size = imgs.size(0)
            total_samples += batch_size
            
            logits = model(imgs, qpos)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).cpu().numpy().flatten()
            labels = labels.cpu().numpy().flatten()
            
            y_true.extend(labels)
            y_pred.extend(preds)
            
    # [新增] 3. 计算 FPS
    end_time = time.time()
    total_time = end_time - start_time
    fps = total_samples / total_time
            
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*30)
    print(f"Dataset Evaluation Results:")
    # [新增] 4. 打印 FPS 信息
    print(f"Inference Time: {total_time:.2f}s")
    print(f"Throughput    : {fps:.2f} FPS (Samples/sec)")
    print("-" * 30)
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

# ================= 3. 核心功能：单文件推理 + 可视化 + 定量分析 =================

def draw_confusion_matrix_on_canvas(canvas, x_start, y_start, width, height, counts):
    """
    辅助函数：在画布指定区域绘制漂亮的混淆矩阵
    counts: {'TP': int, 'TN': int, 'FP': int, 'FN': int}
    """
    tn, fp, fn, tp = counts['TN'], counts['FP'], counts['FN'], counts['TP']
    
    # 定义颜色 (BGR)
    color_bg = (50, 50, 50)       # 深灰背景
    color_correct = (0, 100, 0)   # 深绿 (正确区域背景)
    color_wrong = (0, 0, 100)     # 深红 (错误区域背景)
    color_text = (255, 255, 255)  # 白色文字
    color_border = (200, 200, 200) # 边框颜色

    # 绘制背景
    cv2.rectangle(canvas, (x_start, y_start), (x_start + width, y_start + height), color_bg, -1)
    
    # 矩阵是 2x2 的格子
    #   Pred 0 | Pred 1
    # GT 0 [ TN ] | [ FP ]
    # GT 1 [ FN ] | [ TP ]
    
    cell_w = width // 2
    cell_h = (height - 40) // 2 # 留出顶部 40px 写标题
    
    # 标题 "Confusion Matrix (Real-time)"
    cv2.putText(canvas, "Confusion Matrix", (x_start + 10, y_start + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_text, 2)

    # 定义四个格子的坐标
    # Grid Start (relative to x_start, y_start)
    grid_y = y_start + 40
    
    cells = [
        {'val': tn, 'label': 'TN', 'x': x_start,          'y': grid_y,          'color': color_correct},
        {'val': fp, 'label': 'FP', 'x': x_start + cell_w, 'y': grid_y,          'color': color_wrong},
        {'val': fn, 'label': 'FN', 'x': x_start,          'y': grid_y + cell_h, 'color': color_wrong},
        {'val': tp, 'label': 'TP', 'x': x_start + cell_w, 'y': grid_y + cell_h, 'color': color_correct}
    ]

    for cell in cells:
        # 1. 绘制色块
        cv2.rectangle(canvas, (cell['x'], cell['y']), (cell['x'] + cell_w, cell['y'] + cell_h), cell['color'], -1)
        # 2. 绘制边框
        cv2.rectangle(canvas, (cell['x'], cell['y']), (cell['x'] + cell_w, cell['y'] + cell_h), color_border, 1)
        # 3. 绘制文字 (居中)
        text = f"{cell['label']}: {cell['val']}"
        font_scale = 1.0
        thickness = 2
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = cell['x'] + (cell_w - text_size[0]) // 2
        text_y = cell['y'] + (cell_h + text_size[1]) // 2
        cv2.putText(canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_text, thickness)


def inference_single_file(model, file_path, output_dir, qpos_stats, device, save_video=True):
    model.eval()
    results = [] 
    
    # 实时统计计数器
    realtime_metrics = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    
    history_frames = [0, 4, 8]
    camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    q_mean = qpos_stats['mean']
    q_std = qpos_stats['std']

    print(f"\nInferencing single file: {file_path}")
    
    # 视频初始化
    video_writer = None
    if save_video:
        output_video_dir = os.path.join(output_dir, "output_video")
        os.makedirs(output_video_dir, exist_ok=True)
        video_name = os.path.join(output_video_dir, os.path.basename(file_path).replace('.hdf5', '_inference.mp4'))
        h_single, w_single = 480, 640 
        gap = 20 
        video_w = w_single * 2 + gap * 3
        video_h = h_single * 2 + gap * 3
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_name, fourcc, 30.0, (video_w, video_h))
        print(f"Video will be saved to: {video_name}")

    with h5py.File(file_path, 'r') as f:
        length = f['action'].shape[0]
        if 'keyframe' in f:
            gt_labels_raw = f['keyframe'][:]
            gt_labels = (gt_labels_raw > 0.5).astype(int)
        else:
            gt_labels = np.zeros(length, dtype=int)
            print("Warning: No 'keyframe' found, GT=0.")
        
        all_qpos = f['observations/qpos'][:]
        all_imgs = {}
        for cam in camera_names:
            all_imgs[cam] = f[f'observations/images/{cam}'][:]
            
        start_time = time.time()
        
        for t in range(length):
            frames_qpos = []
            frames_images = []
            
            # --- 1. 数据准备 ---
            for delta in history_frames:
                curr_t = max(0, t - delta)
                qpos = all_qpos[curr_t]
                qpos = (qpos - q_mean) / q_std
                frames_qpos.append(torch.from_numpy(qpos).float())
                
                all_cam_tensor_imgs = []
                for cam_name in camera_names:
                    raw_img = all_imgs[cam_name][curr_t]
                    img = torch.from_numpy(raw_img).permute(2, 0, 1).float() / 255.0
                    img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
                    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    img = normalize(img)
                    all_cam_tensor_imgs.append(img)
                
                timestep_imgs = torch.stack(all_cam_tensor_imgs, axis=0)
                frames_images.append(timestep_imgs)
            
            visual_input = torch.stack(frames_images, dim=0).unsqueeze(0).to(device)
            qpos_input = torch.stack(frames_qpos, dim=0).view(-1).unsqueeze(0).to(device)
            
            # --- 2. 推理 ---
            with torch.no_grad():
                logits = model(visual_input, qpos_input)
                prob = torch.sigmoid(logits).item()
                pred = 1 if prob > 0.5 else 0
                results.append(pred)
            
            # --- 3. 更新实时混淆矩阵 ---
            gt_val = int(gt_labels[t])
            if gt_val == 1 and pred == 1: realtime_metrics['TP'] += 1
            elif gt_val == 0 and pred == 0: realtime_metrics['TN'] += 1
            elif gt_val == 0 and pred == 1: realtime_metrics['FP'] += 1
            elif gt_val == 1 and pred == 0: realtime_metrics['FN'] += 1

            # --- 4. 绘制视频帧 ---
            if save_video and video_writer is not None:
                img_high = cv2.cvtColor(all_imgs['cam_high'][t], cv2.COLOR_RGB2BGR)
                img_left = cv2.cvtColor(all_imgs['cam_left_wrist'][t], cv2.COLOR_RGB2BGR)
                img_right = cv2.cvtColor(all_imgs['cam_right_wrist'][t], cv2.COLOR_RGB2BGR)
                
                h, w, _ = img_high.shape 
                canvas = np.zeros((h * 2 + gap * 3, w * 2 + gap * 3, 3), dtype=np.uint8)
                
                # 贴图
                canvas[gap:gap+h, gap:gap+w] = img_high              # Top-Left
                canvas[2*gap+h : 2*gap+2*h, gap : gap+w] = img_left  # Bottom-Left
                canvas[2*gap+h : 2*gap+2*h, 2*gap+w : 2*gap+2*w] = img_right # Bottom-Right (Right side)
                
                # === 右上角 (Top-Right) 分成两部分 ===
                # 左半部分：文字信息
                # 右半部分：混淆矩阵
                
                info_area_start_x = 2*gap + w
                info_area_start_y = gap
                info_area_width = w
                info_area_height = h
                
                # 1. 绘制文字信息 (占用左边 40% 宽度)
                text_start_x = info_area_start_x + 20
                text_start_y = info_area_start_y + 80
                line_height = 60
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                
                cv2.putText(canvas, f"Frame: {t}/{length}", (text_start_x, text_start_y), font, font_scale, (255, 255, 255), thickness)
                
                color_gt = (0, 255, 0)
                color_pred = (0, 165, 255)
                color_wrong = (0, 0, 255)
                
                cv2.putText(canvas, f"GT: {gt_val}", (text_start_x, text_start_y + line_height *2), font, font_scale, color_gt, thickness)
                cv2.putText(canvas, f"Pred: {pred}", (text_start_x, text_start_y + line_height * 3), font, font_scale, color_pred, thickness)
                cv2.putText(canvas, f"Prob: {prob:.2f}", (text_start_x, text_start_y + line_height * 4), font, font_scale, color_pred, thickness)
                
                status_text = "CORRECT" if gt_val == pred else "WRONG"
                status_color = color_gt if gt_val == pred else color_wrong
                cv2.putText(canvas, status_text, (text_start_x, text_start_y + line_height * 5), font, font_scale, status_color, thickness)
                
                # 2. 绘制混淆矩阵 (占用右边 60% 宽度)
                cm_width = int(info_area_width * 0.55)
                cm_height = int(info_area_height * 0.6) # 高度占60%
                cm_start_x = info_area_start_x + info_area_width - cm_width # 靠右对齐，留20px边距
                cm_start_y = info_area_start_y + 150 # 稍微往下一点
                
                draw_confusion_matrix_on_canvas(canvas, cm_start_x, cm_start_y, cm_width, cm_height, realtime_metrics)
                
                video_writer.write(canvas)

            if (t+1) % 50 == 0:
                print(f"Processed frame {t+1}/{length}...")

        fps = length / (time.time() - start_time)
        print(f"Inference finished. Average FPS: {fps:.2f}")
        
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to {video_name}")

    # 计算最终指标
    print("\n" + "="*30)
    print(f"Final Results for {os.path.basename(file_path)}:")
    
    acc = accuracy_score(gt_labels, results)
    precision, recall, f1, _ = precision_recall_fscore_support(gt_labels, results, average='binary', zero_division=0)
    
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("Confusion Matrix (Final):")
    print(f"TN: {realtime_metrics['TN']} | FP: {realtime_metrics['FP']}")
    print(f"FN: {realtime_metrics['FN']} | TP: {realtime_metrics['TP']}")
    print("="*30 + "\n")

    return results


def inference_single_file_ori(model, file_path, qpos_stats, device, save_video=True):
    model.eval()
    results = [] # 存储预测结果
    
    history_frames = [0, 4, 8]
    camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    q_mean = qpos_stats['mean']
    q_std = qpos_stats['std']

    print(f"\nInferencing single file: {file_path}")
    
    # 视频初始化
    video_writer = None
    if save_video:
        video_name = os.path.basename(file_path).replace('.hdf5', '_eval_grid.mp4')
        # 假设单图尺寸 480x640
        h_single, w_single = 480, 640 
        
        # [修改] 定义间隔 (Gap)
        gap = 20 
        
        # [修改] 画布大小：2图宽 + 3间隔
        video_w = w_single * 2 + gap * 3
        video_h = h_single * 2 + gap * 3
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_name, fourcc, 30.0, (video_w, video_h))
        print(f"Video will be saved to: {video_name} (Resolution: {video_w}x{video_h})")

    with h5py.File(file_path, 'r') as f:
        length = f['action'].shape[0]
        # 读取 GT Label (并二值化，防止它是多分类)
        if 'keyframe' in f:
            gt_labels_raw = f['keyframe'][:]
            gt_labels = (gt_labels_raw > 0.5).astype(int) # 0/1 int array
        else:
            gt_labels = np.zeros(length, dtype=int)
            print("Warning: No 'keyframe' found in HDF5, GT assumed to be 0.")
        
        all_qpos = f['observations/qpos'][:]
        all_imgs = {}
        for cam in camera_names:
            all_imgs[cam] = f[f'observations/images/{cam}'][:]
            
        start_time = time.time()
        
        for t in range(length):
            frames_qpos = []
            frames_images = []
            
            # --- 1. 数据准备 ---
            for delta in history_frames:
                curr_t = max(0, t - delta)
                qpos = all_qpos[curr_t]
                qpos = (qpos - q_mean) / q_std
                frames_qpos.append(torch.from_numpy(qpos).float())
                
                all_cam_tensor_imgs = []
                for cam_name in camera_names:
                    raw_img = all_imgs[cam_name][curr_t]
                    img = torch.from_numpy(raw_img).permute(2, 0, 1).float() / 255.0
                    img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
                    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    img = normalize(img)
                    all_cam_tensor_imgs.append(img)
                
                timestep_imgs = torch.stack(all_cam_tensor_imgs, axis=0)
                frames_images.append(timestep_imgs)
            
            visual_input = torch.stack(frames_images, dim=0).unsqueeze(0).to(device)
            qpos_input = torch.stack(frames_qpos, dim=0).view(-1).unsqueeze(0).to(device)
            
            # --- 2. 推理 ---
            with torch.no_grad():
                logits = model(visual_input, qpos_input)
                prob = torch.sigmoid(logits).item()
                pred = 1 if prob > 0.5 else 0
                results.append(pred)
            
            # --- 3. 绘制视频帧 (带间隔) ---
            if save_video and video_writer is not None:
                img_high = cv2.cvtColor(all_imgs['cam_high'][t], cv2.COLOR_RGB2BGR)
                img_left = cv2.cvtColor(all_imgs['cam_left_wrist'][t], cv2.COLOR_RGB2BGR)
                img_right = cv2.cvtColor(all_imgs['cam_right_wrist'][t], cv2.COLOR_RGB2BGR)
                
                h, w, _ = img_high.shape # 480, 640
                
                # 创建黑色画布
                canvas = np.zeros((h * 2 + gap * 3, w * 2 + gap * 3, 3), dtype=np.uint8)
                
                # 布局坐标计算
                # Row 1, Col 1: High (x: gap, y: gap)
                canvas[gap:gap+h, gap:gap+w] = img_high
                
                # Row 2, Col 1: Left (x: gap, y: 2*gap + h)
                canvas[2*gap+h : 2*gap+2*h, gap : gap+w] = img_left
                
                # Row 2, Col 2: Right (x: 2*gap + w, y: 2*gap + h)
                canvas[2*gap+h : 2*gap+2*h, 2*gap+w : 2*gap+2*w] = img_right
                
                # Row 1, Col 2: Info Panel (x: 2*gap + w, y: gap)
                # 信息面板区域实际上就是右上方剩下的黑块，我们直接在上面写字
                info_start_x = 2*gap + w + 50
                info_start_y = gap + 100
                line_height = 80
                
                gt_val = int(gt_labels[t])
                
                # 字体配置
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.3
                thickness = 3
                
                cv2.putText(canvas, f"Frame: {t}/{length}", (info_start_x, info_start_y), font, font_scale, (255, 255, 255), thickness)
                
                # GT & Pred Colors
                color_gt = (0, 255, 0)
                color_pred = (0, 165, 255)
                color_wrong = (0, 0, 255)
                
                cv2.putText(canvas, f"GT: {gt_val}", (info_start_x, info_start_y + line_height), font, font_scale, color_gt, thickness)
                cv2.putText(canvas, f"Pred: {pred}", (info_start_x, info_start_y + line_height * 2), font, font_scale, color_pred, thickness)
                cv2.putText(canvas, f"Prob: {prob:.4f}", (info_start_x, info_start_y + line_height * 3), font, font_scale, color_pred, thickness)
                
                # Status
                status_text = "CORRECT" if gt_val == pred else "WRONG"
                status_color = color_gt if gt_val == pred else color_wrong
                cv2.putText(canvas, status_text, (info_start_x, info_start_y + line_height * 4), font, font_scale, status_color, thickness)
                
                video_writer.write(canvas)

            if (t+1) % 50 == 0:
                print(f"Processed frame {t+1}/{length}...")

        fps = length / (time.time() - start_time)
        print(f"Inference finished. Average FPS: {fps:.2f}")
        
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to {video_name}")

    # --- 4. [修改] 计算并打印定量指标 ---
    print("\n" + "="*30)
    print(f"Single File Quantitative Results ({os.path.basename(file_path)}):")
    
    acc = accuracy_score(gt_labels, results)
    precision, recall, f1, _ = precision_recall_fscore_support(gt_labels, results, average='binary', zero_division=0)
    cm = confusion_matrix(gt_labels, results)
    
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("-" * 30)
    print("Confusion Matrix:")
    # 防止混淆矩阵形状不对（例如GT全是0的情况，cm可能是1x1）
    if cm.shape == (2, 2):
        print(f"TN: {cm[0,0]} | FP: {cm[0,1]}")
        print(f"FN: {cm[1,0]} | TP: {cm[1,1]}")
    else:
        print(cm)
    print("="*30 + "\n")

    return results


class RobotKeyframeDataset(Dataset):
    def __init__(self, data_dir, qpos_stats, history_frames=[0, 4, 8], is_train=False):
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, '*.hdf5')))
        if is_train:
            self.file_paths = self.file_paths[:int(len(self.file_paths)*0.8)]
        else:
            self.file_paths = self.file_paths[int(len(self.file_paths)*0.8):]
            
        self.history_frames = history_frames
        self.camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
        self.index_map = []
        self.qpos_mean, self.qpos_std = qpos_stats['mean'], qpos_stats['std']
        
        # 只扫描部分文件以加快 demo 速度，如果你想跑全量，去掉这里的切片
        # self.file_paths = self.file_paths[:10] 
        
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

# ================= 4. 主入口 =================

def main():
    # 配置路径
    data_dir = '/home/ps/wx_ws/code/2512_keyframes/RoboTwin/policy/ACT/processed_data/sim-blocks_ranking_rgb/aloha_clean_500-200'
    ckpt_path = '/home/ps/wx_ws/code/2512_keyframes/RoboTwin/policy/ACT/cls_ckpt/test_1342/act_style_model_ep0.pth'
    
    # 待测试的单个文件路径
    test_file_path = os.path.join(data_dir, 'episode_190.hdf5') 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 重新计算统计量
    all_files = sorted(glob.glob(os.path.join(data_dir, '*.hdf5')))
    train_files_len = int(len(all_files) * 0.8)
    print("Recalculating Norm Stats...")
    qpos_mean, qpos_std = get_norm_stats(data_dir, train_files_len)
    stats = {'mean': qpos_mean, 'std': qpos_std}
    
    # 2. 加载模型
    print(f"Loading model from {ckpt_path}...")
    model = ACTStyleClassifier(hidden_dim=256)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    count_parameters(model)

    # ----------------------------------------------------
    # 模式选择
    # ----------------------------------------------------
    
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
    # [Mode 2] 单文件推理 + 视频生成 + 定量计算


    print(f"\n[Mode 2] Single File Inference with Video & Metrics...")
    output_dir = os.path.dirname(ckpt_path)
    for i in tqdm(range(160, 200)):
        test_file_path = os.path.join(data_dir, f'episode_{i}.hdf5')
        pred_list = inference_single_file(model, test_file_path, output_dir, stats, device, save_video=True)


if __name__ == "__main__":
    main()