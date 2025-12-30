import os
import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from PIL import Image
import time
import datetime

# ================= 1. Backbone Utilities (参考 backbone.py) =================

class FrozenBatchNorm2d(torch.nn.Module):
    """
    参考 backbone.py: 冻结统计数据和参数的 BatchNorm。
    这对于迁移学习非常重要，有助于稳定特征提取。
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # 简单的推理模式 BN
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class BackboneBase(nn.Module):
    """参考 backbone.py: 封装 ResNet 并获取中间层"""
    def __init__(self, backbone: nn.Module, num_channels: int):
        super().__init__()
        # ACT 默认提取 layer4 的特征
        return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs

def build_backbone(name='resnet18', train_backbone=True):
    """构建类似 ACT 的 Backbone"""
    # 替换标准 BN 为 FrozenBN
    norm_layer = FrozenBatchNorm2d
    backbone = getattr(torchvision.models, name)(
        replace_stride_with_dilation=[False, False, False],
        pretrained=True, 
        norm_layer=norm_layer
    )
    
    # 冻结层参数 (参考 backbone.py 的逻辑)
    # 这里的逻辑稍微简化：如果 train_backbone=False，冻结所有；
    # 否则通常冻结 layer1 及之前的层。这里为了分类任务简单，如果不微调则全冻结。
    if not train_backbone:
        for name, parameter in backbone.named_parameters():
            parameter.requires_grad_(False)
            
    num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
    return BackboneBase(backbone, num_channels)

# ================= 2. 数据集 (参考 utils.py) =================

class RobotKeyframeDataset(Dataset):
    def __init__(self, data_dir, qpos_stats, history_frames=[0, 4, 8], is_train=True):
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, '*.hdf5')))
        # 简单的划分，实际请根据 utils.py logic 做 train/val split
        if is_train:
            self.file_paths = self.file_paths[:int(len(self.file_paths)*0.8)]
        else:
            self.file_paths = self.file_paths[int(len(self.file_paths)*0.8):]
            
        self.history_frames = history_frames
        self.camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
        self.index_map = []
        
        # 预扫描
        print(f"Scanning files for {'Train' if is_train else 'Val'}...")
        for f_idx, f_path in enumerate(self.file_paths):
            with h5py.File(f_path, 'r') as f:
                length = f['action'].shape[0]
                for i in range(length):
                    self.index_map.append((f_idx, i))
        print(f"Total samples: {len(self.index_map)}")
        
        # get norm state
        # self.qpos_mean, self.qpos_std = get_norm_stats(data_dir, int(len(self.file_paths)*0.8))
        self.qpos_mean, self.qpos_std = qpos_stats['mean'], qpos_stats['std']

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, t = self.index_map[idx]
        file_path = self.file_paths[file_idx]
        
        frames_qpos = []
        frames_images = [] # 将存储 (Time, Cam, C, H, W)
        
        with h5py.File(file_path, 'r') as f:
            label = float(f['keyframe'][t])
            label = 1.0 if label > 0.5 else 0.0
            
            # 处理时间步 T, T-4, T-8
            for delta in self.history_frames:
                curr_t = max(0, t - delta)
                
                # 1. Qpos
                qpos = f['observations/qpos'][curr_t] # (14,)
                qpos = (qpos - self.qpos_mean) / self.qpos_std # 归一化
                frames_qpos.append(torch.from_numpy(qpos).float())
                
                # 2. Images (参考 utils.py: 提取所有相机并 stack)
                image_dict = {}
                image_dict['cam_high'] = f['observations/images/cam_high'][curr_t]
                image_dict['cam_left_wrist'] = f['observations/images/cam_left_wrist'][curr_t]
                image_dict['cam_right_wrist'] = f['observations/images/cam_right_wrist'][curr_t]
                
                all_cam_images = []
                for cam_name in self.camera_names:
                    # Resize logic: 原始 ACT 往往在 load 时不做 resize，但为了 ResNet 输入通常需要
                    # 这里假设先转换 tensor 再处理，或者手动 resize
                    img = torch.from_numpy(image_dict[cam_name]) # (H, W, C)
                    # 变为 (C, H, W) 并归一化 (参考 utils.py: image_data / 255.0)
                    img = img.permute(2, 0, 1).float() / 255.0
                    
                    # 简单的 Resize (因为 480x640 太大了)
                    img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
                    
                    # 标准化 (ImageNet stats) - 虽然 ACT utils.py 没做这个，但用 ResNet 最好做
                    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    img = normalize(img)
                    
                    all_cam_images.append(img)
                
                # Stack cameras: (3_cam, C, H, W) -> 参考 utils.py line 58
                timestep_imgs = torch.stack(all_cam_images, axis=0)
                frames_images.append(timestep_imgs)

        # 最终堆叠时间步
        # visual_input: (3_time, 3_cam, C, H, W)
        visual_input = torch.stack(frames_images, dim=0)
        
        # qpos_input: (3_time, 14) -> flatten -> (42,)
        qpos_input = torch.stack(frames_qpos, dim=0).view(-1)
        
        return visual_input, qpos_input, torch.tensor([label], dtype=torch.float32)


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]  # Assuming this is a numpy array
            # action = root["/action"][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        # all_action_data.append(torch.from_numpy(action))

    # Pad all tensors to the maximum size
    max_qpos_len = max(q.size(0) for q in all_qpos_data)
    # max_action_len = max(a.size(0) for a in all_action_data)

    padded_qpos = []
    for qpos in all_qpos_data:
        current_len = qpos.size(0)
        if current_len < max_qpos_len:
            # Pad with the last element
            pad = qpos[-1:].repeat(max_qpos_len - current_len, 1)
            qpos = torch.cat([qpos, pad], dim=0)
        padded_qpos.append(qpos)

    # padded_action = []
    # for action in all_action_data:
    #     current_len = action.size(0)
    #     if current_len < max_action_len:
    #         pad = action[-1:].repeat(max_action_len - current_len, 1)
    #         action = torch.cat([action, pad], dim=0)
    #     padded_action.append(action)

    all_qpos_data = torch.stack(padded_qpos)
    # all_action_data = torch.stack(padded_action)
    # all_action_data = all_action_data

    # # normalize action data
    # action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    # action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    # action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    # stats = {
    #     "action_mean": action_mean.numpy().squeeze(),
    #     "action_std": action_std.numpy().squeeze(),
    #     "qpos_mean": qpos_mean.numpy().squeeze(),
    #     "qpos_std": qpos_std.numpy().squeeze(),
    #     "example_qpos": qpos,
    # }

    return qpos_mean.numpy().squeeze(), qpos_std.numpy().squeeze()

# ================= 3. 模型 (参考 detr_vae.py) =================

class ACTStyleClassifier(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        # 1. 构建 Backbone (参考 detr_vae.py: build_backbone)
        # 我们这里只初始化一个 backbone 实例，用于所有相机 (Siamese Network)
        self.backbone = build_backbone('resnet18', train_backbone=True)
        
        # 2. Input Projection (参考 detr_vae.py: self.input_proj)
        # ResNet18 layer4 输出 channel 为 512，ACT 使用 1x1 卷积投影到 hidden_dim
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)
        
        # 3. 聚合层 (分类头)
        # 我们不使用 Transformer，而是将特征展平
        # 每个时间步有 3 个相机。每个相机经过 input_proj 后是 (hidden_dim, H', W')
        # 我们进行 Global Average Pooling 变成 (hidden_dim)
        # 输入向量总长 = (3_time * 3_cam * hidden_dim) + (3_time * 14_qpos)
        
        # [新增] 空间映射层：替代 GAP
        # 输入: hidden_dim * 7 * 7
        # 输出: hidden_dim (或者你可以保持更大，看显存情况)
        self.spatial_proj = nn.Sequential(
            nn.Linear(hidden_dim * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1) # 加上 Dropout 防止过拟合，因为参数变多了
        )
        
        self.num_cameras = 3
        self.num_frames = 3
        self.feature_dim_per_cam = hidden_dim # GAP 之后的维度
        
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
        # images: (Batch, Time, Cam, C, H, W)
        # qpos: (Batch, 42)
        
        b, t, n_cam, c, h, w = images.shape
        
        # 参考 detr_vae.py: 遍历相机 (虽然为了并行我们通常 reshape，但为了逻辑清晰模仿 ACT 写法)
        # 为了高效，我们把 Batch 和 Time 合并，但把 Cam 维度保持像 DETR 那样处理
        
        # Reshape: (B*T, Cam, C, H, W)
        images_flat_time = images.view(b * t, n_cam, c, h, w)
        
        all_cam_features = []
        
        # --- 核心 ACT 逻辑 ---
        # 参考 detr_vae.py line 86-90 loop
        for cam_id in range(n_cam):
            # 1. 提取特征: self.backbones[0](image[:, cam_id])
            # 注意：backbone 返回的是 dict (IntermediateLayerGetter)
            features_dict = self.backbone(images_flat_time[:, cam_id])
            features = features_dict['0'] # 获取 layer4 输出
            
            # 2. 投影: self.input_proj(features)
            # shape: (B*T, hidden_dim, H', W') torch.Size([96, 256, 7, 7])
            projected = self.input_proj(features)
            
            # --- [修改开始] ---
            # 1. 展平: (B*T, 256 * 49)
            flat_features = projected.flatten(1) 
            
            # 2. 线性降维 (替代 GAP): (B*T, 256)
            # 这里保留了空间信息并压缩了维度
            cam_feature = self.spatial_proj(flat_features) 
            # --- [修改结束] ---
            
            # # 3. 这里的差异：ACT 之后保留空间维度进 Transformer
            # # 我们这里做 Global Average Pooling 变成分类特征
            # # shape: (B*T, hidden_dim) torch.Size([96, 256])
            # gap_features = torch.mean(projected, dim=[2, 3])
            
            # all_cam_features.append(gap_features)
            
            all_cam_features.append(cam_feature)
        # --------------------
        
        # 拼接所有相机特征: (B*T, 3_cam * hidden_dim)
        visual_features = torch.cat(all_cam_features, dim=1)
        
        # 恢复 Batch 和 Time 维度
        # (B, T, 3_cam * hidden_dim)
        visual_features = visual_features.view(b, t, -1)
        
        # 展平时间维度
        # (B, T * 3_cam * hidden_dim)
        visual_flat = visual_features.view(b, -1)
        
        # 拼接 Qpos 并分类
        # torch.Size([32, 2346])
        combined = torch.cat([visual_flat, qpos], dim=1)
        # torch.Size([32, 1])
        logits = self.classifier(combined)
        
        return logits

# ================= 4. 训练主循环 =================
def format_time(elapsed):
    """格式化时间为 hh:mm:ss"""
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def main():
    # 配置
    data_dir = '/home/ps/wx_ws/code/2512_keyframes/RoboTwin/policy/ACT/processed_data/sim-blocks_ranking_rgb/aloha_clean_500-200' # 请修改为你的路径
    batch_size = 32
    lr = 1e-4
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ---------------------------------------------------------
    # [新增] 1. 先计算训练集的统计量 (或者你可以手动算一次然后写死在这里)
    # ---------------------------------------------------------
    print("Calculating dataset stats (this might take a while)...")
    all_files = sorted(glob.glob(os.path.join(data_dir, '*.hdf5')))
    train_files_len = int(len(all_files) * 0.8)
    
    # 注意：这里我们只用训练集数量的文件来计算
    qpos_mean, qpos_std = get_norm_stats(data_dir, train_files_len)
    
    stats = {'mean': qpos_mean, 'std': qpos_std}
    print(f"Stats calculated. Mean: {stats['mean'][:4]}..., Std: {stats['std'][:4]}...") 
    # ---------------------------------------------------------
    
    # 初始化
    train_dataset = RobotKeyframeDataset(data_dir, stats, is_train=True)
    val_dataset = RobotKeyframeDataset(data_dir, stats, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 实例化模型 (Hidden dim 设为 256 或 512 均可)
    model = ACTStyleClassifier(hidden_dim=256).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) # AdamW 通常比 Adam 更好
    
    print("Start Training...")
    
    # === ETA 计时器初始化 ===
    total_steps = len(train_loader) * epochs
    start_time = time.time()
    global_step = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for i, (imgs, qpos, labels) in enumerate(train_loader):
            global_step += 1
            imgs, qpos, labels = imgs.to(device), qpos.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs, qpos)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # [建议] 加回梯度裁剪，防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # if i % 10 == 0:
            #     print(f"Epoch {epoch}, Iter {i}, Loss: {loss.item():.4f}")
            
            # --- ETA 计算与打印 ---
            if i % 10 == 0:
                elapsed = time.time() - start_time
                avg_time_per_step = elapsed / global_step
                remaining_steps = total_steps - global_step
                remaining_time = remaining_steps * avg_time_per_step
                
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f} | "
                      f"Elapsed: {format_time(elapsed)}, ETA: {format_time(remaining_time)}")
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, qpos, labels in val_loader:
                imgs, qpos, labels = imgs.to(device), qpos.to(device), labels.to(device)
                outputs = model(imgs, qpos)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        acc = 100 * correct / total
        print(f"Epoch {epoch} Finished. Avg Loss: {total_loss/len(train_loader):.4f}, Val Acc: {acc:.2f}%")
        
        # 保存模型
        save_dir = "/home/ps/wx_ws/code/2512_keyframes/RoboTwin/policy/ACT/cls_ckpt/test_1342"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{save_dir}/act_style_model_ep{epoch}.pth")

if __name__ == "__main__":
    main()