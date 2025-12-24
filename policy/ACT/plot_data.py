import numpy as np
import os
import matplotlib.pyplot as plt

def plot_joint_angles_2x7(npy_file_path):
    # 1. 加载数据
    try:
        data = np.load(npy_file_path)
        print(f"成功加载文件: {npy_file_path}, 数据形状: {data.shape}")
    except FileNotFoundError:
        print("错误: 找不到指定的文件，请检查路径。")
        return

    # 2. 提取数据
    time_steps = data[:, 0]
    joint_angles = data[:, 1:17]

    num_joints = joint_angles.shape[1]
    if num_joints != 16:
        print(f"警告: 预期提取14列数据，但实际提取了 {num_joints} 列。")
        # 为了保证后续绘图不报错，截断或填充数据
        if num_joints > 16:
             joint_angles = joint_angles[:, :16]
        elif num_joints < 16:
             # 如果少于14个，就只画这几个，后面代码会自动处理
             pass

    # ================== 修改重点 ==================
    # 3. 绘制图像
    # 修改为 2行7列。
    # figsize 需要调整为宽长高短的比例，例如 (24, 8)
    fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(24, 8), sharex=True, sharey=False)
    
    # 将 2x7 的数组展平为 1维数组 (长度14)
    # 顺序是：先第一行7个，再第二行7个，符合你的需求
    axes_flat = axes.flatten()

    for i in range(16):
        # 如果实际关节数据少于14个，提前退出循环
        if i >= num_joints:
             # 隐藏多余的空子图
             axes_flat[i].axis('off')
             continue
            
        ax = axes_flat[i]
        # 绘制曲线
        ax.plot(time_steps, joint_angles[:, i], linewidth=1.5)
        
        ax.set_title(f'Joint {i+1}', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # --- 标签逻辑修改 ---
        # Y轴标签：只在每一行的最左侧显示 (索引 0 和 7)
        if i % 8 == 0:
            ax.set_ylabel('Angle (rad)')
            
        # X轴标签：只在第二行显示 (索引 7 到 13)
        if i >= 8:
             ax.set_xlabel('Timestep (s)')
    
    # ==============================================

    plt.suptitle('Dual-Arm Robot Joint Angles (Top: Left Arm / Bottom: Right Arm)', fontsize=16)
    # 调整整体布局，避免标题和子图重叠
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    output_dir = os.path.dirname(npy_file_path)
    # output_filename = os.path.join(output_dir, 'joint_angles.png')
    output_filename = os.path.join(output_dir, 'endpose.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"绘图完成，已保存为 {output_filename}")
    # plt.show() # 如果需要显示窗口请取消注释

if __name__ == "__main__":
    file_name = '/home/ps/wx_ws/code/2512_keyframes/RoboTwin/policy/ACT/processed_data/sim-blocks_ranking_rgb/aloha_clean_500-vis-img-endpose/000/qpos_endpose.npy' # 请替换为你的文件名
    plot_joint_angles_2x7(file_name)