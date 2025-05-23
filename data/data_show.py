import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def smooth_multiple_files(file_list, weight=0.65, save_dir='/home/wangjunjie/DRL-for-Path-Planning/data/'):
    """
    对多个文件的轨迹曲线进行平滑并绘制在同一张图中。

    参数:
    - file_list (list): 包含多个CSV文件路径的列表。
    - weight (float): 平滑参数，控制平滑的程度。
    - save_dir (str): 保存平滑结果文件的目录。
    """
    plt.figure(figsize=(8, 6))

    # 遍历文件列表，处理每个文件
    for file_path in file_list:
        data = pd.read_csv(file_path, header=0, names=['Step', 'Value'], dtype={'Step': int, 'Value': float})
        scalar = data['Value'].values
        last = scalar[0]
        smoothed = []
        for point in scalar:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val

        # 保存平滑结果到文件
        filename = os.path.basename(file_path).replace('.csv', '_smoothed.csv')
        smoothed_df = pd.DataFrame({'Step': data['Step'].values, 'Value': smoothed})
        smoothed_df.to_csv(os.path.join(save_dir, filename), index=False)

        # 绘制每个文件的平滑前后对比
        # plt.plot(data['Step'], data['Value'], label=f'Original {os.path.basename(file_path)}', alpha=0.4)
        plt.plot(data['Step'], smoothed, label=f' {os.path.basename(file_path)}', alpha=1.0)

    # 添加标签和标题
    plt.title(' Reward/ Episode')
    plt.xlabel('Step')
    plt.ylabel('Value')

    # 添加图例
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 显示图表
    plt.show()

if __name__ == '__main__':
    # 假设文件列表包含多个文件路径
    file_list = [
        'data/Letter-1-Reward_Episode.csv',
        'data/Letter-2-Reward_Episode.csv',
        'data/Letter-3-Reward_Episode.csv',
        'data/Letter-4-Reward_Episode.csv'
    ]
    smooth_multiple_files(file_list)
