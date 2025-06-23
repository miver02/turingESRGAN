#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESRGAN数据集处理脚本
用于处理test20203文件夹中的细胞图像数据集，准备ESRGAN训练数据
"""

import os
import sys
import argparse
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import json
import random

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_cell_image(cell_dir, output_hr_dir, output_lr_dir, scale=4, min_size=256):
    """处理单个细胞图像，提取为HR图像并生成对应的LR图像"""
    cell_path = Path(cell_dir)
    
    # 查找mat.png文件
    mat_file = cell_path / "mat.png"
    if not mat_file.exists():
        logger.warning(f"在 {cell_dir} 中未找到mat.png文件")
        return False
    
    # 读取图像
    img = cv2.imread(str(mat_file))
    if img is None:
        logger.warning(f"无法读取图像: {mat_file}")
        return False
    
    # 检查图像尺寸
    h, w = img.shape[:2]
    if h < min_size or w < min_size:
        logger.debug(f"图像尺寸太小 ({w}x{h}): {mat_file}")
        return False
    
    # 生成输出文件名
    # 使用目录结构作为文件名前缀，确保唯一性
    # 例如：slide1_cell1_mat.png
    parts = cell_path.parts
    slide_name = parts[-2]  # 例如 "slide1"
    cell_name = parts[-1]   # 例如 "cell1"
    output_name = f"{slide_name}_{cell_name}_mat.png"
    
    # 保存HR图像
    hr_path = output_hr_dir / output_name
    cv2.imwrite(str(hr_path), img)
    
    # 生成并保存LR图像
    h_lr, w_lr = h // scale, w // scale
    img_lr = cv2.resize(img, (w_lr, h_lr), interpolation=cv2.INTER_CUBIC)
    lr_path = output_lr_dir / output_name
    cv2.imwrite(str(lr_path), img_lr)
    
    return True

def process_dataset(dataset_dir, output_dir, scale=4, min_size=256, val_split=0.1, test_split=0.1):
    """处理整个数据集，将细胞图像分为训练、验证和测试集"""
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录结构
    train_hr_dir = output_path / "train" / "hr"
    train_lr_dir = output_path / "train" / "lr"
    val_hr_dir = output_path / "val" / "hr"
    val_lr_dir = output_path / "val" / "lr"
    test_hr_dir = output_path / "test" / "hr"
    test_lr_dir = output_path / "test" / "lr"
    
    for dir_path in [train_hr_dir, train_lr_dir, val_hr_dir, val_lr_dir, test_hr_dir, test_lr_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 收集所有细胞目录
    cell_dirs = []
    for slide_dir in dataset_path.glob("slide*"):
        if slide_dir.is_dir():
            for cell_dir in slide_dir.glob("cell*"):
                if cell_dir.is_dir():
                    cell_dirs.append(cell_dir)
    
    logger.info(f"找到 {len(cell_dirs)} 个细胞目录")
    
    # 随机打乱并分割数据集
    random.shuffle(cell_dirs)
    total = len(cell_dirs)
    val_count = int(total * val_split)
    test_count = int(total * test_split)
    train_count = total - val_count - test_count
    
    train_dirs = cell_dirs[:train_count]
    val_dirs = cell_dirs[train_count:train_count+val_count]
    test_dirs = cell_dirs[train_count+val_count:]
    
    logger.info(f"分割数据集: 训练集 {len(train_dirs)}, 验证集 {len(val_dirs)}, 测试集 {len(test_dirs)}")
    
    # 处理训练集
    success_count = 0
    for cell_dir in tqdm(train_dirs, desc="处理训练集"):
        if process_cell_image(cell_dir, train_hr_dir, train_lr_dir, scale, min_size):
            success_count += 1
    
    logger.info(f"成功处理 {success_count}/{len(train_dirs)} 个训练样本")
    
    # 处理验证集
    success_count = 0
    for cell_dir in tqdm(val_dirs, desc="处理验证集"):
        if process_cell_image(cell_dir, val_hr_dir, val_lr_dir, scale, min_size):
            success_count += 1
    
    logger.info(f"成功处理 {success_count}/{len(val_dirs)} 个验证样本")
    
    # 处理测试集
    success_count = 0
    for cell_dir in tqdm(test_dirs, desc="处理测试集"):
        if process_cell_image(cell_dir, test_hr_dir, test_lr_dir, scale, min_size):
            success_count += 1
    
    logger.info(f"成功处理 {success_count}/{len(test_dirs)} 个测试样本")
    
    # 统计处理结果
    train_hr_count = len(list(train_hr_dir.glob("*.png")))
    val_hr_count = len(list(val_hr_dir.glob("*.png")))
    test_hr_count = len(list(test_hr_dir.glob("*.png")))
    
    logger.info(f"数据集处理完成:")
    logger.info(f"- 训练集: {train_hr_count} 张图像")
    logger.info(f"- 验证集: {val_hr_count} 张图像")
    logger.info(f"- 测试集: {test_hr_count} 张图像")
    
    return {
        "train_hr_dir": str(train_hr_dir),
        "train_lr_dir": str(train_lr_dir),
        "val_hr_dir": str(val_hr_dir),
        "val_lr_dir": str(val_lr_dir),
        "test_hr_dir": str(test_hr_dir),
        "test_lr_dir": str(test_lr_dir),
        "train_count": train_hr_count,
        "val_count": val_hr_count,
        "test_count": test_hr_count
    }

def create_config_file(dataset_info, output_dir, scale=4):
    """创建训练配置文件"""
    config = {
        'experiment_name': 'esrgan_cell_images',
        'experiment_dir': f'./experiments/esrgan_cell_images',
        'scale': scale,
        'pretrained_model': './weights/RealESRGAN_x4plus.pth',
        
        'network': {
            'nf': 64,
            'nb': 23,
            'gc': 32
        },
        
        'datasets': {
            'hr_dir': dataset_info['train_hr_dir'],
            'lr_dir': dataset_info['train_lr_dir'],
            'val_hr_dir': dataset_info['val_hr_dir'],
            'val_lr_dir': dataset_info['val_lr_dir'],
            'patch_size': 128
        },
        
        'training': {
            'epochs': 100,
            'batch_size': 4,
            'num_workers': 4,
            'lr': 1e-4,
            'lr_steps': [50, 75, 90],
            'lr_gamma': 0.5,
            'save_freq': 1000,
            'save_epoch_freq': 10,
            'val_freq': 5  # 每5个epoch进行一次验证
        }
    }
    
    config_path = Path(output_dir) / "cell_train_config.yml"
    import yaml
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"创建配置文件: {config_path}")
    return str(config_path)

def main():
    parser = argparse.ArgumentParser(description='ESRGAN细胞图像数据集处理')
    parser.add_argument('--dataset_dir', type=str, required=True, help='数据集目录路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录路径')
    parser.add_argument('--scale', type=int, default=4, help='缩放倍数')
    parser.add_argument('--min_size', type=int, default=256, help='最小图像尺寸')
    parser.add_argument('--val_split', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--test_split', type=float, default=0.1, help='测试集比例')
    
    args = parser.parse_args()
    
    logger.info("开始处理细胞图像数据集...")
    dataset_info = process_dataset(
        args.dataset_dir, 
        args.output_dir, 
        args.scale, 
        args.min_size,
        args.val_split,
        args.test_split
    )
    
    config_path = create_config_file(dataset_info, args.output_dir, args.scale)
    
    logger.info(f"数据集处理完成，配置文件已保存到: {config_path}")
    logger.info(f"可以使用以下命令开始训练:")
    logger.info(f"python train.py --config {config_path}")

if __name__ == '__main__':
    main() 