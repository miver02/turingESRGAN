#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESRGAN细胞图像训练脚本
使用test20203数据集训练ESRGAN模型
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import subprocess

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='ESRGAN细胞图像训练脚本')
    parser.add_argument('--dataset_dir', type=str, default='./test20203', help='数据集目录路径')
    parser.add_argument('--output_dir', type=str, default='./datasets/cell_dataset', help='处理后的数据集输出目录')
    parser.add_argument('--scale', type=int, default=4, choices=[2, 4], help='缩放倍数')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--pretrained', type=str, default='./weights/RealESRGAN_x4plus.pth', help='预训练模型路径')
    parser.add_argument('--no_pretrained', action='store_true', help='不使用预训练模型')
    parser.add_argument('--skip_data_processing', action='store_true', help='跳过数据处理步骤')
    
    args = parser.parse_args()
    
    # 检查数据集目录是否存在
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        logger.error(f"数据集目录不存在: {args.dataset_dir}")
        return
    
    # 检查预训练模型是否存在
    if not args.no_pretrained and not os.path.exists(args.pretrained):
        logger.warning(f"预训练模型不存在: {args.pretrained}")
        logger.warning("将从头开始训练模型")
        args.no_pretrained = True
    
    # 步骤1: 处理数据集
    if not args.skip_data_processing:
        logger.info("步骤1: 处理细胞图像数据集...")
        
        # 检查dataset_processor.py是否存在
        if not os.path.exists('dataset_processor.py'):
            logger.error("找不到dataset_processor.py脚本")
            return
        
        # 运行数据处理脚本
        cmd = [
            'python', 'dataset_processor.py',
            '--dataset_dir', str(dataset_dir),
            '--output_dir', args.output_dir,
            '--scale', str(args.scale),
            '--min_size', '128',  # 最小图像尺寸
            '--val_split', '0.1',  # 10%验证集
            '--test_split', '0.1'  # 10%测试集
        ]
        
        logger.info(f"执行命令: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"数据处理失败: {e}")
            return
        
        logger.info("数据集处理完成")
        
        # 配置文件路径
        config_path = os.path.join(args.output_dir, 'cell_train_config.yml')
        if not os.path.exists(config_path):
            logger.error(f"找不到配置文件: {config_path}")
            return
    else:
        # 如果跳过数据处理，检查配置文件是否存在
        config_path = os.path.join(args.output_dir, 'cell_train_config.yml')
        if not os.path.exists(config_path):
            logger.error(f"找不到配置文件: {config_path}")
            logger.error("请先运行数据处理步骤或提供正确的配置文件路径")
            return
    
    # 步骤2: 修改配置文件中的参数
    logger.info("步骤2: 更新训练配置...")
    
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新配置
    config['training']['epochs'] = args.epochs
    config['training']['batch_size'] = args.batch_size
    
    if args.no_pretrained:
        config['pretrained_model'] = None
    else:
        config['pretrained_model'] = args.pretrained
    
    # 保存更新后的配置
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"配置已更新: {config_path}")
    
    # 步骤3: 开始训练
    logger.info("步骤3: 开始训练ESRGAN模型...")
    
    # 检查train.py是否存在
    if not os.path.exists('train.py'):
        logger.error("找不到train.py脚本")
        return
    
    # 运行训练脚本
    cmd = [
        'python', 'train.py',
        '--config', config_path
    ]
    
    logger.info(f"执行命令: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"训练失败: {e}")
        return
    
    logger.info("训练完成!")
    logger.info(f"模型保存在: {config['experiment_dir']}/models/")
    
    # 步骤4: 测试模型
    logger.info("步骤4: 测试模型性能...")
    
    # 获取测试集路径
    test_hr_dir = os.path.join(args.output_dir, 'test', 'hr')
    test_lr_dir = os.path.join(args.output_dir, 'test', 'lr')
    
    if not os.path.exists(test_hr_dir) or not os.path.exists(test_lr_dir):
        logger.warning("找不到测试集，跳过测试步骤")
        return
    
    # 获取最终模型路径
    final_model_path = os.path.join(config['experiment_dir'], 'models', 'final_model.pth')
    best_model_path = os.path.join(config['experiment_dir'], 'models', 'best_model.pth')
    
    model_path = best_model_path if os.path.exists(best_model_path) else final_model_path
    
    if not os.path.exists(model_path):
        logger.warning(f"找不到训练好的模型: {model_path}")
        return
    
    # 检查inference.py是否存在
    if not os.path.exists('inference.py'):
        logger.warning("找不到inference.py脚本，跳过测试步骤")
        return
    
    # 创建结果目录
    results_dir = os.path.join(config['experiment_dir'], 'test_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 运行推理脚本
    cmd = [
        'python', 'inference.py',
        '--model', model_path,
        '--input', test_lr_dir,
        '--output', results_dir,
        '--scale', str(args.scale)
    ]
    
    logger.info(f"执行命令: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"测试失败: {e}")
        return
    
    logger.info("测试完成!")
    logger.info(f"超分辨率结果保存在: {results_dir}")
    logger.info(f"可以使用以下命令对新图像进行超分辨率处理:")
    logger.info(f"python inference.py --model {model_path} --input <输入图像或目录> --output <输出路径> --scale {args.scale}")

if __name__ == '__main__':
    main() 