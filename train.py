#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESRGAN模型训练脚本
支持从预训练模型开始训练自定义的ESRGAN模型
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ESRGANDataset(torch.utils.data.Dataset):
    """ESRGAN数据集类"""
    
    def __init__(self, hr_dir, lr_dir, scale=4, patch_size=128):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.scale = scale
        self.patch_size = patch_size
        
        # 获取所有图像文件
        self.hr_images = sorted([f for f in self.hr_dir.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        self.lr_images = sorted([f for f in self.lr_dir.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        
        assert len(self.hr_images) == len(self.lr_images), "HR和LR图像数量不匹配"
        logger.info(f"找到 {len(self.hr_images)} 对训练图像")
    
    def __len__(self):
        return len(self.hr_images)
    
    def __getitem__(self, idx):
        # 读取HR和LR图像
        hr_path = self.hr_images[idx]
        lr_path = self.lr_images[idx]
        
        hr_img = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)
        lr_img = cv2.imread(str(lr_path), cv2.IMREAD_COLOR)
        
        if hr_img is None or lr_img is None:
            raise ValueError(f"无法读取图像: {hr_path} 或 {lr_path}")
        
        # 转换为RGB
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        
        # 随机裁剪
        hr_h, hr_w = hr_img.shape[:2]
        lr_h, lr_w = lr_img.shape[:2]
        
        # 确保尺寸匹配
        if hr_h != lr_h * self.scale or hr_w != lr_w * self.scale:
            # 调整LR图像尺寸
            lr_img = cv2.resize(lr_img, (hr_w // self.scale, hr_h // self.scale), interpolation=cv2.INTER_CUBIC)
        
        # 随机裁剪patch
        if hr_h > self.patch_size and hr_w > self.patch_size:
            lr_patch_size = self.patch_size // self.scale
            
            # 随机选择裁剪位置
            top = np.random.randint(0, hr_h - self.patch_size + 1)
            left = np.random.randint(0, hr_w - self.patch_size + 1)
            
            hr_patch = hr_img[top:top+self.patch_size, left:left+self.patch_size]
            
            lr_top = top // self.scale
            lr_left = left // self.scale
            lr_patch = lr_img[lr_top:lr_top+lr_patch_size, lr_left:lr_left+lr_patch_size]
        else:
            hr_patch = hr_img
            lr_patch = lr_img
        
        # 数据增强：随机翻转和旋转
        if np.random.random() > 0.5:
            hr_patch = np.fliplr(hr_patch)
            lr_patch = np.fliplr(lr_patch)
        
        if np.random.random() > 0.5:
            hr_patch = np.flipud(hr_patch)
            lr_patch = np.flipud(lr_patch)
        
        # 转换为tensor并归一化到[0,1]
        hr_tensor = torch.from_numpy(hr_patch.transpose(2, 0, 1)).float() / 255.0
        lr_tensor = torch.from_numpy(lr_patch.transpose(2, 0, 1)).float() / 255.0
        
        return lr_tensor, hr_tensor

class RRDBBlock(nn.Module):
    """RRDB块"""
    
    def __init__(self, nf, gc=32):
        super(RRDBBlock, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)
    
    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class ResidualDenseBlock_5C(nn.Module):
    """5层卷积的残差密集块"""
    
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDBNet(nn.Module):
    """RRDB网络"""
    
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDBBlock, nf=nf, gc=gc)
        
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # 上采样
        if scale == 4:
            self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        elif scale == 2:
            self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.scale = scale
    
    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        
        if self.scale == 4:
            fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        elif self.scale == 2:
            fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out

import functools
import torch.nn.functional as F

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ESRGANTrainer:
    """ESRGAN训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.exp_dir = Path(config['experiment_dir'])
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.exp_dir / 'models'
        self.models_dir.mkdir(exist_ok=True)
        
        # 初始化模型
        self.setup_model()
        
        # 初始化数据加载器
        self.setup_dataloader()
        
        # 初始化优化器和损失函数
        self.setup_optimizer()
        
        # 训练状态
        self.current_iter = 0
        self.current_epoch = 0
    
    def setup_model(self):
        """设置模型"""
        self.model = RRDBNet(
            in_nc=3,
            out_nc=3,
            nf=self.config['network']['nf'],
            nb=self.config['network']['nb'],
            gc=self.config['network']['gc'],
            scale=self.config['scale']
        ).to(self.device)
        
        # 加载预训练模型
        if self.config.get('pretrained_model'):
            pretrained_path = self.config['pretrained_model']
            if os.path.exists(pretrained_path):
                logger.info(f"加载预训练模型: {pretrained_path}")
                checkpoint = torch.load(pretrained_path, map_location=self.device)
                self.model.load_state_dict(checkpoint, strict=False)
            else:
                logger.warning(f"预训练模型不存在: {pretrained_path}")
    
    def setup_dataloader(self):
        """设置数据加载器"""
        dataset = ESRGANDataset(
            hr_dir=self.config['datasets']['hr_dir'],
            lr_dir=self.config['datasets']['lr_dir'],
            scale=self.config['scale'],
            patch_size=self.config['datasets']['patch_size']
        )
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
        
        logger.info(f"数据集大小: {len(dataset)}")
        logger.info(f"批次大小: {self.config['training']['batch_size']}")
    
    def setup_optimizer(self):
        """设置优化器和损失函数"""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['lr'],
            betas=(0.9, 0.999)
        )
        
        # 使用L1损失
        self.criterion = nn.L1Loss()
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.config['training']['lr_steps'],
            gamma=self.config['training']['lr_gamma']
        )
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(pbar):
            lr_imgs = lr_imgs.to(self.device)
            hr_imgs = hr_imgs.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            sr_imgs = self.model(lr_imgs)
            
            # 计算损失
            loss = self.criterion(sr_imgs, hr_imgs)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            self.current_iter += 1
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg Loss': f'{epoch_loss/(batch_idx+1):.6f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # 保存检查点
            if self.current_iter % self.config['training']['save_freq'] == 0:
                self.save_checkpoint()
        
        return epoch_loss / len(self.dataloader)
    
    def save_checkpoint(self):
        """保存检查点"""
        checkpoint_path = self.models_dir / f'model_iter_{self.current_iter}.pth'
        torch.save(self.model.state_dict(), checkpoint_path)
        logger.info(f"保存检查点: {checkpoint_path}")
    
    def train(self):
        """开始训练"""
        logger.info("开始训练...")
        logger.info(f"总epoch数: {self.config['training']['epochs']}")
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # 训练一个epoch
            avg_loss = self.train_epoch()
            
            # 更新学习率
            self.scheduler.step()
            
            logger.info(f"Epoch {epoch} 完成, 平均损失: {avg_loss:.6f}")
            
            # 保存最终模型
            if (epoch + 1) % self.config['training']['save_epoch_freq'] == 0:
                final_model_path = self.models_dir / f'model_epoch_{epoch+1}.pth'
                torch.save(self.model.state_dict(), final_model_path)
                logger.info(f"保存epoch模型: {final_model_path}")
        
        # 保存最终模型
        final_model_path = self.models_dir / 'final_model.pth'
        torch.save(self.model.state_dict(), final_model_path)
        logger.info(f"训练完成! 最终模型保存至: {final_model_path}")

def create_default_config():
    """创建默认配置"""
    config = {
        'experiment_name': 'my_esrgan',
        'experiment_dir': './experiments/my_esrgan',
        'scale': 4,
        'pretrained_model': './weights/RealESRGAN_x4plus.pth',
        
        'network': {
            'nf': 64,
            'nb': 23,
            'gc': 32
        },
        
        'datasets': {
            'hr_dir': './datasets/train/hr',
            'lr_dir': './datasets/train/lr',
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
            'save_epoch_freq': 10
        }
    }
    return config

def prepare_dataset(hr_dir, scale=4):
    """准备训练数据集"""
    hr_path = Path(hr_dir)
    lr_path = hr_path.parent / 'lr'
    lr_path.mkdir(exist_ok=True)
    
    logger.info(f"从 {hr_path} 生成LR图像到 {lr_path}")
    
    hr_images = list(hr_path.glob('*'))
    hr_images = [f for f in hr_images if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    
    for hr_img_path in tqdm(hr_images, desc="生成LR图像"):
        # 读取HR图像
        hr_img = cv2.imread(str(hr_img_path))
        if hr_img is None:
            continue
        
        # 生成LR图像
        h, w = hr_img.shape[:2]
        lr_h, lr_w = h // scale, w // scale
        lr_img = cv2.resize(hr_img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        
        # 保存LR图像
        lr_img_path = lr_path / hr_img_path.name
        cv2.imwrite(str(lr_img_path), lr_img)
    
    logger.info(f"生成了 {len(hr_images)} 对训练图像")

def main():
    parser = argparse.ArgumentParser(description='ESRGAN训练脚本')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--hr_dir', type=str, help='HR图像目录')
    parser.add_argument('--prepare_data', action='store_true', help='准备训练数据')
    parser.add_argument('--scale', type=int, default=4, help='缩放倍数')
    
    args = parser.parse_args()
    
    if args.prepare_data and args.hr_dir:
        # 准备数据集
        prepare_dataset(args.hr_dir, args.scale)
        return
    
    # 加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = create_default_config()
        
        # 保存默认配置
        config_path = 'train_config.yml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"创建默认配置文件: {config_path}")
    
    # 检查数据集
    hr_dir = Path(config['datasets']['hr_dir'])
    lr_dir = Path(config['datasets']['lr_dir'])
    
    if not hr_dir.exists() or not lr_dir.exists():
        logger.error("训练数据集不存在!")
        logger.info("请先准备训练数据:")
        logger.info("1. 将HR图像放入 ./datasets/train/hr/ 目录")
        logger.info("2. 运行: python train.py --prepare_data --hr_dir ./datasets/train/hr")
        return
    
    # 开始训练
    trainer = ESRGANTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
