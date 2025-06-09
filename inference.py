#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESRGAN推理脚本
使用训练好的ESRGAN模型进行图像超分辨率处理
"""

import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import functools
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ESRGANInference:
    """ESRGAN推理器"""
    
    def __init__(self, model_path, scale=4, device=None):
        self.scale = scale
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = self.load_model(model_path)
        self.model.eval()
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"模型缩放倍数: {self.scale}x")
    
    def load_model(self, model_path):
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 创建模型
        model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=self.scale)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 处理不同的权重格式
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 移除可能的前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        model.to(self.device)
        
        logger.info(f"成功加载模型: {model_path}")
        return model
    
    def preprocess_image(self, img):
        """预处理图像"""
        # 转换为RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 转换为tensor并归一化
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img_tensor
    
    def postprocess_image(self, tensor):
        """后处理图像"""
        # 转换回numpy数组
        img = tensor.squeeze(0).cpu().numpy()
        img = img.transpose(1, 2, 0)
        
        # 限制像素值范围
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        
        # 转换为BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img
    
    def enhance_image(self, img):
        """增强单张图像"""
        with torch.no_grad():
            # 预处理
            lr_tensor = self.preprocess_image(img)
            
            # 推理
            sr_tensor = self.model(lr_tensor)
            
            # 后处理
            sr_img = self.postprocess_image(sr_tensor)
            
            return sr_img
    
    def enhance_image_file(self, input_path, output_path):
        """增强图像文件"""
        # 读取图像
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"无法读取图像: {input_path}")
        
        logger.info(f"处理图像: {input_path}")
        logger.info(f"原始尺寸: {img.shape[1]}x{img.shape[0]}")
        
        # 增强图像
        sr_img = self.enhance_image(img)
        
        logger.info(f"增强后尺寸: {sr_img.shape[1]}x{sr_img.shape[0]}")
        
        # 保存结果
        cv2.imwrite(output_path, sr_img)
        logger.info(f"保存结果: {output_path}")
    
    def enhance_directory(self, input_dir, output_dir):
        """批量增强目录中的图像"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            image_files.extend(input_path.glob(ext))
            image_files.extend(input_path.glob(ext.upper()))
        
        if not image_files:
            logger.warning(f"在 {input_dir} 中未找到图像文件")
            return
        
        logger.info(f"找到 {len(image_files)} 个图像文件")
        
        # 批量处理
        for img_file in tqdm(image_files, desc="处理图像"):
            try:
                output_file = output_path / img_file.name
                self.enhance_image_file(str(img_file), str(output_file))
            except Exception as e:
                logger.error(f"处理 {img_file} 时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='ESRGAN图像超分辨率推理')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--input', type=str, required=True, help='输入图像或目录路径')
    parser.add_argument('--output', type=str, required=True, help='输出图像或目录路径')
    parser.add_argument('--scale', type=int, default=4, choices=[2, 4], help='缩放倍数')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='计算设备')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # 创建推理器
    try:
        inference = ESRGANInference(args.model, args.scale, device)
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return
    
    # 处理输入
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 单个文件
        try:
            inference.enhance_image_file(args.input, args.output)
            logger.info("处理完成!")
        except Exception as e:
            logger.error(f"处理失败: {e}")
    
    elif input_path.is_dir():
        # 目录
        try:
            inference.enhance_directory(args.input, args.output)
            logger.info("批量处理完成!")
        except Exception as e:
            logger.error(f"批量处理失败: {e}")
    
    else:
        logger.error(f"输入路径不存在: {args.input}")

if __name__ == '__main__':
    main() 