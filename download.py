#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESRGAN模型下载脚本
支持下载多种预训练的ESRGAN模型
"""

import os
import requests
import argparse
from pathlib import Path
from tqdm import tqdm

# 模型下载链接配置
MODEL_URLS = {
    # Real-ESRGAN 模型
    'RealESRGAN_x4plus': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        'description': '通用4倍超分辨率模型，适用于真实世界图像'
    },
    'RealESRGAN_x4plus_anime_6B': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
        'description': '专门针对动漫图像优化的4倍超分辨率模型'
    },
    'RealESRGAN_x2plus': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        'description': '2倍超分辨率模型'
    },
    'realesr-animevideov3': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth',
        'description': '专门用于动漫视频的超分辨率模型'
    },
    'realesr-general-x4v3': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
        'description': '通用场景的小型4倍超分辨率模型'
    },
    
    # 原始 ESRGAN 模型
    'RRDB_ESRGAN_x4': {
        'url': 'https://github.com/xinntao/ESRGAN/releases/download/v0.0.0/RRDB_ESRGAN_x4.pth',
        'description': '原始ESRGAN 4倍超分辨率模型'
    },
    'RRDB_PSNR_x4': {
        'url': 'https://github.com/xinntao/ESRGAN/releases/download/v0.0.0/RRDB_PSNR_x4.pth',
        'description': '高PSNR性能的4倍超分辨率模型'
    }
}

def download_file(url, filename, description=""):
    """
    下载文件并显示进度条
    """
    print(f"正在下载: {description}")
    print(f"URL: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✅ 下载完成: {filename}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 下载失败: {e}")
        return False

def create_directories():
    """
    创建必要的目录结构
    """
    directories = ['models', 'weights', 'inputs', 'results']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True) # Path(dir_name)用pathlib.Path创建目录,mkdir(exist_ok=True)检测目录是否存在,存在就跳过,不存在就创建
        print(f"📁 创建目录: {dir_name}")

def list_available_models():
    """
    列出所有可用的模型
    """
    print("\n🎯 可用的ESRGAN模型:")
    print("=" * 60)
    for i, (model_name, model_info) in enumerate(MODEL_URLS.items(), 1): # 利用enumerate函数遍历字典,并为每个元素赋予一个从1开始的编号
        print(f"{i:2d}. {model_name}")
        print(f"    描述: {model_info['description']}")
        print()

def download_model(model_name, save_dir='weights'):
    """
    下载指定的模型
    """
    if model_name not in MODEL_URLS:
        print(f"❌ 未找到模型: {model_name}")
        return False
    
    model_info = MODEL_URLS[model_name]
    url = model_info['url']
    filename = os.path.join(save_dir, f"{model_name}.pth")
    
    # 检查文件是否已存在
    if os.path.exists(filename):
        print(f"⚠️  模型已存在: {filename}")
        overwrite = input("是否重新下载? (y/N): ").lower().strip()
        if overwrite != 'y':
            print("跳过下载")
            return True
    
    return download_file(url, filename, model_info['description'])


def main():
    parser = argparse.ArgumentParser(description='ESRGAN模型下载工具')
    parser.add_argument('--model', '-m', type=str, help='指定要下载的模型名称')
    parser.add_argument('--list', '-l', action='store_true', help='列出所有可用模型')
    parser.add_argument('--dir', '-d', type=str, default='weights', help='模型保存目录 (默认: weights)') # 当文件夹需要储存完成的模型结构文件时,使用models文件夹;当文件夹需要储存模型的权重参数文件时,通常使用weights文件夹
    
    args = parser.parse_args()
    
    # 创建目录结构
    create_directories()
    
    if args.list:
        list_available_models()
    elif args.model:
        download_model(args.model, args.dir)
    else:
        # 交互式模式
        print("🎨 ESRGAN模型下载工具")
        print("=" * 40)
        
        while True:
            print("\n请选择操作:")
            print("1. 列出所有可用模型")
            print("2. 下载指定模型")
            print("3. 退出")
            
            choice = input("\n请输入选择 (1-3): ").strip()
            
            if choice == '1':
                list_available_models()
            elif choice == '2':
                list_available_models()
                model_choice = input("请输入模型名称和下载路径(可选): ").strip()
                download_model(model_choice, args.dir)
            elif choice == '3':
                print("👋 再见!")
                break
            else:
                print("❌ 无效选择，请重新输入")

if __name__ == '__main__':
    main()
