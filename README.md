# ESRGAN 模型下载、训练和推理工具

这是一个完整的ESRGAN（Enhanced Super-Resolution Generative Adversarial Networks）工具包，支持模型下载、自定义训练和图像超分辨率推理。

## 功能特性

- 🚀 **自动下载预训练模型** - 支持多种ESRGAN预训练模型
- 🎯 **自定义模型训练** - 使用自己的数据集训练ESRGAN模型
- 🖼️ **图像超分辨率推理** - 使用训练好的模型进行图像增强
- 📊 **进度显示** - 实时显示下载和训练进度
- 🔧 **灵活配置** - 支持多种参数配置和模型选择

## 安装依赖

```bash
pip install -r requirements.txt
```

## 1. 模型下载

### 查看可用模型
```bash
python download.py --list
```

### 下载特定模型
```bash
# 下载通用4x超分辨率模型（推荐）
python download.py --model RealESRGAN_x4plus

# 下载动漫优化模型
python download.py --model RealESRGAN_x4plus_anime_6B

# 下载2x超分辨率模型
python download.py --model RealESRGAN_x2plus
```

### 下载所有模型
```bash
python download.py --all
```

## 2. 模型训练

### 准备训练数据

1. **准备高分辨率图像**
   ```bash
   mkdir -p datasets/train/hr
   # 将你的高分辨率图像放入 datasets/train/hr/ 目录
   ```

2. **生成低分辨率图像**
   ```bash
   # 自动从HR图像生成对应的LR图像
   python train.py --prepare_data --hr_dir ./datasets/train/hr --scale 4
   ```

### 开始训练

1. **使用默认配置训练**
   ```bash
   python train.py
   ```

2. **使用自定义配置训练**
   ```bash
   # 首次运行会生成默认配置文件 train_config.yml
   # 编辑配置文件后再次运行
   python train.py --config train_config.yml
   ```

### 训练配置说明

训练配置文件 `train_config.yml` 包含以下主要参数：

```yaml
# 实验设置
experiment_name: 'my_esrgan'
experiment_dir: './experiments/my_esrgan'
scale: 4  # 缩放倍数
pretrained_model: './weights/RealESRGAN_x4plus.pth'  # 预训练模型路径

# 网络结构
network:
  nf: 64    # 特征通道数
  nb: 23    # RRDB块数量
  gc: 32    # 增长通道数

# 数据集设置
datasets:
  hr_dir: './datasets/train/hr'  # 高分辨率图像目录
  lr_dir: './datasets/train/lr'  # 低分辨率图像目录
  patch_size: 128                # 训练patch大小

# 训练参数
training:
  epochs: 100          # 训练轮数
  batch_size: 4        # 批次大小
  num_workers: 4       # 数据加载线程数
  lr: 1e-4            # 学习率
  lr_steps: [50, 75, 90]  # 学习率衰减步骤
  lr_gamma: 0.5        # 学习率衰减系数
  save_freq: 1000      # 保存检查点频率
  save_epoch_freq: 10  # 保存epoch模型频率
```

## 3. 图像推理

### 处理单张图像
```bash
python inference.py --model ./weights/RealESRGAN_x4plus.pth --input ./inputs/test.jpg --output ./results/test_4x.jpg --scale 4
```

### 批量处理图像
```bash
python inference.py --model ./weights/RealESRGAN_x4plus.pth --input ./inputs/ --output ./results/ --scale 4
```

### 使用自训练模型
```bash
python inference.py --model ./experiments/my_esrgan/models/final_model.pth --input ./inputs/test.jpg --output ./results/test_custom.jpg --scale 4
```

## 可用模型

| 模型名称 | 描述 | 适用场景 |
|---------|------|----------|
| RealESRGAN_x4plus | 通用4x超分辨率模型 | 真实世界图像（推荐） |
| RealESRGAN_x4plus_anime_6B | 动漫优化4x模型 | 动漫、插画图像 |
| RealESRGAN_x2plus | 2x超分辨率模型 | 轻量级处理 |
| realesr-animevideov3 | 动漫视频模型 | 动漫视频帧 |
| realesr-general-x4v3 | 小型通用4x模型 | 快速处理 |
| RRDB_ESRGAN_x4 | 原始ESRGAN模型 | 经典ESRGAN |
| RRDB_PSNR_x4 | 高PSNR模型 | 高保真度需求 |

## 目录结构

```
turingESRGAN/
├── download.py          # 模型下载脚本
├── train.py            # 模型训练脚本
├── inference.py        # 推理脚本
├── requirements.txt    # 依赖包列表
├── README.md          # 说明文档
├── train_config.yml   # 训练配置文件（自动生成）
├── datasets/          # 训练数据集
│   └── train/
│       ├── hr/        # 高分辨率图像
│       └── lr/        # 低分辨率图像
├── weights/           # 预训练模型
├── models/           # 下载的模型文件
├── inputs/           # 输入图像
├── results/          # 输出结果
└── experiments/      # 训练实验
    └── my_esrgan/
        └── models/   # 训练的模型
```

## 训练建议

### 数据准备
- **图像质量**：使用高质量、清晰的图像作为训练数据
- **数据量**：建议至少1000张以上的训练图像
- **图像尺寸**：HR图像建议大于512x512像素
- **数据多样性**：包含不同场景、纹理的图像

### 训练参数
- **批次大小**：根据GPU显存调整，4-8为常用值
- **学习率**：从1e-4开始，可根据训练情况调整
- **预训练模型**：建议从预训练模型开始微调
- **训练轮数**：通常需要50-200个epoch

### 硬件要求
- **GPU**：建议使用NVIDIA GPU，至少4GB显存
- **内存**：建议16GB以上系统内存
- **存储**：训练过程需要足够的磁盘空间

## 常见问题

### 1. 内存不足
- 减小批次大小（batch_size）
- 减小训练patch大小（patch_size）
- 减少数据加载线程数（num_workers）

### 2. 训练速度慢
- 使用GPU加速
- 增加批次大小
- 使用更快的存储设备

### 3. 模型效果不佳
- 增加训练数据量
- 延长训练时间
- 调整学习率
- 使用更好的预训练模型

### 4. 推理结果异常
- 检查模型文件是否完整
- 确认输入图像格式正确
- 验证模型和缩放倍数匹配

## 相关资源

- [Real-ESRGAN GitHub](https://github.com/xinntao/Real-ESRGAN)
- [ESRGAN 论文](https://arxiv.org/abs/1809.00219)
- [BasicSR 框架](https://github.com/XPixelGroup/BasicSR)

## 许可证

本项目遵循 MIT 许可证。使用的预训练模型可能有不同的许可证要求，请查看相应的模型文档。 