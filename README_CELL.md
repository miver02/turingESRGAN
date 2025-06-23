# ESRGAN 细胞图像超分辨率训练指南

本文档介绍如何使用test20203数据集训练ESRGAN模型，用于细胞图像的超分辨率处理。

## 数据集结构

test20203数据集包含多个幻灯片（slide）文件夹，每个幻灯片文件夹中包含多个细胞（cell）文件夹。每个细胞文件夹中包含以下文件：

- `mat.png`: 细胞图像
- `gray.json`: 灰度信息
- `labels.json`: 标签信息

我们主要使用`mat.png`图像进行训练。

## 训练流程

我们提供了三个主要脚本来处理整个训练流程：

1. `dataset_processor.py`: 处理原始数据集，生成训练、验证和测试集
2. `train.py`: 训练ESRGAN模型
3. `inference.py`: 使用训练好的模型进行推理
4. `train_cell_model.py`: 整合以上流程的一键式训练脚本

### 一键式训练

最简单的方法是使用`train_cell_model.py`脚本，它会自动执行所有步骤：

```bash
python train_cell_model.py --dataset_dir ./test20203 --output_dir ./datasets/cell_dataset
```

参数说明：
- `--dataset_dir`: 原始数据集目录，默认为`./test20203`
- `--output_dir`: 处理后的数据集输出目录，默认为`./datasets/cell_dataset`
- `--scale`: 缩放倍数，可选2或4，默认为4
- `--epochs`: 训练轮数，默认为100
- `--batch_size`: 批次大小，默认为4
- `--pretrained`: 预训练模型路径，默认为`./weights/RealESRGAN_x4plus.pth`
- `--no_pretrained`: 不使用预训练模型，从头开始训练
- `--skip_data_processing`: 跳过数据处理步骤，直接使用已处理的数据集进行训练

### 步骤详解

如果您想了解每个步骤的详细过程，可以按照以下步骤手动执行：

#### 1. 数据处理

```bash
python dataset_processor.py --dataset_dir ./test20203 --output_dir ./datasets/cell_dataset
```

这将处理原始数据集，并生成训练、验证和测试集，同时创建配置文件`cell_train_config.yml`。

#### 2. 训练模型

```bash
python train.py --config ./datasets/cell_dataset/cell_train_config.yml
```

训练过程会自动保存检查点和最佳模型。

#### 3. 推理测试

```bash
python inference.py --model ./experiments/esrgan_cell_images/models/best_model.pth --input ./datasets/cell_dataset/test/lr --output ./results
```

## 修改后的训练脚本特性

我们对原始ESRGAN训练脚本进行了以下改进：

1. **验证集评估**: 训练过程中定期在验证集上评估模型性能（PSNR指标）
2. **最佳模型保存**: 自动保存验证集上性能最佳的模型
3. **数据增强**: 随机裁剪、翻转等数据增强技术
4. **自适应数据处理**: 针对细胞图像特点的数据处理流程
5. **批量测试**: 支持对整个测试集进行批量推理

## 训练建议

1. **GPU加速**: 强烈建议使用GPU进行训练，CPU训练会非常慢
2. **预训练模型**: 使用预训练模型可以大幅加速收敛
3. **批次大小**: 根据GPU显存调整批次大小，显存不足时可以减小批次大小
4. **训练轮数**: 通常需要50-200个epoch，可以根据验证集性能提前停止
5. **学习率**: 默认学习率为1e-4，如果训练不稳定可以尝试降低

## 结果评估

训练完成后，可以使用以下指标评估模型性能：

1. **PSNR (Peak Signal-to-Noise Ratio)**: 峰值信噪比，越高越好
2. **视觉质量**: 主观评估超分辨率图像的视觉质量
3. **边缘保留**: 评估细胞边缘和细节的保留程度

## 常见问题

1. **内存不足**:
   - 减小批次大小（batch_size）
   - 减小训练patch大小（patch_size）
   - 减少数据加载线程数（num_workers）

2. **训练速度慢**:
   - 使用GPU加速
   - 使用SSD存储数据集
   - 增加数据加载线程数

3. **模型效果不佳**:
   - 增加训练数据量
   - 延长训练时间
   - 调整学习率
   - 尝试不同的预训练模型

## 自定义训练配置

如需自定义训练配置，可以编辑`cell_train_config.yml`文件：

```yaml
# 实验设置
experiment_name: 'esrgan_cell_images'
experiment_dir: './experiments/esrgan_cell_images'
scale: 4  # 缩放倍数
pretrained_model: './weights/RealESRGAN_x4plus.pth'  # 预训练模型路径

# 网络结构
network:
  nf: 64    # 特征通道数
  nb: 23    # RRDB块数量
  gc: 32    # 增长通道数

# 数据集设置
datasets:
  hr_dir: './datasets/cell_dataset/train/hr'  # 高分辨率图像目录
  lr_dir: './datasets/cell_dataset/train/lr'  # 低分辨率图像目录
  val_hr_dir: './datasets/cell_dataset/val/hr'  # 验证集高分辨率图像目录
  val_lr_dir: './datasets/cell_dataset/val/lr'  # 验证集低分辨率图像目录
  patch_size: 128  # 训练patch大小

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
  val_freq: 5          # 验证频率（每5个epoch验证一次）
``` 