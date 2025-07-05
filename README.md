# MicroISP - 图像信号处理器

一个基于PyTorch实现的轻量级图像信号处理(ISP)神经网络，用于RAW图像到RGB图像的转换和增强。

## 项目简介

MicroISP是一个专门用于图像信号处理的深度学习模型，能够将原始RAW格式图像转换为高质量的RGB图像。该模型采用了注意力机制和多尺度处理，在保持较少参数量的同时实现了优秀的图像增强效果。

## 主要特性

- **轻量级架构**: 基于MicroISP设计，参数量较少，适合移动端部署
- **注意力机制**: 集成了自定义的注意力模块，提升关键特征的处理能力
- **多尺度处理**: 采用双生成器结构，实现多层次的图像处理
- **多种损失函数**: 结合MSE、VGG感知损失和MS-SSIM损失，保证图像质量

## 文件结构

```
code_isp/
├── MicroISP.py          # 主要模型文件(空文件)
├── model.py             # MicroISP模型实现
├── model_pynet.py       # PyNET模型实现
├── train_model.py       # 模型训练脚本
├── load_data.py         # 数据加载器
├── evaluate_accuracy.py # 模型评估脚本
├── utils.py             # 工具函数
├── vgg.py              # VGG损失函数
├── msssim.py           # MS-SSIM损失函数
├── dng_to_png.py       # DNG格式转换工具
├── raw_images/         # 训练数据目录
│   ├── train/          # 训练数据
│   └── val/            # 验证数据
├── models/             # 保存的模型文件
├── results/            # 输出结果
└── visual/             # 可视化数据
```

## 依赖环境

- Python 3.7+
- PyTorch 1.7+
- torchvision
- numpy
- opencv-python
- imageio
- tqdm

## 使用方法

### 训练模型

```bash
python train_model.py
```

训练参数可以在`train_model.py`中修改：
- `batch_size`: 批处理大小 (默认: 16)
- `learning_rate`: 学习率 (默认: 0.001)
- `num_train_epochs`: 训练轮数 (默认: 100)
- `dataset_dir`: 数据集路径 (默认: "./raw_images")

### 评估模型

```bash
python evaluate_accuracy.py
```

### 数据格式转换

```bash
python dng_to_png.py
```

## 模型架构

MicroISP模型包含以下主要组件：

1. **注意力模块 (AttentionModule)**: 
   - 使用多层卷积和全局平均池化
   - 生成注意力权重来增强重要特征

2. **生成器模块 (Generator)**:
   - 多个卷积块处理
   - 残差连接和注意力机制
   - 像素重排(Pixel Shuffle)进行上采样

3. **MicroISP主网络**:
   - 双生成器架构
   - 多尺度特征处理
   - 端到端训练

## 损失函数

模型使用三种损失函数的组合：

- **MSE损失**: 像素级重建损失
- **VGG感知损失**: 基于VGG-19的特征损失
- **MS-SSIM损失**: 多尺度结构相似性损失

总损失 = MSE损失 + VGG损失 + 0.4 × (1 - MS-SSIM)

## 数据集

支持的数据格式：
- RAW格式图像 (4通道输入)
- RGB格式图像 (3通道输出)
- 支持Fujifilm和MediaTek RAW格式

## 训练配置

当前配置适用于快速测试：
- 训练样本: 100张
- 验证样本: 10张
- 可视化样本: 1张

如需完整训练，请修改`train_model.py`中的数据集大小设置。

## 许可证

Copyright 2020 by Andrey Ignatov. All Rights Reserved.

## 贡献

欢迎提交Issues和Pull Requests来改进项目。

## 联系方式

如有问题或建议，请通过GitHub Issues联系。 