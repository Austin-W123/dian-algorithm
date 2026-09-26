# Dian Team 2026 Algorithm Recruitment

Dian 团队 2026 秋招算法题个人项目。

本项目按照 Level 0 → Level 4 的任务路线，逐步完成从深度学习环境搭建、MLP、CNN、经典 CNN 架构到 U-Net 图像到图像任务的学习与实践。

各阶段的代码、实验过程、结果分析和知识总结均记录在对应 Level 的 README 中，本 README 仅作为整个项目的概览与导航。

---

# 1. 项目结构

```text
dian-algorithm/
│
├── level0_environment/
│   └── 环境搭建与测试
│
├── level1_mlp/
│   ├── main.py
│   ├── README.md
│   └── outputs/
│
├── level2_cnn/
│   ├── main.py
│   ├── README.md
│   └── outputs/
│
├── level3_classic_cnn/
│   ├── alexnet.py
│   ├── vgg.py
│   ├── resnet.py
│   ├── README.md
│   └── outputs/
│
├── level4_unet/
│   ├── README.md
│   ├── ...
│   └── outputs/
│
├── .gitignore
└── README.md
```

每个 Level 独立保存对应的：

```text
代码实现
实验结果
结果分析
方法说明
学习总结
```

详细内容请进入对应目录查看 README。

---

# 2. 学习路线

本项目按照任务要求逐步推进：

```text
Level 0
深度学习环境搭建
        ↓
Level 1
MLP + MNIST
        ↓
Level 2
CNN + MNIST
        ↓
Level 3
AlexNet / VGG / ResNet
        ↓
Level 4
U-Net + Handwriting Removal
```

对应的知识路线可以概括为：

```text
Environment
    ↓
MLP
    ↓
CNN
    ↓
Classic CNN Architectures
    ↓
Encoder-Decoder
    ↓
U-Net
    ↓
Image-to-Image
```

从 Level 1 到 Level 4，任务也逐渐从：

```text
Image Classification
```

发展到：

```text
Image-to-Image
```

即从“判断一张图片属于什么类别”，逐步过渡到“根据输入图片生成目标图片”。

---

# 3. Level 0：Environment Setup

## 完成任务

完成后续深度学习实验所需的开发与 GPU 训练环境搭建。

## 实现方式

在 Windows 上配置 WSL2 + Ubuntu + Miniconda，并创建独立 Conda 环境安装 PyTorch 与 CUDA 版本依赖，最终完成 GPU 可用性测试。

主要环境：

```text
Ubuntu      24.04
Python      3.11.9
PyTorch     2.13.0+cu130
torchvision 0.28.0+cu130
GPU         NVIDIA GeForce RTX 5060 Laptop GPU
```

## 结果

```text
CUDA Available = True
```

后续 Level 1 ～ Level 4 均能够使用 GPU 完成模型训练。

## 简单结论

Level 0 建立了整个项目统一、可用的深度学习实验环境，为后续模型训练提供基础。

> 详细环境配置与测试过程见 `level0_environment/`。

---

# 4. Level 1：MLP + MNIST

## 完成任务

使用多层感知机 MLP 完成 MNIST 手写数字分类任务。

## 实现方式

将 `28 × 28` MNIST 图像 Flatten 为 784 维向量，通过：

```text
784 → 128 → 10
```

的全连接神经网络进行分类，并使用 ReLU、CrossEntropyLoss 和 SGD 完成训练。

## 结果

```text
Test Accuracy = 96.36%
Parameters    = 101,770
```

## 简单结论

MLP 能够完成 MNIST 分类，并建立了 Forward、Loss、Backward、Optimizer 等基本神经网络训练流程。

> 详细代码、实验结果与分析见 `level1_mlp/README.md`。

---

# 5. Level 2：CNN + MNIST

## 完成任务

在 Level 1 的基础上使用卷积神经网络 CNN 完成 MNIST 分类，并与 MLP 进行比较。

## 实现方式

使用两层卷积进行图像特征提取，并结合 ReLU、MaxPooling 和 Fully Connected Layer 完成分类。

## 结果

```text
MLP Test Accuracy = 96.36%
CNN Test Accuracy = 98.70%

Improvement = +2.34 percentage points

CNN Parameters    = 421,642
Training Time     = 44.51 s
```

## 简单结论

CNN 能够利用图像的空间结构提取局部特征，在本次 MNIST 实验中取得了比 MLP 更高的分类准确率。

> 详细网络结构、训练过程、错误样本与结果分析见 `level2_cnn/README.md`。

---

# 6. Level 3：AlexNet / VGG / ResNet

## 完成任务

在基础 CNN 之上实现 AlexNet-style、VGG-style 和 ResNet-18-style 三种经典 CNN 架构，并进行对比实验。

## 实现方式

针对 `28 × 28` MNIST 图像分别实现三个适配版本，在统一的数据划分和主要训练配置下比较不同网络结构的 Accuracy、Parameters 和 Training Time。

## 结果

| Network | Test Accuracy | Parameters | Training Time |
|---|---:|---:|---:|
| Level 2 CNN | 98.70% | 421,642 | 44.51 s |
| AlexNet-style | 99.33% | 3,564,490 | 58.28 s |
| VGG-style | 99.06% | 3,048,394 | 61.99 s |
| ResNet-18-style | 99.18% | 2,797,034 | 101.42 s |

## 简单结论

三种经典架构在本次实验中均取得了 99% 以上的 Test Accuracy，同时实验也表明网络更深、结构更复杂并不意味着准确率一定严格更高，需要结合模型规模、训练时间和实验结果共同分析。

> 详细网络结构、训练曲线、错误样本与架构对比见 `level3_classic_cnn/README.md`。

---

# 7. Level 4：U-Net Handwriting Removal

## 完成任务

使用 U-Net 完成手写试卷 / 作业图片中的手写痕迹去除任务，将带有手写内容的图片转换为较干净的目标图片。

## 实现方式

使用成对的 Input / Target 图像训练 U-Net。针对高分辨率且尺寸不统一的文档图片，采用预处理、`256 × 256` Patch Training 和组合损失进行训练；完整图片推理时使用重叠 Patch 与 Weighted Blending 完成图像重建。

## 结果

最终测试集：

```text
Original Image Pairs = 242
Test Patches         = 1210

Global L1            = 0.032711
Global MSE           = 0.016915
Changed-Region L1    = 0.150156

PSNR                  = 20.73 dB
SSIM                  = 0.8770
```

完整图片推理能够实现：

```text
Original Image
      ↓
Preprocessing
      ↓
Overlapping Patches
      ↓
U-Net
      ↓
Weighted Blending
      ↓
Cleaned Image
```

## 简单结论

Level 4 将前面学习的 CNN 特征提取进一步扩展到 Image-to-Image 任务，完成了从数据处理、模型训练、损失设计、测试评估到完整图片推理的整体流程。

> 详细数据处理、U-Net 结构、Loss 设计、实验过程、指标和可视化结果见 `level4_unet/README.md`。

---

# 8. 总结

本项目按照：

```text
Environment
↓
MLP
↓
CNN
↓
AlexNet / VGG / ResNet
↓
U-Net
↓
Image-to-Image
```

的路线逐步完成了 Level 0 ～ Level 4。

项目从 MNIST 分类任务出发，通过 MLP 建立基本神经网络训练流程，再通过 CNN 学习图像空间特征提取；随后实现 AlexNet、VGG 和 ResNet，进一步理解不同深层 CNN 的架构设计；最终使用 U-Net 将前面学习的知识应用到手写痕迹去除这一图像到图像任务中。

整个过程中不仅记录最终实验结果，也保留了各阶段的代码、训练过程、结果分析以及实验中遇到的问题。

各 Level 的详细内容请参阅对应目录中的 README。

