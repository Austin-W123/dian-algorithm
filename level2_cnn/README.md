# Level 2：基于 CNN 的 MNIST 手写数字分类

## 1. 任务目标

Level 2 的目标是使用卷积神经网络（Convolutional Neural Network, CNN）完成 MNIST 手写数字分类，并与 Level 1 的 MLP 进行比较。

Level 1 的 MLP 在分类前直接将二维图片展开：

```text
[1, 28, 28]
      ↓ Flatten
[784]
      ↓
MLP
      ↓
10 Classes
```

这种方法可以完成分类，但 Flatten 后不再显式保留图片原本的二维空间结构。

Level 2 改为：

```text
MNIST Image
     ↓
Convolution
     ↓
Feature Maps
     ↓
Pooling
     ↓
Convolution
     ↓
Feature Maps
     ↓
Pooling
     ↓
Fully Connected
     ↓
10 Classes
```

本阶段主要完成：

- 使用 `Conv2d` 搭建 CNN；
- 理解 Kernel、Channel、Feature Map；
- 理解 Stride 与 Padding；
- 使用 Max Pooling 进行下采样；
- 跟踪 CNN 中 Tensor Shape 的变化；
- 使用 Adam 训练 CNN；
- 记录 Loss 与 Accuracy 曲线；
- 统计模型参数量与训练时间；
- 在独立 Test Set 上测试；
- 可视化错误分类样本；
- 完成单张图片推理；
- 与 Level 1 的 MLP 进行实验比较。

最终结果：

```text
Test Accuracy = 98.70%
```

相比 Level 1：

```text
MLP = 96.36%
CNN = 98.70%

提升 = 2.34 个百分点
```

---

# 2. 项目文件结构

Level 2 项目结构：

```text
level2_cnn/
├── main.py
├── README.md
└── outputs/
    ├── cnn_mnist.pth
    ├── loss_curve.png
    ├── accuracy_curve.png
    ├── wrong_samples.png
    └── inference_example.png
```

MNIST 数据直接复用：

```text
level1_mlp/data/
```

因此 Level 2 不重复保存数据集。

## 2.1 `main.py`

`main.py` 是本阶段的核心程序，负责完整的 CNN 实验流程：

```text
定义 CNN
    ↓
加载 MNIST
    ↓
Train / Validation Split
    ↓
DataLoader
    ↓
创建 CNN
    ↓
统计参数量
    ↓
Train / Validation
    ↓
Test
    ↓
保存模型
    ↓
Loss / Accuracy Curve
    ↓
Wrong Samples
    ↓
Single-image Inference
```

---

## 2.2 `outputs/cnn_mnist.pth`

保存训练完成后的 CNN 参数：

```text
outputs/cnn_mnist.pth
```

保存方式：

```python
torch.save(model.state_dict(), model_path)
```

模型参数可以在重新创建相同 CNN 结构后加载，用于后续推理。

---

## 2.3 `outputs/loss_curve.png`

保存：

```text
Train Loss
Validation Loss
```

随 Epoch 的变化。

用于观察：

```text
模型是否正常收敛
Validation Loss 是否开始反弹
是否出现过拟合趋势
```

---

## 2.4 `outputs/accuracy_curve.png`

保存：

```text
Train Accuracy
Validation Accuracy
```

随 Epoch 的变化。

它可以更加直观地观察 CNN 在训练集和验证集上的分类能力。

---

## 2.5 `outputs/wrong_samples.png`

保存 Test Set 中前 16 个错误分类样本。

每个样本显示：

```text
True: 真实标签
Pred: 模型预测
```

通过错误样本可以观察 CNN 容易混淆的数字形态。

---

## 2.6 `outputs/inference_example.png`

保存单张 MNIST 图片推理结果。

本次样本：

```text
True:      7
Predicted: 7
```

预测正确。

---

# 3. 实验结果与结果分析

## 3.1 训练结果

CNN 共训练：

```text
5 Epochs
```

实验结果：

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Time |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.1883 | 94.16% | 0.0788 | 97.78% | 10.90 s |
| 2 | 0.0513 | 98.42% | 0.0497 | 98.60% | 8.61 s |
| 3 | 0.0355 | 98.92% | 0.0454 | **98.78%** | 8.60 s |
| 4 | 0.0277 | 99.13% | 0.0534 | 98.40% | 8.21 s |
| 5 | 0.0203 | 99.36% | 0.0459 | 98.52% | 8.20 s |

总训练时间：

```text
44.51 s
```

可以看到：

```text
Train Loss:
0.1883 → 0.0203

Train Accuracy:
94.16% → 99.36%
```

说明 CNN 在训练集上持续学习。

Validation Accuracy 在第 3 个 Epoch 达到最高：

```text
98.78%
```

随后：

```text
Epoch 4 = 98.40%
Epoch 5 = 98.52%
```

与此同时 Train Accuracy 仍继续上升。

因此后期已经出现：

```text
Train Performance ↑
Validation Performance ≈ stable / slight fluctuation
```

说明模型存在轻微的过拟合趋势，但整体 Validation Accuracy 仍保持在较高水平。

---

## 3.2 Loss Curve

Loss Curve 保存在：

```text
outputs/loss_curve.png
```

整体趋势：

```text
Train Loss
0.1883
   ↓
0.0203
```

Validation Loss：

```text
0.0788
  ↓
0.0497
  ↓
0.0454
  ↑
0.0534
  ↓
0.0459
```

Train Loss 持续下降，而 Validation Loss 在后期出现一定波动。

这与 Accuracy 曲线观察到的现象一致：

> 模型在 Training Set 上继续变好，但 Validation Set 的性能已经基本趋于稳定。

---

## 3.3 Accuracy Curve

Accuracy Curve 保存在：

```text
outputs/accuracy_curve.png
```

Train Accuracy：

```text
94.16%
  ↓
98.42%
  ↓
98.92%
  ↓
99.13%
  ↓
99.36%
```

Validation Accuracy：

```text
97.78%
  ↓
98.60%
  ↓
98.78%
  ↓
98.40%
  ↓
98.52%
```

CNN 在第 1 个 Epoch 就已经取得：

```text
97.78%
```

的 Validation Accuracy。

说明在当前训练配置下，CNN 很快就学习到了适用于 MNIST 分类的有效特征。

---

## 3.4 最终 Test Set

训练完成后，在独立 Test Set 上得到：

```text
Test Accuracy = 98.70%
```

Test Set 共：

```text
10000 images
```

因此 CNN 能够正确分类其中绝大多数手写数字。

最终结果明显高于 Level 1 的：

```text
MLP Test Accuracy = 96.36%
```

---

## 3.5 CNN 与 MLP 对比

Level 1 与 Level 2：

| Model | Test Accuracy | Parameters |
|---|---:|---:|
| MLP | 96.36% | 101,770 |
| CNN | **98.70%** | 421,642 |

Accuracy 提升：

```text
98.70% - 96.36%
= 2.34 个百分点
```

但参数量：

```text
MLP = 101,770
CNN = 421,642
```

CNN 约为：

```text
421642 / 101770
≈ 4.14
```

即本实验 CNN 参数量约为 MLP 的 **4.14 倍**。

因此不能简单得出：

```text
CNN 参数一定比 MLP 少
```

这样的结论。

本实验中 CNN 参数较多的重要原因是：

```text
64 × 7 × 7
    ↓ Flatten
3136
    ↓
Linear(3136, 128)
```

这个全连接层本身包含大量参数。

所以更准确的结论是：

> CNN 的优势主要来自其更适合处理图像的结构，而不是本实验中拥有更少的总参数。

---

## 3.6 为什么 CNN 表现更好

MLP：

```text
Image
  ↓
Flatten
  ↓
784-dimensional Vector
```

CNN：

```text
Image
  ↓
Local Convolution
  ↓
Feature Maps
  ↓
Pooling
  ↓
Higher-level Features
```

CNN 在特征提取阶段保留二维空间关系，并通过卷积核学习局部结构。

例如手写数字中的：

```text
边缘
转折
横线
竖线
弧线
局部形状
```

都属于局部空间特征。

CNN 可以逐层将这些简单特征组合成更加复杂的数字特征，因此更加适合图像分类。

---

## 3.7 收敛表现比较

Level 1 MLP 的 Validation Accuracy：

| Epoch | MLP |
|---:|---:|
| 1 | 90.64% |
| 2 | 93.52% |
| 3 | 94.60% |
| 4 | 95.70% |
| 5 | 95.86% |

Level 2 CNN：

| Epoch | CNN |
|---:|---:|
| 1 | 97.78% |
| 2 | 98.60% |
| 3 | 98.78% |
| 4 | 98.40% |
| 5 | 98.52% |

从当前实验可以看到：

```text
CNN 在第 1 Epoch：
97.78%

MLP 在第 5 Epoch：
95.86%
```

从 Epoch 级别的 Accuracy 变化来看，当前 CNN 实验更快达到较高分类准确率。

但是这里需要注意：

```text
MLP Optimizer = SGD
CNN Optimizer = Adam
```

因此这不是严格控制所有变量后的模型结构对比。

更准确的表述是：

> 在本项目实际采用的两套训练配置下，CNN 表现出了更快达到较高 Validation Accuracy 的现象。

---

## 3.8 错误样本分析

错误样本保存在：

```text
outputs/wrong_samples.png
```

观察这些样本可以发现，部分错误图片具有：

```text
字迹倾斜
笔画连接
形状不标准
局部结构模糊
不同数字外观相似
```

例如模型可能产生：

```text
4 → 9
6 → 0
7 → 3
5 → 3
8 → 2
3 → 5
```

这说明即使整体 Test Accuracy 已达到：

```text
98.70%
```

模型仍然会在一些视觉形态非常模糊或具有歧义的样本上出错。

错误样本分析比只观察 Accuracy 更有意义，因为它能够帮助判断：

> 模型到底在哪些类型的数据上失败。

---

## 3.9 单张图片推理

程序从 Test Set 取出一张图片：

```text
True Label      = 7
Predicted Label = 7
```

结果正确。

完整流程：

```text
Single Image
[1, 28, 28]
      ↓
unsqueeze(0)
      ↓
[1, 1, 28, 28]
      ↓
CNN
      ↓
[1, 10] Logits
      ↓
argmax
      ↓
Predicted Class = 7
```

说明训练后的 CNN 已经可以完成：

```text
输入图片
→ Forward
→ Logits
→ Predicted Class
```

的完整推理过程。

---

# 4. 数据集

## 4.1 MNIST

Level 2 继续使用 MNIST。

每张图片：

```text
Channel = 1
Height  = 28
Width   = 28
```

因此输入 Tensor：

```text
[1, 28, 28]
```

与 Level 1 不同的是：

Level 1 很快将图片 Flatten：

```text
[1, 28, 28]
→
[784]
```

而 Level 2 在卷积阶段保持：

```text
[Channel, Height, Width]
```

这种空间结构。

---

## 4.2 数据划分

继续采用：

```text
Train      = 55000
Validation = 5000
Test       = 10000
```

并使用：

```text
Random Seed = 42
```

进行 Train / Validation 划分。

这样与 Level 1 保持相同的数据规模和划分方式，有利于进行实验比较。

---

## 4.3 DataLoader

Batch Size：

```text
64
```

因此正常 Training Batch：

```text
[64, 1, 28, 28]
```

设置：

```text
Train:
shuffle = True

Validation:
shuffle = False

Test:
shuffle = False
```

Train Set 在每个 Epoch 中打乱样本顺序，而 Validation / Test 不参与参数更新。

---

# 5. 方法与实验策略

## 5.1 整体 Pipeline

Level 2 的训练流程：

```text
MNIST
  ↓
DataLoader
  ↓
[B, 1, 28, 28]
  ↓
Conv2d
  ↓
ReLU
  ↓
MaxPool
  ↓
Conv2d
  ↓
ReLU
  ↓
MaxPool
  ↓
Flatten
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
10 Logits
  ↓
CrossEntropyLoss
  ↓
Backward
  ↓
Adam
```

---

## 5.2 CNN 网络结构

实际网络：

```text
Input
[B, 1, 28, 28]

        ↓

Conv2d(
    in_channels=1,
    out_channels=32,
    kernel_size=3,
    stride=1,
    padding=1
)

[B, 32, 28, 28]

        ↓
      ReLU
        ↓
MaxPool2d(2)

[B, 32, 14, 14]

        ↓

Conv2d(
    in_channels=32,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1
)

[B, 64, 14, 14]

        ↓
      ReLU
        ↓
MaxPool2d(2)

[B, 64, 7, 7]

        ↓
      Flatten

[B, 3136]

        ↓
Linear(3136, 128)
        ↓
      ReLU
        ↓
Linear(128, 10)

[B, 10]
```

参数量：

```text
Total Parameters     = 421,642
Trainable Parameters = 421,642
```

---

## 5.3 卷积核与 Feature Map

卷积层使用小型 Kernel 在图片上滑动。

第一层：

```text
Input Channels  = 1
Output Channels = 32
Kernel Size     = 3 × 3
```

意味着网络使用：

```text
32 个可学习卷积核
```

从输入灰度图中提取特征。

得到：

```text
32 个 Feature Maps
```

第二层：

```text
32 Channels
    ↓
Conv2d
    ↓
64 Channels
```

随着网络加深，不同 Channel 可以学习不同类型的特征表示。

---

## 5.4 Local Connectivity 与 Weight Sharing

CNN 与 MLP 的重要区别之一是：

```text
MLP:
一个神经元与上一层大量输入建立连接

CNN:
卷积核只观察一个局部区域
```

例如：

```text
3 × 3 Kernel
```

一次只处理局部的 `3 × 3` 区域。

这叫：

```text
Local Connectivity
```

同时，同一个卷积核会在整张图片上反复使用：

```text
同一组 Kernel Parameters
→ 扫描整张图片
```

这叫：

```text
Weight Sharing
```

因此 CNN 能够学习：

> 某种局部特征是否出现在图片的不同位置。

---

## 5.5 Stride 与 Padding

卷积层设置：

```text
kernel_size = 3
stride      = 1
padding     = 1
```

对于本实验的 `3 × 3` 卷积：

```text
28 × 28
   ↓
Conv
   ↓
28 × 28
```

空间尺寸保持不变。

因此第一层：

```text
[B, 1, 28, 28]
→
[B, 32, 28, 28]
```

改变的是：

```text
Channel
```

而不是 Height / Width。

---

## 5.6 Max Pooling

每个卷积阶段后使用：

```text
MaxPool2d(
    kernel_size=2,
    stride=2
)
```

因此：

```text
28 × 28
→
14 × 14
→
7 × 7
```

Pooling 的作用是进行空间下采样。

可以理解为：

```text
保留局部较强响应
+
减少 Feature Map 空间尺寸
```

所以整个 CNN 的空间尺寸变化为：

```text
28 × 28
   ↓
14 × 14
   ↓
7 × 7
```

---

## 5.7 Channel 与空间尺寸

本网络整体变化：

```text
Input:
1 × 28 × 28

↓

32 × 28 × 28

↓

32 × 14 × 14

↓

64 × 14 × 14

↓

64 × 7 × 7
```

可以看到：

```text
Spatial Size:
28 → 14 → 7

Channels:
1 → 32 → 64
```

这是一种典型 CNN 特征提取过程：

> 空间尺寸逐渐减小，而 Feature Channels 增加，用更多通道表示逐渐丰富的特征。

---

## 5.8 Flatten 与全连接分类

卷积结束后：

```text
[B, 64, 7, 7]
```

Flatten：

```text
64 × 7 × 7
=
3136
```

得到：

```text
[B, 3136]
```

随后：

```text
Linear(3136, 128)
ReLU
Linear(128, 10)
```

卷积部分负责：

```text
Feature Extraction
```

全连接部分负责：

```text
Classification
```

因此可以把 CNN 分成：

```text
Feature Extractor
+
Classifier
```

两部分理解。

---

## 5.9 Adam Optimizer

Level 1 使用：

```text
SGD
Learning Rate = 0.1
```

Level 2 使用：

```text
Adam
Learning Rate = 0.001
```

Adam 会根据梯度的一阶矩和二阶矩信息自适应调整不同参数的更新幅度。

在本实验中选择 Adam，主要目的是：

```text
减少调参时间
+
在较少 Epoch 内获得稳定训练结果
```

需要注意，由于 Level 1 与 Level 2 使用不同 Optimizer，所以不能把收敛速度差异全部归因于：

```text
MLP vs CNN
```

---

## 5.10 训练策略

最终配置：

| Parameter | Value |
|---|---|
| Python | `3.11.9` |
| PyTorch | `2.13.0+cu130` |
| Device | CUDA GPU |
| Batch Size | `64` |
| Epochs | `5` |
| Optimizer | Adam |
| Learning Rate | `0.001` |
| Loss | CrossEntropyLoss |
| Random Seed | `42` |

训练闭环仍然与 Level 1 一致：

```text
Forward
  ↓
CrossEntropyLoss
  ↓
Backward
  ↓
Gradient
  ↓
Adam
  ↓
Parameter Update
```

变化最大的部分不是训练框架，而是：

```text
Model Architecture
```

从：

```text
MLP
```

变成：

```text
CNN
```

---

# 6. 总结

Level 2 在 Level 1 的 PyTorch 分类训练框架基础上，将模型从 MLP 升级为 CNN。

模型结构：

```text
Input
 ↓
Conv 1→32
 ↓
MaxPool
 ↓
Conv 32→64
 ↓
MaxPool
 ↓
Flatten
 ↓
FC 3136→128
 ↓
FC 128→10
```

最终结果：

```text
Test Accuracy       = 98.70%
Parameters          = 421,642
Training Time       = 44.51 s
Best Val Accuracy   = 98.78%
```

相比 Level 1：

```text
MLP Test Accuracy = 96.36%
CNN Test Accuracy = 98.70%

Improvement = 2.34 percentage points
```

同时：

```text
MLP Parameters = 101,770
CNN Parameters = 421,642
```

说明本实验 CNN 的优势并不是来自更少的总参数，而是其结构更适合处理图像。

从 Level 1 到 Level 2，最重要的变化可以概括为：

```text
Level 1:
Image
 ↓
Flatten
 ↓
MLP
 ↓
Class

Level 2:
Image
 ↓
Convolution
 ↓
Spatial Feature Extraction
 ↓
Pooling
 ↓
Higher-level Features
 ↓
Classifier
 ↓
Class
```

通过本阶段，进一步理解了 CNN 为什么适合图像：

```text
局部连接
+
权重共享
+
二维空间结构
+
层次化特征提取
```

并第一次通过错误样本、Loss Curve、Accuracy Curve 和 MLP/CNN 对比，不只是观察“最终 Accuracy”，而是从多个角度分析模型的训练与泛化表现。

---

## 关键词总结

| 关键词 | 本 Level 中的含义 |
|---|---|
| CNN | 使用卷积进行空间特征提取的神经网络 |
| Kernel / Filter | 在局部区域进行计算的可学习卷积核 |
| Conv2d | PyTorch 二维卷积层 |
| Channel | Feature Map 的通道维度 |
| Feature Map | 卷积核提取得到的特征表示 |
| Local Connectivity | 卷积只连接局部区域 |
| Weight Sharing | 同一个卷积核参数在整张图片上共享 |
| Receptive Field | 一个特征所能感受到的输入区域 |
| Kernel Size | 卷积核尺寸 |
| Stride | 卷积核每次移动的距离 |
| Padding | 在输入边缘补充像素 |
| Max Pooling | 取局部最大值进行下采样 |
| Downsampling | 减小 Feature Map 空间尺寸 |
| Flatten | 将卷积 Feature Map 展开成向量 |
| Feature Extractor | CNN 中负责提取图像特征的部分 |
| Classifier | 根据提取的特征完成分类的部分 |
| Adam | 本实验使用的自适应优化器 |
| Overfitting | Training 继续改善而 Validation 不再同步改善 |
| Wrong Samples | 模型在 Test Set 中分类错误的样本 |
| Generalization | 模型对未参与训练数据的表现能力 |