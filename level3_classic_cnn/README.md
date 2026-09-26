# Level 3：经典 CNN 架构 —— AlexNet、VGG 与 ResNet

# 1. 任务目标

Level 3 在 Level 2 基础 CNN 的基础上，进一步实现并比较三种经典卷积神经网络架构：

- AlexNet-style
- VGG-style
- ResNet-18-style

三个模型均针对 `28 × 28` 的 MNIST 灰度图像进行了适配，而不是直接照搬原始 ImageNet 网络。

本阶段主要目标是理解：

```text
基础 CNN
   ↓
更深的卷积网络
   ↓
AlexNet
   ↓
VGG
   ↓
ResNet
```

并通过实际训练比较不同网络的：

```text
Test Accuracy
Parameters
Training Time
Loss / Accuracy Curve
Wrong Samples
```

本阶段三个模型统一采用相同的数据划分和主要训练配置，从而将实验重点放在：

```text
Network Architecture
```

最终 Test Accuracy：

```text
AlexNet-style   = 99.33%
VGG-style       = 99.06%
ResNet-18-style = 99.18%
```

作为对照，Level 2 Baseline CNN 为：

```text
98.70%
```

---

# 2. 项目文件结构

Level 3 的核心结构：

```text
level3_classic_cnn/
├── alexnet.py
├── vgg.py
├── resnet.py
├── README.md
│
└── outputs/
    ├── alexnet/
    │   ├── alexnet_mnist.pth
    │   ├── loss_curve.png
    │   ├── accuracy_curve.png
    │   ├── wrong_samples.png
    │   └── inference_example.png
    │
    ├── vgg/
    │   ├── vgg_mnist.pth
    │   ├── loss_curve.png
    │   ├── accuracy_curve.png
    │   ├── wrong_samples.png
    │   └── inference_example.png
    │
    └── resnet/
        ├── resnet_mnist.pth
        ├── loss_curve.png
        ├── accuracy_curve.png
        ├── wrong_samples.png
        └── inference_example.png
```

> `.pth` 文件为训练得到的模型参数文件，体积较大；其余输出主要用于记录和分析实验结果。

---

## 2.1 `alexnet.py`

实现适配 MNIST 的：

```text
AlexNet-style CNN
```

主要包含：

```text
5 个卷积层
MaxPooling
ReLU
Fully Connected Classifier
Dropout
```

同时负责：

```text
数据加载
训练
Validation
Test
模型保存
Loss Curve
Accuracy Curve
错误样本可视化
单张图片推理
```

---

## 2.2 `vgg.py`

实现：

```text
VGG-style CNN
```

核心特点是将多个：

```text
3 × 3 Conv
```

连续堆叠形成卷积 Block。

主要用于理解：

```text
Repeated Small Convolutions
        ↓
Deeper Feature Extraction
```

以及 VGG 规则化的网络设计思想。

---

## 2.3 `resnet.py`

实现：

```text
ResNet-18-style CNN
```

这是三个模型中结构上最特殊的一个。

主要包含：

```text
BasicBlock
Residual Connection
Shortcut
1 × 1 Conv
BatchNorm
Adaptive Average Pooling
```

并保留 ResNet-18 的 BasicBlock 数量：

```text
[2, 2, 2, 2]
```

同时针对 MNIST 的小尺寸输入修改网络 Stem。

---

## 2.4 `outputs/`

三个模型分别拥有自己的实验输出目录：

```text
outputs/alexnet/
outputs/vgg/
outputs/resnet/
```

每个目录保存：

```text
模型参数
Loss Curve
Accuracy Curve
Wrong Samples
Inference Example
```

这样可以分别观察三个模型的训练过程和最终效果。

---

# 3. 实验结果与结果分析

## 3.1 总体结果

首先比较 Level 2 Baseline CNN 和 Level 3 三种经典架构：

| Network | Test Accuracy | Parameters | Training Time | Best Val Accuracy |
|---|---:|---:|---:|---:|
| Baseline CNN | 98.70% | 421,642 | 44.51 s | 98.78% |
| AlexNet-style | **99.33%** | 3,564,490 | 58.28 s | 99.12% |
| VGG-style | 99.06% | 3,048,394 | 61.99 s | 98.94% |
| ResNet-18-style | 99.18% | 2,797,034 | 101.42 s | **99.18%** |

从本次实验可以看到：

```text
三种 Level 3 网络
Test Accuracy
均超过 Level 2 Baseline CNN
```

但与此同时：

```text
Parameters ↑
Training Cost ↑
```

而 Accuracy 的提升相对有限。

这说明：

> 在 MNIST 这样的相对简单任务上，基础 CNN 已经能够取得很高的准确率。增加网络复杂度仍然能够进一步提升结果，但模型规模和计算成本的增长并不会等比例转化为 Accuracy 提升。

同时，本次实验也说明：

```text
网络更深 / 更复杂
≠
Test Accuracy 一定更高
```

例如本次实验中 AlexNet-style 的 Test Accuracy 为 `99.33%`，高于 VGG-style 和 ResNet-18-style。

这只是当前：

```text
MNIST
+
当前网络实现
+
当前超参数
+
5 Epoch
```

下得到的实验结果，并不代表 AlexNet 在一般情况下优于 VGG 或 ResNet。

---

## 3.2 AlexNet-style

参数量：

```text
3,564,490
```

完整训练结果：

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Time |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.3644 | 87.79% | 0.1004 | 96.96% | 14.03 s |
| 2 | 0.0715 | 98.09% | 0.0423 | 98.82% | 11.12 s |
| 3 | 0.0536 | 98.63% | 0.0402 | 98.84% | 11.17 s |
| 4 | 0.0421 | 98.93% | 0.0414 | 98.94% | 11.08 s |
| 5 | 0.0337 | 99.11% | 0.0322 | 99.12% | 10.88 s |

最终：

```text
Test Accuracy       = 99.33%
Best Val Accuracy   = 99.12%
Total Training Time = 58.28 s
```

Loss 整体持续下降：

```text
Train Loss:
0.3644 → 0.0337

Validation Loss:
0.1004 → 0.0322
```

Accuracy 也持续提高：

```text
Train Accuracy:
87.79% → 99.11%

Validation Accuracy:
96.96% → 99.12%
```

说明 AlexNet-style 在 5 个 Epoch 内保持了较稳定的收敛过程。

相比 Level 2：

```text
Baseline CNN = 98.70%
AlexNet      = 99.33%

提升 = 0.63 个百分点
```

但模型参数量也由：

```text
421,642
```

增加到：

```text
3,564,490
```

约为 Baseline CNN 的 8.45 倍。

因此 AlexNet-style 获得了更高的分类准确率，同时也付出了明显更高的模型规模成本。

---

## 3.3 VGG-style

参数量：

```text
3,048,394
```

训练结果：

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Time |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.4777 | 83.67% | 0.0914 | 97.18% | 14.11 s |
| 2 | 0.0858 | 97.81% | 0.0778 | 98.04% | 11.99 s |
| 3 | 0.0612 | 98.37% | 0.0605 | 98.14% | 11.87 s |
| 4 | 0.0492 | 98.76% | 0.0527 | 98.74% | 12.12 s |
| 5 | 0.0431 | 98.93% | 0.0385 | 98.94% | 11.90 s |

最终：

```text
Test Accuracy       = 99.06%
Best Val Accuracy   = 98.94%
Total Training Time = 61.99 s
```

从曲线可以看到：

```text
Train Loss
Validation Loss
```

整体都持续下降。

同时：

```text
Train Accuracy      = 98.93%
Validation Accuracy = 98.94%
```

在第 5 个 Epoch 非常接近。

本次训练没有表现出明显的 Train / Validation 分离。

相比 Level 2：

```text
98.70% → 99.06%

提升 = 0.36 个百分点
```

VGG-style 的重点并不仅仅是最终 Accuracy，而是展示了一种非常规则的深层 CNN 设计方式：

```text
多个 3×3 Conv
      ↓
形成 Block
      ↓
MaxPool
      ↓
增加 Channel
      ↓
继续堆叠
```

---

## 3.4 ResNet-18-style

参数量：

```text
2,797,034
```

训练结果：

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Time |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.1093 | 96.71% | 0.0460 | 98.52% | 21.71 s |
| 2 | 0.0446 | 98.59% | 0.0476 | 98.70% | 20.80 s |
| 3 | 0.0350 | 98.91% | 0.0358 | 98.78% | 19.39 s |
| 4 | 0.0276 | 99.15% | 0.0470 | 98.78% | 19.28 s |
| 5 | 0.0248 | 99.25% | 0.0289 | 99.18% | 20.25 s |

最终：

```text
Test Accuracy       = 99.18%
Best Val Accuracy   = 99.18%
Total Training Time = 101.42 s
```

Train Loss：

```text
0.1093 → 0.0248
```

持续下降。

Validation Loss：

```text
0.0460
→ 0.0476
→ 0.0358
→ 0.0470
→ 0.0289
```

其中第 4 个 Epoch 出现了一次比较明显的回升。

但是与此同时：

```text
Val Accuracy:
98.52%
→ 98.70%
→ 98.78%
→ 98.78%
→ 99.18%
```

并没有出现持续下降。

第 5 个 Epoch：

```text
Val Loss     = 0.0289
Val Accuracy = 99.18%
```

均取得本次训练中的较好结果。

因此更合适的判断是：

> ResNet 的 Validation Loss 在训练过程中出现了短暂波动，但没有形成持续恶化趋势，整体训练仍然正常收敛。

不能仅根据第 4 个 Epoch 的一次 Loss 上升，就直接判断训练失败或者已经发生严重过拟合。

---

## 3.5 参数量与训练时间

三个 Level 3 网络：

```text
AlexNet:
3,564,490 parameters
58.28 s

VGG:
3,048,394 parameters
61.99 s

ResNet:
2,797,034 parameters
101.42 s
```

一个值得注意的现象是：

```text
ResNet 参数量
<
AlexNet 参数量
```

但：

```text
ResNet Training Time
>
AlexNet Training Time
```

说明：

> 参数量不是决定训练速度的唯一因素。

训练时间还会受到：

```text
网络深度
卷积运算数量
Feature Map 尺寸
Residual Block 数量
BatchNorm
GPU 运算特征
```

等因素影响。

因此不能简单认为：

```text
Parameters 少
=
训练一定更快
```

---

## 3.6 错误样本

三个模型均保存：

```text
wrong_samples.png
```

用于显示 Test Set 中的错误分类样本。

从错误样本可以看到，一些数字本身具有较强的视觉歧义，例如：

```text
2 → 7
6 → 0
3 → 5
7 → 1
7 → 9
8 → 2
```

常见情况包括：

```text
书写不规范
笔画倾斜
局部笔画缺失
不同数字形态接近
样本本身存在一定歧义
```

这说明：

> 即使 Test Accuracy 已经达到 99% 左右，剩余错误仍然值得分析，因为 Accuracy 本身不能告诉我们模型究竟在哪些输入上失败。

---

## 3.7 单张图片推理

三个模型均对 Test Set 中的单张图片进行了推理。

示例：

```text
True: 7
Pred: 7
```

完整过程：

```text
Single MNIST Image
        ↓
unsqueeze(0)
        ↓
Batch Dimension
        ↓
Model Forward
        ↓
10 Logits
        ↓
argmax
        ↓
Predicted Class
```

三个模型都能够完成从：

```text
训练
→ 保存参数
→ 测试
→ 单张图片推理
```

的完整流程。

---

# 4. 数据集

## 4.1 MNIST

Level 3 继续使用 MNIST 手写数字数据集。

输入：

```text
1 × 28 × 28
```

类别：

```text
0 ~ 9
```

这是一个：

```text
10-Class Image Classification
```

任务。

---

## 4.2 数据划分

三个模型均采用：

| Dataset | Number |
|---|---:|
| Train | 55,000 |
| Validation | 5,000 |
| Test | 10,000 |

Train / Validation 划分固定：

```python
torch.Generator().manual_seed(42)
```

因此三个模型使用相同的数据划分。

数据直接复用：

```text
../level1_mlp/data
```

不重复下载或在 Level 3 中保存一份新的 MNIST。

---

## 4.3 为什么保持相同数据划分

Level 3 的重点是比较：

```text
Network Architecture
```

因此需要尽量减少其他变量变化。

三个模型统一：

```text
Dataset
Train / Validation Split
Batch Size
Epochs
Optimizer
Learning Rate
Loss Function
Random Seed
```

这样实验之间的主要差异集中在：

```text
AlexNet-style
vs
VGG-style
vs
ResNet-18-style
```

需要注意，这仍然只是本项目中的实验比较，并不等价于对三种经典架构进行完整、严格的大规模 benchmark。

---

# 5. 方法与实验策略

## 5.1 统一训练配置

Level 3 三个模型采用：

| Configuration | Value |
|---|---|
| Dataset | MNIST |
| Train | 55,000 |
| Validation | 5,000 |
| Test | 10,000 |
| Batch Size | 64 |
| Epochs | 5 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Random Seed | 42 |
| Device | CUDA |

训练流程仍然与 Level 2 相同：

```text
Image
  ↓
Forward
  ↓
Logits
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

也就是说，从 Level 2 到 Level 3：

```text
训练框架没有发生本质变化
```

真正变化的是：

```text
Feature Extraction Architecture
```

---

## 5.2 AlexNet-style

原始 AlexNet 面向大型 RGB 图像。

MNIST 只有：

```text
1 × 28 × 28
```

因此本实验没有机械复制原始 AlexNet，而是保留其核心设计思想并适配 MNIST。

结构：

```text
Input
[B, 1, 28, 28]

      ↓ Conv 1→64
      ↓ ReLU
      ↓ MaxPool

[B, 64, 14, 14]

      ↓ Conv 64→192
      ↓ ReLU
      ↓ MaxPool

[B, 192, 7, 7]

      ↓ Conv 192→384
      ↓ ReLU

      ↓ Conv 384→256
      ↓ ReLU

      ↓ Conv 256→256
      ↓ ReLU
      ↓ MaxPool

[B, 256, 3, 3]

      ↓ Flatten

[B, 2304]

      ↓ Linear 2304→512
      ↓ ReLU
      ↓ Dropout

      ↓ Linear 512→256
      ↓ ReLU
      ↓ Dropout

      ↓ Linear 256→10

[B, 10]
```

相比 Level 2 CNN：

```text
2 Conv
```

AlexNet-style 使用更多卷积层进行特征提取。

分类器中还加入：

```text
Dropout(p=0.5)
```

用于在训练阶段随机屏蔽部分神经元输出，从而起到正则化作用。

---

## 5.3 VGG-style

VGG-style 的核心设计是：

```text
重复堆叠 3×3 Conv
```

而不是每进行一次卷积就立刻池化。

本实验结构可以概括为：

```text
Input

 ↓

Block 1
Conv 1→64
ReLU
Conv 64→64
ReLU
MaxPool

 ↓

Block 2
Conv 64→128
ReLU
Conv 128→128
ReLU
MaxPool

 ↓

Block 3
Conv 128→256
ReLU
Conv 256→256
ReLU
Conv 256→256
ReLU
MaxPool

 ↓

Classifier

 ↓

10 Classes
```

整体思想是：

```text
同一空间尺度
连续进行多次特征提取
        ↓
再进行下采样
        ↓
增加 Channel
        ↓
继续提取更复杂特征
```

相比 Level 2 的基础 CNN，VGG-style 更加系统地展示了：

```text
Deep CNN
=
Repeated Convolution Blocks
```

的设计方式。

---

## 5.4 ResNet-18-style

随着网络不断加深，仅仅继续堆叠普通卷积层会使网络优化更加困难。

ResNet 引入：

```text
Residual Connection
```

核心表达：

```text
H(x) = F(x) + x
```

其中：

```text
F(x)
```

是主分支通过卷积学习的变换；

```text
x
```

通过 Shortcut 传递。

本实验使用 ResNet-18 的 BasicBlock 数量：

```text
[2, 2, 2, 2]
```

但针对 MNIST 修改输入 Stem：

```text
3 × 3 Conv
stride = 1
```

而不是在网络刚开始时进行较强下采样。

原因是 MNIST 本身只有：

```text
28 × 28
```

如果过早大幅缩小空间尺寸，会过快损失图像空间信息。

---

## 5.5 BasicBlock 与 Residual Connection

BasicBlock 的核心结构：

```text
x -----------------------------+
|                               |
|                               |
↓                               |
Conv 3×3                        |
↓                               |
BatchNorm                       |
↓                               |
ReLU                            |
↓                               |
Conv 3×3                        |
↓                               |
BatchNorm                       |
↓                               |
F(x)                            |
|                               |
+------------- Add <------------+
               ↓
             ReLU
```

最终：

```text
Output = F(x) + Shortcut(x)
```

代码逻辑可以概括为：

```python
identity = self.shortcut(x)

out = self.conv1(x)
out = self.bn1(out)
out = self.relu(out)

out = self.conv2(out)
out = self.bn2(out)

out = out + identity
out = self.relu(out)
```

这也是 ResNet 和普通 Sequential CNN 最明显的结构区别。

---

## 5.6 Shortcut 与 1×1 Conv

如果：

```text
F(x)
```

与：

```text
x
```

Shape 相同，可以直接：

```text
F(x) + x
```

此时 Shortcut 使用：

```text
Identity
```

但是当：

```text
Channel 改变
```

或者：

```text
Stride = 2
```

时，两边 Shape 不一致。

例如：

```text
[B, 32, 28, 28]
        ↓
[B, 64, 14, 14]
```

此时不能直接进行逐元素相加。

因此 Shortcut 使用：

```text
1 × 1 Conv
+
对应的 Stride
```

把输入调整为相同 Shape：

```text
Main Branch:
[B, 64, 14, 14]

Shortcut:
[B, 64, 14, 14]

        ↓

Element-wise Addition
```

需要特别区分：

```text
ResNet Shortcut
→ Addition
```

而后续 U-Net 中常见的 Skip Connection：

```text
U-Net Skip Connection
→ Concatenation
```

二者虽然都跨越网络层传递信息，但具体操作和目的并不完全相同。

---

## 5.7 Batch Normalization

ResNet BasicBlock 中大量使用：

```text
BatchNorm2d
```

典型组合：

```text
Conv
 ↓
BatchNorm
 ↓
ReLU
```

BatchNorm 会对中间特征进行规范化处理，并具有可学习的缩放和平移参数。

在本项目中，它是 ResNet BasicBlock 的重要组成部分。

同时这也意味着：

```text
model.train()
```

与：

```text
model.eval()
```

在 ResNet 中非常重要，因为 BatchNorm 在训练阶段和评估阶段的行为不同。

---

## 5.8 Global Average Pooling

ResNet 最后没有直接将：

```text
[B, 256, 4, 4]
```

展开成：

```text
[B, 4096]
```

再连接大型全连接层。

而是使用：

```python
nn.AdaptiveAvgPool2d((1, 1))
```

得到：

```text
[B, 256, 4, 4]

        ↓

[B, 256, 1, 1]

        ↓ Flatten

[B, 256]

        ↓ Linear

[B, 10]
```

这种方式可以显著减小分类器部分的参数规模。

这也是为什么本实验中：

```text
ResNet Parameters = 2,797,034
```

反而低于：

```text
AlexNet Parameters = 3,564,490
```

尽管 ResNet 的网络计算结构更加复杂。

---

## 5.9 三种网络的核心区别

可以把 Level 3 三种架构概括为：

| Architecture | 本实验中重点理解的思想 |
|---|---|
| AlexNet-style | 更深的 Conv + ReLU + Pooling，并在分类器中使用 Dropout |
| VGG-style | 使用规则的多个 `3×3 Conv` 组成深层卷积 Block |
| ResNet-18-style | 使用 Residual Block、Shortcut 和 `F(x)+x` 优化深层网络 |

它们并不是三套完全不同的机器学习理论。

共同基础仍然是：

```text
CNN
+
Forward
+
Loss
+
Backward
+
Optimizer
```

区别主要在：

```text
如何组织卷积层
如何传递特征
如何构建更深的网络
```

---

## 5.10 从 Level 2 到 Level 3

Level 2：

```text
Input
 ↓
Conv
 ↓
Pool
 ↓
Conv
 ↓
Pool
 ↓
FC
 ↓
Output
```

Level 3：

```text
             ┌→ AlexNet-style
             │   更深的卷积网络
             │
Baseline CNN ├→ VGG-style
             │   规则堆叠 3×3 Conv
             │
             └→ ResNet-style
                 Residual + Shortcut
```

因此 Level 3 的重点已经不再是：

```text
“CNN 是什么？”
```

而是进一步思考：

```text
CNN 怎样变深？

深层网络怎样组织？

不同架构为什么要这样设计？

更复杂的模型是否一定得到更高 Accuracy？
```

---

# 6. 总结

Level 3 在 Level 2 基础 CNN 的基础上，实现并训练了：

```text
AlexNet-style
VGG-style
ResNet-18-style
```

三个模型均使用：

```text
MNIST
Train      = 55,000
Validation = 5,000
Test       = 10,000

Batch Size    = 64
Epochs        = 5
Learning Rate = 0.001
Optimizer     = Adam
Loss          = CrossEntropyLoss
```

最终结果：

```text
Baseline CNN:
98.70%

AlexNet-style:
99.33%

VGG-style:
99.06%

ResNet-18-style:
99.18%
```

本阶段最重要的收获并不是单纯将 MNIST Accuracy 从：

```text
98.70%
```

提升到：

```text
99%+
```

而是通过三个实际实现理解了深层 CNN 的不同设计思路：

```text
AlexNet
→ 更深的卷积特征提取
→ Dropout

VGG
→ 规则堆叠 3×3 Conv
→ Block 化结构

ResNet
→ Residual Learning
→ Shortcut
→ F(x) + x
→ BatchNorm
→ Global Average Pooling
```

同时实验也说明：

```text
模型更复杂
≠
准确率一定严格更高

参数更多
≠
训练一定更慢或更快

单独观察 Test Accuracy
≠
完整理解模型表现
```

还需要结合：

```text
Validation
Loss Curve
Accuracy Curve
Parameters
Training Time
Wrong Samples
```

共同分析。

从整个项目学习路径来看：

```text
Level 1
MLP
↓
建立基本神经网络训练流程

Level 2
CNN
↓
学习图像空间特征提取

Level 3
AlexNet / VGG / ResNet
↓
理解深层 CNN 的结构设计

Level 4
U-Net
↓
从 Image Classification
进入 Image-to-Image
```

因此 Level 3 也是从：

```text
“会使用 CNN”
```

进一步走向：

```text
“理解 CNN 网络为什么可以有不同的架构设计”
```

的重要阶段。

---

## 关键词总结

| 关键词 | 本 Level 中的含义 |
|---|---|
| AlexNet | 经典深层 CNN 架构，本实验实现 MNIST 适配版本 |
| VGG | 使用规则的小卷积核堆叠构建深层 CNN |
| ResNet | 使用 Residual Learning 构建深层网络 |
| Dropout | 训练时随机屏蔽部分神经元输出进行正则化 |
| VGG Block | 多个 `3×3 Conv + ReLU` 后进行 Pooling |
| BasicBlock | ResNet 中的基础残差模块 |
| Residual | 主分支学习的 `F(x)` |
| Shortcut | 将输入跨层传递到残差加法位置 |
| Identity | Shape 相同时直接传递输入 |
| 1×1 Conv | Shortcut 中用于调整 Channel / 空间尺寸 |
| Addition | ResNet 中主分支与 Shortcut 的逐元素相加 |
| BatchNorm | ResNet 中用于中间特征规范化的模块 |
| Global Average Pooling | 将每个 Channel 的空间信息压缩到 `1×1` |
| AdaptiveAvgPool2d | PyTorch 中实现自适应平均池化的模块 |
| Deep CNN | 通过更多卷积层构建更深的特征提取网络 |
| Wrong Samples | Test Set 中模型预测错误的样本 |
| Baseline | 用于后续实验比较的基础模型 |
```