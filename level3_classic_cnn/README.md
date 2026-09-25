# Level 3：经典 CNN 网络 —— AlexNet、VGG、ResNet

## 1. 任务目标

本阶段在 Level 2 CNN 的基础上继续学习经典卷积神经网络架构。

主要目标：

1. 理解 CNN、AlexNet、VGG、ResNet、U-Net 的发展关系与核心思想。
2. 理解不同网络之间的结构差异。
3. 在 Level 2 MNIST CNN 代码基础上实现：
   - AlexNet-style
   - VGG-style
   - ResNet-18-style
4. 使用相同的 MNIST 数据集和训练配置进行实验。
5. 将三个经典网络与 Level 2 Baseline CNN 从以下方面进行比较：
   - Test Accuracy
   - 参数量
   - 收敛过程
   - 训练时间
6. 理解 ResNet 中 Residual Connection / Shortcut 的实现方式。
7. 为后续 U-Net 图像到图像任务建立基础。

---

# 2. 从 CNN 到经典网络

Level 2 已经实现了一个基础 CNN：

```text
Input
 ↓
Conv
 ↓
ReLU
 ↓
Pool
 ↓
Conv
 ↓
ReLU
 ↓
Pool
 ↓
Flatten
 ↓
FC
 ↓
Output
```

AlexNet、VGG 和 ResNet 并不是脱离 CNN 的全新网络类型。

它们本质上仍然属于卷积神经网络，只是使用了不同的网络结构设计。

可以将它们的发展关系简单理解为：

```text
CNN
 ↓
利用局部连接和权重共享提取图像空间特征

AlexNet
 ↓
证明深层 CNN 可以在大型视觉任务中取得很强的效果

VGG
 ↓
使用大量规则堆叠的 3×3 卷积构建更深网络

网络继续加深
 ↓
深层网络优化变得更加困难

ResNet
 ↓
引入 Residual Connection
通过 F(x) + x 改善深层网络的优化

分类任务
 ↓
最终只需要类别信息

像素级 / 图像到图像任务
 ↓
还需要恢复空间信息

U-Net
 ↓
Encoder + Decoder + Skip Connection
```

---

# 3. 数据集

实验继续使用 MNIST 手写数字数据集。

图像：

```text
1 × 28 × 28
```

类别：

```text
0 ~ 9
```

数据划分：

| Dataset | Number |
|---|---:|
| Train | 55,000 |
| Validation | 5,000 |
| Test | 10,000 |

为了保证实验之间具有可比性，数据划分使用：

```python
torch.Generator().manual_seed(42)
```

因此 Baseline CNN、AlexNet、VGG、ResNet 使用相同的数据划分。

---

# 4. 实验配置

为了尽量进行控制变量比较，四个网络使用相近的训练设置。

Level 3 三个网络统一使用：

```text
Batch Size     = 64
Learning Rate  = 0.001
Epochs         = 5
Optimizer      = Adam
Loss Function  = CrossEntropyLoss
Random Seed    = 42
Device         = CUDA
```

训练基本流程没有因为网络结构变化而改变：

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
Update Parameters
```

因此本阶段最主要的变量是：

```text
Network Architecture
```

---

# 5. Baseline CNN

Level 2 的 CNN 作为本阶段实验 Baseline。

网络结构：

```text
Input
[B, 1, 28, 28]

 ↓ Conv 1→32

[B, 32, 28, 28]

 ↓ ReLU
 ↓ MaxPool

[B, 32, 14, 14]

 ↓ Conv 32→64

[B, 64, 14, 14]

 ↓ ReLU
 ↓ MaxPool

[B, 64, 7, 7]

 ↓ Flatten

[B, 3136]

 ↓ Linear

[B, 128]

 ↓ ReLU
 ↓ Linear

[B, 10]
```

Baseline CNN 的特点是结构简单，仅使用两个卷积层进行特征提取。

---

# 6. AlexNet-style

## 6.1 核心思想

AlexNet 是经典深层 CNN 架构之一。

与简单 CNN 相比，其主要特点包括：

- 更深的卷积结构
- 多层 Conv + ReLU
- MaxPooling
- Fully Connected Classifier
- Dropout

由于原始 AlexNet 面向大型 RGB 图像，而 MNIST 只有：

```text
1 × 28 × 28
```

因此本实验没有机械复制原始 AlexNet，而是实现了：

```text
AlexNet-style adapted for MNIST
```

保留其核心结构思想，同时调整输入通道和网络尺寸。

---

## 6.2 网络结构

```text
Input
[B, 1, 28, 28]

 ↓ Conv

[B, 64, 28, 28]

 ↓ ReLU
 ↓ MaxPool

[B, 64, 14, 14]

 ↓ Conv

[B, 192, 14, 14]

 ↓ ReLU
 ↓ MaxPool

[B, 192, 7, 7]

 ↓ Conv

[B, 384, 7, 7]

 ↓ ReLU

 ↓ Conv

[B, 256, 7, 7]

 ↓ ReLU

 ↓ Conv

[B, 256, 7, 7]

 ↓ ReLU
 ↓ MaxPool

[B, 256, 3, 3]

 ↓ Flatten

[B, 2304]

 ↓ FC
 ↓ ReLU
 ↓ Dropout
 ↓ FC
 ↓ ReLU
 ↓ Dropout
 ↓ FC

[B, 10]
```

---

## 6.3 Dropout

AlexNet-style 分类器中加入：

```python
nn.Dropout(p=0.5)
```

Dropout 在训练阶段随机屏蔽部分神经元输出，使网络减少对某些特定神经元组合的依赖，从而起到正则化作用。

因此：

```python
model.train()
```

和：

```python
model.eval()
```

不仅是形式上的区别。

训练模式下 Dropout 生效，而评估模式下采用对应的推理行为。

---

# 7. VGG-style

## 7.1 核心思想

VGG 的一个重要特点是：

```text
大量规则堆叠的 3×3 Conv
```

相比 AlexNet，VGG 的网络结构更加规则。

典型 Block：

```text
Conv 3×3
 ↓
ReLU
 ↓
Conv 3×3
 ↓
ReLU
 ↓
MaxPool
```

多个小卷积连续堆叠，可以逐层提取更加复杂的特征，同时在卷积层之间加入非线性激活。

本实验实现：

```text
VGG-style adapted for MNIST
```

---

## 7.2 网络结构

```text
Input
[B, 1, 28, 28]

========== Block 1 ==========

Conv 1→64
 ↓
ReLU
 ↓
Conv 64→64
 ↓
ReLU
 ↓
MaxPool

[B, 64, 14, 14]

========== Block 2 ==========

Conv 64→128
 ↓
ReLU
 ↓
Conv 128→128
 ↓
ReLU
 ↓
MaxPool

[B, 128, 7, 7]

========== Block 3 ==========

Conv 128→256
 ↓
ReLU
 ↓
Conv 256→256
 ↓
ReLU
 ↓
Conv 256→256
 ↓
ReLU
 ↓
MaxPool

[B, 256, 3, 3]

========== Classifier ==========

Flatten
 ↓
FC
 ↓
ReLU
 ↓
Dropout
 ↓
FC
 ↓
ReLU
 ↓
Dropout
 ↓
FC

[B, 10]
```

---

# 8. ResNet-18-style

## 8.1 为什么需要 ResNet？

随着 CNN 不断加深，深层网络的优化会变得更加困难。

需要注意：

```text
网络更深
≠
一定更容易训练
≠
一定得到更高准确率
```

ResNet 的核心思想是 Residual Learning。

普通网络学习：

```text
H(x)
```

ResNet 将其写成：

```text
H(x) = F(x) + x
```

其中：

```text
F(x)
```

由卷积层学习，而：

```text
x
```

通过 Shortcut 直接传递。

---

# 9. BasicBlock

本实验中的 BasicBlock 可以表示为：

```text
x --------------------------+
|                            |
↓                            |
Conv 3×3                     |
↓                            |
BatchNorm                    |
↓                            |
ReLU                         |
↓                            |
Conv 3×3                     |
↓                            |
BatchNorm                    |
↓                            |
F(x)                         |
|                            |
+----------- Add <-----------+
             ↓
           ReLU
```

最终：

```text
Output = F(x) + x
```

代码中的核心逻辑：

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

这是 ResNet 与普通 Sequential CNN 最关键的代码差异之一。

---

# 10. Shortcut 与 1×1 Conv

如果：

```text
F(x)
```

和：

```text
x
```

具有相同 Shape，则 Shortcut 可以直接使用：

```python
nn.Identity()
```

但是当网络进行下采样或改变 Channel 时，例如：

```text
x:

[B, 64, 14, 14]

F(x):

[B, 128, 7, 7]
```

两者无法直接相加。

此时 Shortcut 使用：

```text
1×1 Conv
```

并配合相应的 stride：

```text
[B, 64, 14, 14]

 ↓ 1×1 Conv
 ↓ stride = 2

[B, 128, 7, 7]
```

这样：

```text
F(x) + Shortcut(x)
```

两侧 Shape 相同，可以逐元素相加。

---

# 11. ResNet-18-style 网络结构

本实验保留 ResNet-18 的经典 BasicBlock 数量：

```text
[2, 2, 2, 2]
```

但为了适配 28×28 MNIST，修改了输入 Stem。

原始 ResNet 面向较大的 RGB 图像，而 MNIST 图像非常小，因此本实验使用：

```text
3×3 Conv
stride = 1
```

避免在网络入口处过早进行强烈下采样。

整体数据流：

```text
Input
[B, 1, 28, 28]

 ↓ Stem

[B, 32, 28, 28]

 ↓ Layer 1
   2 × BasicBlock

[B, 32, 28, 28]

 ↓ Layer 2
   2 × BasicBlock
   Downsampling

[B, 64, 14, 14]

 ↓ Layer 3
   2 × BasicBlock
   Downsampling

[B, 128, 7, 7]

 ↓ Layer 4
   2 × BasicBlock
   Downsampling

[B, 256, 4, 4]

 ↓ Adaptive Average Pooling

[B, 256, 1, 1]

 ↓ Flatten

[B, 256]

 ↓ FC

[B, 10]
```

---

# 12. Global Average Pooling

ResNet 中使用：

```python
nn.AdaptiveAvgPool2d((1, 1))
```

例如：

```text
[B, 256, 4, 4]

 ↓ Global Average Pooling

[B, 256, 1, 1]
```

然后：

```text
Flatten

↓

[B, 256]
```

再进入最终分类层。

相比直接将整个 Feature Map Flatten 后连接一个很大的 Fully Connected Layer，这种方式可以减少分类器部分的参数量。

---

# 13. Batch Normalization

ResNet 中大量使用：

```python
nn.BatchNorm2d(...)
```

BatchNorm 对中间激活进行规范化处理，同时具有可学习的缩放和平移参数。

在深层网络中，BatchNorm 通常有助于改善训练稳定性和优化过程。

因此 ResNet BasicBlock 中常见：

```text
Conv
 ↓
BatchNorm
 ↓
ReLU
```

这样的组合。

---

# 14. 四个网络的共同点

虽然四个网络结构不同，但训练机制完全可以使用同一个框架：

```python
optimizer.zero_grad()

outputs = model(images)

loss = criterion(outputs, labels)

loss.backward()

optimizer.step()
```

因此：

```text
Baseline CNN
AlexNet
VGG
ResNet
```

并不是四套完全不同的机器学习理论。

它们共同遵循：

```text
Input Image
 ↓
Feature Extraction
 ↓
Classification
 ↓
Logits
 ↓
Loss
 ↓
Backward
 ↓
Gradient
 ↓
Optimizer
 ↓
Parameter Update
```

最主要的区别在于：

```text
Feature Extraction Architecture
```

---

# 15. 实验结果

## 15.1 Baseline CNN

Level 2 实验结果：

```text
Parameters:
421,642

Test Accuracy:
98.70%

Total Training Time:
44.51s
```

---

## 15.2 AlexNet-style

参数量：

```text
3,564,490
```

训练结果：

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Time |
|---|---:|---:|---:|---:|---:|
| 1 | 0.3644 | 87.79% | 0.1004 | 96.96% | 14.03s |
| 2 | 0.0715 | 98.09% | 0.0423 | 98.82% | 11.12s |
| 3 | 0.0536 | 98.63% | 0.0402 | 98.84% | 11.17s |
| 4 | 0.0421 | 98.93% | 0.0414 | 98.94% | 11.08s |
| 5 | 0.0337 | 99.11% | 0.0322 | 99.12% | 10.88s |

最终：

```text
Test Accuracy:
99.33%

Total Training Time:
58.28s
```

---

## 15.3 VGG-style

参数量：

```text
3,048,394
```

训练结果：

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Time |
|---|---:|---:|---:|---:|---:|
| 1 | 0.4777 | 83.67% | 0.0914 | 97.18% | 14.11s |
| 2 | 0.0858 | 97.81% | 0.0778 | 98.04% | 11.99s |
| 3 | 0.0612 | 98.37% | 0.0605 | 98.14% | 11.87s |
| 4 | 0.0492 | 98.76% | 0.0527 | 98.74% | 12.12s |
| 5 | 0.0431 | 98.93% | 0.0385 | 98.94% | 11.90s |

最终：

```text
Test Accuracy:
99.06%

Total Training Time:
61.99s
```

---

## 15.4 ResNet-18-style

参数量：

```text
2,797,034
```

训练结果：

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Time |
|---|---:|---:|---:|---:|---:|
| 1 | 0.1093 | 96.71% | 0.0460 | 98.52% | 21.71s |
| 2 | 0.0446 | 98.59% | 0.0476 | 98.70% | 20.80s |
| 3 | 0.0350 | 98.91% | 0.0358 | 98.78% | 19.39s |
| 4 | 0.0276 | 99.15% | 0.0470 | 98.78% | 19.28s |
| 5 | 0.0248 | 99.25% | 0.0289 | 99.18% | 20.25s |

最终：

```text
Test Accuracy:
99.18%

Total Training Time:
101.42s
```

---

# 16. 四网络最终对比

| Network | Test Accuracy | Parameters | Training Time | Epoch 1 Val Acc | Best Val Acc |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | 98.70% | 421,642 | 44.51s | 97.78% | 98.78% |
| AlexNet-style | 99.33% | 3,564,490 | 58.28s | 96.96% | 99.12% |
| VGG-style | 99.06% | 3,048,394 | 61.99s | 97.18% | 98.94% |
| ResNet-18-style | 99.18% | 2,797,034 | 101.42s | 98.52% | 99.18% |

---

# 17. 实验结果分析

## 17.1 Baseline CNN 与复杂网络

Baseline CNN 只有：

```text
421,642 parameters
```

但已经取得：

```text
98.70% Test Accuracy
```

说明 MNIST 是一个相对简单的图像分类任务，一个规模较小的 CNN 已经能够取得较高准确率。

更复杂的网络确实进一步提高了本次实验中的最终测试准确率，但提升幅度并没有与参数量增长成比例。

---

## 17.2 AlexNet-style

AlexNet-style：

```text
Test Accuracy = 99.33%
Parameters    = 3,564,490
Training Time = 58.28s
```

相比 Baseline CNN：

```text
Accuracy:
98.70% → 99.33%

提升：
0.63 个百分点
```

但参数量约增加到 Baseline 的：

```text
8.45 倍
```

因此更复杂的网络带来了更强的特征提取能力，同时也增加了模型规模和计算成本。

在本次实验中，AlexNet-style 获得了四个模型中最高的 Test Accuracy。

这只是当前 MNIST、当前超参数和当前训练轮数下的实验结果，并不意味着 AlexNet 在所有任务上都优于其他网络。

---

## 17.3 VGG-style

VGG-style：

```text
Test Accuracy = 99.06%
Parameters    = 3,048,394
Training Time = 61.99s
```

其 Test Accuracy 高于 Baseline CNN，但低于本次实验中的 AlexNet-style 和 ResNet-18-style。

这说明：

```text
网络更深 / 结构更复杂
```

并不意味着：

```text
在所有数据集和实验条件下准确率一定严格更高
```

VGG 的重要价值之一在于其规则的网络设计：

```text
Repeated 3×3 Convolution Blocks
```

这种结构使深层 CNN 的组织方式非常清晰。

---

## 17.4 ResNet-18-style

ResNet-18-style：

```text
Test Accuracy = 99.18%
Parameters    = 2,797,034
Training Time = 101.42s
```

ResNet 参数量少于本实验中的 AlexNet-style 和 VGG-style，但训练时间反而最长。

这说明：

```text
Parameter Count
```

和：

```text
Actual Computational Cost
```

不是同一个概念。

ResNet 包含大量卷积、BatchNorm 和 Residual Block，因此即使参数数量较少，也可能需要更多计算。

ResNet 的核心价值不是保证在任何任务上都得到最高 Accuracy，而是通过 Residual Connection 改善深层网络的优化，使构建和训练更深网络成为可能。

---

# 18. ResNet 第 4 个 Epoch 的验证波动

ResNet 在 Epoch 3 → Epoch 4 时出现：

```text
Train Loss:
0.0350 → 0.0276

Train Acc:
98.91% → 99.15%
```

训练集继续改善。

但 Validation：

```text
Val Loss:
0.0358 → 0.0470

Val Acc:
98.78% → 98.78%
```

Validation Loss 出现上升。

这表示该轮参数更新进一步改善了训练集表现，但没有同步改善验证集上的泛化表现。

这种现象可能表现出暂时性的过拟合倾向，也可能属于正常训练波动。

不能仅凭一个 Epoch 就判断模型已经进入持续性过拟合，因为 Epoch 5：

```text
Val Loss:
0.0470 → 0.0289

Val Acc:
98.78% → 99.18%
```

验证性能再次明显改善，并达到当前最佳结果。

因此本实验更适合将 Epoch 4 描述为一次验证性能波动，而不是代码错误或确定性的持续过拟合。

---

# 19. 为什么 Accuracy 不变但 Loss 会变化？

Epoch 3 和 Epoch 4：

```text
Val Accuracy:
98.78% → 98.78%
```

基本不变。

但：

```text
Val Loss:
0.0358 → 0.0470
```

明显变化。

原因是 Accuracy 主要关心：

```text
预测类别是否正确
```

而 CrossEntropyLoss 还与模型输出的类别相对置信程度有关。

例如两个模型都预测正确：

```text
Model A:
正确类别相对分数只略高于其他类别

Model B:
正确类别相对分数远高于其他类别
```

两者 Accuracy 都记作正确，但 Loss 可以不同。

同样，如果模型对一个错误答案非常自信，也可能产生较大的 Loss。

因此：

```text
Loss
```

和：

```text
Accuracy
```

衡量的是不同方面，训练时同时观察两者更加合理。

---

# 20. 参数量并不代表一切

本实验参数量：

```text
Baseline CNN
421,642

ResNet
2,797,034

VGG
3,048,394

AlexNet
3,564,490
```

但训练时间：

```text
Baseline CNN
44.51s

AlexNet
58.28s

VGG
61.99s

ResNet
101.42s
```

参数最多的 AlexNet 并不是训练最慢的模型。

因此模型复杂度不能只通过：

```text
Number of Parameters
```

判断。

实际计算成本还受到以下因素影响：

```text
卷积层数量
Feature Map 大小
Channel 数量
卷积运算次数
BatchNorm
网络连接方式
硬件并行效率
```

等因素影响。

---

# 21. CNN、AlexNet、VGG、ResNet 的代码关系

本阶段最重要的理解之一：

四种网络不需要四套训练系统。

训练框架：

```python
outputs = model(images)

loss = criterion(outputs, labels)

optimizer.zero_grad()

loss.backward()

optimizer.step()
```

可以基本保持不变。

主要变化的是：

```python
model
```

即：

```text
Network Architecture
```

Baseline CNN：

```text
简单 Conv + Pool
```

AlexNet：

```text
更多 Conv
+
Pooling
+
Dropout
```

VGG：

```text
重复的 3×3 Conv Blocks
```

ResNet：

```text
BasicBlock
+
Residual Connection
+
Shortcut
+
BatchNorm
```

因此可以将 PyTorch 图像分类程序理解为两个部分：

```text
通用训练框架
+
可替换的网络模型
```

---

# 22. 图像在 CNN 中的完整处理流程

一张图像从文件进入网络直到参数更新，可以表示为：

```text
Image File
 ↓
Decode
 ↓
Pixel Data
 ↓
ToTensor
 ↓
[C, H, W]
 ↓
DataLoader
 ↓
[B, C, H, W]
 ↓
GPU
 ↓
Convolution
 ↓
Feature Map
 ↓
Activation
 ↓
Downsampling / Pooling
 ↓
Deeper Feature Representation
 ↓
Classifier
 ↓
Logits
 ↓
CrossEntropyLoss
 ↓
Autograd
 ↓
Backward
 ↓
Gradients
 ↓
Optimizer
 ↓
Update Kernels / Weights
 ↓
Next Batch
```

其中卷积核参数并不是人工指定成：

```text
边缘检测器
纹理检测器
数字检测器
```

而是在训练过程中通过：

```text
Loss
 ↓
Backward
 ↓
Gradient
 ↓
Optimizer
```

逐渐学习出来的。

---

# 23. 卷积中的 Channel

对于：

```python
nn.Conv2d(
    in_channels=Cin,
    out_channels=Cout,
    kernel_size=K
)
```

卷积权重 Shape 为：

```text
[Cout, Cin, K, K]
```

一个输出 Filter 会同时观察所有输入 Channel，并产生一个输出 Feature Map。

因此：

```text
out_channels
```

决定输出 Feature Map 的数量。

例如：

```python
nn.Conv2d(
    64,
    128,
    kernel_size=3
)
```

表示：

```text
64 input channels
 ↓
128 output filters
 ↓
128 output feature maps
```

所以卷积不仅能够提取空间特征，也可以改变 Channel 数量。

---

# 24. Downsampling 与 Channel

Downsampling 的本质是：

```text
H ↓
W ↓
```

例如：

```text
28×28
 ↓
14×14
 ↓
7×7
```

Pooling 可以实现 Downsampling，但：

```text
Downsampling != Pooling
```

因为 stride > 1 的卷积也可以进行下采样。

很多 CNN 常采用：

```text
H, W ↓
C ↑
```

例如：

```text
[B, 32, 28, 28]

↓

[B, 64, 14, 14]

↓

[B, 128, 7, 7]
```

但需要注意：

```text
Downsampling 本身并不规定 Channel 必须增加。
```

Channel 的改变通常由卷积层决定。

---

# 25. ResNet Skip Connection 与 U-Net Skip Connection

Skip Connection 是一个更广泛的概念。

ResNet 中：

```text
x
↓
F(x)

同时：

x ──────────────┐
                ↓
             F(x)+x
```

通常采用：

```text
Addition
```

核心目标之一是改善深层网络优化和信息传播。

而经典 U-Net 中：

```text
Encoder Feature
        │
        └────────────→ Decoder
```

通常采用：

```python
torch.cat(...)
```

即：

```text
Concatenation
```

其主要作用是将 Encoder 中较高分辨率的空间细节传递给 Decoder。

因此：

```text
ResNet Skip:
通常 Add

U-Net Skip:
经典结构通常 Concatenate
```

两者都属于 Skip Connection，但目的和实现方式不同。

---

# 26. 从分类到 U-Net

当前网络：

```text
CNN
AlexNet
VGG
ResNet
```

最终目标都是：

```text
Image
 ↓
10-dimensional logits
 ↓
Class
```

也就是说：

```text
Image → Label
```

但后续手写内容擦除任务需要：

```text
带手写内容的图像
 ↓
Network
 ↓
清理后的图像
```

即：

```text
Image → Image
```

分类网络会不断压缩空间信息，而图像恢复任务最终还需要恢复原始空间分辨率。

因此需要：

```text
Encoder
 ↓
Bottleneck
 ↓
Decoder
```

Encoder：

```text
Downsampling
H/W ↓
提取高层特征
```

Decoder：

```text
Upsampling
H/W ↑
恢复图像空间结构
```

但是单纯 Upsampling 无法自动恢复 Encoder 下采样过程中损失的全部细节。

因此 U-Net 使用：

```text
Skip Connection
```

将 Encoder 的高分辨率 Feature Map 直接传递到对应 Decoder 层。

整体结构：

```text
Encoder
   │
   │ Skip
   ├──────────────────────┐
   ↓                      │
Downsampling              │
   ↓                      │
Encoder                   │
   │                      │
   ├───────────────┐      │
   ↓               │      │
Bottleneck         │      │
   ↓               │      │
Decoder ←──────────┘      │
   ↓                      │
Upsampling                │
   ↓                      │
Decoder ←─────────────────┘
   ↓
Output Image
```

这为下一阶段的图像到图像任务建立了网络结构基础。

---

# 27. 本阶段总结

通过 Level 3，我完成了从基础 CNN 到经典深层 CNN 架构的学习和实验。

主要理解：

```text
1. AlexNet、VGG、ResNet 本质上仍然属于 CNN。

2. 网络架构决定 Conv、Pooling、Channel 和连接方式如何组织。

3. AlexNet 使用更深的卷积结构，并在分类器中使用 Dropout。

4. VGG 使用大量规则堆叠的 3×3 Conv。

5. 网络更深并不意味着一定更容易优化。

6. ResNet 使用 Residual Learning：

   H(x) = F(x) + x

7. Shortcut Shape 不一致时，可以使用 1×1 Conv 调整。

8. ResNet 的 Skip Connection 通常采用 Addition。

9. U-Net 的 Skip Connection 通常采用 Concatenation。

10. Downsampling 主要改变空间分辨率 H/W，
    Channel 的改变由网络结构另外决定。

11. 参数量并不能完全代表实际训练计算成本。

12. 更复杂的网络并不保证在简单数据集上获得同比例的性能提升。

13. Loss 和 Accuracy 衡量的是不同方面，
    Validation Loss 的短暂上升并不一定意味着训练失败。

14. 不同 CNN 架构可以复用相同的：

    Dataset
    DataLoader
    Loss
    Backward
    Optimizer
    Train / Validation / Test

15. 网络实验应该尽量控制变量，
    再比较不同 Architecture 的实际表现。
```

最终，本阶段建立了：

```text
MLP
 ↓
CNN
 ↓
AlexNet
 ↓
VGG
 ↓
ResNet
 ↓
Encoder / Decoder
 ↓
U-Net
```

这一完整的网络结构学习路径，为后续手写内容擦除的图像到图像任务做准备。

---

# 28. 项目结构

```text
level3_classic_cnn/
├── README.md
├── alexnet.py
├── vgg.py
├── resnet.py
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

---

# 29. 最终实验结果

```text
Baseline CNN
Test Accuracy: 98.70%

AlexNet-style
Test Accuracy: 99.33%

VGG-style
Test Accuracy: 99.06%

ResNet-18-style
Test Accuracy: 99.18%
```

本次实验表明：

在 MNIST 这种相对简单的数据集上，基础 CNN 已经能够达到较高准确率。

更复杂的经典 CNN 架构可以进一步提高模型表现，但同时增加了参数量或计算成本，并且模型复杂度与最终准确率之间并不存在简单的单调关系。

因此选择网络结构时，需要综合考虑：

```text
任务难度
数据规模
准确率
参数量
训练时间
计算资源
泛化能力
```

而不能仅以“网络是否更深”作为判断标准。