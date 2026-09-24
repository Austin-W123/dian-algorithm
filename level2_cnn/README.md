# Level 2 - CNN MNIST 图像分类

## 1. 任务目标

本阶段使用卷积神经网络（Convolutional Neural Network, CNN）完成 MNIST 手写数字分类任务，并与 Level 1 中实现的 MLP 进行比较。

主要目标：

- 理解图像的局部空间结构
- 理解卷积核、步长（Stride）、填充（Padding）等基本概念
- 理解卷积层中的局部连接与权重共享
- 使用 CNN 完成 MNIST 分类
- 比较 CNN 与 MLP 的准确率
- 比较 CNN 与 MLP 的参数量
- 分析模型的收敛情况
- 对 CNN 的错误分类样本进行可视化

---

## 2. 数据集

使用 MNIST 手写数字数据集。

数据划分：

| 数据集 | 数量 |
|---|---:|
| Training Set | 55,000 |
| Validation Set | 5,000 |
| Test Set | 10,000 |

为了保证与 Level 1 的实验具有较好的可比性，本实验使用相同的随机种子 `42` 进行训练集和验证集划分。

输入图片尺寸：

```text
[1, 28, 28]
```

其中：

- `1`：灰度图通道数
- `28 × 28`：图像空间尺寸

---

## 3. 为什么使用 CNN

Level 1 中的 MLP 在输入网络之前，需要首先将：

```text
[1, 28, 28]
```

展平为：

```text
[784]
```

这种方式能够完成分类任务，但是没有显式利用图像中像素之间的二维空间关系。

CNN 使用卷积核在图像的局部区域上进行计算，并在整张图片上共享卷积核参数，因此能够利用图像的局部空间结构。

CNN 的主要特点包括：

1. 局部连接（Local Connectivity）
2. 权重共享（Weight Sharing）
3. 多个卷积核学习不同的特征表示
4. 逐层提取更加复杂的图像特征

---

## 4. CNN 网络结构

本实验使用的网络结构如下：

```text
Input
[1, 28, 28]

↓ Conv2d(1, 32, kernel_size=3, padding=1)

[32, 28, 28]

↓ ReLU
↓ MaxPool2d(2)

[32, 14, 14]

↓ Conv2d(32, 64, kernel_size=3, padding=1)

[64, 14, 14]

↓ ReLU
↓ MaxPool2d(2)

[64, 7, 7]

↓ Flatten

3136

↓ Linear(3136, 128)
↓ ReLU
↓ Linear(128, 10)

10 logits
```

最终输出 10 个 logits，分别对应数字 `0 ~ 9`。

---

## 5. 训练配置

| 参数 | 设置 |
|---|---|
| Python | 3.11.9 |
| PyTorch | 2.13.0+cu130 |
| Device | CUDA GPU |
| Batch Size | 64 |
| Epochs | 5 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss |
| Random Seed | 42 |

本实验使用 Adam 优化器，以较少的训练轮数快速获得稳定的 CNN 训练结果。

---

## 6. CNN 参数量

程序统计得到：

```text
Total Parameters: 421642
Trainable Parameters: 421642
```

CNN 总可训练参数量为：

```text
421,642
```

虽然卷积层通过局部连接和权重共享减少了单个卷积操作所需的参数，但本实验 CNN 后部包含：

```text
Linear(64 × 7 × 7, 128)
```

即：

```text
Linear(3136, 128)
```

该全连接层包含大量参数，因此本实验 CNN 的总参数量实际上高于 Level 1 中的 MLP。

这说明：

> CNN 是否具有更少的总参数取决于具体网络结构，不能简单认为 CNN 的总参数量一定小于 MLP。

---

## 7. 训练结果

5 个 Epoch 的训练结果如下：

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Time |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.1883 | 94.16% | 0.0788 | 97.78% | 10.90s |
| 2 | 0.0513 | 98.42% | 0.0497 | 98.60% | 8.61s |
| 3 | 0.0355 | 98.92% | 0.0454 | 98.78% | 8.60s |
| 4 | 0.0277 | 99.13% | 0.0534 | 98.40% | 8.21s |
| 5 | 0.0203 | 99.36% | 0.0459 | 98.52% | 8.20s |

总训练时间：

```text
44.51 s
```

可以看到训练 Loss 持续下降，训练准确率持续提高。

Validation Accuracy 在第 3 个 Epoch 达到本次训练中的最高值：

```text
98.78%
```

之后训练准确率继续提高，但验证准确率没有继续同步提高，说明模型后期出现了轻微的过拟合趋势。

---

## 8. 最终测试结果

CNN 在 Test Set 上的最终结果：

```text
Test Accuracy: 98.70%
```

测试准确率明显超过任务参考目标。

---

## 9. CNN 与 MLP 对比

Level 1 中 MLP 的测试准确率：

```text
96.36%
```

Level 2 中 CNN 的测试准确率：

```text
98.70%
```

对比如下：

| Model | Test Accuracy | Parameters |
|---|---:|---:|
| MLP | 96.36% | 101,770 |
| CNN | 98.70% | 421,642 |

CNN 相比 MLP：

```text
98.70% - 96.36% = 2.34%
```

测试准确率提高了约 **2.34 个百分点**。

CNN 的参数量约为 MLP 的：

```text
421642 / 101770 ≈ 4.14
```

即约 **4.14 倍**。

因此，本实验中 CNN 的性能提升并不是因为模型总参数更少。

CNN 的主要优势来自其针对图像结构设计的归纳偏置：

- 使用局部连接提取邻近像素之间的关系
- 使用卷积核进行权重共享
- 保留二维空间结构进行特征提取
- 通过多层卷积逐渐形成更高级的特征表示

---

## 10. 收敛情况比较

Level 1 中 MLP 的验证准确率：

| Epoch | MLP Val Accuracy |
|---:|---:|
| 1 | 90.64% |
| 2 | 93.52% |
| 3 | 94.60% |
| 4 | 95.70% |
| 5 | 95.86% |

Level 2 中 CNN 的验证准确率：

| Epoch | CNN Val Accuracy |
|---:|---:|
| 1 | 97.78% |
| 2 | 98.60% |
| 3 | 98.78% |
| 4 | 98.40% |
| 5 | 98.52% |

可以看到，在当前实验配置下，CNN 在第 1 个 Epoch 就已经达到 **97.78%** 的验证准确率，而 MLP 在训练 5 个 Epoch 后验证准确率为 **95.86%**。

因此，从 Epoch 级别的 Accuracy 变化来看，本实验中的 CNN 更快达到较高的分类准确率。

需要注意的是，两个实验使用的优化器不同：

- MLP：SGD
- CNN：Adam

因此这个结果能够反映当前两组实验配置下的实际收敛表现，但并不是严格控制所有变量后的模型结构对比。

另外，Level 1 实验中没有记录训练时间，因此这里不直接比较 MLP 与 CNN 的训练秒数。

---

## 11. 错误样本分析

程序从测试集中保存了前 16 个预测错误的样本：

```text
outputs/wrong_samples.png
```

每张图片同时显示：

```text
True: 真实标签
Pred: CNN 预测标签
```

通过观察错误样本，可以分析模型容易混淆的数字形态。

部分 MNIST 图片本身可能存在：

- 字迹倾斜
- 笔画连接
- 数字形状不标准
- 不同数字之间外观相似

这些情况会增加分类难度。

---

## 12. 训练曲线

程序保存了 Loss 曲线：

```text
outputs/loss_curve.png
```

以及 Accuracy 曲线：

```text
outputs/accuracy_curve.png
```

Loss 曲线用于观察训练过程中 Train Loss 和 Validation Loss 的变化。

Accuracy 曲线用于观察 Train Accuracy 和 Validation Accuracy 的变化。

从本次实验可以看到：

- Train Loss 总体持续下降
- Train Accuracy 持续提高
- Validation Accuracy 很快达到较高水平
- 第 3 个 Epoch 后验证性能基本趋于稳定

---

## 13. 单张图片推理

程序对测试集中的单张图片进行了推理。

结果：

```text
真实标签：7
预测标签：7
```

说明模型能够完成从单张 MNIST 图片到分类结果的完整推理流程。

推理结果保存在：

```text
outputs/inference_example.png
```

---

## 14. 模型保存

训练后的模型参数保存为：

```text
outputs/cnn_mnist.pth
```

保存方式：

```python
torch.save(model.state_dict(), "outputs/cnn_mnist.pth")
```

模型保存后可以通过重新建立相同的 CNN 网络结构，再使用：

```python
model.load_state_dict(
    torch.load("outputs/cnn_mnist.pth")
)
```

加载已经训练好的参数。

---

## 15. 项目结构

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

MNIST 数据集直接复用：

```text
level1_mlp/data/
```

避免重复保存数据。

---

## 16. 实验总结

通过本实验，我完成了从 MLP 到 CNN 的进一步学习。

Level 1 中，MLP 将二维图像直接 Flatten 后进行全连接计算。

Level 2 中，CNN 首先使用卷积层在二维图像上提取局部特征，再逐步形成更高级的特征表示，最后使用全连接层完成分类。

本实验 CNN 最终取得：

```text
98.70%
```

的测试准确率，高于 MLP 的：

```text
96.36%
```

提高了约：

```text
2.34 个百分点
```

在当前实验配置下，CNN 也表现出了更快达到较高验证准确率的现象。

同时，本实验说明 CNN 的优势不能简单归结为“参数更少”。本实验 CNN 的参数量实际上高于 MLP：

```text
MLP: 101,770
CNN: 421,642
```

但 CNN 依然取得了更好的分类结果。

更重要的原因在于 CNN 的结构更加适合图像数据：

1. **局部连接**：重点学习邻近像素之间的关系
2. **权重共享**：同一个卷积核可以扫描整张图片
3. **空间结构**：在特征提取阶段保留图像的二维结构
4. **层次化特征提取**：多层卷积能够逐渐学习更加复杂的特征

因此，相比直接 Flatten 图像的 MLP，CNN 更适合处理具有明显空间结构的图像数据。

---

## 17. 本阶段掌握的主要知识

通过 Level 2，主要学习和实践了以下内容：

- CNN（Convolutional Neural Network）
- 卷积核（Kernel / Filter）
- 局部感受野（Local Receptive Field）
- 权重共享（Weight Sharing）
- Feature Map
- Channel
- `nn.Conv2d`
- Kernel Size
- Stride
- Padding
- `nn.MaxPool2d`
- CNN 中 Tensor Shape 的变化
- CNN 参数量统计
- Adam Optimizer
- Train / Validation / Test
- Loss Curve
- Accuracy Curve
- 错误样本可视化
- CNN 与 MLP 的实验对比

至此完成 Level 2 的 CNN MNIST 图像分类实验。