# Level 1：基于 MLP 的 MNIST 手写数字分类

# 1. 任务目标

Level 1 的目标是使用 PyTorch 搭建一个多层感知机（MLP），完成 MNIST 手写数字分类任务。

模型输入是一张 `28 × 28` 的灰度手写数字图片，输出该图片属于数字 `0~9` 中哪一类。

整体任务可以表示为：

```text
MNIST Image
    ↓
Tensor
    ↓
MLP
    ↓
10-dimensional Logits
    ↓
Predicted Class
```

本实验主要完成：

- MNIST 数据集加载与预处理；
- Train / Validation / Test 数据划分；
- 使用 DataLoader 按 Batch 加载数据；
- 将 `28 × 28` 图像 Flatten 为 784 维向量；
- 使用 MLP 完成十分类；
- 使用 ReLU 引入非线性；
- 使用 CrossEntropyLoss 计算分类损失；
- 使用 SGD 和反向传播更新模型参数；
- 使用 GPU 进行模型训练；
- 记录 Train / Validation Loss 与 Accuracy；
- 在独立 Test Set 上计算最终准确率；
- 保存训练后的模型参数；
- 绘制 Loss Curve；
- 完成单张 MNIST 图片推理。

最终模型在 MNIST Test Set 上达到：

```text
Test Accuracy = 96.36%
```

---

# 2. 项目文件结构

Level 1 的项目结构：

```text
level1_mlp/
├── main.py
├── README.md
├── data/
│   └── MNIST/
│
└── outputs/
    ├── mlp_mnist.pth
    ├── loss_curve.png
    └── inference_example.png
```

## 2.1 `main.py`

`main.py` 是 Level 1 的核心程序。

整个 MLP 实验都在该文件中完成，包括：

```text
定义 MLP
    ↓
设置训练参数
    ↓
加载 MNIST
    ↓
划分 Train / Validation
    ↓
创建 DataLoader
    ↓
创建模型
    ↓
训练
    ↓
Validation
    ↓
Test
    ↓
保存模型
    ↓
绘制 Loss Curve
    ↓
单张图片推理
```

也就是说，`main.py` 包含了一个完整的 PyTorch 图像分类训练流程。

---

## 2.2 `data/`

```text
data/
└── MNIST/
```

用于保存通过 `torchvision.datasets.MNIST` 下载的 MNIST 数据集。

数据集文件不需要上传到 Git 仓库，因为程序可以自动重新下载。

---

## 2.3 `outputs/mlp_mnist.pth`

```text
outputs/mlp_mnist.pth
```

保存训练完成后的 MLP 参数。

模型使用：

```python
torch.save(model.state_dict(), model_path)
```

保存的是模型的 `state_dict`，也就是网络中需要学习的参数。

以后只要重新创建相同结构的 MLP，再加载该参数文件，就可以继续进行推理。

---

## 2.4 `outputs/loss_curve.png`

```text
outputs/loss_curve.png
```

保存训练过程中：

```text
Train Loss
Validation Loss
```

随 Epoch 变化的曲线。

它主要用于观察：

```text
模型是否正常学习
Loss 是否下降
Train / Validation 是否出现明显分离
是否存在明显过拟合趋势
```

---

## 2.5 `outputs/inference_example.png`

```text
outputs/inference_example.png
```

保存单张 MNIST 图片的推理结果。

本次样本：

```text
True:      7
Predicted: 7
```

说明模型对该样本预测正确。

---

# 3. 实验结果与结果分析

## 3.1 训练结果

本实验共训练：

```text
5 Epochs
```

完整训练结果如下：

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.4549 | 87.85% | 0.3121 | 90.64% |
| 2 | 0.2340 | 93.37% | 0.2214 | 93.52% |
| 3 | 0.1778 | 94.96% | 0.1828 | 94.60% |
| 4 | 0.1423 | 95.97% | 0.1560 | 95.70% |
| 5 | 0.1193 | 96.63% | 0.1394 | 95.86% |

可以看到，在 5 个 Epoch 中：

```text
Train Loss:
0.4549 → 0.1193

Validation Loss:
0.3121 → 0.1394
```

同时：

```text
Train Accuracy:
87.85% → 96.63%

Validation Accuracy:
90.64% → 95.86%
```

总体趋势表现为：

```text
Epoch ↑
   ↓
Loss ↓
Accuracy ↑
```

说明 MLP 在训练过程中成功学习到了能够区分 MNIST 手写数字的特征。

---

## 3.2 Loss Curve

训练过程中记录了：

```text
Train Loss
Validation Loss
```

并绘制：

```text
outputs/loss_curve.png
```

从 Loss Curve 可以看到：

```text
Train Loss
   ↓

Validation Loss
   ↓
```

两条曲线都随着 Epoch 增加持续下降。

第 5 个 Epoch：

```text
Train Loss      = 0.1193
Validation Loss = 0.1394
```

Validation Loss 略高于 Train Loss 是正常现象，因为模型参数是直接根据 Training Set 优化的，而 Validation Set 不参与参数更新。

当前 5 个 Epoch 内没有观察到 Validation Loss 明显反弹，因此没有出现非常明显的过拟合现象。

不过最后：

```text
Train Accuracy      = 96.63%
Validation Accuracy = 95.86%
```

已经出现一定差距。

这说明随着训练继续进行，需要继续观察 Validation Loss 和 Validation Accuracy，而不能只看 Training Accuracy。

---

## 3.3 最终 Test Set 结果

训练完成后，模型在独立 Test Set 上进行最终测试。

结果：

```text
Test Accuracy = 96.36%
```

即：

```text
10000 张测试图片中
约 96.36% 被模型正确分类
```

Test Set 不参与训练，也不用于模型参数更新，因此这个结果能够更客观地反映模型对未参与训练数据的分类能力。

最终：

```text
Test Accuracy = 96.36%
```

达到 Level 1 对 MNIST 分类准确率的要求。

---

## 3.4 Train / Validation / Test 对比

最终几个主要 Accuracy：

```text
Train Accuracy      = 96.63%
Validation Accuracy = 95.86%
Test Accuracy       = 96.36%
```

三个结果比较接近。

这说明在当前实验设置下，模型不仅能够拟合 Training Set，也能够在 Validation Set 和 Test Set 上保持相近的分类能力。

三者的作用不同：

```text
Train
→ 用来学习参数

Validation
→ 用来观察训练过程

Test
→ 最终独立评价
```

因此不能用 Test Set 参与模型训练。

---

## 3.5 单张图片推理

训练完成以后，从 MNIST Test Set 中取出第一张图片进行单张推理。

结果：

```text
True Label      = 7
Predicted Label = 7
```

预测正确。

对应图片保存为：

```text
outputs/inference_example.png
```

单张图片原本的 Tensor Shape：

```text
[1, 28, 28]
```

但是神经网络通常按照 Batch 进行计算，因此首先增加 Batch Dimension：

```text
[1, 28, 28]
      ↓
unsqueeze(0)
      ↓
[1, 1, 28, 28]
```

然后输入模型：

```text
Image
  ↓
MLP
  ↓
10 Logits
  ↓
argmax
  ↓
Predicted Class = 7
```

---

# 4. 数据集

## 4.1 MNIST

本实验使用 MNIST 手写数字数据集。

MNIST 是一个经典的手写数字分类数据集，共包含数字：

```text
0 1 2 3 4 5 6 7 8 9
```

因此这是一个：

```text
10-Class Classification
```

任务。

每张图片：

```text
Height  = 28
Width   = 28
Channel = 1
```

Tensor Shape：

```text
[1, 28, 28]
```

其中 `1` 表示灰度图只有一个 Channel。

---

## 4.2 原始数据数量

MNIST 原始数据：

```text
Training Images = 60000
Test Images     = 10000
```

实验中进一步将原始的 60000 张 Training Images 划分为：

```text
Train      = 55000
Validation = 5000
```

因此最终使用：

```text
Train      = 55000
Validation = 5000
Test       = 10000
```

---

## 4.3 为什么需要 Validation Set

如果只有：

```text
Train
+
Test
```

那么训练过程中就缺少一个独立数据集来观察模型对未参与参数更新的数据表现。

因此从原始 Training Set 中划出：

```text
5000
```

张图片作为 Validation Set。

三个数据集的职责为：

```text
Train
→ 参与 Forward
→ 计算 Loss
→ Backward
→ 更新参数

Validation
→ Forward
→ 计算 Loss / Accuracy
→ 不更新参数

Test
→ 训练全部结束后
→ 最终评价模型
```

---

## 4.4 数据预处理

数据通过：

```python
ToTensor()
```

转换为 PyTorch Tensor。

原始图片：

```text
28 × 28
```

进入 Dataset 后得到：

```text
[1, 28, 28]
```

随后 DataLoader 将多张图片组成 Batch。

本实验：

```text
Batch Size = 64
```

因此正常情况下，一个 Batch 的输入 Shape 为：

```text
[64, 1, 28, 28]
```

---

## 4.5 DataLoader

Dataset 负责：

```text
“有哪些数据？”
```

DataLoader 负责：

```text
“怎样把这些数据一批一批送进模型？”
```

本实验设置：

```text
Train:
batch_size = 64
shuffle    = True

Validation:
batch_size = 64
shuffle    = False

Test:
batch_size = 64
shuffle    = False
```

Training Set 使用：

```text
shuffle=True
```

可以在每个 Epoch 中打乱训练数据顺序。

Validation 和 Test 不需要通过随机顺序来训练参数，因此：

```text
shuffle=False
```

---

# 5. 方法与实验策略

## 5.1 整体训练 Pipeline

整个 Level 1 可以概括为：

```text
MNIST Dataset
      ↓
ToTensor
      ↓
DataLoader
      ↓
Batch
      ↓
MLP Forward
      ↓
Logits
      ↓
CrossEntropyLoss
      ↓
Backward
      ↓
Gradient
      ↓
SGD
      ↓
Update Parameters
```

每个 Epoch 都会重复这一过程。

---

## 5.2 MLP 网络结构

本实验使用一个隐藏层的 MLP。

网络结构：

```text
Input
[Batch, 1, 28, 28]
        ↓
      Flatten
        ↓
[Batch, 784]
        ↓
Linear(784, 128)
        ↓
       ReLU
        ↓
Linear(128, 10)
        ↓
      Logits
[Batch, 10]
```

可以简写为：

```text
784 → 128 → 10
```

---

## 5.3 Flatten

原始 MNIST Batch：

```text
[B, 1, 28, 28]
```

MLP 中的 `Linear` 层接收的是一维 Feature Vector，因此需要先进行：

```python
torch.flatten(x, start_dim=1)
```

得到：

```text
[B, 784]
```

因为：

```text
1 × 28 × 28 = 784
```

例如：

```text
[64, 1, 28, 28]
        ↓
Flatten
        ↓
[64, 784]
```

需要注意：

> Flatten 只改变数据的组织形式，不会增加新的特征，也没有需要学习的参数。

---

## 5.4 第一层 Linear

第一层：

```text
Linear(784, 128)
```

输入：

```text
[B, 784]
```

输出：

```text
[B, 128]
```

数学上可以表示为：

```text
z = xW + b
```

其中：

```text
x = 输入
W = Weight
b = Bias
```

`W` 和 `b` 就是模型在训练过程中需要学习的参数。

---

## 5.5 ReLU

第一层 Linear 后使用：

```text
ReLU
```

定义：

```text
ReLU(x) = max(0, x)
```

即：

```text
x > 0 → 保留
x < 0 → 变成 0
```

如果多个 Linear 层之间完全没有非线性激活函数，那么多层线性变换最终仍然可以等价为一个线性变换。

因此 ReLU 的重要作用是：

> 为神经网络引入非线性表达能力。

---

## 5.6 输出层

第二层：

```text
Linear(128, 10)
```

输出：

```text
[B, 10]
```

因为 MNIST 一共有：

```text
10 Classes
```

分别对应：

```text
0~9
```

例如模型可能输出：

```text
[-1.2, 0.5, 2.1, -0.3, 0.7, 1.1, -2.0, 4.8, 0.2, -1.0]
```

这些数叫：

```text
Logits
```

它们不是概率。

最终预测：

```python
torch.argmax(outputs, dim=1)
```

也就是选择 Logit 最大的位置作为预测类别。

如果最大值位于索引：

```text
7
```

那么：

```text
Predicted Class = 7
```

---

## 5.7 CrossEntropyLoss

分类训练使用：

```python
nn.CrossEntropyLoss()
```

模型直接输出：

```text
Raw Logits
```

不需要在 `forward()` 中手动添加 Softmax。

可以把 PyTorch 的 CrossEntropyLoss 理解为内部完成了与：

```text
LogSoftmax
+
Negative Log Likelihood
```

对应的计算。

因此正确流程为：

```text
Linear
  ↓
Raw Logits
  ↓
CrossEntropyLoss
```

而不是先在模型中手动 Softmax 再交给 CrossEntropyLoss。

---

## 5.8 Forward

训练一个 Batch 时：

```python
outputs = model(images)
```

会调用：

```python
forward()
```

数据依次经过：

```text
Images
  ↓
Flatten
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
Logits
```

这整个过程叫：

```text
Forward Propagation
```

---

## 5.9 Loss

得到 Logits 后：

```python
loss = criterion(outputs, labels)
```

Loss 用来衡量：

```text
模型当前预测
```

和：

```text
真实 Label
```

之间的差距。

训练的核心目标就是：

```text
不断调整参数
      ↓
Loss 逐渐下降
```

---

## 5.10 Backward

本实验训练一个 Batch 的核心代码：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

对应：

```text
zero_grad()
    ↓
清空上一轮梯度

loss.backward()
    ↓
反向传播
计算当前参数梯度

optimizer.step()
    ↓
根据梯度更新参数
```

因此完整过程：

```text
Forward
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

---

## 5.11 为什么要 `zero_grad()`

PyTorch 默认会：

```text
累积 Gradient
```

也就是说，如果不清空：

```text
当前 Batch 的 Gradient
+
上一 Batch 的 Gradient
```

会累积在一起。

但普通训练中，我们希望每个 Batch 根据自己计算得到的 Loss 更新一次参数。

因此每个 Batch 都需要：

```python
optimizer.zero_grad()
```

清除上一轮梯度。

---

## 5.12 SGD

本实验使用：

```text
Optimizer = SGD
Learning Rate = 0.1
```

SGD 的核心思想可以简单表示为：

```text
New Parameter
=
Old Parameter
-
Learning Rate × Gradient
```

即：

```text
θ_new = θ_old - η∇L
```

其中：

```text
θ  = 模型参数
η  = Learning Rate
∇L = Loss 对参数的梯度
```

梯度告诉模型：

```text
参数向哪个方向变化
会使 Loss 增大
```

因此 SGD 向相反方向更新参数，使 Loss 尽量下降。

---

## 5.13 Batch 与 Epoch

本实验：

```text
Batch Size = 64
Epoch      = 5
```

Batch 表示：

> 一次送入模型多少个样本。

Epoch 表示：

> 整个 Training Set 被完整学习一遍。

因此：

```text
1 Epoch
≠
1 次参数更新
```

一个 Epoch 中会包含很多 Batch。

每处理一个 Training Batch：

```text
Forward
→ Loss
→ Backward
→ Update
```

都会进行一次参数更新。

---

## 5.14 `model.train()` 与 `model.eval()`

训练阶段：

```python
model.train()
```

Validation / Test / Inference：

```python
model.eval()
```

本 Level 的 MLP 只有 Linear 和 ReLU，因此两种模式不会像 Dropout、BatchNorm 模型那样产生明显行为差异。

但是保持这种写法非常重要，因为以后使用：

```text
Dropout
BatchNorm
```

时，Train Mode 和 Eval Mode 的行为会不同。

因此这是一个应该从 Level 1 就建立的标准 PyTorch 训练习惯。

---

## 5.15 `torch.no_grad()`

Validation、Test 和单张图片推理都使用：

```python
with torch.no_grad():
```

因为这些阶段：

```text
只需要 Forward
不需要 Backward
```

所以没有必要建立用于反向传播的计算图。

这样可以：

```text
减少显存 / 内存开销
提高推理效率
```

---

## 5.16 实验配置

最终训练配置：

| Parameter | Value |
|---|---|
| Python | `3.11.9` |
| PyTorch | `2.13.0+cu130` |
| Input Size | `28 × 28` |
| Input Dimension | `784` |
| Hidden Dimension | `128` |
| Output Dimension | `10` |
| Batch Size | `64` |
| Epoch | `5` |
| Learning Rate | `0.1` |
| Optimizer | `SGD` |
| Loss | `CrossEntropyLoss` |
| Device | `CUDA GPU` |

网络：

```text
784
 ↓
128
 ↓
10
```

训练：

```text
Forward
 ↓
CrossEntropyLoss
 ↓
Backward
 ↓
SGD
```

---

# 6. 总结

本 Level 使用 PyTorch 完成了第一个完整的神经网络图像分类任务。

最终模型：

```text
MLP
784 → 128 → 10
```

训练配置：

```text
Batch Size    = 64
Epoch         = 5
Learning Rate = 0.1
Optimizer     = SGD
Loss          = CrossEntropyLoss
```

最终结果：

```text
Train Accuracy      = 96.63%
Validation Accuracy = 95.86%
Test Accuracy       = 96.36%
```

通过本实验，完整经历了：

```text
Dataset
   ↓
DataLoader
   ↓
Batch
   ↓
Tensor
   ↓
Forward
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
   ↓
Validation
   ↓
Test
   ↓
Inference
```

这套流程也是后续 Level 2、Level 3 和 Level 4 的基础。

Level 1 最重要的意义并不只是：

```text
MNIST Accuracy = 96.36%
```

而是第一次建立了完整的深度学习训练框架：

> 数据进入模型，通过 Forward 得到预测；Loss 衡量预测与真实答案之间的差距；Backward 根据 Loss 计算梯度；Optimizer 根据梯度更新参数；经过多个 Batch 和 Epoch 后，模型逐渐学会完成分类任务。

---

## 关键词总结

| 关键词 | 本 Level 中的含义 |
|---|---|
| Tensor | PyTorch 中用于存储和计算数据的多维数组 |
| Dataset | 定义数据集中的样本 |
| DataLoader | 按 Batch 加载 Dataset |
| Batch | 一次送入模型的一组样本 |
| Epoch | 完整遍历一次 Training Set |
| MLP | 由全连接层构成的多层感知机 |
| Flatten | 将 `1 × 28 × 28` 图像展开成 784 维向量 |
| Linear | 全连接层，执行可学习的线性变换 |
| ReLU | 为网络引入非线性 |
| Logits | 模型输出的分类分数 |
| CrossEntropyLoss | 多分类任务使用的损失函数 |
| Forward | 输入经过网络得到预测结果 |
| Loss | 衡量 Prediction 与 Label 的差距 |
| Backward | 根据 Loss 反向计算梯度 |
| Gradient | 参数变化对 Loss 的影响方向和大小 |
| SGD | 根据梯度更新模型参数的优化器 |
| Learning Rate | 控制每次参数更新步长 |
| Train Set | 用于学习模型参数 |
| Validation Set | 用于观察训练过程 |
| Test Set | 用于最终独立评价 |
| `model.train()` | 将模型切换到训练模式 |
| `model.eval()` | 将模型切换到评估模式 |
| `torch.no_grad()` | 推理时关闭梯度计算 |
| `state_dict` | 保存模型可学习参数的字典 |
| Inference | 使用训练好的模型预测新输入 |