# Level 1 - MLP MNIST 手写数字分类

## 1. 任务目标

使用 PyTorch 搭建多层感知机（MLP），完成 MNIST 手写数字分类任务。

本实验完成了：

- MNIST 数据集加载与预处理
- 将 28×28 灰度图像 Flatten 为 784 维向量
- 使用 MLP 完成 0~9 十分类
- 使用 CrossEntropyLoss 计算分类损失
- 使用 SGD 和反向传播训练模型
- 划分 Train / Validation / Test 数据集
- 使用 GPU 进行模型训练
- 保存训练后的模型参数
- 绘制并保存 Train / Validation Loss 曲线
- 在测试集上计算最终准确率
- 完成单张 MNIST 图片推理

最终测试准确率达到 **96.36%**。

---

## 2. 数据集

使用 MNIST 手写数字数据集。

原始数据：

- 训练数据：60000 张
- 测试数据：10000 张
- 图像尺寸：28×28
- 图像通道数：1
- 分类数量：10（数字 0~9）

实验中将原始训练集进一步划分为：

- Train：55000 张
- Validation：5000 张
- Test：10000 张

---

## 3. MLP 网络结构

网络结构：

```text
Input
[Batch, 1, 28, 28]
        ↓
Flatten
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

模型使用一个隐藏层。

第一层全连接层将 784 维输入映射到 128 维隐藏特征，经过 ReLU 激活函数后，再通过第二层全连接层输出 10 个 logits，分别对应数字 0~9。

---

## 4. 训练配置

| 参数 | 设置 |
|---|---|
| Python | 3.11.9 |
| PyTorch | 2.13.0+cu130 |
| Batch Size | 64 |
| Epoch | 5 |
| Learning Rate | 0.1 |
| Optimizer | SGD |
| Loss Function | CrossEntropyLoss |
| Hidden Dimension | 128 |
| Device | CUDA GPU |

---

## 5. 训练结果

训练过程如下：

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
|---|---:|---:|---:|---:|
| 1 | 0.4549 | 87.85% | 0.3121 | 90.64% |
| 2 | 0.2340 | 93.37% | 0.2214 | 93.52% |
| 3 | 0.1778 | 94.96% | 0.1828 | 94.60% |
| 4 | 0.1423 | 95.97% | 0.1560 | 95.70% |
| 5 | 0.1193 | 96.63% | 0.1394 | 95.86% |

最终测试集结果：

```text
Test Accuracy: 96.36%
```

达到任务要求的测试准确率 **≥90%**。

从训练结果可以看到：

- Train Loss 从 0.4549 下降到 0.1193
- Validation Loss 从 0.3121 下降到 0.1394
- Train Accuracy 从 87.85% 提升到 96.63%
- Validation Accuracy 从 90.64% 提升到 95.86%

随着训练进行，Loss 持续下降，Accuracy 持续提高，说明 MLP 成功学习到了 MNIST 手写数字的分类特征。

---

## 6. Loss 曲线

训练过程中记录了 Train Loss 和 Validation Loss，并将曲线保存为：

```text
outputs/loss_curve.png
```

Loss 曲线用于观察模型在不同 Epoch 下的训练情况。

从实验结果可以看到，Train Loss 和 Validation Loss 均随着 Epoch 增加而持续下降。

---

## 7. 单张图片推理

训练完成后，从 MNIST 测试集中取出单张图片进行推理。

本次实验结果：

```text
真实标签：7
预测标签：7
```

模型成功预测该测试样本。

推理结果图片保存在：

```text
outputs/inference_example.png
```

---

## 8. 模型保存

训练完成后的模型参数保存在：

```text
outputs/mlp_mnist.pth
```

模型使用 PyTorch 的 `state_dict` 进行保存，可以在之后重新加载模型参数进行推理。

---

## 9. 项目结构

```text
level1_mlp/
├── main.py
├── README.md
├── data/
│   └── MNIST 数据集
└── outputs/
    ├── mlp_mnist.pth
    ├── loss_curve.png
    └── inference_example.png
```

其中：

- `main.py`：MLP 的训练、验证、测试以及单张图片推理
- `README.md`：Level 1 实验记录
- `data/`：MNIST 数据集，不上传至 Git 仓库
- `outputs/mlp_mnist.pth`：训练完成后的模型参数
- `outputs/loss_curve.png`：训练与验证 Loss 曲线
- `outputs/inference_example.png`：单张图片推理结果

---

## 10. 实验总结

本实验使用 PyTorch 实现了一个简单的 MLP，并完成 MNIST 手写数字分类任务。

通过本实验理解和实践了以下内容：

- Tensor 与图像数据表示
- Dataset 与 DataLoader
- Batch 与 Epoch
- Flatten
- 全连接层 Linear
- ReLU 非线性激活函数
- Logits 与分类预测
- CrossEntropyLoss
- Forward 前向传播
- Backward 反向传播
- Gradient 梯度
- SGD 参数更新
- Train / Validation / Test 的区别
- GPU 训练
- 模型保存
- 单张图片推理

MLP 的基本数据流为：

```text
MNIST Image
    ↓
Tensor
    ↓
Batch
    ↓
Flatten
    ↓
Linear(784, 128)
    ↓
ReLU
    ↓
Linear(128, 10)
    ↓
Logits
    ↓
CrossEntropyLoss
    ↓
Backward
    ↓
SGD 更新参数
```

最终模型在 MNIST 测试集上的准确率达到：

**96.36%**

满足 Level 1 测试准确率 **≥90%** 的要求。