# Level 4：基于 U-Net 的手写笔迹擦除

## 1. 任务目标

Level 4 的目标是完成一个图像到图像（Image-to-Image）的手写笔迹擦除任务：

> 输入一张包含印刷内容和手写笔迹的试卷 / 作业图片，输出尽可能去除手写内容、同时保留原始印刷内容的干净图片。

与前面 Level 1～3 的图像分类任务不同，本任务不再输出类别：

```text
Level 1～3:
Image → Network → Class

Level 4:
Image → U-Net → Image
```

因此模型需要对图像中的每一个像素进行预测。

训练数据采用成对形式：

```text
Input X：包含手写笔迹的图片
Target Y：对应的干净图片
Prediction Ŷ = U-Net(X)
```

训练目标是使：

```text
Ŷ ≈ Y
```

---

# 2. 数据集

数据集位于：

```text
/mnt/d/User/University/QQ/Dian/deli
```

原始目录结构：

```text
deli/
├── 20250211/
│   └── dataset/
│       ├── input/
│       └── output/
├── 20250212/
│   └── dataset/
│       ├── input/
│       └── output/
└── 20250213/
    └── dataset/
        ├── input/
        └── output/
```

其中：

```text
input/
```

保存带有手写笔迹的图片，

```text
output/
```

保存对应的干净目标图片。

Input 与 Output 使用相同文件名形成一一对应关系。

数据量统计：

| 数据目录 | Input | Output |
|---|---:|---:|
| 20250211 | 812 | 812 |
| 20250212 | 900 | 900 |
| 20250213 | 700 | 700 |
| Total | 2412 | 2412 |

因此总共有：

```text
2412 pairs
4824 JPG images
```

数据内容包括：

- 中文试卷
- 英文试卷
- 数学题
- 几何图形
- 表格
- 填空题
- 作业批改
- 黑色 / 彩色手写笔迹
- 大面积手写内容
- 无手写内容的样本
- 不同分辨率和长宽比的图片

这些数据具有较明显的真实拍摄特征，例如：

- 光照不均匀
- 页面阴影
- 背景发灰
- 透页
- 拍摄角度差异
- 图片尺寸不统一

因此该任务不仅是简单的像素复制，而需要模型学习：

> 哪些内容属于应该保留的印刷信息，哪些内容属于应该被擦除的手写信息。

---

# 3. 数据检查

正式训练之前首先使用：

```text
inspect_dataset.py
```

检查数据集。

主要检查内容包括：

- Input / Output 是否一一对应
- 是否存在缺失图片
- 图片是否能够正常解码
- Input / Output 尺寸是否一致
- 图片颜色模式
- 图片尺寸分布
- 横图 / 竖图比例

数据检查的意义是：

> 在训练神经网络之前，首先确认监督信号本身是可靠的。

对于 Image-to-Image 任务尤其重要，因为 Input 与 Target 如果发生错位，像素级 Loss 将失去意义。

---

# 4. 为什么采用 Patch Training

原始图片尺寸差异很大，并且部分图片分辨率较高。

如果直接将完整图片输入 U-Net：

```text
[B, C, H, W]
```

显存消耗会随着 H 和 W 快速增加。

本实验使用的 GPU 为：

```text
NVIDIA RTX 5060 Laptop GPU
VRAM: 8 GB
```

因此采用 Patch Training，将完整图片裁剪为固定大小的局部区域进行训练。

主要 Patch Size：

```text
256 × 256
```

Patch Training 的优点：

1. 控制 GPU 显存占用
2. 保留局部高分辨率细节
3. 不需要将整张高分辨率试卷强行缩小
4. 可以从一张图片生成多个训练样本
5. 适合 U-Net 的局部图像恢复任务

相比直接 Whole Image Resize：

```text
Whole Image Resize
    ↓
可能破坏长宽比
    ↓
细小文字和笔迹被压缩
```

Patch Training 更适合文档图像：

```text
High Resolution Document
        ↓
256 × 256 Patches
        ↓
U-Net
```

训练时 Input 与 Target 必须使用完全相同的 Crop 坐标，否则：

```text
Input Patch ≠ Target 对应区域
```

像素级监督将失效。

---

# 5. Difference Mask

为了分析真正发生变化的区域，定义：

```text
Difference = |Input - Target|
```

并通过阈值得到 Difference Mask：

```text
Mask = |Input - Target| > threshold
```

本实验中使用：

```text
Difference Threshold = 0.1
```

Difference Mask 的意义是：

> 找出 Input 和 Target 中真正发生明显变化的像素。

这些区域通常对应：

- 手写文字
- 批改痕迹
- 手写线条
- 被擦除的标记

需要注意：

> Difference Mask 并不是完美的“手写区域分割标签”。

因为 Input 和 Target 之间还可能存在：

- 扫描差异
- 光照差异
- 对齐误差
- 背景变化
- 印刷文字边缘差异

因此 Difference Mask 更准确的含义是：

> Input 与 Target 的显著变化区域。

---

# 6. 图像预处理

最终使用的输入流程为：

```text
RGB
 ↓
Grayscale (L)
 ↓
Background Normalization
 ↓
Document Enhancement
 ↓
Tensor
```

即：

```text
RGB → L → Background Normalization → Document Enhancement
```

这样做的目的主要是降低真实拍摄图片中的：

- 光照不均
- 灰色背景
- 页面阴影

使模型更加关注：

```text
文字 / 笔迹 / 图形结构
```

而不是学习无关的背景亮度变化。

---

# 7. U-Net 网络结构

本任务采用 U-Net。

U-Net 的基本结构为：

```text
Input
  │
  ▼
Encoder
  │
  ▼
Bottleneck
  │
  ▼
Decoder
  │
  ▼
Output
```

同时 Encoder 和 Decoder 之间存在 Skip Connection：

```text
Encoder ───────────────┐
   ↓                   │
Encoder ──────────┐    │
   ↓              │    │
Bottleneck        │    │
   ↓              │    │
Decoder ◄─────────┘    │
   ↓                   │
Decoder ◄──────────────┘
   ↓
Output
```

最终模型参数量：

```text
Total Parameters: 7,762,465
Trainable Parameters: 7,762,465
```

---

# 8. Encoder 与 Downsampling

Encoder 的主要作用是逐渐提取更加抽象的图像特征。

典型变化：

```text
H, W ↓
Channels ↑
```

例如：

```text
256 × 256
    ↓
128 × 128
    ↓
64 × 64
    ↓
32 × 32
```

随着空间尺寸下降，每个神经元能够感受到更大的有效感受野。

因此网络可以从：

```text
局部像素
```

逐渐学习：

```text
边缘
↓
笔画
↓
字符结构
↓
手写 / 印刷模式
```

---

# 9. Bottleneck

Bottleneck 位于 U-Net 最深处。

它拥有：

```text
较小的空间尺寸
+
较高层的语义特征
```

可以理解为：

> U-Net 对当前 Patch 的高度压缩特征表示。

---

# 10. Decoder 与 Upsampling

Decoder 与分类 CNN 最大的区别之一是：

分类任务最终只需要：

```text
Class Logits
```

而本任务最终必须恢复：

```text
完整图像
```

因此需要逐步 Upsampling：

```text
32 × 32
 ↓
64 × 64
 ↓
128 × 128
 ↓
256 × 256
```

最终恢复到输入 Patch 的空间分辨率。

---

# 11. Skip Connection

如果只有：

```text
Encoder → Bottleneck → Decoder
```

Downsampling 会损失大量高分辨率细节。

例如：

- 小字体
- 字符边缘
- 表格线
- 几何图形
- 数学符号

因此 U-Net 将 Encoder 中的高分辨率 Feature Map 直接传递给 Decoder。

经典 U-Net 通常采用：

```python
torch.cat(...)
```

即 Concatenation。

例如：

```text
Decoder Feature:
[B, 64, 128, 128]

Encoder Feature:
[B, 64, 128, 128]

Concatenate
        ↓
[B, 128, 128, 128]
```

这样 Decoder 在恢复图像时既能够使用：

```text
深层语义信息
```

也能够使用：

```text
浅层空间细节
```

这对于文档恢复尤其重要。

---

# 12. U-Net Skip Connection 与 ResNet Shortcut 的区别

Level 3 中 ResNet 也使用了 Skip Connection，但目的和实现方式有所不同。

ResNet：

```text
F(x) + x
```

通常使用：

```text
Addition
```

主要解决深层网络优化和信息传播问题。

U-Net：

```text
Encoder Feature + Decoder Feature
```

经典实现通常使用：

```text
Concatenation
```

主要用于：

> 将 Encoder 中的高分辨率细节传递给 Decoder。

因此：

```text
ResNet Shortcut
→ 更关注深层网络优化

U-Net Skip Connection
→ 更关注空间细节恢复
```

---

# 13. 训练流程

虽然任务从 Classification 变成了 Image-to-Image，但神经网络训练的基本闭环没有改变：

```text
Input
 ↓
Forward
 ↓
Prediction
 ↓
Loss
 ↓
Backward
 ↓
Gradient
 ↓
Optimizer
 ↓
Update Parameters
```

核心代码仍然是：

```python
optimizer.zero_grad()

prediction = model(input_image)

loss = criterion(prediction, target_image)

loss.backward()

optimizer.step()
```

因此 Level 1～4 的核心训练思想实际上是统一的：

```text
Forward
→ Loss
→ Backward
→ Optimizer
```

变化的是：

```text
模型结构
+
输入形式
+
输出形式
+
Loss
+
Evaluation Metric
```

---

# 14. Loss Function

分类任务使用：

```text
CrossEntropyLoss
```

但图像恢复任务输出的是像素，因此需要使用像素级 Loss。

实验中研究了：

```text
L1 Loss
MSE Loss
Weighted Loss
Changed-Region Loss
```

---

## 14.1 Global L1

L1：

```text
L1 = mean(|Prediction - Target|)
```

衡量预测图片和 Target 的平均绝对像素误差。

L1 对异常值相对不敏感，并且在图像恢复任务中通常能够得到较稳定的结果。

---

## 14.2 Global MSE

MSE：

```text
MSE = mean((Prediction - Target)^2)
```

平方操作使较大的像素误差受到更强惩罚。

---

## 14.3 Changed-Region Loss

整个页面中：

```text
背景 + 印刷内容
```

通常占据绝大多数像素，而真正需要修改的手写区域可能只占较少比例。

如果只优化 Global Loss，模型很容易通过：

```text
尽量保持 Input 不变
```

获得较低的平均误差。

因此进一步对 Difference Mask 对应的区域计算 Loss：

```text
Changed MSE
```

让模型更加关注：

```text
真正需要被修改的区域
```

---

# 15. Weighted Loss

实验使用了组合 Loss：

```text
Loss =
L1
+ 0.5 × Global MSE
+ 2.0 × Changed-Region MSE
```

其思想为：

```text
Global L1
→ 保证整体图像接近 Target

Global MSE
→ 对较大的像素错误进行额外惩罚

Changed-Region MSE
→ 强化真正需要擦除区域的学习
```

最终主模型：

```text
outputs/train_weighted_loss/best_unet_weighted.pth
```

模型大小约：

```text
30 MB
```

---

# 16. Changed-MSE Fine-tuning 实验

为了进一步减少手写残影，在 Weighted Loss 模型基础上进行了 Changed-MSE Fine-tuning。

配置：

```text
Checkpoint:
outputs/train_weighted_loss/best_unet_weighted.pth

Loss:
L1 + 0.5 × Global MSE + 2.0 × Changed-Region MSE

Difference Threshold:
0.1

Learning Rate:
0.0002

Epochs:
3
```

训练结果：

| Epoch | Train Total | Val Total | Val L1 | Val Changed-MSE | PSNR | SSIM |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.169180 | 0.146476 | 0.030286 | 0.055205 | 20.40 dB | 0.8995 |
| 2 | 0.164830 | 0.144460 | 0.032820 | 0.052772 | 19.97 dB | 0.8880 |
| 3 | 0.159636 | 0.139675 | 0.029952 | 0.052033 | 20.53 dB | 0.9016 |

Best Epoch：

```text
3
```

Best Validation Loss：

```text
0.139675
```

Fine-tuning Time：

```text
203.20 s
```

虽然 Changed-MSE Fine-tuning 的 Validation Loss 继续下降，但是实际图片的视觉效果并没有稳定优于原来的 Weighted Loss 模型。

因此最终没有简单地使用：

```text
Validation Loss 最低
```

作为唯一选择标准。

最终完整测试与完整图推理仍采用：

```text
outputs/train_weighted_loss/best_unet_weighted.pth
```

这个实验说明：

> Loss 数值变小并不一定意味着最终视觉效果一定更好。

对于 Image-to-Image 任务，需要结合：

```text
Loss
+
PSNR
+
SSIM
+
实际视觉效果
```

综合分析。

---

# 17. PSNR

PSNR（Peak Signal-to-Noise Ratio）用于衡量重建图像和 Target 之间的像素误差。

其核心与 MSE 有关：

```text
PSNR = 10 log10(MAX² / MSE)
```

对于归一化到：

```text
[0, 1]
```

的图像：

```text
MAX = 1
```

一般而言：

```text
PSNR 越高
→ Prediction 与 Target 的像素误差越小
```

---

# 18. SSIM

SSIM（Structural Similarity Index）从结构角度比较两张图片。

相比只比较逐像素误差的 MSE，SSIM更加关注：

- 局部亮度
- 对比度
- 图像结构

其取值越接近：

```text
1
```

通常表示结构越相似。

文档恢复中 SSIM 很有意义，因为我们不仅希望像素接近，还希望：

```text
文字结构
表格结构
图形结构
```

能够得到保留。

---

# 19. Final Test

最终模型使用：

```text
outputs/train_weighted_loss/best_unet_weighted.pth
```

在独立 Test Set 上进行最终评估。

Test 数据：

```text
Test Original Pairs: 242
Test Patches: 1210
```

Difference Threshold：

```text
0.1
```

最终结果：

| Metric | Result |
|---|---:|
| Global L1 | 0.032711 |
| Global MSE | 0.016915 |
| Changed-Region L1 | 0.150156 |
| Changed Pixel Ratio | 10.11% |
| PSNR | 20.73 dB |
| SSIM | 0.8770 |

可以发现：

```text
Changed Pixel Ratio = 10.11%
```

说明真正发生明显变化的像素只占整个测试 Patch 的一部分。

这也解释了为什么仅使用 Global Loss 容易忽略手写区域。

同时：

```text
Global L1 = 0.032711
Changed-Region L1 = 0.150156
```

Changed Region 的误差明显高于整体误差。

说明当前任务真正困难的部分并不是：

```text
复制大面积背景
```

而是：

```text
正确恢复被手写笔迹覆盖的位置。
```

---

# 20. Final Test 输出

最终测试保存：

```text
outputs/final_test/
├── final_test_results.txt
├── final_test_metrics.csv
├── final_test_predictions.png
├── psnr_distribution.png
└── ssim_distribution.png
```

其中：

```text
final_test_metrics.csv
```

保存每个 Test Patch 的指标，可以进一步分析模型表现的分布。

PSNR / SSIM Distribution 也说明不同 Patch 的恢复难度存在明显差异。

简单区域通常能够获得较高指标，而：

- 大面积手写
- 手写覆盖印刷内容
- 图形与手写重叠
- 复杂背景

属于更困难的情况。

---

# 21. 完整图片推理

训练阶段采用：

```text
256 × 256 Patch
```

但实际应用需要处理：

```text
任意尺寸完整图片
```

因此实现：

```text
infer_full_image.py
```

完整图推理流程：

```text
Original Image
      ↓
Padding
      ↓
Overlapping Patches
      ↓
U-Net Prediction
      ↓
Weighted Blending
      ↓
Crop Padding
      ↓
Final Clean Image
```

---

# 22. Overlapping Patch Inference

如果直接将图片切成互不重叠的 Patch：

```text
Patch | Patch | Patch
```

Patch 边界可能出现明显接缝。

因此使用：

```text
Patch Size = 256
Stride = 192
Overlap = 64
```

即相邻 Patch 存在：

```text
64 pixels
```

的重叠区域。

然后对重叠预测进行 Weighted Blending。

这样可以减轻：

```text
Patch Boundary Artifact
```

最终多张完整图片测试中没有观察到明显的规则性 Patch 网格接缝，说明 Overlapping Patch + Weighted Blending 能够较好地完成完整图拼接。

---

# 23. 完整图测试

为了避免只观察单张“容易样本”，从 Test Set 中使用固定随机种子抽取多张图片进行完整图推理。

测试图片包含：

- 横版试卷
- 竖版试卷
- 大尺寸完整页面
- 中文题目
- 英文阅读
- 数学题
- 几何图
- 大量手写计算
- 少量手写标记
- 不同背景和拍摄质量

完整图实验表明模型已经能够完成基本的：

```text
Handwriting Removal
+
Printed Content Preservation
```

特别是在：

```text
印刷内容清晰
+
手写与印刷内容差异较明显
```

的情况下，模型能够明显削弱甚至基本移除手写笔迹。

---

# 24. 完整图实验现象

通过多张完整图片可以观察到以下现象。

## 24.1 成功现象

### 手写内容明显削弱

大量：

- 手写答案
- 批改标记
- 长线
- 计算过程
- 手写英文

能够被明显削弱或擦除。

### 印刷文字总体得到保留

模型通常能够区分：

```text
Printed Text
```

和：

```text
Handwriting
```

大部分印刷题干能够继续保持清晰。

### 支持不同尺寸

测试图片尺寸包括例如：

```text
952 × 301
952 × 1315
1167 × 656
1280 × 1810
1280 × 1811
1485 × 726
2488 × 3303
```

均能够通过 Patch Inference 处理。

这说明推理流程已经不再受训练 Patch Size 限制。

---

# 25. 当前模型的局限

模型仍然不是完美的手写擦除系统。

## 25.1 手写残影

部分手写笔迹被擦除后仍然留下：

```text
浅灰色轮廓
```

这说明模型已经识别到这些区域需要改变，但预测像素还没有完全恢复到 Target 的背景。

这与：

```text
Changed-Region L1 = 0.150156
```

相吻合。

---

## 25.2 手写与印刷重叠

如果手写内容直接覆盖在印刷文字上，任务本身更加困难。

模型不仅需要：

```text
删除 handwriting
```

还需要推断：

```text
被 handwriting 遮挡的原始 printed content
```

这已经包含一定程度的 Image Restoration / Inpainting 问题。

---

## 25.3 图形内容可能被削弱

对于：

- 几何图形
- 细线
- 表格
- 特殊符号

如果其视觉特征与手写线条相似，模型有可能发生误判。

这说明：

> “手写”和“应该保留的线条”并不能仅依靠简单的颜色或边缘强度完全区分。

---

## 25.4 大面积手写更加困难

当一张图片存在大量密集手写内容时，模型需要恢复的区域明显增大。

此时：

```text
Input 与 Target 差异更大
```

恢复难度也随之增加。

---

# 26. 为什么没有继续无限优化

实验中已经尝试：

```text
Global Loss
↓
Weighted Loss
↓
Difference Mask
↓
Changed-Region MSE
↓
Fine-tuning
```

继续增大 Changed Region 权重并不保证视觉效果持续改善。

过度强调擦除区域可能导致：

```text
手写擦得更干净
```

但同时增加：

```text
印刷内容 / 图形被误伤
```

的风险。

因此该任务实际上存在一个重要 Trade-off：

```text
Handwriting Removal
        ↕
Content Preservation
```

最终选择保留 Weighted Loss 主模型，是因为它在当前实验中提供了相对合理的综合效果。

---

# 27. 从 Level 1 到 Level 4 的知识链

整个任务形成了一条完整的深度学习学习路线。

## Level 1：MLP

```text
28 × 28
 ↓
Flatten
 ↓
784
 ↓
Linear
 ↓
ReLU
 ↓
Linear
 ↓
10 Classes
```

学习：

- Tensor
- Dataset
- DataLoader
- Forward
- Loss
- Backward
- Optimizer
- SGD
- CrossEntropyLoss

---

## Level 2：CNN

```text
Image
 ↓
Conv
 ↓
ReLU
 ↓
Pool
 ↓
Feature Map
 ↓
Classifier
```

学习：

- Locality
- Convolution Kernel
- Feature Map
- Weight Sharing
- Pooling
- Spatial Structure

---

## Level 3：AlexNet / VGG / ResNet

进一步学习深层 CNN 架构：

```text
CNN
 ↓
AlexNet
 ↓
VGG
 ↓
ResNet
```

理解：

- 深度
- Repeated 3×3 Conv
- Dropout
- BatchNorm
- Residual Learning
- Shortcut
- Global Average Pooling

ResNet：

```text
y = F(x) + x
```

---

## Level 4：U-Net

任务从：

```text
Image → Class
```

变成：

```text
Image → Image
```

因此需要：

```text
Encoder
+
Bottleneck
+
Decoder
+
Skip Connection
```

最终形成：

```text
Document
 ↓
Preprocessing
 ↓
Patch
 ↓
U-Net
 ↓
Clean Patch
 ↓
Overlapping Weighted Blending
 ↓
Clean Document
```

---

# 28. 本实验最重要的理解

通过 Level 4，可以看到一个完整深度学习项目不仅仅是：

```text
model = UNet()
```

真正完整的系统包括：

```text
Dataset
    ↓
Data Inspection
    ↓
Preprocessing
    ↓
Train / Validation / Test Split
    ↓
Patch Sampling
    ↓
U-Net
    ↓
Loss Design
    ↓
Backpropagation
    ↓
Validation
    ↓
PSNR / SSIM
    ↓
Checkpoint Selection
    ↓
Final Test
    ↓
Full Image Inference
    ↓
Failure Analysis
```

模型只是整个 Pipeline 中的一部分。

---

# 29. 进一步改进方向

当前模型仍有进一步改进空间。

## 29.1 更精确的 Difference Mask

目前：

```text
|Input - Target| > threshold
```

会同时检测：

- 手写变化
- 背景变化
- 对齐误差
- 印刷边缘变化

未来可以研究更可靠的手写区域 Mask。

---

## 29.2 Difference-aware Sampling

随机 Patch Sampling 可能抽到大量：

```text
Input ≈ Target
```

的 Patch。

可以根据：

```text
Difference Ratio
```

提高包含明显手写区域 Patch 的采样概率。

这样模型能够看到更多真正需要学习的区域。

---

## 29.3 更强的数据增强

可以加入：

- Brightness
- Contrast
- Perspective
- Blur
- Noise
- JPEG Compression

模拟真实手机拍照场景。

但 Input 与 Target 的几何变换必须保持同步。

---

## 29.4 更大的 Patch

例如：

```text
512 × 512
```

可以获得更多上下文信息，但显存消耗也会显著增加。

---

## 29.5 GAN

可以进一步尝试 Conditional GAN。

Generator：

```text
Handwritten Document
        ↓
Clean Document
```

Discriminator 判断：

```text
Prediction
```

是否接近真实 Clean Document。

GAN 有可能提高视觉真实性，但训练通常更加不稳定。

---

## 29.6 Diffusion

Diffusion Model 也可以用于图像恢复 / Inpainting。

潜在优势：

- 更强的生成能力
- 更复杂的内容恢复能力

但代价包括：

- 更高训练成本
- 更慢推理速度
- 更复杂的工程实现

对于当前数据规模和任务要求，U-Net 是更直接且计算成本更低的方案。

---

# 30. 项目输出

主要输出包括：

```text
outputs/
├── train_weighted_loss/
│   ├── best_unet_weighted.pth
│   ├── final_unet_weighted.pth
│   ├── weighted_loss_curve.png
│   ├── loss_components.png
│   ├── psnr_curve.png
│   ├── ssim_curve.png
│   └── weighted_predictions.png
│
├── finetune_changed_mse/
│   ├── best_unet_changed_mse.pth
│   ├── final_unet_changed_mse.pth
│   ├── finetune_loss_curve.png
│   ├── loss_components.png
│   ├── psnr_curve.png
│   ├── ssim_curve.png
│   └── changed_mse_predictions.png
│
├── final_test/
│   ├── final_test_results.txt
│   ├── final_test_metrics.csv
│   ├── final_test_predictions.png
│   ├── psnr_distribution.png
│   └── ssim_distribution.png
│
└── full_image_inference/
    ├── *_cleaned.png
    ├── *_preprocessed.png
    └── *_comparison.png
```

---

# 31. 最终结果

最终主模型：

```text
outputs/train_weighted_loss/best_unet_weighted.pth
```

参数量：

```text
7,762,465
```

最终 Test Set：

```text
242 original image pairs
1210 test patches
```

最终指标：

```text
Global L1:          0.032711
Global MSE:         0.016915
Changed-Region L1: 0.150156
Changed Pixel Ratio: 10.11%
PSNR:               20.73 dB
SSIM:               0.8770
```

完整图推理：

```text
Patch Size: 256
Stride:     192
Overlap:    64
```

最终完成：

```text
Handwritten Document
        ↓
Document Preprocessing
        ↓
Overlapping Patch Inference
        ↓
U-Net
        ↓
Weighted Blending
        ↓
Handwriting-Removed Document
```

---

# 32. 总结

本 Level 完成了一个完整的 U-Net 手写笔迹擦除系统。

不仅实现了模型训练，还完成了：

- 数据集检查
- 成对数据读取
- 图像预处理
- Patch Training
- U-Net Encoder / Decoder
- Skip Connection
- L1 / MSE / Weighted Loss
- Difference Mask
- Changed-Region Loss
- Fine-tuning 实验
- PSNR
- SSIM
- 独立 Test Set
- Checkpoint Selection
- 完整图片推理
- Overlapping Patch
- Weighted Blending
- 多张真实图片定性测试
- 失败案例分析

最终实验也说明：

> 图像恢复任务不能只依赖单一 Loss 或单一指标评价。

一个模型即使 Validation Loss 更低，也不一定具有更好的实际视觉效果。

对于手写擦除任务，需要同时考虑：

```text
Handwriting Removal
+
Printed Content Preservation
+
PSNR / SSIM
+
Visual Quality
```

当前模型仍存在浅灰残影、复杂图形误伤以及大面积手写恢复困难等问题，但已经完成了从训练数据到完整图片推理的端到端流程，并能够在多种真实文档场景中实现明显的手写笔迹削弱与擦除。