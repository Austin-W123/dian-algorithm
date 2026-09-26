# Level 4：基于 U-Net 的手写笔迹擦除

# 1. 任务目标

Level 4 的目标是完成一个 **Image-to-Image（图像到图像）** 的手写笔迹擦除任务：

> 输入一张包含印刷内容和手写笔迹的试卷 / 作业图片，输出尽可能去除手写内容、同时保留原始印刷内容的干净图片。

与 Level 1～3 的图像分类不同，本任务的输出不再是类别，而是一张与输入对应的图像：

```text
Level 1～3：
Image → Network → Class

Level 4：
Handwritten Document → U-Net → Clean Document
```

训练数据采用成对形式：

```text
Input X      ：包含手写笔迹的图片
Target Y     ：对应的干净图片
Prediction Ŷ ：U-Net(X)

训练目标：Ŷ ≈ Y
```

因此，模型需要学习的不只是“图片属于什么类别”，而是对图像中的像素进行恢复：尽量擦除手写痕迹，同时保留题干、表格、数学符号、几何图形等原始内容。

最终完成的流程包括：

- 成对数据读取与检查；
- 文档图像预处理；
- `256 × 256` Patch Training；
- U-Net 模型训练；
- Difference Mask 与 Changed-Region 分析；
- L1 / MSE / Weighted Loss 实验；
- PSNR / SSIM 定量评估；
- 独立 Test Set 测试；
- 任意尺寸完整图片推理；
- Overlapping Patch + Weighted Blending 拼接；
- 多张真实文档的定性测试与失败案例分析。

---

# 2. 项目文件结构

Level 4 的核心代码、数据划分和实验输出集中在 `level4_unet/` 中。

```text
level4_unet/
├── README.md
├── requirements.txt
│
├── inspect_dataset.py
├── create_splits.py
├── analyze_train_difference.py
│
├── dataset.py
├── dataset_preprocessed.py
├── preprocessing.py
│
├── unet.py
│
├── sanity_train.py
├── train.py
├── train_grayscale.py
├── train_preprocessed.py
├── train_weighted_loss.py
├── finetune_changed_mse.py
│
├── test_dataset.py
├── test_preprocessed_dataset.py
├── test_document_preprocessing.py
├── test_unet.py
├── test_recursive_inference.py
├── test_final.py
│
├── infer_full_image.py
│
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
│
└── outputs/
    ├── sanity/
    ├── train/
    ├── train_grayscale/
    ├── train_preprocessed/
    ├── train_weighted_loss/
    ├── finetune_changed_mse/
    ├── difference_analysis/
    ├── document_preprocessing/
    ├── final_test/
    └── full_image_inference/
```

## 2.1 数据检查与划分

| 文件 | 作用 |
|---|---|
| `inspect_dataset.py` | 在正式训练前检查 Input / Output 是否一一对应、图片能否正常读取、尺寸是否一致，并了解数据的尺寸和颜色模式等基本情况。 |
| `create_splits.py` | 将原始的成对数据划分为 Train / Validation / Test，并把划分结果保存到 `splits/`。 |
| `analyze_train_difference.py` | 分析 Input 与 Target 之间的像素差异，观察真正发生变化的区域及其比例，为 Difference Mask 和 Changed-Region Loss 的设计提供依据。 |

## 2.2 Dataset 与预处理

| 文件 | 作用 |
|---|---|
| `dataset.py` | 实现基础的成对图片 Dataset，保证 Input 与 Target 使用相同的裁剪位置，生成训练所需 Patch。 |
| `dataset_preprocessed.py` | 在 Dataset 中加入文档预处理流程，为后续训练提供预处理后的单通道输入。 |
| `preprocessing.py` | 实现灰度化、Background Normalization、Document Enhancement 等文档图像预处理操作。 |

## 2.3 模型

| 文件 | 作用 |
|---|---|
| `unet.py` | 定义本任务使用的 U-Net，包括 Encoder、Bottleneck、Decoder 和 Skip Connection。最终模型参数量为 `7,762,465`。 |

## 2.4 训练与实验

| 文件 | 作用 |
|---|---|
| `sanity_train.py` | 在正式进行较长训练前进行小规模 Sanity Check，确认 Dataset、模型、Loss 和反向传播能够正常工作。 |
| `train.py` | 基础训练版本，用于建立最初的 Image-to-Image 训练闭环。 |
| `train_grayscale.py` | 灰度输入实验，用于比较输入表示方式。 |
| `train_preprocessed.py` | 加入 Background Normalization 等文档预处理，并保持 L1 Loss，单独观察预处理带来的影响。 |
| `train_weighted_loss.py` | 最终主训练方案。使用 L1、Global MSE 与 Changed-Region MSE 的组合损失，使模型同时关注整体恢复和真正需要修改的区域。 |
| `finetune_changed_mse.py` | 在 Weighted Loss 模型基础上继续针对 Changed Region 进行微调，用于研究能否进一步减少手写残影。 |

## 2.5 测试与推理

| 文件 | 作用 |
|---|---|
| `test_dataset.py` | 检查基础 Dataset 的配对、Patch 尺寸等是否正确。 |
| `test_preprocessed_dataset.py` | 检查加入预处理后的 Dataset 输出是否符合预期。 |
| `test_document_preprocessing.py` | 可视化文档预处理各阶段结果，检查预处理是否改善背景、光照和文档结构。 |
| `test_unet.py` | 检查 U-Net 的输入输出尺寸和基本前向传播。 |
| `test_recursive_inference.py` | 对相关推理方案进行实验并记录指标。 |
| `test_final.py` | 使用最终选择的 Checkpoint 在独立 Test Set 上计算 Global L1、Global MSE、Changed-Region L1、PSNR、SSIM 等最终指标。 |
| `infer_full_image.py` | 对任意尺寸完整图片进行推理：Padding → Overlapping Patches → U-Net → Weighted Blending → Crop，最终得到完整清理图片。 |

## 2.6 数据划分与实验输出

`splits/` 保存固定的数据划分：

```text
splits/
├── train.txt
├── val.txt
└── test.txt
```

这样不同实验可以使用相同的数据划分，避免由于每次随机划分不同而影响实验比较。

`outputs/` 保存训练曲线、预测图、指标文件和推理结果。

其中最重要的目录为：

```text
outputs/
├── train_weighted_loss/
│   ├── best_unet_weighted.pth
│   ├── weighted_loss_curve.png
│   ├── loss_components.png
│   ├── psnr_curve.png
│   ├── ssim_curve.png
│   └── weighted_predictions.png
│
├── final_test/
│   ├── final_test_results.txt
│   ├── final_test_metrics.csv
│   ├── final_test_predictions.png
│   ├── psnr_distribution.png
│   └── ssim_distribution.png
│
└── full_image_inference/
    ├── *_preprocessed.png
    ├── *_cleaned.png
    └── *_comparison.png
```

最终正式使用的模型为：

```text
outputs/train_weighted_loss/best_unet_weighted.pth
```

模型文件大小约 `30 MB`。

---

# 3. 实验结果与结果分析

## 3.1 最终模型

最终测试与完整图推理使用：

```text
outputs/train_weighted_loss/best_unet_weighted.pth
```

模型参数量：

```text
Total Parameters:     7,762,465
Trainable Parameters: 7,762,465
```

最终没有选择 Changed-MSE Fine-tuning 后的模型。

原因并不是 Fine-tuning 没有使 Validation Loss 下降，而是：

> 更低的 Validation Loss 并没有稳定对应更好的完整图片视觉效果。

因此 Checkpoint 的选择不能只看一个数字，而是综合考虑：

```text
Loss
+ PSNR
+ SSIM
+ 实际视觉效果
```

最终保留 Weighted Loss 主训练得到的模型作为正式模型。

---

## 3.2 独立 Test Set

最终 Test Set 包含：

```text
242 original image pairs
1210 test patches
```

Difference Threshold：

```text
0.1
```

最终结果：

| Metric | Result |
|---|---:|
| Global L1 | `0.032711` |
| Global MSE | `0.016915` |
| Changed-Region L1 | `0.150156` |
| Changed Pixel Ratio | `10.11%` |
| PSNR | `20.73 dB` |
| SSIM | `0.8770` |

---

## 3.3 Global L1 与 Global MSE

最终：

```text
Global L1  = 0.032711
Global MSE = 0.016915
```

Global L1 表示在整个测试 Patch 上，Prediction 与 Target 的平均绝对像素误差。

Global MSE 则会对较大的像素偏差给予更高的惩罚，可以补充 L1 对整体误差的评价。

但是 Global 指标存在一个重要问题：

文档图片中，大面积区域本身就不需要发生改变。

例如：

```text
空白背景
印刷题干
表格
原本正确的图形
```

这些区域在 Input 和 Target 中可能本来就非常接近。

因此，即使模型只是很好地保留这些区域，也可以得到较低的 Global Loss。

所以：

> Global Loss 可以衡量整体恢复质量，但不能单独说明“手写到底擦得怎么样”。

---

## 3.4 Changed Pixel Ratio 与 Changed-Region L1

最终测试结果：

```text
Changed Pixel Ratio = 10.11%
```

说明按照当前 Difference Mask 的定义，真正发生明显变化的像素只占测试 Patch 的一部分。

同时：

```text
Global L1         = 0.032711
Changed-Region L1 = 0.150156
```

Changed Region 的误差明显高于整体误差。

这说明任务真正困难的部分并不是复制大面积本来就正确的背景，而是：

> 在手写出现的位置正确删除笔迹，并尽可能恢复原本应该存在的背景或印刷内容。

这也是后续引入 Changed-Region MSE 的直接原因。

---

## 3.5 PSNR

最终：

```text
PSNR = 20.73 dB
```

对于归一化到 `[0, 1]` 的图像：

```text
PSNR = 10 × log10(1 / MSE)
```

PSNR 与 MSE 直接相关。

一般来说：

```text
MSE 越小
→ PSNR 越高
→ Prediction 与 Target 的整体像素差异越小
```

因此 PSNR 可以从整体像素恢复的角度评价模型。

但是对于本项目来说，PSNR 仍然不能单独判断手写是否真的被正确擦除，因为背景区域也会对这一指标产生较大影响。

---

## 3.6 SSIM

最终：

```text
SSIM = 0.8770
```

SSIM 即 Structural Similarity Index，重点比较两张图片的结构相似性，包括局部亮度、对比度和结构信息。

对于文档恢复任务，这个指标很重要。

因为我们不仅希望：

```text
Prediction 的像素值 ≈ Target
```

还希望：

```text
题干结构
数学符号
表格
几何图形
字符边缘
```

能够被正确保留下来。

因此 SSIM 可以作为 PSNR 和像素级 Loss 的补充。

---

## 3.7 完整图片推理结果

训练阶段模型只接收：

```text
256 × 256 Patch
```

但最终推理流程已经可以处理不同尺寸的完整图片。

完整图片实验中观察到的主要效果：

- 手写答案能够被明显削弱或擦除；
- 批改标记能够被一定程度去除；
- 手写计算过程能够被明显减少；
- 手写英文和部分长线能够被处理；
- 大部分印刷题干能够继续保持清晰；
- 不同尺寸和长宽比的图片均可以通过 Patch Inference 处理；
- 使用 Overlapping Patch + Weighted Blending 后，没有观察到明显的规则性 Patch 网格接缝。

这说明最终系统已经从：

```text
只能预测 256 × 256 Patch
```

扩展到了：

```text
可以处理实际完整文档
```

---

## 3.8 当前模型的局限

### 1. 手写残影

部分笔迹被擦除以后仍然会留下浅灰色轮廓。

这与：

```text
Changed-Region L1 = 0.150156
```

相对应。

说明模型知道这些区域需要改变，但还不能总是准确恢复到 Target。

### 2. 手写与印刷内容重叠

如果手写直接覆盖题干，任务就不只是“删除一个像素区域”。

模型还需要推断：

```text
被手写遮住的地方
原来应该是什么？
```

这时任务实际上已经带有 Image Restoration / Inpainting 的性质。

### 3. 细线和图形可能被误伤

例如：

```text
几何图形
表格线
特殊数学符号
细小印刷字符
```

这些内容与手写线条在局部视觉上可能比较相似。

因此模型有时会把本应保留的细线一起削弱。

### 4. 大面积密集手写更加困难

需要恢复的区域越大：

```text
Input 与 Target 差异越大
→ 模型需要预测的信息越多
→ 恢复难度越高
```

因此，本任务存在一个非常重要的 Trade-off：

```text
Handwriting Removal
        ↕
Content Preservation
```

并不是：

```text
擦得越多越好
```

而是：

> 在去除手写和保护原始内容之间取得平衡。

---

# 4. 数据集

## 4.1 数据组织

原始数据由三批成对文档图片组成：

```text
deli/
├── 20250211/
│   └── dataset/
│       ├── input/
│       └── output/
│
├── 20250212/
│   └── dataset/
│       ├── input/
│       └── output/
│
└── 20250213/
    └── dataset/
        ├── input/
        └── output/
```

其中：

```text
input/
```

存放包含手写笔迹的图片。

```text
output/
```

存放对应的干净目标图片。

Input 与 Output 使用相同文件名形成一一对应关系。

---

## 4.2 数据数量

数据量如下：

| 数据目录 | Input | Output |
|---|---:|---:|
| `20250211` | 812 | 812 |
| `20250212` | 900 | 900 |
| `20250213` | 700 | 700 |
| **Total** | **2412** | **2412** |

因此总共有：

```text
2412 pairs
4824 JPG images
```

也就是说：

```text
2412 张 Input
+
2412 张 Target
=
4824 张图片
```

---

## 4.3 数据特点

数据不是统一尺寸的标准数据集，而是真实试卷 / 作业场景。

内容包括：

- 中文；
- 英文；
- 数学题；
- 几何图形；
- 表格；
- 填空题；
- 手写答案；
- 计算过程；
- 批改痕迹。

同时，真实拍摄图片还存在：

- 图片尺寸不同；
- 长宽比不同；
- 光照不均匀；
- 页面阴影；
- 背景发灰；
- 透页；
- 拍摄角度差异；
- 手写颜色不同；
- 手写密度不同。

因此任务并不是简单地：

```text
找到某一种颜色
→ 删除
```

而是需要模型学习：

```text
哪些视觉结构应该保留？
哪些视觉结构应该去除？
```

---

## 4.4 为什么必须检查 Pair

Image-to-Image 使用像素级监督：

```text
Input Patch ↔ Target Patch
```

如果 Input 与 Target 文件配错：

```text
Input A
↓
Target B
```

模型得到的监督信息就是错误的。

同样，如果 Input 与 Target 裁剪位置不同：

```text
Input  ：左上角
Target ：右下角
```

Loss 比较的就不再是同一个位置。

因此正式训练前必须检查：

```text
文件名是否对应
图片是否缺失
图片能否正常读取
Input / Target 尺寸是否一致
颜色模式
图片尺寸分布
```

---

## 4.5 Train / Validation / Test

项目将数据划分固定保存到：

```text
splits/train.txt
splits/val.txt
splits/test.txt
```

三部分分别承担不同作用：

```text
Train
→ 用于反向传播
→ 更新模型参数

Validation
→ 不参与参数更新
→ 观察训练效果
→ 选择 Checkpoint

Test
→ 不参与训练和模型选择
→ 最终独立评价
```

最终 Test Set 包含：

```text
242 对原始图片
```

并在最终 Patch 测试中形成：

```text
1210 个 Test Patches
```

---

# 5. 方法与实验策略

## 5.1 整体 Pipeline

最终系统可以概括为：

```text
Paired Dataset
      ↓
Data Inspection
      ↓
Train / Validation / Test Split
      ↓
Document Preprocessing
      ↓
256 × 256 Patch Sampling
      ↓
U-Net
      ↓
Weighted Loss
      ↓
Validation
      ↓
Checkpoint Selection
      ↓
Final Test
      ↓
Full-image Overlapping Patch Inference
      ↓
Weighted Blending
      ↓
Handwriting-Removed Document
```

在这个任务中，模型只是整个 Pipeline 中的一部分。

真正决定最终效果的还有：

```text
数据质量
预处理
Patch Sampling
Loss
评价方法
Checkpoint 选择
完整图推理方式
```

---

## 5.2 文档预处理

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

也就是：

```text
RGB
→ L
→ Background Normalization
→ Document Enhancement
```

真实文档图片可能存在：

```text
光照不均
背景发灰
页面阴影
不同拍摄环境
```

如果直接把这些差异全部交给模型学习，模型需要同时处理：

```text
背景亮度变化
+
手写擦除
+
内容恢复
```

因此预处理的主要目标是：

> 尽量降低与任务无关的背景变化，使模型更加关注文字、笔迹和图形结构。

预处理后模型使用单通道输入，因此 U-Net 配置为：

```text
in_channels  = 1
out_channels = 1
```

---

## 5.3 为什么采用 256 × 256 Patch Training

原始图片尺寸差异较大，而且部分图片分辨率较高。

实验设备 GPU 显存为：

```text
8 GB
```

如果直接把完整高分辨率图片输入 U-Net：

```text
High Resolution Image
        ↓
Large Feature Maps
        ↓
Large GPU Memory Usage
```

U-Net 在 Encoder 和 Decoder 中需要保存大量 Feature Map，因此完整高分辨率训练的显存开销很大。

另一种方法是：

```text
把整张图片缩小
```

但如果缩得太小：

```text
小字体
细线
手写笔画
字符边缘
```

都可能丢失。

因此最终采用：

```text
Patch Size = 256 × 256
```

Patch Training 的主要优点：

1. 控制显存占用；
2. 保留局部高分辨率细节；
3. 不需要把整张文档强行缩小；
4. 一张原图可以提供多个局部训练样本；
5. 比较适合 U-Net 的局部图像恢复。

训练时最重要的一点是：

```text
Input 与 Target
必须使用完全相同的 Crop 坐标
```

即：

```text
Input Image
    ↓ 同一坐标
Input Patch

Target Image
    ↓ 同一坐标
Target Patch
```

否则像素级监督将失去意义。

---

## 5.4 U-Net

本任务采用 U-Net。

整体结构：

```text
Input
  ↓
Encoder
  ↓
Bottleneck
  ↓
Decoder
  ↓
Output
```

Encoder 负责逐渐提取更高层的特征：

```text
Spatial Size ↓
Channels ↑
```

即：

```text
H、W 越来越小
Feature Channels 越来越多
```

Bottleneck 位于网络最深处，用于表示经过高度压缩的语义特征。

Decoder 则执行相反的过程：

```text
Spatial Size ↑
Channels ↓
```

逐渐恢复空间分辨率。

最终输出：

```text
256 × 256 Prediction
```

与 Target Patch 进行像素级比较。

最终模型参数量：

```text
7,762,465
```

---

## 5.5 Skip Connection

如果只有：

```text
Encoder
↓
Bottleneck
↓
Decoder
```

经过多次 Downsampling 以后，一些高分辨率信息容易丢失。

例如：

```text
小字体
字符边缘
表格线
几何图形
细小数学符号
```

这些恰恰是文档恢复任务中需要保留的重要信息。

因此 U-Net 使用 Skip Connection：

```text
Encoder Feature ─────────────┐
                             ↓
Decoder Feature → Concatenate → Convolution
```

Encoder 中保存的高分辨率 Feature Map 会直接传递给对应的 Decoder。

这样 Decoder 不需要完全依靠 Bottleneck 恢复所有细节。

它和 Level 3 学习的 ResNet Shortcut 有区别：

```text
ResNet：
F(x) + x
→ Addition
→ 主要帮助深层网络优化和梯度传播

U-Net：
Encoder Feature + Decoder Feature
→ Concatenation
→ 主要帮助恢复空间细节
```

---

## 5.6 Difference Mask

文档图像中真正需要修改的像素比例并不高。

为了定位 Input 与 Target 中发生明显变化的位置，定义：

```text
Difference = |Input - Target|
```

然后：

```text
Mask = |Input - Target| > 0.1
```

其中：

```text
Difference Threshold = 0.1
```

Mask 中的像素通常包括：

```text
手写答案
批改痕迹
手写线条
计算过程
其他 Input / Target 明显不同的位置
```

但是需要注意：

> Difference Mask 并不是严格意义上的“手写语义分割标签”。

因为 Input 和 Target 之间还可能存在：

```text
光照差异
轻微对齐误差
背景变化
印刷边缘差异
```

这些区域同样可能被 Difference Mask 捕获。

因此更准确的定义是：

> Difference Mask 表示 Input 与 Target 之间的显著变化区域。

---

## 5.7 为什么不能只使用 Global Loss

最终测试中：

```text
Changed Pixel Ratio = 10.11%
```

也就是说，大部分像素实际上属于：

```text
不需要修改的区域
```

如果模型直接：

```text
Prediction ≈ Input
```

那么大量背景和印刷内容仍然是正确的。

于是即使模型没有很好地擦除手写：

```text
Global Loss
```

也可能看起来比较低。

因此：

```text
低 Global Loss
≠
手写一定擦得好
```

这也是本项目后期最重要的实验发现之一。

---

## 5.8 L1 与 MSE 的取舍

### L1 Loss

L1 定义：

```text
L1 = mean(|Prediction - Target|)
```

它计算平均绝对误差。

特点：

```text
所有像素误差近似线性惩罚
对极端误差没有 MSE 那么敏感
训练相对稳定
```

因此可以作为整体图像恢复的基础 Loss。

### MSE Loss

MSE 定义：

```text
MSE = mean((Prediction - Target)²)
```

由于误差会被平方：

```text
较大的误差
→ 得到更大的惩罚
```

所以 MSE 更强调那些预测错误比较大的像素。

两者并不是：

```text
L1 好
MSE 不好
```

或者：

```text
MSE 好
L1 不好
```

而是关注点不同：

```text
L1
→ 稳定约束整体图像

MSE
→ 更强调较大的像素错误
```

---

## 5.9 最终 Weighted Loss

最终主训练方案使用：

```text
Loss =
L1
+ 0.5 × Global MSE
+ 2.0 × Changed-Region MSE
```

即：

```text
L_total =
L_L1
+ 0.5 L_global_MSE
+ 2.0 L_changed_MSE
```

三部分分别承担不同任务。

### Global L1

```text
L1
```

作用：

> 保证整个 Prediction 不要偏离 Target 太远。

也就是说，不能为了擦手写而把整张图片改坏。

### Global MSE

```text
0.5 × Global MSE
```

作用：

> 对整个图像中较大的像素错误给予额外惩罚。

### Changed-Region MSE

```text
2.0 × Changed-Region MSE
```

作用：

> 让模型更加重视 Input 与 Target 真正发生变化的位置。

因此整个 Weighted Loss 实际上是在同时解决两个互相制约的问题：

```text
整体内容不能乱改
+
手写区域又必须有足够强的修改动力
```

最终主模型保存为：

```text
outputs/train_weighted_loss/best_unet_weighted.pth
```

---

## 5.10 Changed-MSE Fine-tuning 实验

为了进一步减少残留手写，在 Weighted Loss Checkpoint 基础上继续进行了 Changed-MSE Fine-tuning。

实验设置：

```text
Epochs        = 3
Learning Rate = 0.0002
Threshold     = 0.1
```

实验结果：

| Epoch | Train Total | Val Total | Val L1 | Val Changed-MSE | PSNR | SSIM |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.169180 | 0.146476 | 0.030286 | 0.055205 | 20.40 dB | 0.8995 |
| 2 | 0.164830 | 0.144460 | 0.032820 | 0.052772 | 19.97 dB | 0.8880 |
| 3 | 0.159636 | **0.139675** | 0.029952 | **0.052033** | **20.53 dB** | **0.9016** |

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

从数值上看：

```text
Validation Loss
```

继续下降。

但是在完整图片上进行视觉检查以后，Fine-tuning 模型并没有稳定优于原来的 Weighted Loss Checkpoint。

原因是继续加强 Changed Region 后：

```text
手写擦除能力 ↑
```

的同时也可能导致：

```text
印刷文字误伤 ↑
细线误伤 ↑
图形结构损失 ↑
```

因此最终没有简单地选择：

```text
Validation Loss 最低的模型
```

而是仍然选择原来的：

```text
best_unet_weighted.pth
```

这个实验带来的一个重要认识是：

> 对 Image-to-Image 任务，Validation Loss 最低并不自动等于实际视觉效果最好。

---

## 5.11 完整图片推理

训练阶段：

```text
256 × 256 Patch
        ↓
      U-Net
        ↓
256 × 256 Prediction
```

但是实际使用时输入的是：

```text
完整试卷 / 作业图片
```

因此 `infer_full_image.py` 实现：

```text
Original Image
      ↓
Preprocessing
      ↓
Padding
      ↓
Overlapping 256 × 256 Patches
      ↓
U-Net Prediction
      ↓
Weighted Blending
      ↓
Crop Padding
      ↓
Final Clean Image
```

这样就可以让只在固定 Patch 上训练的网络处理任意尺寸的完整图片。

---

## 5.12 为什么使用 Overlap + Weighted Blending

如果直接把图片切成互不重叠的 Patch：

```text
┌───────┬───────┬───────┐
│ Patch │ Patch │ Patch │
├───────┼───────┼───────┤
│ Patch │ Patch │ Patch │
└───────┴───────┴───────┘
```

Patch 边缘位置缺少周围上下文。

因此不同 Patch 在边缘位置可能给出不同预测。

重新拼接以后就可能出现：

```text
Patch Boundary Artifact
```

也就是规则性的网格接缝。

最终使用：

```text
Patch Size = 256
Stride     = 192
Overlap    = 64
```

也就是说，相邻 Patch 之间有：

```text
64 pixels
```

的重叠。

然后对重叠区域进行 Weighted Blending：

```text
Patch A ──┐
          ├─ Weighted Average → Final Pixel
Patch B ──┘
```

这样同一个位置可以由多个 Patch 的预测共同决定。

最终完整图实验中没有观察到明显的规则性网格接缝，说明这一方法能够较好地把 Patch 模型扩展到完整文档。

---

## 5.13 最终评价策略

本项目没有只依赖一个指标。

而是同时使用：

```text
Global L1 / MSE
→ 看整体像素误差

Changed-Region L1
→ 看真正发生变化区域的恢复难度

PSNR
→ 看整体像素重建质量

SSIM
→ 看结构相似性

Full-image Visual Quality
→ 看手写是否真正减少
→ 看印刷内容是否被保护
```

对于手写擦除任务，最终真正关心的是：

```text
Handwriting Removal
+
Printed Content Preservation
+
Quantitative Metrics
+
Visual Quality
```

而不是单纯：

```text
哪个 Loss 最小
```

---

# 6. 总结

本 Level 完成了一个从成对训练数据到完整图片推理的 U-Net 手写笔迹擦除系统。

与前面的分类任务相比，本实验最大的变化是任务从：

```text
Image → Class
```

进入：

```text
Image → Image
```

因此除了神经网络本身，还必须同时处理：

```text
成对数据对齐
文档图像预处理
Patch Sampling
像素级 Loss
图像质量评价
Checkpoint Selection
完整图片拼接
```

最终模型在独立 Test Set 上得到：

```text
Global L1:           0.032711
Global MSE:          0.016915
Changed-Region L1:   0.150156
Changed Pixel Ratio: 10.11%
PSNR:                20.73 dB
SSIM:                0.8770
```

完整图推理采用：

```text
Patch Size = 256 × 256
Stride     = 192
Overlap    = 64
Weighted Blending
```

实际结果表明，模型能够明显削弱或去除多种手写内容，同时总体保留印刷题干，并能够处理不同尺寸的真实文档图片。

当前模型仍然存在：

```text
浅灰色手写残影
手写覆盖印刷内容时恢复困难
复杂图形可能被误伤
细线可能被削弱
大面积密集手写处理困难
```

因此，本实验最重要的认识之一是：

> 图像恢复不能只追求更低的 Loss。真正的目标是在“需要改变的区域”和“必须保护的区域”之间取得平衡。

---

## 关键词总结

| 关键词 | 本项目中的含义 |
|---|---|
| Image-to-Image | 输入图片，输出恢复后的图片 |
| Paired Dataset | Input 与 Target 一一对应的监督数据 |
| Patch Training | 使用 `256 × 256` 局部区域训练 |
| U-Net | Encoder + Bottleneck + Decoder 的图像恢复网络 |
| Skip Connection | 将 Encoder 的高分辨率特征传给 Decoder |
| Background Normalization | 减弱不均匀背景与光照影响 |
| Difference Mask | 根据 Input / Target 差异定位显著变化区域 |
| L1 | 稳定约束整体像素误差 |
| MSE | 更强惩罚较大的像素误差 |
| Changed-Region MSE | 强化真正需要修改区域的学习 |
| Weighted Loss | 综合整体恢复与局部擦除目标 |
| PSNR | 基于像素误差的图像质量指标 |
| SSIM | 更关注结构相似性的图像质量指标 |
| Checkpoint Selection | 综合 Loss、指标和视觉效果选择模型 |
| Overlapping Patch | 完整图推理时使用相互重叠的 Patch |
| Weighted Blending | 平滑融合重叠 Patch，减少边界接缝 |
| Trade-off | 手写擦除强度与印刷内容保护之间的权衡 |