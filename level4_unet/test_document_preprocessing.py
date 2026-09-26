from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageFilter

from dataset import HandwritingDataset


# ============================================================
# 1. Config
# ============================================================

DATA_ROOT = "/mnt/d/User/University/QQ/Dian/deli"
VAL_SPLIT = "splits/val.txt"

OUTPUT_DIR = Path(
    "outputs/document_preprocessing"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PATCH_SIZE = 256

# 先可视化多少组 Validation Patch
NUM_EXAMPLES = 8


# ============================================================
# 2. Validation Dataset
#
# 继续使用冻结的 Validation Split。
#
# 不碰 Test Set。
#
# 注意：
# 这里使用 RGB Input，
# 因为我们要从最原始的 RGB 信息开始比较不同预处理方法。
# ============================================================

val_dataset = HandwritingDataset(
    data_root=DATA_ROOT,
    split_file=VAL_SPLIT,
    patch_size=PATCH_SIZE,
    mode="val",
    input_grayscale=False,
)

print("=" * 70)
print("Document Preprocessing Test")
print("=" * 70)

print(
    "Validation Patches：",
    len(val_dataset),
)

assert len(val_dataset) == 241 * 5


# ============================================================
# 3. Tensor → PIL
#
# Dataset 返回：
#
# Input:
# [3,H,W]
#
# Target:
# [1,H,W]
#
# range:
# [0,1]
# ============================================================

def rgb_tensor_to_pil(tensor):

    array = (
        tensor
        .permute(1, 2, 0)
        .numpy()
    )

    array = np.clip(
        array * 255.0,
        0,
        255,
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        array,
        mode="RGB",
    )


def gray_tensor_to_pil(tensor):

    array = (
        tensor
        .squeeze(0)
        .numpy()
    )

    array = np.clip(
        array * 255.0,
        0,
        255,
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        array,
        mode="L",
    )


# ============================================================
# 4. Method 1
#
# Simple Grayscale
#
# RGB
# ↓
# L
#
# 只去掉颜色，
# 不处理阴影 / 背景 / 光照。
# ============================================================

def simple_grayscale(image):

    return image.convert("L")


# ============================================================
# 5. Method 2
#
# Grayscale + Autocontrast
#
# Autocontrast 会重新拉伸灰度范围：
#
# 暗 → 更接近黑
# 亮 → 更接近白
#
# 但它仍然是全局操作，
# 不一定能解决局部阴影。
# ============================================================

def autocontrast_grayscale(image):

    gray = image.convert("L")

    enhanced = ImageOps.autocontrast(
        gray,
        cutoff=1,
    )

    return enhanced


# ============================================================
# 6. Method 3
#
# Background Normalization
#
# 核心思想：
#
# 原图 Gray 可以粗略理解成：
#
#     内容 + 低频背景 / 光照
#
# 我们先用较大的 Gaussian Blur：
#
#     Gray → Background
#
# Blur 会模糊文字和笔画，
# 但保留大范围的：
#
#     阴影
#     光照
#     纸张亮度变化
#
# 然后：
#
#     normalized
#       =
#     gray - background + 255
#
# 如果某区域只是因为阴影整体变暗，
# background 也会相应变暗，
# 两者相减以后阴影会被削弱。
#
# 而文字这种高频局部结构仍然保留。
# ============================================================

def background_normalization(
    image,
    blur_radius=15,
):

    gray = image.convert("L")

    background = gray.filter(
        ImageFilter.GaussianBlur(
            radius=blur_radius
        )
    )

    gray_np = np.asarray(
        gray,
        dtype=np.float32,
    )

    background_np = np.asarray(
        background,
        dtype=np.float32,
    )

    normalized = (
        gray_np
        - background_np
        + 255.0
    )

    normalized = np.clip(
        normalized,
        0,
        255,
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        normalized,
        mode="L",
    )


# ============================================================
# 7. Method 4
#
# Document Enhanced
#
# Background Normalization
#          ↓
# Autocontrast
#
# 第一阶段：
# 尽量去除低频背景 / 光照。
#
# 第二阶段：
# 重新拉开文字与白背景之间的对比。
#
# 注意：
# 仍然没有进行硬二值化。
#
# 所以手写、打印、Logo 等结构
# 理论上都有机会继续保留。
# ============================================================

def document_enhanced(image):

    normalized = background_normalization(
        image,
        blur_radius=15,
    )

    enhanced = ImageOps.autocontrast(
        normalized,
        cutoff=1,
    )

    return enhanced


# ============================================================
# 8. 保存单张处理结果
#
# 方便我们放大观察。
# ============================================================

def save_single_example(
    index,
    original,
    gray,
    autocontrast,
    normalized,
    enhanced,
    target,
):

    example_dir = (
        OUTPUT_DIR
        / f"example_{index + 1:02d}"
    )

    example_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    original.save(
        example_dir
        / "01_original.png"
    )

    gray.save(
        example_dir
        / "02_gray.png"
    )

    autocontrast.save(
        example_dir
        / "03_autocontrast.png"
    )

    normalized.save(
        example_dir
        / "04_background_normalized.png"
    )

    enhanced.save(
        example_dir
        / "05_document_enhanced.png"
    )

    target.save(
        example_dir
        / "06_target.png"
    )


# ============================================================
# 9. Visualization
#
# 每一行：
#
# Original
# Gray
# Autocontrast
# Background Normalized
# Document Enhanced
# Target
# ============================================================

fig, axes = plt.subplots(
    NUM_EXAMPLES,
    6,
    figsize=(
        22,
        NUM_EXAMPLES * 4,
    ),
)


for index in range(
    NUM_EXAMPLES
):

    input_tensor, target_tensor = (
        val_dataset[index]
    )

    original = rgb_tensor_to_pil(
        input_tensor
    )

    target = gray_tensor_to_pil(
        target_tensor
    )


    # --------------------------------------------------------
    # 四种 preprocessing
    # --------------------------------------------------------

    gray = simple_grayscale(
        original
    )

    auto = autocontrast_grayscale(
        original
    )

    normalized = background_normalization(
        original,
        blur_radius=15,
    )

    enhanced = document_enhanced(
        original
    )


    # --------------------------------------------------------
    # 保存每张单独图片
    # --------------------------------------------------------

    save_single_example(
        index=index,
        original=original,
        gray=gray,
        autocontrast=auto,
        normalized=normalized,
        enhanced=enhanced,
        target=target,
    )


    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    axes[index, 0].imshow(
        original
    )

    axes[index, 0].set_title(
        f"Original {index + 1}"
    )

    axes[index, 0].axis(
        "off"
    )


    # --------------------------------------------------------
    # Gray
    # --------------------------------------------------------

    axes[index, 1].imshow(
        gray,
        cmap="gray",
        vmin=0,
        vmax=255,
    )

    axes[index, 1].set_title(
        f"Gray {index + 1}"
    )

    axes[index, 1].axis(
        "off"
    )


    # --------------------------------------------------------
    # Autocontrast
    # --------------------------------------------------------

    axes[index, 2].imshow(
        auto,
        cmap="gray",
        vmin=0,
        vmax=255,
    )

    axes[index, 2].set_title(
        f"Autocontrast {index + 1}"
    )

    axes[index, 2].axis(
        "off"
    )


    # --------------------------------------------------------
    # Background Normalization
    # --------------------------------------------------------

    axes[index, 3].imshow(
        normalized,
        cmap="gray",
        vmin=0,
        vmax=255,
    )

    axes[index, 3].set_title(
        f"Background Norm {index + 1}"
    )

    axes[index, 3].axis(
        "off"
    )


    # --------------------------------------------------------
    # Document Enhanced
    # --------------------------------------------------------

    axes[index, 4].imshow(
        enhanced,
        cmap="gray",
        vmin=0,
        vmax=255,
    )

    axes[index, 4].set_title(
        f"Document Enhanced {index + 1}"
    )

    axes[index, 4].axis(
        "off"
    )


    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    axes[index, 5].imshow(
        target,
        cmap="gray",
        vmin=0,
        vmax=255,
    )

    axes[index, 5].set_title(
        f"Target {index + 1}"
    )

    axes[index, 5].axis(
        "off"
    )


plt.tight_layout()


comparison_path = (
    OUTPUT_DIR
    / "preprocessing_comparison.png"
)


plt.savefig(
    comparison_path,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 10. Finished
# ============================================================

print(
    "\n比较图已保存：",
    comparison_path,
)

print(
    "单独样本目录：",
    OUTPUT_DIR,
)

print(
    "\n"
    + "=" * 70
)

print(
    "Document Preprocessing Test Finished!"
)

print(
    "=" * 70
)