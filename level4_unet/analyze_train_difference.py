from pathlib import Path
import random
import csv

import numpy as np
from PIL import Image, ImageOps

import matplotlib.pyplot as plt


# ============================================================
# 1. 配置
# ============================================================

DATA_ROOT = Path(
    "/mnt/d/User/University/QQ/Dian/deli"
)

TRAIN_SPLIT = Path(
    "splits/train.txt"
)

OUTPUT_DIR = Path(
    "outputs/difference_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


PATCH_SIZE = 256

# 每张训练图片模拟多少次 Random Crop
RANDOM_PATCHES_PER_IMAGE = 5

# 灰度像素差异阈值。
#
# Input / Target 都转换到 0~255 灰度空间。
#
# abs(Input - Target) > 30
# 才认为这个像素发生了“明显变化”。
DIFF_THRESHOLD = 30

SEED = 42

rng = random.Random(SEED)


# ============================================================
# 2. 读取已经冻结的 Train Split
# ============================================================

def load_train_pairs():

    pairs = []

    with open(
        TRAIN_SPLIT,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 2:

                raise ValueError(
                    f"Split 格式错误："
                    f"第 {line_number} 行\n"
                    f"{line}"
                )

            input_relative = parts[0]
            target_relative = parts[1]

            input_path = (
                DATA_ROOT
                / input_relative
            )

            target_path = (
                DATA_ROOT
                / target_relative
            )

            if not input_path.exists():

                raise FileNotFoundError(
                    f"Input 不存在："
                    f"{input_path}"
                )

            if not target_path.exists():

                raise FileNotFoundError(
                    f"Target 不存在："
                    f"{target_path}"
                )

            pairs.append(
                (
                    input_path,
                    target_path,
                )
            )

    return pairs


# ============================================================
# 3. 小图 Padding
#
# 与 Dataset 的思想保持一致：
#
# 如果图片小于 256×256，
# 先补成至少 256×256。
# ============================================================

def pad_if_needed(
    input_image,
    target_image,
):

    width, height = input_image.size

    pad_width = max(
        0,
        PATCH_SIZE - width,
    )

    pad_height = max(
        0,
        PATCH_SIZE - height,
    )

    if (
        pad_width == 0
        and pad_height == 0
    ):
        return input_image, target_image

    left = pad_width // 2
    right = pad_width - left

    top = pad_height // 2
    bottom = pad_height - top

    padding = (
        left,
        top,
        right,
        bottom,
    )

    # 分析阶段 Input 已经转成 L，
    # 因此白色就是 255。
    input_image = ImageOps.expand(
        input_image,
        border=padding,
        fill=255,
    )

    target_image = ImageOps.expand(
        target_image,
        border=padding,
        fill=255,
    )

    return input_image, target_image


# ============================================================
# 4. 计算 Difference Ratio
# ============================================================

def calculate_difference_ratio(
    input_array,
    target_array,
):

    # 转成有符号类型。
    #
    # 如果直接 uint8 相减，
    # 可能出现下溢问题。
    input_array = input_array.astype(
        np.int16
    )

    target_array = target_array.astype(
        np.int16
    )

    difference = np.abs(
        input_array - target_array
    )

    difference_mask = (
        difference > DIFF_THRESHOLD
    )

    ratio = difference_mask.mean()

    return (
        ratio,
        difference,
        difference_mask,
    )


# ============================================================
# 5. Random Patch Difference Ratio
# ============================================================

def sample_random_patch_ratios(
    input_array,
    target_array,
):

    height, width = input_array.shape

    ratios = []

    for _ in range(
        RANDOM_PATCHES_PER_IMAGE
    ):

        left = rng.randint(
            0,
            width - PATCH_SIZE,
        )

        top = rng.randint(
            0,
            height - PATCH_SIZE,
        )

        input_patch = input_array[
            top:top + PATCH_SIZE,
            left:left + PATCH_SIZE,
        ]

        target_patch = target_array[
            top:top + PATCH_SIZE,
            left:left + PATCH_SIZE,
        ]

        ratio, _, _ = (
            calculate_difference_ratio(
                input_patch,
                target_patch,
            )
        )

        ratios.append(ratio)

    return ratios


# ============================================================
# 6. 主分析
# ============================================================

pairs = load_train_pairs()


print("=" * 70)
print("Train Difference Analysis")
print("=" * 70)

print(
    "Train Pair 数量：",
    len(pairs),
)

print(
    "Difference Threshold：",
    DIFF_THRESHOLD,
)

print(
    "Patch Size：",
    PATCH_SIZE,
)

print(
    "Random Patches / Image：",
    RANDOM_PATCHES_PER_IMAGE,
)


assert len(pairs) == 1929


image_results = []

all_patch_ratios = []


for index, (
    input_path,
    target_path,
) in enumerate(
    pairs,
    start=1,
):

    # --------------------------------------------------------
    # Input → 灰度
    # Target → 灰度
    # --------------------------------------------------------

    with Image.open(input_path) as image:

        input_image = image.convert("L")

    with Image.open(target_path) as image:

        target_image = image.convert("L")


    if input_image.size != target_image.size:

        raise ValueError(
            "尺寸不一致：\n"
            f"{input_path}\n"
            f"{target_path}"
        )


    # --------------------------------------------------------
    # 为 Random Patch 模拟做 Padding
    # --------------------------------------------------------

    input_image, target_image = (
        pad_if_needed(
            input_image,
            target_image,
        )
    )


    input_array = np.array(
        input_image,
        dtype=np.uint8,
    )

    target_array = np.array(
        target_image,
        dtype=np.uint8,
    )


    # --------------------------------------------------------
    # 整张图片 Difference Ratio
    # --------------------------------------------------------

    image_ratio, _, _ = (
        calculate_difference_ratio(
            input_array,
            target_array,
        )
    )


    image_results.append(
        {
            "input_path": str(
                input_path.relative_to(
                    DATA_ROOT
                )
            ),

            "target_path": str(
                target_path.relative_to(
                    DATA_ROOT
                )
            ),

            "difference_ratio": (
                image_ratio
            ),
        }
    )


    # --------------------------------------------------------
    # 模拟当前 Random Crop 策略
    # --------------------------------------------------------

    patch_ratios = (
        sample_random_patch_ratios(
            input_array,
            target_array,
        )
    )

    all_patch_ratios.extend(
        patch_ratios
    )


    if (
        index % 200 == 0
        or index == len(pairs)
    ):

        print(
            f"已分析："
            f"{index}/{len(pairs)}"
        )


# ============================================================
# 7. 转成 NumPy
# ============================================================

image_ratios = np.array(
    [
        item["difference_ratio"]
        for item in image_results
    ],
    dtype=np.float64,
)

patch_ratios = np.array(
    all_patch_ratios,
    dtype=np.float64,
)


# ============================================================
# 8. 通用统计函数
# ============================================================

def print_statistics(
    name,
    ratios,
):

    print("\n" + "=" * 70)

    print(name)

    print("=" * 70)


    print(
        f"数量："
        f"{len(ratios)}"
    )

    print(
        f"最小值："
        f"{ratios.min() * 100:.4f}%"
    )

    print(
        f"最大值："
        f"{ratios.max() * 100:.4f}%"
    )

    print(
        f"平均值："
        f"{ratios.mean() * 100:.4f}%"
    )

    print(
        f"中位数："
        f"{np.median(ratios) * 100:.4f}%"
    )


    print("\nPercentiles：")

    for percentile in [
        10,
        25,
        50,
        75,
        90,
        95,
    ]:

        value = np.percentile(
            ratios,
            percentile,
        )

        print(
            f"P{percentile:02d}: "
            f"{value * 100:.4f}%"
        )


    # --------------------------------------------------------
    # 分桶
    # --------------------------------------------------------

    bins = [
        (0.00, 0.001),
        (0.001, 0.01),
        (0.01, 0.05),
        (0.05, 0.20),
        (0.20, 1.01),
    ]

    labels = [
        "< 0.1%",
        "0.1% ~ 1%",
        "1% ~ 5%",
        "5% ~ 20%",
        ">= 20%",
    ]


    print("\nDifference Ratio 分布：")


    for (
        lower,
        upper,
    ), label in zip(
        bins,
        labels,
    ):

        if lower == 0:

            mask = (
                ratios < upper
            )

        else:

            mask = (
                (ratios >= lower)
                &
                (ratios < upper)
            )

        count = int(
            mask.sum()
        )

        percentage = (
            count
            / len(ratios)
            * 100
        )

        print(
            f"{label:12s} "
            f"{count:6d} "
            f"({percentage:6.2f}%)"
        )


# ============================================================
# 9. 打印整图统计
# ============================================================

print_statistics(
    "Whole Image Difference Ratio",
    image_ratios,
)


# ============================================================
# 10. 打印 Random Patch 统计
# ============================================================

print_statistics(
    "Random 256x256 Patch Difference Ratio",
    patch_ratios,
)


# ============================================================
# 11. 额外统计：
# Random Patch 有效变化概率
# ============================================================

print("\n" + "=" * 70)

print(
    "Random Patch 有效变化概率"
)

print("=" * 70)


# Patch 内至少 1% 像素发生明显变化
for threshold in [
    0.001,
    0.01,
    0.05,
    0.10,
]:

    count = int(
        (
            patch_ratios >= threshold
        ).sum()
    )

    percentage = (
        count
        / len(patch_ratios)
        * 100
    )

    print(
        f"Difference Ratio >= "
        f"{threshold * 100:.1f}%："
        f"{count} / {len(patch_ratios)} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 12. 保存每张 Train 图片的统计 CSV
# ============================================================

csv_path = (
    OUTPUT_DIR
    / "train_image_difference.csv"
)


with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "input_path",
            "target_path",
            "difference_ratio",
            "difference_percent",
        ]
    )

    for item in image_results:

        ratio = item[
            "difference_ratio"
        ]

        writer.writerow(
            [
                item["input_path"],
                item["target_path"],
                ratio,
                ratio * 100,
            ]
        )


print(
    "\nCSV 已保存：",
    csv_path,
)


# ============================================================
# 13. 保存整图 Difference Ratio Histogram
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    image_ratios * 100,
    bins=50,
)

plt.xlabel(
    "Difference Ratio (%)"
)

plt.ylabel(
    "Number of Images"
)

plt.title(
    "Train Whole-Image Difference Ratio"
)

plt.tight_layout()


image_hist_path = (
    OUTPUT_DIR
    / "whole_image_difference_hist.png"
)


plt.savefig(
    image_hist_path,
    dpi=150,
)

plt.close()


print(
    "整图 Histogram 已保存：",
    image_hist_path,
)


# ============================================================
# 14. 保存 Random Patch Histogram
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    patch_ratios * 100,
    bins=50,
)

plt.xlabel(
    "Difference Ratio (%)"
)

plt.ylabel(
    "Number of Random Patches"
)

plt.title(
    "Random 256x256 Patch Difference Ratio"
)

plt.tight_layout()


patch_hist_path = (
    OUTPUT_DIR
    / "random_patch_difference_hist.png"
)


plt.savefig(
    patch_hist_path,
    dpi=150,
)

plt.close()


print(
    "Random Patch Histogram 已保存：",
    patch_hist_path,
)


# ============================================================
# 15. 输出差异最小 / 中间 / 最大样本
#
# 这里只打印路径，不读取 Val / Test。
# ============================================================

sorted_results = sorted(
    image_results,
    key=lambda item: (
        item["difference_ratio"]
    ),
)


print("\n" + "=" * 70)

print(
    "Difference Ratio 最小的 10 张 Train 图片"
)

print("=" * 70)


for item in sorted_results[:10]:

    print(
        f"{item['difference_ratio'] * 100:8.4f}%  "
        f"{item['input_path']}"
    )


print("\n" + "=" * 70)

print(
    "Difference Ratio 最大的 10 张 Train 图片"
)

print("=" * 70)


for item in sorted_results[-10:]:

    print(
        f"{item['difference_ratio'] * 100:8.4f}%  "
        f"{item['input_path']}"
    )


print("\n" + "=" * 70)

print("Analysis Finished")

print("=" * 70)