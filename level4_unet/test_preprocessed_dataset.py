import os

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset_preprocessed import PreprocessedHandwritingDataset


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = "/mnt/d/User/University/QQ/Dian/deli"

VAL_SPLIT = "splits/val.txt"

PATCH_SIZE = 256
PATCHES_PER_IMAGE = 5

BLUR_RADIUS = 15

OUTPUT_DIR = "outputs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# Visualization Helper
# ============================================================

def show_tensor(
    ax,
    tensor,
    title,
):
    """
    显示 [1, H, W] 灰度 Tensor。

    Tensor range:
        [0, 1]
    """

    image = (
        tensor
        .squeeze(0)
        .cpu()
        .numpy()
    )

    ax.imshow(
        image,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title(
        title
    )

    ax.axis(
        "off"
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "Document Enhanced Dataset Test"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Dataset
    #
    # 注意：
    #
    # 这里必须使用 dataset_preprocessed.py 中的
    # PreprocessedHandwritingDataset。
    #
    # 不能再使用原来的 dataset.py / DeliDataset。
    # --------------------------------------------------------

    dataset = PreprocessedHandwritingDataset(
        data_root=DATA_ROOT,
        split_file=VAL_SPLIT,
        patch_size=PATCH_SIZE,
        mode="val",
        patches_per_image=PATCHES_PER_IMAGE,
        blur_radius=BLUR_RADIUS,
    )

    print(
        "Validation Samples：",
        len(dataset),
    )

    # 我们冻结的 Validation Pair = 241。
    #
    # 每张图片取 5 个固定 Patch。
    #
    # 因此：
    #
    # 241 × 5 = 1205

    expected_samples = (
        241
        * PATCHES_PER_IMAGE
    )

    assert len(dataset) == expected_samples, (
        f"Validation Samples 错误："
        f"{len(dataset)} != {expected_samples}"
    )

    print(
        "Validation Sample 数量检查通过！"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    inputs, targets = next(
        iter(loader)
    )

    # --------------------------------------------------------
    # Tensor Information
    # --------------------------------------------------------

    print(
        "\n===== Tensor ====="
    )

    print(
        "Input shape：",
        inputs.shape,
    )

    print(
        "Target shape：",
        targets.shape,
    )

    print(
        "Input dtype：",
        inputs.dtype,
    )

    print(
        "Target dtype：",
        targets.dtype,
    )

    print(
        "Input range：",
        float(inputs.min()),
        "~",
        float(inputs.max()),
    )

    print(
        "Target range：",
        float(targets.min()),
        "~",
        float(targets.max()),
    )

    # --------------------------------------------------------
    # Shape Assertions
    #
    # Document preprocessing 后：
    #
    # Input:
    # [B, 1, 256, 256]
    #
    # Target:
    # [B, 1, 256, 256]
    # --------------------------------------------------------

    assert inputs.ndim == 4
    assert targets.ndim == 4

    assert inputs.shape[0] == 4
    assert targets.shape[0] == 4

    assert inputs.shape[1] == 1
    assert targets.shape[1] == 1

    assert inputs.shape[-2:] == (
        PATCH_SIZE,
        PATCH_SIZE,
    )

    assert targets.shape[-2:] == (
        PATCH_SIZE,
        PATCH_SIZE,
    )

    # --------------------------------------------------------
    # Range Assertions
    # --------------------------------------------------------

    assert float(inputs.min()) >= 0.0
    assert float(inputs.max()) <= 1.0

    assert float(targets.min()) >= 0.0
    assert float(targets.max()) <= 1.0

    print(
        "\nTensor 检查通过！"
    )

    # --------------------------------------------------------
    # Visualization
    #
    # 左：
    #
    # RGB
    # ↓
    # L
    # ↓
    # Background Normalization
    # ↓
    # Document Enhancement
    #
    #
    # 右：
    #
    # 原始 Target
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(8, 16),
    )

    for i in range(4):

        show_tensor(
            axes[i, 0],
            inputs[i],
            f"Document Enhanced Input {i + 1}",
        )

        show_tensor(
            axes[i, 1],
            targets[i],
            f"Target {i + 1}",
        )

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        "document_enhanced_dataset_test.png",
    )

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\n可视化已保存：",
        save_path,
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "Document Enhanced Dataset Test Finished!"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()