import os
import csv
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from skimage.metrics import structural_similarity

from dataset_preprocessed import PreprocessedHandwritingDataset
from unet import UNet


# ============================================================
# 1. Config
# ============================================================

torch.manual_seed(42)


DATA_ROOT = (
    "/mnt/d/User/University/QQ/Dian/deli"
)

TEST_SPLIT = "splits/test.txt"

CHECKPOINT_PATH = (
    "outputs/train_weighted_loss/"
    "best_unet_weighted.pth"
)

OUTPUT_DIR = "outputs/final_test"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


PATCH_SIZE = 256

BATCH_SIZE = 8

TEST_PATCHES_PER_IMAGE = 5

BLUR_RADIUS = 15

# 与 train_weighted_loss.py 保持一致
DIFF_THRESHOLD = 0.1


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "使用设备：",
    device,
)

print(
    "Input Mode： "
    "RGB → L → Background Normalization "
    "→ Document Enhancement"
)

print(
    "Final Checkpoint：",
    CHECKPOINT_PATH,
)

print(
    "Difference Threshold：",
    DIFF_THRESHOLD,
)


# ============================================================
# 2. Dataset
#
# Final Test 必须使用：
#
# dataset_preprocessed.py
#
# 与 train_weighted_loss.py 保持完全一致的输入处理。
#
# Test：
#   不随机 Crop
#   使用固定 patch
#
# 因此每次运行得到的是同一组测试 patch。
# ============================================================

test_dataset = PreprocessedHandwritingDataset(
    data_root=DATA_ROOT,
    split_file=TEST_SPLIT,
    patch_size=PATCH_SIZE,
    mode="test",
    patches_per_image=TEST_PATCHES_PER_IMAGE,
    blur_radius=BLUR_RADIUS,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


print()
print("===== Final Test Dataset =====")

print(
    "Test 原图 Pair：",
    len(test_dataset.pairs),
)

print(
    "Test Patches：",
    len(test_dataset),
)

print(
    "Test Batches：",
    len(test_loader),
)


# ============================================================
# 3. Model
#
# 输入：
# [B, 1, 256, 256]
#
# 输出：
# [B, 1, 256, 256]
# ============================================================

model = UNet(
    in_channels=1,
    out_channels=1,
).to(device)


total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print()
print("===== U-Net Parameters =====")

print(
    "Total Parameters：",
    total_params,
)

print(
    "Trainable Parameters：",
    trainable_params,
)


# ============================================================
# 4. Load Best Model
# ============================================================

if not os.path.exists(
    CHECKPOINT_PATH
):
    raise FileNotFoundError(
        f"Checkpoint 不存在："
        f"{CHECKPOINT_PATH}"
    )


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)


# ------------------------------------------------------------
# 兼容两种保存方式：
#
# 1.
# torch.save(
#     model.state_dict(),
#     path
# )
#
# 2.
# torch.save(
#     {
#         "model_state_dict": ...
#     },
#     path
# )
# ------------------------------------------------------------

if isinstance(
    checkpoint,
    dict,
) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model.eval()


print()
print(
    "成功加载主模型：",
    CHECKPOINT_PATH,
)


# ============================================================
# 5. Metric Functions
# ============================================================


def calculate_psnr(
    prediction,
    target,
):
    """
    prediction / target:
        [B, 1, H, W]

    图像范围：
        [0, 1]
    """

    mse = F.mse_loss(
        prediction,
        target,
        reduction="none",
    )

    mse = mse.mean(
        dim=(1, 2, 3)
    )

    psnr = 10.0 * torch.log10(
        1.0
        / (mse + 1e-10)
    )

    return psnr


def calculate_ssim_batch(
    prediction,
    target,
):
    """
    使用 skimage 逐张计算 SSIM。

    返回：
        list[float]
    """

    prediction_np = (
        prediction
        .detach()
        .cpu()
        .numpy()
    )

    target_np = (
        target
        .detach()
        .cpu()
        .numpy()
    )

    values = []

    for i in range(
        prediction_np.shape[0]
    ):

        pred_img = (
            prediction_np[i, 0]
        )

        target_img = (
            target_np[i, 0]
        )

        value = (
            structural_similarity(
                target_img,
                pred_img,
                data_range=1.0,
            )
        )

        values.append(
            float(value)
        )

    return values


def changed_region_l1_per_image(
    prediction,
    input_image,
    target,
    threshold,
):
    """
    Difference Mask：

        |Input - Target| > threshold

    只统计这些 Changed Region 中
    Prediction 与 Target 的 L1。

    返回：
        changed_l1
        changed_ratio

    shape:
        [B]
    """

    difference = torch.abs(
        input_image
        - target
    )

    mask = (
        difference
        > threshold
    ).float()


    absolute_error = torch.abs(
        prediction
        - target
    )


    numerator = (
        absolute_error
        * mask
    ).sum(
        dim=(1, 2, 3)
    )


    denominator = (
        mask.sum(
            dim=(1, 2, 3)
        )
    )


    changed_l1 = (
        numerator
        / (
            denominator
            + 1e-8
        )
    )


    total_pixels = (
        input_image.shape[1]
        * input_image.shape[2]
        * input_image.shape[3]
    )


    changed_ratio = (
        denominator
        / total_pixels
    )


    return (
        changed_l1,
        changed_ratio,
        mask,
    )


# ============================================================
# 6. Final Test
# ============================================================

all_global_l1 = []

all_global_mse = []

all_changed_l1 = []

all_changed_ratio = []

all_psnr = []

all_ssim = []


# 保存一些固定测试样本用于最终可视化
visual_examples = []

MAX_VISUAL_EXAMPLES = 8


print()
print(
    "===== Final Test Started ====="
)


processed = 0


with torch.no_grad():

    for batch_index, batch in enumerate(
        test_loader
    ):

        # ----------------------------------------------------
        # Dataset 当前应返回：
        #
        # input_tensor,
        # target_tensor
        #
        # shape:
        # [B, 1, 256, 256]
        # ----------------------------------------------------

        if isinstance(
            batch,
            (list, tuple),
        ) and len(batch) >= 2:

            inputs = batch[0]
            targets = batch[1]

        else:

            raise ValueError(
                "Dataset 返回格式异常。"
                "需要至少返回 "
                "(input, target)。"
            )


        inputs = inputs.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        predictions = model(
            inputs
        )

        # 如果 UNet 最后一层已经使用 Sigmoid，
        # clamp 不会改变正常结果。
        #
        # 如果没有 Sigmoid，
        # 则保证图像指标计算范围在 [0, 1]。
        predictions = torch.clamp(
            predictions,
            0.0,
            1.0,
        )


        # ----------------------------------------------------
        # Global L1
        # ----------------------------------------------------

        global_l1 = F.l1_loss(
            predictions,
            targets,
            reduction="none",
        ).mean(
            dim=(1, 2, 3)
        )


        # ----------------------------------------------------
        # Global MSE
        # ----------------------------------------------------

        global_mse = F.mse_loss(
            predictions,
            targets,
            reduction="none",
        ).mean(
            dim=(1, 2, 3)
        )


        # ----------------------------------------------------
        # Changed Region
        # ----------------------------------------------------

        (
            changed_l1,
            changed_ratio,
            difference_mask,
        ) = (
            changed_region_l1_per_image(
                prediction=predictions,
                input_image=inputs,
                target=targets,
                threshold=DIFF_THRESHOLD,
            )
        )


        # ----------------------------------------------------
        # PSNR
        # ----------------------------------------------------

        psnr = calculate_psnr(
            predictions,
            targets,
        )


        # ----------------------------------------------------
        # SSIM
        # ----------------------------------------------------

        ssim_values = (
            calculate_ssim_batch(
                predictions,
                targets,
            )
        )


        # ----------------------------------------------------
        # 保存 metric
        # ----------------------------------------------------

        all_global_l1.extend(
            global_l1
            .detach()
            .cpu()
            .tolist()
        )

        all_global_mse.extend(
            global_mse
            .detach()
            .cpu()
            .tolist()
        )

        all_changed_l1.extend(
            changed_l1
            .detach()
            .cpu()
            .tolist()
        )

        all_changed_ratio.extend(
            changed_ratio
            .detach()
            .cpu()
            .tolist()
        )

        all_psnr.extend(
            psnr
            .detach()
            .cpu()
            .tolist()
        )

        all_ssim.extend(
            ssim_values
        )


        # ----------------------------------------------------
        # 保存前 8 个固定样本
        # ----------------------------------------------------

        if (
            len(visual_examples)
            < MAX_VISUAL_EXAMPLES
        ):

            batch_size_now = (
                inputs.shape[0]
            )

            for i in range(
                batch_size_now
            ):

                if (
                    len(visual_examples)
                    >= MAX_VISUAL_EXAMPLES
                ):
                    break


                visual_examples.append(
                    (
                        inputs[i]
                        .detach()
                        .cpu(),

                        difference_mask[i]
                        .detach()
                        .cpu(),

                        predictions[i]
                        .detach()
                        .cpu(),

                        targets[i]
                        .detach()
                        .cpu(),
                    )
                )


        processed += (
            inputs.shape[0]
        )


        if (
            processed % 200 < BATCH_SIZE
            or processed
            == len(test_dataset)
        ):

            print(
                f"已处理："
                f"{processed}/"
                f"{len(test_dataset)} "
                f"patches"
            )


# ============================================================
# 7. Aggregate Results
# ============================================================


mean_global_l1 = float(
    np.mean(
        all_global_l1
    )
)

mean_global_mse = float(
    np.mean(
        all_global_mse
    )
)

mean_changed_l1 = float(
    np.mean(
        all_changed_l1
    )
)

mean_changed_ratio = float(
    np.mean(
        all_changed_ratio
    )
)

mean_psnr = float(
    np.mean(
        all_psnr
    )
)

mean_ssim = float(
    np.mean(
        all_ssim
    )
)


print()
print(
    "=" * 70
)

print(
    "FINAL TEST RESULTS"
)

print(
    "=" * 70
)

print(
    f"Global L1："
    f"{mean_global_l1:.6f}"
)

print(
    f"Global MSE："
    f"{mean_global_mse:.6f}"
)

print(
    f"Changed-Region L1："
    f"{mean_changed_l1:.6f}"
)

print(
    f"Changed Pixel Ratio："
    f"{mean_changed_ratio * 100:.2f}%"
)

print(
    f"PSNR："
    f"{mean_psnr:.2f} dB"
)

print(
    f"SSIM："
    f"{mean_ssim:.4f}"
)


# ============================================================
# 8. Save Summary TXT
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "final_test_results.txt",
)


with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "Level 4 Final Test Results\n"
    )

    f.write(
        "=" * 50
        + "\n"
    )

    f.write(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH}\n"
    )

    f.write(
        "Input Mode: "
        "RGB -> L -> "
        "Background Normalization -> "
        "Document Enhancement\n"
    )

    f.write(
        f"Test Original Pairs: "
        f"{len(test_dataset.pairs)}\n"
    )

    f.write(
        f"Test Patches: "
        f"{len(test_dataset)}\n"
    )

    f.write(
        f"Difference Threshold: "
        f"{DIFF_THRESHOLD}\n"
    )

    f.write(
        "\n"
    )

    f.write(
        f"Global L1: "
        f"{mean_global_l1:.6f}\n"
    )

    f.write(
        f"Global MSE: "
        f"{mean_global_mse:.6f}\n"
    )

    f.write(
        f"Changed-Region L1: "
        f"{mean_changed_l1:.6f}\n"
    )

    f.write(
        f"Changed Pixel Ratio: "
        f"{mean_changed_ratio * 100:.2f}%\n"
    )

    f.write(
        f"PSNR: "
        f"{mean_psnr:.2f} dB\n"
    )

    f.write(
        f"SSIM: "
        f"{mean_ssim:.4f}\n"
    )


# ============================================================
# 9. Save Per-Patch CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "final_test_metrics.csv",
)


with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow(
        [
            "patch_index",
            "global_l1",
            "global_mse",
            "changed_region_l1",
            "changed_pixel_ratio",
            "psnr_db",
            "ssim",
        ]
    )


    for i in range(
        len(all_global_l1)
    ):

        writer.writerow(
            [
                i,
                all_global_l1[i],
                all_global_mse[i],
                all_changed_l1[i],
                all_changed_ratio[i],
                all_psnr[i],
                all_ssim[i],
            ]
        )


# ============================================================
# 10. Visualization
#
# 每一行：
#
# Input
# Difference Mask
# Prediction
# Target
# ============================================================

num_examples = len(
    visual_examples
)


fig, axes = plt.subplots(
    num_examples,
    4,
    figsize=(
        16,
        4 * num_examples,
    ),
)


if num_examples == 1:
    axes = np.expand_dims(
        axes,
        axis=0,
    )


for i, (
    input_image,
    mask,
    prediction,
    target,
) in enumerate(
    visual_examples
):

    input_np = (
        input_image[0]
        .numpy()
    )

    mask_np = (
        mask[0]
        .numpy()
    )

    prediction_np = (
        prediction[0]
        .numpy()
    )

    target_np = (
        target[0]
        .numpy()
    )


    axes[i, 0].imshow(
        input_np,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[i, 0].set_title(
        f"Input {i + 1}"
    )

    axes[i, 0].axis(
        "off"
    )


    axes[i, 1].imshow(
        mask_np,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[i, 1].set_title(
        f"Difference Mask {i + 1}"
    )

    axes[i, 1].axis(
        "off"
    )


    axes[i, 2].imshow(
        prediction_np,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[i, 2].set_title(
        f"Prediction {i + 1}"
    )

    axes[i, 2].axis(
        "off"
    )


    axes[i, 3].imshow(
        target_np,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[i, 3].set_title(
        f"Target {i + 1}"
    )

    axes[i, 3].axis(
        "off"
    )


plt.tight_layout()


visual_path = os.path.join(
    OUTPUT_DIR,
    "final_test_predictions.png",
)


plt.savefig(
    visual_path,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 11. Metric Distribution
#
# Final Test 不只有 mean。
#
# 画分布可以观察：
# 是否存在少量特别困难的 patch。
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    all_psnr,
    bins=30,
)

plt.xlabel(
    "PSNR (dB)"
)

plt.ylabel(
    "Number of Patches"
)

plt.title(
    "Final Test PSNR Distribution"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "psnr_distribution.png",
    ),
    dpi=150,
)

plt.close()


plt.figure(
    figsize=(10, 6)
)

plt.hist(
    all_ssim,
    bins=30,
)

plt.xlabel(
    "SSIM"
)

plt.ylabel(
    "Number of Patches"
)

plt.title(
    "Final Test SSIM Distribution"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ssim_distribution.png",
    ),
    dpi=150,
)

plt.close()


# ============================================================
# 12. Finished
# ============================================================

print()
print(
    "===== Files Saved ====="
)

print(
    summary_path
)

print(
    csv_path
)

print(
    visual_path
)

print(
    os.path.join(
        OUTPUT_DIR,
        "psnr_distribution.png",
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "ssim_distribution.png",
    )
)


print()
print(
    "=" * 70
)

print(
    "Final Test Finished!"
)

print(
    "=" * 70
)

print(
    "输出目录：",
    OUTPUT_DIR,
)