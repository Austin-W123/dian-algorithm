import os
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from skimage.metrics import structural_similarity

from dataset_preprocessed import (
    PreprocessedHandwritingDataset
)

from unet import UNet


# ============================================================
# 1. Config
# ============================================================

torch.manual_seed(42)

DATA_ROOT = (
    "/mnt/d/User/University/QQ/Dian/deli"
)

TRAIN_SPLIT = "splits/train.txt"
VAL_SPLIT = "splits/val.txt"

OUTPUT_DIR = (
    "outputs/train_weighted_loss"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


PATCH_SIZE = 256

BATCH_SIZE = 8

LEARNING_RATE = 0.001

# ------------------------------------------------------------
# 本轮仍然先训练 5 Epoch。
#
# 目的不是直接追求最终模型，
# 而是验证：
#
# 普通 L1
#       ↓
# L1 + MSE + Changed-Region L1
#
# 是否能够让模型更加关注需要真正修改的区域。
# ------------------------------------------------------------

NUM_EPOCHS = 5

VAL_PATCHES_PER_IMAGE = 5

BLUR_RADIUS = 15


# ============================================================
# Loss 超参数
# ============================================================

# 全图 MSE 的权重
LAMBDA_MSE = 0.5

# 变化区域额外 L1 的权重
LAMBDA_CHANGED = 2.0

# ------------------------------------------------------------
# Difference Mask Threshold
#
# Input / Target 都在 [0, 1]。
#
# 如果：
#
# |Input - Target| > 0.10
#
# 则认为该像素属于“明显变化区域”。
#
# 注意：
# 这里得到的不是人工标注的 handwriting mask，
# 而是由 paired data 自动构造的 difference mask。
# ------------------------------------------------------------

DIFF_THRESHOLD = 0.10


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
    "RGB → L → Background Normalization → Document Enhancement"
)

print(
    "Loss： "
    "L1 + "
    f"{LAMBDA_MSE} × MSE + "
    f"{LAMBDA_CHANGED} × Changed-Region L1"
)

print(
    "Difference Threshold：",
    DIFF_THRESHOLD,
)


# ============================================================
# 2. Dataset
# ============================================================

train_dataset = (
    PreprocessedHandwritingDataset(
        data_root=DATA_ROOT,
        split_file=TRAIN_SPLIT,
        patch_size=PATCH_SIZE,
        mode="train",
        blur_radius=BLUR_RADIUS,
    )
)


val_dataset = (
    PreprocessedHandwritingDataset(
        data_root=DATA_ROOT,
        split_file=VAL_SPLIT,
        patch_size=PATCH_SIZE,
        mode="val",
        patches_per_image=VAL_PATCHES_PER_IMAGE,
        blur_radius=BLUR_RADIUS,
    )
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)


print()
print("===== Dataset =====")

print(
    "Train 原图 Pair：",
    len(train_dataset.pairs),
)

print(
    "Validation Patches：",
    len(val_dataset),
)

print(
    "Train Batches：",
    len(train_loader),
)

print(
    "Validation Batches：",
    len(val_loader),
)


# ============================================================
# 3. Model
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
# 4. Loss
# ============================================================

l1_criterion = nn.L1Loss()

mse_criterion = nn.MSELoss()


def weighted_loss(
    prediction,
    input_image,
    target,
):
    """
    总 Loss：

        L1
        +
        lambda_mse * MSE
        +
        lambda_changed * Changed-Region L1


    Difference Mask：

        |Input - Target| > threshold


    目的：

        普通 L1 / MSE
        保证整张图片总体重建质量。

        Changed-Region L1
        让模型额外关注 Input 和 Target
        真正不同的位置。

    注意：

        Difference Mask 并不等价于
        “真正的手写 Mask”。

        因为 Input 经过 Document Enhancement，
        Input / Target 之间仍可能存在
        对比度、背景、扫描风格等差异。

        所以这是一个弱监督近似。
    """

    # --------------------------------------------------------
    # Global L1
    # --------------------------------------------------------

    global_l1 = l1_criterion(
        prediction,
        target,
    )


    # --------------------------------------------------------
    # Global MSE
    # --------------------------------------------------------

    global_mse = mse_criterion(
        prediction,
        target,
    )


    # --------------------------------------------------------
    # Difference Map
    #
    # [B, 1, H, W]
    # --------------------------------------------------------

    difference = torch.abs(
        input_image - target
    )


    # --------------------------------------------------------
    # Binary Difference Mask
    #
    # 不参与反向传播。
    # --------------------------------------------------------

    with torch.no_grad():

        changed_mask = (
            difference
            > DIFF_THRESHOLD
        ).float()


    # --------------------------------------------------------
    # Prediction Error
    # --------------------------------------------------------

    pixel_l1 = torch.abs(
        prediction - target
    )


    # --------------------------------------------------------
    # Changed-Region L1
    #
    # 只计算 Mask = 1 的像素。
    # --------------------------------------------------------

    changed_pixels = (
        changed_mask.sum()
    )


    if changed_pixels.item() > 0:

        changed_l1 = (
            pixel_l1
            * changed_mask
        ).sum() / (
            changed_pixels
            + 1e-8
        )

    else:

        # 极少数 patch 可能没有明显变化区域。
        #
        # 此时 Changed Loss 设为 0，
        # 仍然使用 Global L1 + MSE 训练。

        changed_l1 = (
            prediction.sum()
            * 0.0
        )


    # --------------------------------------------------------
    # Total Loss
    # --------------------------------------------------------

    total_loss = (
        global_l1
        +
        LAMBDA_MSE
        * global_mse
        +
        LAMBDA_CHANGED
        * changed_l1
    )


    return (
        total_loss,
        global_l1,
        global_mse,
        changed_l1,
        changed_mask,
    )


# ============================================================
# 5. Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# 6. Metrics
# ============================================================

def calculate_psnr(
    prediction,
    target,
):
    mse = torch.mean(
        (prediction - target) ** 2
    ).item()

    if mse == 0:
        return 100.0

    return (
        10.0
        * math.log10(
            1.0 / mse
        )
    )


def calculate_ssim(
    prediction,
    target,
):
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

    scores = []

    for i in range(
        prediction_np.shape[0]
    ):

        pred_image = (
            prediction_np[i, 0]
        )

        target_image = (
            target_np[i, 0]
        )

        score = structural_similarity(
            target_image,
            pred_image,
            data_range=1.0,
        )

        scores.append(
            score
        )

    return float(
        np.mean(scores)
    )


# ============================================================
# 7. History
# ============================================================

train_total_history = []
train_l1_history = []
train_mse_history = []
train_changed_history = []

val_total_history = []
val_l1_history = []
val_mse_history = []
val_changed_history = []

val_psnr_history = []
val_ssim_history = []


best_val_loss = float("inf")
best_epoch = 0

total_training_start = time.time()


# ============================================================
# 8. Training
# ============================================================

print()
print(
    "===== Difference-Weighted Training ====="
)


for epoch in range(NUM_EPOCHS):

    epoch_start = time.time()


    # ========================================================
    # Train
    # ========================================================

    model.train()


    train_total_sum = 0.0
    train_l1_sum = 0.0
    train_mse_sum = 0.0
    train_changed_sum = 0.0


    for (
        input_images,
        targets,
    ) in train_loader:

        input_images = (
            input_images
            .to(
                device,
                non_blocking=True,
            )
        )

        targets = (
            targets
            .to(
                device,
                non_blocking=True,
            )
        )


        optimizer.zero_grad()


        predictions = model(
            input_images
        )


        (
            loss,
            global_l1,
            global_mse,
            changed_l1,
            _,
        ) = weighted_loss(
            predictions,
            input_images,
            targets,
        )


        loss.backward()

        optimizer.step()


        train_total_sum += (
            loss.item()
        )

        train_l1_sum += (
            global_l1.item()
        )

        train_mse_sum += (
            global_mse.item()
        )

        train_changed_sum += (
            changed_l1.item()
        )


    train_total = (
        train_total_sum
        / len(train_loader)
    )

    train_l1 = (
        train_l1_sum
        / len(train_loader)
    )

    train_mse = (
        train_mse_sum
        / len(train_loader)
    )

    train_changed = (
        train_changed_sum
        / len(train_loader)
    )


    # ========================================================
    # Validation
    # ========================================================

    model.eval()


    val_total_sum = 0.0
    val_l1_sum = 0.0
    val_mse_sum = 0.0
    val_changed_sum = 0.0

    val_psnr_sum = 0.0
    val_ssim_sum = 0.0

    val_batches = 0


    with torch.no_grad():

        for (
            input_images,
            targets,
        ) in val_loader:

            input_images = (
                input_images
                .to(
                    device,
                    non_blocking=True,
                )
            )

            targets = (
                targets
                .to(
                    device,
                    non_blocking=True,
                )
            )


            predictions = model(
                input_images
            )


            (
                loss,
                global_l1,
                global_mse,
                changed_l1,
                _,
            ) = weighted_loss(
                predictions,
                input_images,
                targets,
            )


            val_total_sum += (
                loss.item()
            )

            val_l1_sum += (
                global_l1.item()
            )

            val_mse_sum += (
                global_mse.item()
            )

            val_changed_sum += (
                changed_l1.item()
            )


            val_psnr_sum += (
                calculate_psnr(
                    predictions,
                    targets,
                )
            )

            val_ssim_sum += (
                calculate_ssim(
                    predictions,
                    targets,
                )
            )

            val_batches += 1


    val_total = (
        val_total_sum
        / val_batches
    )

    val_l1 = (
        val_l1_sum
        / val_batches
    )

    val_mse = (
        val_mse_sum
        / val_batches
    )

    val_changed = (
        val_changed_sum
        / val_batches
    )

    val_psnr = (
        val_psnr_sum
        / val_batches
    )

    val_ssim = (
        val_ssim_sum
        / val_batches
    )


    # ========================================================
    # History
    # ========================================================

    train_total_history.append(
        train_total
    )

    train_l1_history.append(
        train_l1
    )

    train_mse_history.append(
        train_mse
    )

    train_changed_history.append(
        train_changed
    )


    val_total_history.append(
        val_total
    )

    val_l1_history.append(
        val_l1
    )

    val_mse_history.append(
        val_mse
    )

    val_changed_history.append(
        val_changed
    )

    val_psnr_history.append(
        val_psnr
    )

    val_ssim_history.append(
        val_ssim
    )


    # ========================================================
    # Best Model
    #
    # 使用 Total Weighted Loss 选择 checkpoint。
    # ========================================================

    if val_total < best_val_loss:

        best_val_loss = (
            val_total
        )

        best_epoch = (
            epoch + 1
        )


        torch.save(
            model.state_dict(),
            os.path.join(
                OUTPUT_DIR,
                "best_unet_weighted.pth",
            ),
        )


    epoch_time = (
        time.time()
        - epoch_start
    )


    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"| Train Total: {train_total:.6f} "
        f"| Train L1: {train_l1:.6f} "
        f"| Train Changed: {train_changed:.6f} "
        f"| Val Total: {val_total:.6f} "
        f"| Val L1: {val_l1:.6f} "
        f"| Val Changed: {val_changed:.6f} "
        f"| PSNR: {val_psnr:.2f} dB "
        f"| SSIM: {val_ssim:.4f} "
        f"| Time: {epoch_time:.2f}s"
    )


# ============================================================
# 9. Total Time
# ============================================================

total_training_time = (
    time.time()
    - total_training_start
)


# ============================================================
# 10. Save Final Model
# ============================================================

torch.save(
    model.state_dict(),
    os.path.join(
        OUTPUT_DIR,
        "final_unet_weighted.pth",
    ),
)


# ============================================================
# 11. Load Best Model
# ============================================================

model.load_state_dict(
    torch.load(
        os.path.join(
            OUTPUT_DIR,
            "best_unet_weighted.pth",
        ),
        map_location=device,
        weights_only=True,
    )
)

model.eval()


# ============================================================
# 12. Loss Curve
# ============================================================

epochs = range(
    1,
    NUM_EPOCHS + 1,
)


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    train_total_history,
    marker="o",
    label="Train Total",
)

plt.plot(
    epochs,
    val_total_history,
    marker="o",
    label="Validation Total",
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Weighted Loss"
)

plt.title(
    "Difference-Weighted Training / Validation Loss"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "weighted_loss_curve.png",
    ),
    dpi=150,
)

plt.close()


# ============================================================
# 13. Loss Components Curve
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    val_l1_history,
    marker="o",
    label="Validation Global L1",
)

plt.plot(
    epochs,
    val_mse_history,
    marker="o",
    label="Validation Global MSE",
)

plt.plot(
    epochs,
    val_changed_history,
    marker="o",
    label="Validation Changed L1",
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Validation Loss Components"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "loss_components.png",
    ),
    dpi=150,
)

plt.close()


# ============================================================
# 14. PSNR Curve
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    val_psnr_history,
    marker="o",
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "PSNR (dB)"
)

plt.title(
    "Difference-Weighted Validation PSNR"
)

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "psnr_curve.png",
    ),
    dpi=150,
)

plt.close()


# ============================================================
# 15. SSIM Curve
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    val_ssim_history,
    marker="o",
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "SSIM"
)

plt.title(
    "Difference-Weighted Validation SSIM"
)

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ssim_curve.png",
    ),
    dpi=150,
)

plt.close()


# ============================================================
# 16. Prediction Visualization
#
# 显示：
#
# Input
# Difference Mask
# Prediction
# Target
#
# 这张图非常重要：
#
# 它能够直接验证 Difference Mask
# 到底有没有真正覆盖手写区域。
# ============================================================

num_examples = 4


plt.figure(
    figsize=(14, 4 * num_examples)
)


with torch.no_grad():

    for i in range(num_examples):

        input_image, target = (
            val_dataset[i]
        )


        input_batch = (
            input_image
            .unsqueeze(0)
            .to(device)
        )


        target_batch = (
            target
            .unsqueeze(0)
            .to(device)
        )


        prediction = model(
            input_batch
        )


        (
            _,
            _,
            _,
            _,
            changed_mask,
        ) = weighted_loss(
            prediction,
            input_batch,
            target_batch,
        )


        input_np = (
            input_image[0]
            .cpu()
            .numpy()
        )

        target_np = (
            target[0]
            .cpu()
            .numpy()
        )

        prediction_np = (
            prediction[
                0,
                0,
            ]
            .cpu()
            .numpy()
        )

        mask_np = (
            changed_mask[
                0,
                0,
            ]
            .cpu()
            .numpy()
        )


        # Input
        plt.subplot(
            num_examples,
            4,
            i * 4 + 1,
        )

        plt.imshow(
            input_np,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        plt.title(
            f"Input {i + 1}"
        )

        plt.axis("off")


        # Difference Mask
        plt.subplot(
            num_examples,
            4,
            i * 4 + 2,
        )

        plt.imshow(
            mask_np,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        plt.title(
            f"Difference Mask {i + 1}"
        )

        plt.axis("off")


        # Prediction
        plt.subplot(
            num_examples,
            4,
            i * 4 + 3,
        )

        plt.imshow(
            prediction_np,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        plt.title(
            f"Prediction {i + 1}"
        )

        plt.axis("off")


        # Target
        plt.subplot(
            num_examples,
            4,
            i * 4 + 4,
        )

        plt.imshow(
            target_np,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        plt.title(
            f"Target {i + 1}"
        )

        plt.axis("off")


plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "weighted_predictions.png",
    ),
    dpi=150,
)

plt.close()


# ============================================================
# 17. Finished
# ============================================================

print()
print(
    "===== Training Finished ====="
)

print(
    "Best Epoch：",
    best_epoch,
)

print(
    "Best Validation Weighted Loss：",
    f"{best_val_loss:.6f}",
)

print(
    "Total Training Time：",
    f"{total_training_time:.2f}s",
)

print()

print(
    "输出文件已保存到：",
    OUTPUT_DIR,
)