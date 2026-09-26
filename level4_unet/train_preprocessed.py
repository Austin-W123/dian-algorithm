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
    "outputs/train_preprocessed"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


PATCH_SIZE = 256

BATCH_SIZE = 8

LEARNING_RATE = 0.001

# 本轮是 preprocessing ablation。
# 先训练 5 Epoch 判断 Document Enhancement 是否真正有帮助，
# 暂时不直接跑 10 Epoch。
NUM_EPOCHS = 5

VAL_PATCHES_PER_IMAGE = 5

BLUR_RADIUS = 15


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


# ============================================================
# 2. Dataset
#
# 这里不使用旧的 dataset.py。
#
# 当前实验使用：
#
# dataset_preprocessed.py
#       ↓
# PreprocessedHandwritingDataset
#       ↓
# RGB / RGBA
#       ↓
# Grayscale
#       ↓
# Background Normalization
#       ↓
# Document Enhancement
#       ↓
# [1, 256, 256]
#
# Target 本身是 L：
# [1, 256, 256]
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
        patches_per_image=(
            VAL_PATCHES_PER_IMAGE
        ),
        blur_radius=BLUR_RADIUS,
    )
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


print("\n===== Dataset =====")

print(
    "Train 原图 Pair：",
    len(train_dataset),
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
#
# Document Enhancement 输出灰度图 L。
#
# 因此当前模型真正使用：
#
# Input:
# [B, 1, 256, 256]
#
# Output:
# [B, 1, 256, 256]
#
# 不再使用原 RGB baseline 的 3-channel 输入。
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


print("\n===== U-Net Parameters =====")

print(
    "Total Parameters：",
    total_params,
)

print(
    "Trainable Parameters：",
    trainable_params,
)


# ============================================================
# 4. Loss / Optimizer
#
# 本轮严格保持 L1 Loss。
#
# 原因：
# 我们现在只测试 preprocessing 是否有帮助。
#
# 如果同时修改 preprocessing 和 loss，
# 那么最后即使结果变好，也无法判断：
#
# 是 preprocessing 起作用，
# 还是新的 loss 起作用。
#
# 因此当前实验：
#
# Document Enhanced Input + L1 Loss
#
# 下一轮再单独研究：
# L1 / MSE / Combination / Weighted Loss
# ============================================================

criterion = nn.L1Loss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# 5. PSNR
#
# 图像范围为 [0, 1]。
#
# PSNR =
#
#     10 * log10(MAX^2 / MSE)
#
# 此处 MAX = 1，
#
# 所以：
#
#     PSNR = 10 * log10(1 / MSE)
#
# PSNR 越高，一般表示 Prediction 与 Target
# 在像素误差意义上越接近。
# ============================================================

def calculate_psnr(
    prediction,
    target,
):

    mse = torch.mean(
        (
            prediction
            - target
        ) ** 2
    ).item()

    if mse == 0:
        return float("inf")

    return 10 * math.log10(
        1.0 / mse
    )


# ============================================================
# 6. SSIM
#
# SSIM 更关注图像的结构相似性。
#
# Prediction / Target 都是：
#
# [B, 1, H, W]
#
# 所以逐张取出：
#
# [H, W]
#
# 然后计算 SSIM。
# ============================================================

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

    batch_size = (
        prediction_np.shape[0]
    )

    for i in range(
        batch_size
    ):

        pred_image = (
            prediction_np[
                i,
                0,
            ]
        )

        target_image = (
            target_np[
                i,
                0,
            ]
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
# 7. Validation
#
# Validation 不进行：
#
# backward()
# optimizer.step()
#
# 这里只评价模型。
#
# 返回：
#
# Validation L1
# Validation PSNR
# Validation SSIM
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0

    total_psnr = 0.0

    total_ssim = 0.0

    total_samples = 0


    with torch.no_grad():

        for (
            inputs,
            targets,
        ) in val_loader:

            inputs = inputs.to(
                device
            )

            targets = targets.to(
                device
            )

            predictions = model(
                inputs
            )

            loss = criterion(
                predictions,
                targets,
            )

            batch_size = (
                inputs.size(0)
            )


            # --------------------------------------------
            # Validation L1
            # --------------------------------------------

            total_loss += (
                loss.item()
                * batch_size
            )


            # --------------------------------------------
            # PSNR
            #
            # 每一张图单独计算，再求整个 Validation
            # Dataset 的平均值。
            # --------------------------------------------

            for i in range(
                batch_size
            ):

                psnr = calculate_psnr(
                    predictions[
                        i:i+1
                    ],
                    targets[
                        i:i+1
                    ],
                )

                total_psnr += psnr


            # --------------------------------------------
            # SSIM
            #
            # calculate_ssim() 返回当前 batch
            # 所有图片的平均 SSIM。
            # --------------------------------------------

            batch_ssim = calculate_ssim(
                predictions,
                targets,
            )

            total_ssim += (
                batch_ssim
                * batch_size
            )

            total_samples += (
                batch_size
            )


    mean_loss = (
        total_loss
        / total_samples
    )

    mean_psnr = (
        total_psnr
        / total_samples
    )

    mean_ssim = (
        total_ssim
        / total_samples
    )


    return (
        mean_loss,
        mean_psnr,
        mean_ssim,
    )


# ============================================================
# 8. Training
# ============================================================

train_losses = []

val_losses = []

val_psnrs = []

val_ssims = []


best_val_loss = float(
    "inf"
)

best_epoch = 0


training_start = time.time()


print(
    "\n===== "
    "Document-Enhanced "
    "U-Net Training ====="
)


for epoch in range(
    NUM_EPOCHS
):

    epoch_start = time.time()

    model.train()

    running_loss = 0.0

    total_samples = 0


    for (
        inputs,
        targets,
    ) in train_loader:

        inputs = inputs.to(
            device
        )

        targets = targets.to(
            device
        )


        # --------------------------------------------
        # 1. 清空上一轮梯度
        # --------------------------------------------

        optimizer.zero_grad()


        # --------------------------------------------
        # 2. Forward
        #
        # Document Enhanced Input
        #        ↓
        #       U-Net
        #        ↓
        # Prediction
        # --------------------------------------------

        predictions = model(
            inputs
        )


        # --------------------------------------------
        # 3. Loss
        # --------------------------------------------

        loss = criterion(
            predictions,
            targets,
        )


        # --------------------------------------------
        # 4. Backward
        # --------------------------------------------

        loss.backward()


        # --------------------------------------------
        # 5. Optimizer 更新参数
        # --------------------------------------------

        optimizer.step()


        batch_size = (
            inputs.size(0)
        )


        running_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )


    train_loss = (
        running_loss
        / total_samples
    )


    # ========================================================
    # Validation
    # ========================================================

    (
        val_loss,
        val_psnr,
        val_ssim,
    ) = validate()


    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    val_psnrs.append(
        val_psnr
    )

    val_ssims.append(
        val_ssim
    )


    epoch_time = (
        time.time()
        - epoch_start
    )


    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train L1: {train_loss:.6f} | "
        f"Val L1: {val_loss:.6f} | "
        f"PSNR: {val_psnr:.2f} dB | "
        f"SSIM: {val_ssim:.4f} | "
        f"Time: {epoch_time:.2f}s"
    )


    # --------------------------------------------------------
    # Best checkpoint
    #
    # 当前仍然按照最低 Validation L1 保存模型。
    #
    # 注意：
    # 全局 L1 不一定完全代表“手写去除能力”。
    #
    # 因此训练完成后还必须结合 Prediction
    # 可视化进行判断。
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = (
            val_loss
        )

        best_epoch = (
            epoch + 1
        )


        torch.save(
            model.state_dict(),
            os.path.join(
                OUTPUT_DIR,
                "best_unet.pth",
            ),
        )


# ============================================================
# 9. Total Time
# ============================================================

total_training_time = (
    time.time()
    - training_start
)


print(
    "\n===== Training Finished ====="
)

print(
    "Best Epoch：",
    best_epoch,
)

print(
    "Best Validation L1：",
    f"{best_val_loss:.6f}",
)

print(
    "Total Training Time：",
    f"{total_training_time:.2f}s",
)


# ============================================================
# 10. Curves
# ============================================================

epochs = list(
    range(
        1,
        NUM_EPOCHS + 1,
    )
)


# ------------------------------------------------------------
# Loss Curve
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs,
    train_losses,
    marker="o",
    label="Train L1",
)

plt.plot(
    epochs,
    val_losses,
    marker="o",
    label="Validation L1",
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "L1 Loss"
)

plt.title(
    "Document-Enhanced U-Net "
    "Training / Validation Loss"
)

plt.legend()

plt.grid()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "loss_curve.png",
    ),
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ------------------------------------------------------------
# PSNR Curve
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs,
    val_psnrs,
    marker="o",
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "PSNR (dB)"
)

plt.title(
    "Document-Enhanced "
    "Validation PSNR"
)

plt.grid()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "psnr_curve.png",
    ),
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ------------------------------------------------------------
# SSIM Curve
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs,
    val_ssims,
    marker="o",
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "SSIM"
)

plt.title(
    "Document-Enhanced "
    "Validation SSIM"
)

plt.grid()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ssim_curve.png",
    ),
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 11. Load Best Model
#
# 训练最后一个 Epoch 不一定是最好的模型。
#
# 因此这里重新加载 Validation L1 最低的 checkpoint。
# ============================================================

model.load_state_dict(
    torch.load(
        os.path.join(
            OUTPUT_DIR,
            "best_unet.pth",
        ),
        map_location=device,
        weights_only=True,
    )
)

model.eval()


# ============================================================
# 12. Prediction Visualization
#
# 最终必须同时观察：
#
# Document Enhanced Input
# Prediction
# Target
#
# 特别关注：
#
# 1. 黑色手写有没有删除
# 2. 打印文字有没有误删
# 3. 图形 / 表格有没有被破坏
# 4. 背景是否足够干净
#
# 不能只根据 L1 / PSNR / SSIM 判断模型效果。
# ============================================================

example_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)


example_inputs, example_targets = (
    next(
        iter(example_loader)
    )
)


example_inputs = (
    example_inputs.to(
        device
    )
)


with torch.no_grad():

    example_predictions = model(
        example_inputs
    )


example_inputs = (
    example_inputs
    .cpu()
)

example_predictions = (
    example_predictions
    .cpu()
)

example_targets = (
    example_targets
    .cpu()
)


fig, axes = plt.subplots(
    4,
    3,
    figsize=(12, 16),
)


for i in range(4):

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    axes[i, 0].imshow(
        example_inputs[
            i,
            0,
        ],
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[i, 0].set_title(
        f"Document Enhanced Input {i + 1}"
    )

    axes[i, 0].axis(
        "off"
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    axes[i, 1].imshow(
        example_predictions[
            i,
            0,
        ],
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[i, 1].set_title(
        f"Prediction {i + 1}"
    )

    axes[i, 1].axis(
        "off"
    )


    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    axes[i, 2].imshow(
        example_targets[
            i,
            0,
        ],
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[i, 2].set_title(
        f"Target {i + 1}"
    )

    axes[i, 2].axis(
        "off"
    )


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "predictions.png",
    ),
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 13. Save Metrics
#
# 将实验配置和每个 Epoch 的结果保存下来，
# 方便之后比较：
#
# RGB baseline
# Grayscale
# Document Enhanced
# Future Loss experiments
# ============================================================

metrics_path = os.path.join(
    OUTPUT_DIR,
    "metrics.txt",
)


with open(
    metrics_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "Document-Enhanced "
        "U-Net Experiment\n"
    )

    f.write(
        "=" * 60
        + "\n"
    )

    f.write(
        "Input: RGB -> L -> "
        "Background Normalization -> "
        "Document Enhancement\n"
    )

    f.write(
        f"Blur Radius: "
        f"{BLUR_RADIUS}\n"
    )

    f.write(
        "Loss: L1\n"
    )

    f.write(
        f"Patch Size: "
        f"{PATCH_SIZE}\n"
    )

    f.write(
        f"Batch Size: "
        f"{BATCH_SIZE}\n"
    )

    f.write(
        f"Learning Rate: "
        f"{LEARNING_RATE}\n"
    )

    f.write(
        f"Epochs: "
        f"{NUM_EPOCHS}\n\n"
    )


    for i in range(
        NUM_EPOCHS
    ):

        f.write(
            f"Epoch {i + 1}: "
            f"Train L1="
            f"{train_losses[i]:.6f}, "
            f"Val L1="
            f"{val_losses[i]:.6f}, "
            f"PSNR="
            f"{val_psnrs[i]:.2f}, "
            f"SSIM="
            f"{val_ssims[i]:.4f}\n"
        )


    f.write(
        "\n"
    )

    f.write(
        f"Best Epoch: "
        f"{best_epoch}\n"
    )

    f.write(
        f"Best Validation L1: "
        f"{best_val_loss:.6f}\n"
    )

    f.write(
        f"Total Training Time: "
        f"{total_training_time:.2f}s\n"
    )


print(
    "\n输出文件已保存到：",
    OUTPUT_DIR,
)