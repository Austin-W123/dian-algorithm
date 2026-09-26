import os
import time
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

DATA_ROOT = "/mnt/d/User/University/QQ/Dian/deli"

TRAIN_SPLIT = "splits/train.txt"
VAL_SPLIT = "splits/val.txt"

# 上一轮 Difference-Weighted 模型
CHECKPOINT_PATH = (
    "outputs/train_weighted_loss/best_unet_weighted.pth"
)

# 本轮单独保存，绝不覆盖上一轮
OUTPUT_DIR = "outputs/finetune_changed_mse"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


PATCH_SIZE = 256
BATCH_SIZE = 8

# ------------------------------------------------------------
# Fine-tuning：
#
# 上一轮从头训练：
# lr = 0.001
#
# 当前模型已经学会基本文档恢复和一定的手写擦除能力，
# 所以降低学习率，只做精细调整。
# ------------------------------------------------------------

LEARNING_RATE = 0.0002

NUM_EPOCHS = 3

VAL_PATCHES_PER_IMAGE = 5

BLUR_RADIUS = 15


# ============================================================
# Loss Config
#
# 上一轮：
#
# L =
# Global L1
# + 0.5 * Global MSE
# + 2.0 * Changed-region L1
#
#
# 本轮：
#
# L =
# Global L1
# + 0.5 * Global MSE
# + 2.0 * Changed-region MSE
#
# 目的：
# 测试 Changed Region 中使用 MSE，
# 是否能够进一步减少手写擦除后的灰色 ghosting。
# ============================================================

LAMBDA_GLOBAL_MSE = 0.5
LAMBDA_CHANGED = 2.0

DIFFERENCE_THRESHOLD = 0.1


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
    "Fine-tune Checkpoint：",
    CHECKPOINT_PATH,
)

print(
    "Loss： "
    "L1 + "
    f"{LAMBDA_GLOBAL_MSE} × Global MSE + "
    f"{LAMBDA_CHANGED} × Changed-Region MSE"
)

print(
    "Difference Threshold：",
    DIFFERENCE_THRESHOLD,
)

print(
    "Learning Rate：",
    LEARNING_RATE,
)


# ============================================================
# 2. Dataset
# ============================================================

train_dataset = PreprocessedHandwritingDataset(
    data_root=DATA_ROOT,
    split_file=TRAIN_SPLIT,
    patch_size=PATCH_SIZE,
    mode="train",
    patches_per_image=VAL_PATCHES_PER_IMAGE,
    blur_radius=BLUR_RADIUS,
)


val_dataset = PreprocessedHandwritingDataset(
    data_root=DATA_ROOT,
    split_file=VAL_SPLIT,
    patch_size=PATCH_SIZE,
    mode="val",
    patches_per_image=VAL_PATCHES_PER_IMAGE,
    blur_radius=BLUR_RADIUS,
)


# ============================================================
# 3. DataLoader
#
# num_workers=4：
# 并行执行较昂贵的 Document Enhancement preprocessing。
#
# pin_memory=True + non_blocking=True：
# 改善 CPU → CUDA 数据传输。
# ============================================================

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
# 4. Model
# ============================================================

# dataset_preprocessed 输出：
#
# [B, 1, H, W]
#
# 因此 U-Net：
#
# in_channels = 1
# out_channels = 1
# ============================================================

model = UNet(
    in_channels=1,
    out_channels=1,
).to(device)


total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print("\n===== U-Net Parameters =====")

print(
    "Total Parameters：",
    total_parameters,
)

print(
    "Trainable Parameters：",
    trainable_parameters,
)


# ============================================================
# 5. Load Previous Checkpoint
# ============================================================

if not os.path.exists(
    CHECKPOINT_PATH
):
    raise FileNotFoundError(
        f"找不到 checkpoint：{CHECKPOINT_PATH}"
    )


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)


# ------------------------------------------------------------
# 兼容两种常见保存方式：
#
# 方式 1：
# torch.save(model.state_dict(), path)
#
# 方式 2：
# torch.save({
#     "model_state_dict": model.state_dict(),
#     ...
# }, path)
# ------------------------------------------------------------

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:
    model.load_state_dict(
        checkpoint
    )


print(
    "\n成功加载上一轮模型：",
    CHECKPOINT_PATH,
)


# ============================================================
# 6. Optimizer
#
# 注意：
#
# 我们只加载模型参数，
# 不加载上一轮 Adam optimizer 状态。
#
# 当前实验的目标是：
#
# 已训练模型
#     ↓
# 新 Loss
#     ↓
# 较低 LR
#     ↓
# 新的 Fine-tuning 阶段
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# 7. Loss Functions
# ============================================================

def calculate_loss(
    predictions,
    inputs,
    targets,
):
    """
    当前 Fine-tuning Loss：

    Global L1
        +
    0.5 × Global MSE
        +
    2.0 × Changed-region MSE


    Difference Mask：

        |Input - Target| > threshold

    暂时保持上一轮 Mask 定义不变，
    这样本轮实验只改变 Changed Loss：

        L1 → MSE

    从而尽量保持单变量实验。
    """

    # --------------------------------------------------------
    # Global L1
    # --------------------------------------------------------

    global_l1 = F.l1_loss(
        predictions,
        targets,
    )


    # --------------------------------------------------------
    # Global MSE
    # --------------------------------------------------------

    global_mse = F.mse_loss(
        predictions,
        targets,
    )


    # --------------------------------------------------------
    # Difference Mask
    #
    # 当前仍然沿用上一轮：
    #
    # abs(Input - Target) > 0.1
    #
    # 白色区域代表：
    # Input 与 Target 差异明显，需要重点学习。
    # --------------------------------------------------------

    difference = torch.abs(
        inputs - targets
    )

    changed_mask = (
        difference
        > DIFFERENCE_THRESHOLD
    ).float()


    # --------------------------------------------------------
    # Changed-region MSE
    #
    # 注意：
    #
    # 不能直接：
    #
    # mse(pred * mask, target * mask)
    #
    # 因为这样仍然会被整张图片的像素数量平均。
    #
    # 我们只希望对 mask 中的像素求平均。
    # --------------------------------------------------------

    squared_error = (
        predictions - targets
    ) ** 2


    mask_pixels = (
        changed_mask.sum()
    )


    if mask_pixels.item() > 0:

        changed_mse = (
            squared_error
            * changed_mask
        ).sum() / (
            mask_pixels + 1e-8
        )

    else:

        changed_mse = torch.zeros(
            (),
            device=predictions.device,
        )


    # --------------------------------------------------------
    # Total Loss
    # --------------------------------------------------------

    total_loss = (
        global_l1
        + LAMBDA_GLOBAL_MSE
        * global_mse
        + LAMBDA_CHANGED
        * changed_mse
    )


    return (
        total_loss,
        global_l1,
        global_mse,
        changed_mse,
        changed_mask,
    )


# ============================================================
# 8. Metrics
# ============================================================

def calculate_psnr(
    predictions,
    targets,
):
    mse = F.mse_loss(
        predictions,
        targets,
    ).item()

    if mse == 0:
        return float("inf")

    return (
        10.0
        * math.log10(
            1.0 / mse
        )
    )


def calculate_batch_ssim(
    predictions,
    targets,
):
    """
    对 batch 中每张灰度图片计算 SSIM，
    最后取平均。
    """

    predictions_np = (
        predictions
        .detach()
        .cpu()
        .numpy()
    )

    targets_np = (
        targets
        .detach()
        .cpu()
        .numpy()
    )

    scores = []

    for i in range(
        predictions_np.shape[0]
    ):

        pred = np.clip(
            predictions_np[i, 0],
            0.0,
            1.0,
        )

        target = np.clip(
            targets_np[i, 0],
            0.0,
            1.0,
        )

        score = structural_similarity(
            target,
            pred,
            data_range=1.0,
        )

        scores.append(
            score
        )

    return float(
        np.mean(scores)
    )


# ============================================================
# 9. History
# ============================================================

train_total_history = []

train_l1_history = []
train_mse_history = []
train_changed_mse_history = []


val_total_history = []

val_l1_history = []
val_mse_history = []
val_changed_mse_history = []

val_psnr_history = []
val_ssim_history = []


# ============================================================
# 10. Fine-tuning
# ============================================================

print(
    "\n===== Changed-MSE Fine-tuning ====="
)


best_val_loss = float("inf")

best_epoch = -1

total_start_time = time.time()


for epoch in range(
    NUM_EPOCHS
):

    epoch_start_time = time.time()


    # ========================================================
    # Train
    # ========================================================

    model.train()


    train_total_sum = 0.0

    train_l1_sum = 0.0
    train_mse_sum = 0.0

    train_changed_sum = 0.0


    for (
        inputs,
        targets,
    ) in train_loader:

        inputs = inputs.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )


        optimizer.zero_grad()


        predictions = model(
            inputs
        )


        (
            total_loss,
            global_l1,
            global_mse,
            changed_mse,
            _,
        ) = calculate_loss(
            predictions,
            inputs,
            targets,
        )


        total_loss.backward()

        optimizer.step()


        train_total_sum += (
            total_loss.item()
        )

        train_l1_sum += (
            global_l1.item()
        )

        train_mse_sum += (
            global_mse.item()
        )

        train_changed_sum += (
            changed_mse.item()
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


    with torch.no_grad():

        for (
            inputs,
            targets,
        ) in val_loader:

            inputs = inputs.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
                device,
                non_blocking=True,
            )


            predictions = model(
                inputs
            )


            (
                total_loss,
                global_l1,
                global_mse,
                changed_mse,
                _,
            ) = calculate_loss(
                predictions,
                inputs,
                targets,
            )


            val_total_sum += (
                total_loss.item()
            )

            val_l1_sum += (
                global_l1.item()
            )

            val_mse_sum += (
                global_mse.item()
            )

            val_changed_sum += (
                changed_mse.item()
            )


            val_psnr_sum += (
                calculate_psnr(
                    predictions,
                    targets,
                )
            )

            val_ssim_sum += (
                calculate_batch_ssim(
                    predictions,
                    targets,
                )
            )


    val_total = (
        val_total_sum
        / len(val_loader)
    )

    val_l1 = (
        val_l1_sum
        / len(val_loader)
    )

    val_mse = (
        val_mse_sum
        / len(val_loader)
    )

    val_changed = (
        val_changed_sum
        / len(val_loader)
    )

    val_psnr = (
        val_psnr_sum
        / len(val_loader)
    )

    val_ssim = (
        val_ssim_sum
        / len(val_loader)
    )


    # ========================================================
    # Save History
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

    train_changed_mse_history.append(
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

    val_changed_mse_history.append(
        val_changed
    )

    val_psnr_history.append(
        val_psnr
    )

    val_ssim_history.append(
        val_ssim
    )


    # ========================================================
    # Save Best Model
    # ========================================================

    if val_total < best_val_loss:

        best_val_loss = val_total

        best_epoch = (
            epoch + 1
        )

        torch.save(
            model.state_dict(),
            os.path.join(
                OUTPUT_DIR,
                "best_unet_changed_mse.pth",
            ),
        )


    epoch_time = (
        time.time()
        - epoch_start_time
    )


    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}]"
        f" | Train Total: {train_total:.6f}"
        f" | Train L1: {train_l1:.6f}"
        f" | Train Changed-MSE: {train_changed:.6f}"
        f" | Val Total: {val_total:.6f}"
        f" | Val L1: {val_l1:.6f}"
        f" | Val Changed-MSE: {val_changed:.6f}"
        f" | PSNR: {val_psnr:.2f} dB"
        f" | SSIM: {val_ssim:.4f}"
        f" | Time: {epoch_time:.2f}s"
    )


# ============================================================
# 11. Save Final Model
# ============================================================

total_training_time = (
    time.time()
    - total_start_time
)


torch.save(
    model.state_dict(),
    os.path.join(
        OUTPUT_DIR,
        "final_unet_changed_mse.pth",
    ),
)


print(
    "\n===== Fine-tuning Finished ====="
)

print(
    "Best Epoch：",
    best_epoch,
)

print(
    "Best Validation Loss：",
    f"{best_val_loss:.6f}",
)

print(
    "Total Fine-tuning Time：",
    f"{total_training_time:.2f}s",
)


# ============================================================
# 12. Curves
# ============================================================

epochs = list(
    range(
        1,
        NUM_EPOCHS + 1,
    )
)


# ------------------------------------------------------------
# Total Loss
# ------------------------------------------------------------

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
    "Fine-tuning Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Changed-MSE Fine-tuning Loss"
)

plt.grid()

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "finetune_loss_curve.png",
    ),
    dpi=150,
)

plt.close()


# ------------------------------------------------------------
# Validation Components
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    val_l1_history,
    marker="o",
    label="Global L1",
)

plt.plot(
    epochs,
    val_mse_history,
    marker="o",
    label="Global MSE",
)

plt.plot(
    epochs,
    val_changed_mse_history,
    marker="o",
    label="Changed MSE",
)

plt.xlabel(
    "Fine-tuning Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Changed-MSE Validation Loss Components"
)

plt.grid()

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "loss_components.png",
    ),
    dpi=150,
)

plt.close()


# ------------------------------------------------------------
# PSNR
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    val_psnr_history,
    marker="o",
)

plt.xlabel(
    "Fine-tuning Epoch"
)

plt.ylabel(
    "PSNR (dB)"
)

plt.title(
    "Changed-MSE Validation PSNR"
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


# ------------------------------------------------------------
# SSIM
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    val_ssim_history,
    marker="o",
)

plt.xlabel(
    "Fine-tuning Epoch"
)

plt.ylabel(
    "SSIM"
)

plt.title(
    "Changed-MSE Validation SSIM"
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
# 13. Visualization
#
# 使用 best checkpoint，而不是最后一个 epoch。
# ============================================================

best_checkpoint_path = os.path.join(
    OUTPUT_DIR,
    "best_unet_changed_mse.pth",
)


model.load_state_dict(
    torch.load(
        best_checkpoint_path,
        map_location=device,
    )
)

model.eval()


# 固定取 validation dataset 前四个 patch，
# 方便和之前实验视觉比较。

num_examples = min(
    4,
    len(val_dataset),
)


fig, axes = plt.subplots(
    num_examples,
    4,
    figsize=(
        16,
        4 * num_examples,
    ),
)


with torch.no_grad():

    for i in range(
        num_examples
    ):

        input_image, target = (
            val_dataset[i]
        )


        input_batch = (
            input_image
            .unsqueeze(0)
            .to(device)
        )


        prediction = model(
            input_batch
        )[0, 0]


        prediction = (
            prediction
            .clamp(
                0.0,
                1.0,
            )
            .cpu()
            .numpy()
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


        difference = np.abs(
            input_np
            - target_np
        )

        difference_mask = (
            difference
            > DIFFERENCE_THRESHOLD
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


        axes[i, 1].imshow(
            difference_mask,
            cmap="gray",
        )

        axes[i, 1].set_title(
            f"Difference Mask {i + 1}"
        )


        axes[i, 2].imshow(
            prediction,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[i, 2].set_title(
            f"Prediction {i + 1}"
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


        for j in range(4):
            axes[i, j].axis(
                "off"
            )


plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "changed_mse_predictions.png",
    ),
    dpi=150,
)

plt.close()


print(
    "\n输出文件已保存到：",
    OUTPUT_DIR,
)