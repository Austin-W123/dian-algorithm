import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from dataset import HandwritingDataset
from unet import UNet


# ============================================================
# 1. Experiment Config
# ============================================================

SEED = 42


DATA_ROOT = (
    "/mnt/d/User/University/QQ/Dian/deli"
)


TRAIN_SPLIT = (
    "splits/train.txt"
)

VAL_SPLIT = (
    "splits/val.txt"
)


PATCH_SIZE = 256

BATCH_SIZE = 8


# ------------------------------------------------------------
# 本次先跑 5 Epoch。
#
# 目的不是立刻训练最终模型，
# 而是验证：
#
# 去除 Input 的颜色信息后，
# 模型是否能够减少：
#
# “红色 → 删除”
# “黑色 → 保留”
#
# 这种颜色捷径。
# ------------------------------------------------------------

NUM_EPOCHS = 5


LEARNING_RATE = 1e-3


# ------------------------------------------------------------
# 核心实验变量
#
# RGB Baseline:
#     False
#
# 当前 Grayscale Ablation:
#     True
# ------------------------------------------------------------

INPUT_GRAYSCALE = True


# ------------------------------------------------------------
# 绝对不能写成 outputs/train
#
# 否则会覆盖之前已经跑好的 RGB Baseline。
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    "outputs/train_grayscale"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. Random Seed
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# ============================================================
# 3. Device
# ============================================================

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
    "Input Mode：",
    "Grayscale → 3 Channels"
    if INPUT_GRAYSCALE
    else "RGB",
)


# ============================================================
# 4. Dataset
# ============================================================

train_dataset = HandwritingDataset(
    data_root=DATA_ROOT,
    split_file=TRAIN_SPLIT,
    patch_size=PATCH_SIZE,
    mode="train",

    # 本实验唯一核心变化
    input_grayscale=INPUT_GRAYSCALE,
)


val_dataset = HandwritingDataset(
    data_root=DATA_ROOT,
    split_file=VAL_SPLIT,
    patch_size=PATCH_SIZE,
    mode="val",

    # Validation 必须和 Train 使用相同 Input Mode。
    input_grayscale=INPUT_GRAYSCALE,
)


print(
    "\n===== Dataset ====="
)


print(
    "Train 原图 Pair：",
    len(train_dataset),
)


print(
    "Validation Patches：",
    len(val_dataset),
)


# 冻结 Split 后，
# 数量理论上必须保持不变。
assert len(train_dataset) == 1929

assert len(val_dataset) == (
    241 * 5
)


# ============================================================
# 5. DataLoader
# ============================================================

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


print(
    "Train Batches：",
    len(train_loader),
)


print(
    "Validation Batches：",
    len(val_loader),
)


# ============================================================
# 6. Model
# ============================================================

# 注意：
#
# 虽然输入已经变成灰度，
# 但 Dataset 把 Gray 复制成了三个相同通道。
#
# 所以：
#
# Input Shape
# 仍然是：
# [B,3,256,256]
#
# 因此 U-Net 完全不需要修改。

model = UNet(
    in_channels=3,
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


print(
    "\n===== U-Net Parameters ====="
)


print(
    "Total Parameters：",
    total_parameters,
)


print(
    "Trainable Parameters：",
    trainable_parameters,
)


# ============================================================
# 7. Loss + Optimizer
# ============================================================

# 与 RGB Baseline 完全相同。
#
# 这样才能保证这是控制变量实验。

criterion = nn.L1Loss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# 8. PSNR
# ============================================================

def calculate_batch_psnr(
    predictions,
    targets,
):

    # --------------------------------------------------------
    # 对 Batch 中每张图单独计算 MSE。
    #
    # prediction / target:
    #
    # [B,1,H,W]
    #
    # 对 C,H,W 求平均：
    #
    # →
    #
    # [B]
    # --------------------------------------------------------

    mse = torch.mean(
        (
            predictions
            - targets
        ) ** 2,
        dim=(1, 2, 3),
    )


    # 防止出现 MSE = 0，
    # 导致 log10(∞)。
    mse = torch.clamp(
        mse,
        min=1e-10,
    )


    # 像素范围是 [0,1]，
    # 所以 MAX_I = 1。
    #
    # PSNR =
    #
    # 10 * log10(1 / MSE)

    psnr = (
        10.0
        * torch.log10(
            1.0 / mse
        )
    )


    return psnr


# ============================================================
# 9. SSIM
# ============================================================

def calculate_batch_ssim(
    predictions,
    targets,
):

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

        prediction = (
            predictions_np[i, 0]
        )

        target = (
            targets_np[i, 0]
        )


        score = (
            structural_similarity(
                target,
                prediction,
                data_range=1.0,
            )
        )


        scores.append(
            score
        )


    return scores


# ============================================================
# 10. 固定 Validation Batch
# ============================================================

# val_loader:
#
# shuffle=False
#
# Validation Dataset:
#
# fixed positions
#
# 所以这里得到的样本在整个实验过程中保持固定。

fixed_val_inputs, fixed_val_targets = (
    next(iter(val_loader))
)


fixed_val_inputs = (
    fixed_val_inputs.to(
        device
    )
)


fixed_val_targets = (
    fixed_val_targets.to(
        device
    )
)


# ============================================================
# 11. History
# ============================================================

train_losses = []

val_losses = []

val_psnr_history = []

val_ssim_history = []

epoch_times = []


best_val_loss = float(
    "inf"
)

best_epoch = 0


# ============================================================
# 12. Formal Training
# ============================================================

print(
    "\n===== Grayscale Ablation Training ====="
)


total_start_time = (
    time.time()
)


for epoch in range(
    NUM_EPOCHS
):

    epoch_start_time = (
        time.time()
    )


    # ========================================================
    # Train
    # ========================================================

    model.train()


    running_train_loss = 0.0

    train_samples = 0


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


        # ----------------------------------------------------
        # 清除上一 Batch 累积的梯度
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        predictions = model(
            inputs
        )


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            predictions,
            targets,
        )


        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Update Parameters
        # ----------------------------------------------------

        optimizer.step()


        batch_size = (
            inputs.size(0)
        )


        running_train_loss += (
            loss.item()
            * batch_size
        )


        train_samples += (
            batch_size
        )


    train_loss = (
        running_train_loss
        / train_samples
    )


    # ========================================================
    # Validation
    # ========================================================

    model.eval()


    running_val_loss = 0.0

    val_samples = 0

    psnr_sum = 0.0

    ssim_sum = 0.0


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


            running_val_loss += (
                loss.item()
                * batch_size
            )


            val_samples += (
                batch_size
            )


            # =================================================
            # PSNR
            # =================================================

            batch_psnr = (
                calculate_batch_psnr(
                    predictions,
                    targets,
                )
            )


            psnr_sum += (
                batch_psnr
                .sum()
                .item()
            )


            # =================================================
            # SSIM
            # =================================================

            batch_ssim = (
                calculate_batch_ssim(
                    predictions,
                    targets,
                )
            )


            ssim_sum += sum(
                batch_ssim
            )


    val_loss = (
        running_val_loss
        / val_samples
    )


    val_psnr = (
        psnr_sum
        / val_samples
    )


    val_ssim = (
        ssim_sum
        / val_samples
    )


    # ========================================================
    # Record History
    # ========================================================

    train_losses.append(
        train_loss
    )


    val_losses.append(
        val_loss
    )


    val_psnr_history.append(
        val_psnr
    )


    val_ssim_history.append(
        val_ssim
    )


    epoch_time = (
        time.time()
        - epoch_start_time
    )


    epoch_times.append(
        epoch_time
    )


    # ========================================================
    # Best Model
    #
    # 和 RGB Baseline 一样：
    #
    # 根据最低 Validation L1
    # 选择 Best Checkpoint。
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = (
            val_loss
        )


        best_epoch = (
            epoch + 1
        )


        torch.save(
            model.state_dict(),
            OUTPUT_DIR
            / "best_unet.pth",
        )


    # ========================================================
    # Print
    # ========================================================

    print(
        f"Epoch "
        f"[{epoch + 1}/{NUM_EPOCHS}] "
        f"Train L1: "
        f"{train_loss:.6f} | "
        f"Val L1: "
        f"{val_loss:.6f} | "
        f"PSNR: "
        f"{val_psnr:.2f} dB | "
        f"SSIM: "
        f"{val_ssim:.4f} | "
        f"Time: "
        f"{epoch_time:.2f}s"
    )


# ============================================================
# 13. Training Finished
# ============================================================

total_training_time = (
    time.time()
    - total_start_time
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
# 14. 保存 Last Model
# ============================================================

torch.save(
    model.state_dict(),
    OUTPUT_DIR
    / "last_unet.pth",
)


# ============================================================
# 15. Loss Curve
# ============================================================

epochs = range(
    1,
    NUM_EPOCHS + 1,
)


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
    "Grayscale U-Net Training / Validation Loss"
)


plt.legend()

plt.grid(True)

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "loss_curve.png",
    dpi=150,
)


plt.close()


# ============================================================
# 16. PSNR Curve
# ============================================================

plt.figure(
    figsize=(8, 5)
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
    "Grayscale Validation PSNR"
)


plt.grid(True)

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "psnr_curve.png",
    dpi=150,
)


plt.close()


# ============================================================
# 17. SSIM Curve
# ============================================================

plt.figure(
    figsize=(8, 5)
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
    "Grayscale Validation SSIM"
)


plt.grid(True)

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "ssim_curve.png",
    dpi=150,
)


plt.close()


# ============================================================
# 18. 加载 Best Model
# ============================================================

best_model_path = (
    OUTPUT_DIR
    / "best_unet.pth"
)


model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=device,
    )
)


model.eval()


# ============================================================
# 19. 固定 Validation Prediction
# ============================================================

with torch.no_grad():

    fixed_predictions = model(
        fixed_val_inputs
    )


NUM_VISUALS = min(
    4,
    fixed_val_inputs.size(0),
)


fig, axes = plt.subplots(
    NUM_VISUALS,
    3,
    figsize=(
        12,
        NUM_VISUALS * 4,
    ),
)


for i in range(
    NUM_VISUALS
):

    input_image = (
        fixed_val_inputs[i]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


    prediction_image = (
        fixed_predictions[i]
        .squeeze(0)
        .cpu()
        .numpy()
    )


    target_image = (
        fixed_val_targets[i]
        .squeeze(0)
        .cpu()
        .numpy()
    )


    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    axes[i, 0].imshow(
        input_image
    )


    axes[i, 0].set_title(
        f"Grayscale Input {i + 1}"
    )


    axes[i, 0].axis(
        "off"
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    axes[i, 1].imshow(
        prediction_image,
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
        target_image,
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
    OUTPUT_DIR
    / "validation_predictions.png",
    dpi=150,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# 20. Final Output
# ============================================================

print(
    "\n输出文件已保存到：",
    OUTPUT_DIR,
)