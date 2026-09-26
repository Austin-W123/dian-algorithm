import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

from dataset import HandwritingDataset
from unet import UNet


# ============================================================
# 1. 配置
# ============================================================

SEED = 42

DATA_ROOT = (
    "/mnt/d/User/University/QQ/Dian/deli"
)

TRAIN_SPLIT = "splits/train.txt"

PATCH_SIZE = 256

# Sanity Test 故意只用很少的数据
SANITY_SAMPLES = 32

BATCH_SIZE = 4

NUM_EPOCHS = 10

LEARNING_RATE = 1e-3


OUTPUT_DIR = Path(
    "outputs/sanity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. 随机种子
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("使用设备：", device)


# ============================================================
# 4. Dataset
# ============================================================

full_train_dataset = HandwritingDataset(
    data_root=DATA_ROOT,
    split_file=TRAIN_SPLIT,
    patch_size=PATCH_SIZE,
    training=True,
)


# 只取前 32 张已经属于 Train 的 Pair。
#
# 注意：
# 这里没有重新进行 Train / Val / Test Split。
# 只是从已经冻结的 Train 中取一个小子集进行调试。
sanity_dataset = Subset(
    full_train_dataset,
    range(SANITY_SAMPLES),
)


sanity_loader = DataLoader(
    sanity_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)


print(
    "完整 Train Pair：",
    len(full_train_dataset),
)

print(
    "Sanity Pair：",
    len(sanity_dataset),
)

print(
    "Sanity Batches / Epoch：",
    len(sanity_loader),
)


# ============================================================
# 5. Model
# ============================================================

model = UNet(
    in_channels=3,
    out_channels=1,
).to(device)


total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "U-Net Parameters：",
    total_parameters,
)


# ============================================================
# 6. Loss
#
# 第一版使用 L1 Loss：
#
# mean(|prediction - target|)
# ============================================================

criterion = nn.L1Loss()


# ============================================================
# 7. Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# 8. 保存训练前的固定样本
#
# 我们希望训练前后比较的是同一个 Patch。
#
# 因为 Dataset(training=True) 默认 Random Crop，
# 所以这里先把一个 Batch 固定下来。
# ============================================================

fixed_inputs, fixed_targets = next(
    iter(sanity_loader)
)

fixed_inputs = fixed_inputs.to(device)
fixed_targets = fixed_targets.to(device)


# ============================================================
# 9. Training
# ============================================================

epoch_losses = []


print("\n===== Sanity Training =====")


for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0
    total_samples = 0


    for inputs, targets in sanity_loader:

        inputs = inputs.to(device)
        targets = targets.to(device)


        # --------------------------------------------
        # 清空上一轮累积梯度
        # --------------------------------------------

        optimizer.zero_grad()


        # --------------------------------------------
        # Forward
        # --------------------------------------------

        predictions = model(inputs)


        # --------------------------------------------
        # Loss
        # --------------------------------------------

        loss = criterion(
            predictions,
            targets,
        )


        # --------------------------------------------
        # Backward
        # --------------------------------------------

        loss.backward()


        # --------------------------------------------
        # Update
        # --------------------------------------------

        optimizer.step()


        batch_size = inputs.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size


    epoch_loss = (
        running_loss
        / total_samples
    )

    epoch_losses.append(
        epoch_loss
    )


    print(
        f"Epoch "
        f"[{epoch + 1}/{NUM_EPOCHS}] "
        f"L1 Loss: "
        f"{epoch_loss:.6f}"
    )


# ============================================================
# 10. 保存 Sanity Model
# ============================================================

model_path = (
    OUTPUT_DIR
    / "sanity_unet.pth"
)

torch.save(
    model.state_dict(),
    model_path,
)

print(
    "\nSanity 模型已保存：",
    model_path,
)


# ============================================================
# 11. Loss Curve
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    range(
        1,
        NUM_EPOCHS + 1,
    ),
    epoch_losses,
    marker="o",
)

plt.xlabel("Epoch")
plt.ylabel("L1 Loss")
plt.title("U-Net Sanity Training Loss")

plt.grid(True)

plt.tight_layout()


loss_curve_path = (
    OUTPUT_DIR
    / "sanity_loss_curve.png"
)

plt.savefig(
    loss_curve_path,
    dpi=150,
)

plt.close()


print(
    "Loss Curve 已保存：",
    loss_curve_path,
)


# ============================================================
# 12. 固定 Batch 推理
# ============================================================

model.eval()


with torch.no_grad():

    fixed_predictions = model(
        fixed_inputs
    )


# ============================================================
# 13. 保存 Input / Prediction / Target
# ============================================================

NUM_VISUALS = min(
    4,
    fixed_inputs.size(0),
)


fig, axes = plt.subplots(
    NUM_VISUALS,
    3,
    figsize=(
        12,
        NUM_VISUALS * 4,
    ),
)


for i in range(NUM_VISUALS):

    # Input
    input_image = (
        fixed_inputs[i]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


    # Prediction
    prediction_image = (
        fixed_predictions[i]
        .squeeze(0)
        .cpu()
        .numpy()
    )


    # Target
    target_image = (
        fixed_targets[i]
        .squeeze(0)
        .cpu()
        .numpy()
    )


    axes[i, 0].imshow(
        input_image
    )

    axes[i, 0].set_title(
        f"Input {i + 1}"
    )

    axes[i, 0].axis("off")


    axes[i, 1].imshow(
        prediction_image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[i, 1].set_title(
        f"Prediction {i + 1}"
    )

    axes[i, 1].axis("off")


    axes[i, 2].imshow(
        target_image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[i, 2].set_title(
        f"Target {i + 1}"
    )

    axes[i, 2].axis("off")


plt.tight_layout()


prediction_path = (
    OUTPUT_DIR
    / "sanity_predictions.png"
)

plt.savefig(
    prediction_path,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


print(
    "Prediction 可视化已保存：",
    prediction_path,
)


print(
    "\nSanity Training Finished!"
)