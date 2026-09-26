import csv
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
# 1. Config
# ============================================================

DATA_ROOT = "/mnt/d/User/University/QQ/Dian/deli"

VAL_SPLIT = "splits/val.txt"

MODEL_PATH = "outputs/train/best_unet.pth"

OUTPUT_DIR = Path(
    "outputs/recursive_inference"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PATCH_SIZE = 256
BATCH_SIZE = 8

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("使用设备：", device)


# ============================================================
# 2. Validation Dataset
#
# 必须与 RGB Baseline 使用相同的 Validation Dataset。
#
# mode="val":
# 每张原图固定取 5 个位置。
#
# 241 × 5 = 1205 patches
#
# 不使用 Test Set。
# ============================================================

val_dataset = HandwritingDataset(
    data_root=DATA_ROOT,
    split_file=VAL_SPLIT,
    patch_size=PATCH_SIZE,
    mode="val",

    # RGB Baseline
    input_grayscale=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print("\n===== Dataset =====")
print("Validation Patches：", len(val_dataset))
print("Validation Batches：", len(val_loader))

assert len(val_dataset) == 241 * 5


# ============================================================
# 3. Load RGB Baseline U-Net
# ============================================================

model = UNet(
    in_channels=3,
    out_channels=1,
).to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
)

model.load_state_dict(
    state_dict
)

model.eval()


total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print("\n===== Model =====")
print("Checkpoint：", MODEL_PATH)
print("U-Net Parameters：", total_parameters)


# ============================================================
# 4. Loss
# ============================================================

criterion = nn.L1Loss(
    reduction="none"
)


# ============================================================
# 5. PSNR
#
# prediction / target:
# [B,1,H,W]
#
# 返回：
# [B]
# ============================================================

def calculate_batch_psnr(
    predictions,
    targets,
):

    mse = torch.mean(
        (
            predictions
            - targets
        ) ** 2,
        dim=(1, 2, 3),
    )

    mse = torch.clamp(
        mse,
        min=1e-10,
    )

    psnr = (
        10.0
        * torch.log10(
            1.0 / mse
        )
    )

    return psnr


# ============================================================
# 6. SSIM
#
# 每张图单独计算。
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

        score = structural_similarity(
            target,
            prediction,
            data_range=1.0,
        )

        scores.append(
            score
        )

    return np.array(
        scores,
        dtype=np.float64,
    )


# ============================================================
# 7. 1 Channel Prediction → 3 Channel Input
#
# U-Net 第一层要求：
#
# [B,3,H,W]
#
# 但是 Prediction 是：
#
# [B,1,H,W]
#
# 所以：
#
# Gray
# ↓
# Gray / Gray / Gray
#
# 三个通道完全相同。
#
# 注意：
# 这里不是恢复 RGB 颜色。
# 只是为了满足网络输入 shape。
# ============================================================

def prediction_to_input(
    prediction,
):

    return prediction.repeat(
        1,
        3,
        1,
        1,
    )


# ============================================================
# 8. 保存每个 Patch 的统计信息
# ============================================================

all_results = []

global_index = 0


# ============================================================
# 9. 保存第一批用于可视化
# ============================================================

visual_inputs = None
visual_p1 = None
visual_p2 = None
visual_p3 = None
visual_targets = None


# ============================================================
# 10. Recursive Inference
# ============================================================

print(
    "\n===== Recursive Inference ====="
)

print(
    "Input → P1 → P2 → P3"
)


with torch.no_grad():

    for batch_index, (
        inputs,
        targets,
    ) in enumerate(val_loader):

        inputs = inputs.to(
            device
        )

        targets = targets.to(
            device
        )


        # ====================================================
        # Pass 1
        #
        # P1 = f(X)
        # ====================================================

        p1 = model(
            inputs
        )


        # ====================================================
        # Pass 2
        #
        # P2 = f(P1)
        #
        # P1:
        # [B,1,H,W]
        #
        # ↓ repeat
        #
        # [B,3,H,W]
        # ====================================================

        p1_input = prediction_to_input(
            p1
        )

        p2 = model(
            p1_input
        )


        # ====================================================
        # Pass 3
        #
        # P3 = f(P2)
        # ====================================================

        p2_input = prediction_to_input(
            p2
        )

        p3 = model(
            p2_input
        )


        # ====================================================
        # L1
        #
        # reduction="none"
        #
        # 先得到每个像素 Loss，
        # 再对每张图片单独平均。
        #
        # 最终 shape:
        # [B]
        # ====================================================

        l1_p1 = criterion(
            p1,
            targets,
        ).mean(
            dim=(1, 2, 3)
        )

        l1_p2 = criterion(
            p2,
            targets,
        ).mean(
            dim=(1, 2, 3)
        )

        l1_p3 = criterion(
            p3,
            targets,
        ).mean(
            dim=(1, 2, 3)
        )


        # ====================================================
        # PSNR
        # ====================================================

        psnr_p1 = calculate_batch_psnr(
            p1,
            targets,
        )

        psnr_p2 = calculate_batch_psnr(
            p2,
            targets,
        )

        psnr_p3 = calculate_batch_psnr(
            p3,
            targets,
        )


        # ====================================================
        # SSIM
        # ====================================================

        ssim_p1 = calculate_batch_ssim(
            p1,
            targets,
        )

        ssim_p2 = calculate_batch_ssim(
            p2,
            targets,
        )

        ssim_p3 = calculate_batch_ssim(
            p3,
            targets,
        )


        # ====================================================
        # CPU
        # ====================================================

        l1_p1_cpu = (
            l1_p1
            .cpu()
            .numpy()
        )

        l1_p2_cpu = (
            l1_p2
            .cpu()
            .numpy()
        )

        l1_p3_cpu = (
            l1_p3
            .cpu()
            .numpy()
        )

        psnr_p1_cpu = (
            psnr_p1
            .cpu()
            .numpy()
        )

        psnr_p2_cpu = (
            psnr_p2
            .cpu()
            .numpy()
        )

        psnr_p3_cpu = (
            psnr_p3
            .cpu()
            .numpy()
        )


        # ====================================================
        # 保存第一 Batch 用于可视化
        # ====================================================

        if batch_index == 0:

            visual_inputs = (
                inputs
                .detach()
                .cpu()
            )

            visual_p1 = (
                p1
                .detach()
                .cpu()
            )

            visual_p2 = (
                p2
                .detach()
                .cpu()
            )

            visual_p3 = (
                p3
                .detach()
                .cpu()
            )

            visual_targets = (
                targets
                .detach()
                .cpu()
            )


        # ====================================================
        # 保存每一个 Patch 的指标
        # ====================================================

        current_batch_size = (
            inputs.size(0)
        )

        for i in range(
            current_batch_size
        ):

            result = {
                "index": global_index,

                "l1_p1": float(
                    l1_p1_cpu[i]
                ),

                "l1_p2": float(
                    l1_p2_cpu[i]
                ),

                "l1_p3": float(
                    l1_p3_cpu[i]
                ),

                "psnr_p1": float(
                    psnr_p1_cpu[i]
                ),

                "psnr_p2": float(
                    psnr_p2_cpu[i]
                ),

                "psnr_p3": float(
                    psnr_p3_cpu[i]
                ),

                "ssim_p1": float(
                    ssim_p1[i]
                ),

                "ssim_p2": float(
                    ssim_p2[i]
                ),

                "ssim_p3": float(
                    ssim_p3[i]
                ),
            }

            all_results.append(
                result
            )

            global_index += 1


        if (
            (batch_index + 1) % 25 == 0
            or
            (batch_index + 1)
            == len(val_loader)
        ):

            print(
                f"已处理："
                f"{global_index}/"
                f"{len(val_dataset)} "
                f"patches"
            )


# ============================================================
# 11. 汇总指标
# ============================================================

def mean_metric(
    key,
):

    return float(
        np.mean(
            [
                result[key]
                for result in all_results
            ]
        )
    )


mean_l1_p1 = mean_metric(
    "l1_p1"
)

mean_l1_p2 = mean_metric(
    "l1_p2"
)

mean_l1_p3 = mean_metric(
    "l1_p3"
)


mean_psnr_p1 = mean_metric(
    "psnr_p1"
)

mean_psnr_p2 = mean_metric(
    "psnr_p2"
)

mean_psnr_p3 = mean_metric(
    "psnr_p3"
)


mean_ssim_p1 = mean_metric(
    "ssim_p1"
)

mean_ssim_p2 = mean_metric(
    "ssim_p2"
)

mean_ssim_p3 = mean_metric(
    "ssim_p3"
)


# ============================================================
# 12. 统计 P2 / P3 到底改善了多少 Patch
# ============================================================

# L1 越低越好

p2_l1_better = sum(
    result["l1_p2"]
    < result["l1_p1"]
    for result in all_results
)

p3_l1_better_than_p2 = sum(
    result["l1_p3"]
    < result["l1_p2"]
    for result in all_results
)


# PSNR 越高越好

p2_psnr_better = sum(
    result["psnr_p2"]
    > result["psnr_p1"]
    for result in all_results
)

p3_psnr_better_than_p2 = sum(
    result["psnr_p3"]
    > result["psnr_p2"]
    for result in all_results
)


# SSIM 越高越好

p2_ssim_better = sum(
    result["ssim_p2"]
    > result["ssim_p1"]
    for result in all_results
)

p3_ssim_better_than_p2 = sum(
    result["ssim_p3"]
    > result["ssim_p2"]
    for result in all_results
)


num_patches = len(
    all_results
)


# ============================================================
# 13. Print Summary
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "Recursive Inference Results"
)

print(
    "=" * 70
)


print(
    "\n===== Mean Metrics ====="
)


print(
    f"P1 | "
    f"L1: {mean_l1_p1:.6f} | "
    f"PSNR: {mean_psnr_p1:.2f} dB | "
    f"SSIM: {mean_ssim_p1:.4f}"
)


print(
    f"P2 | "
    f"L1: {mean_l1_p2:.6f} | "
    f"PSNR: {mean_psnr_p2:.2f} dB | "
    f"SSIM: {mean_ssim_p2:.4f}"
)


print(
    f"P3 | "
    f"L1: {mean_l1_p3:.6f} | "
    f"PSNR: {mean_psnr_p3:.2f} dB | "
    f"SSIM: {mean_ssim_p3:.4f}"
)


print(
    "\n===== P2 vs P1 ====="
)


print(
    "L1 改善："
    f"{p2_l1_better}/{num_patches} "
    f"("
    f"{100 * p2_l1_better / num_patches:.2f}%"
    f")"
)


print(
    "PSNR 改善："
    f"{p2_psnr_better}/{num_patches} "
    f"("
    f"{100 * p2_psnr_better / num_patches:.2f}%"
    f")"
)


print(
    "SSIM 改善："
    f"{p2_ssim_better}/{num_patches} "
    f"("
    f"{100 * p2_ssim_better / num_patches:.2f}%"
    f")"
)


print(
    "\n===== P3 vs P2 ====="
)


print(
    "L1 改善："
    f"{p3_l1_better_than_p2}/{num_patches} "
    f"("
    f"{100 * p3_l1_better_than_p2 / num_patches:.2f}%"
    f")"
)


print(
    "PSNR 改善："
    f"{p3_psnr_better_than_p2}/{num_patches} "
    f"("
    f"{100 * p3_psnr_better_than_p2 / num_patches:.2f}%"
    f")"
)


print(
    "SSIM 改善："
    f"{p3_ssim_better_than_p2}/{num_patches} "
    f"("
    f"{100 * p3_ssim_better_than_p2 / num_patches:.2f}%"
    f")"
)


# ============================================================
# 14. 保存 CSV
# ============================================================

csv_path = (
    OUTPUT_DIR
    / "recursive_metrics.csv"
)


with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "index",
        "l1_p1",
        "l1_p2",
        "l1_p3",
        "psnr_p1",
        "psnr_p2",
        "psnr_p3",
        "ssim_p1",
        "ssim_p2",
        "ssim_p3",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        all_results
    )


print(
    "\nCSV 已保存：",
    csv_path,
)


# ============================================================
# 15. Metric Comparison Plot
# ============================================================

passes = [
    "P1",
    "P2",
    "P3",
]


# ------------------------------------------------------------
# L1
# ------------------------------------------------------------

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    passes,
    [
        mean_l1_p1,
        mean_l1_p2,
        mean_l1_p3,
    ],
    marker="o",
)

plt.xlabel(
    "Recursive Pass"
)

plt.ylabel(
    "Mean L1 Loss"
)

plt.title(
    "Recursive Inference - L1"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "recursive_l1.png",
    dpi=150,
)

plt.close()


# ------------------------------------------------------------
# PSNR
# ------------------------------------------------------------

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    passes,
    [
        mean_psnr_p1,
        mean_psnr_p2,
        mean_psnr_p3,
    ],
    marker="o",
)

plt.xlabel(
    "Recursive Pass"
)

plt.ylabel(
    "PSNR (dB)"
)

plt.title(
    "Recursive Inference - PSNR"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "recursive_psnr.png",
    dpi=150,
)

plt.close()


# ------------------------------------------------------------
# SSIM
# ------------------------------------------------------------

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    passes,
    [
        mean_ssim_p1,
        mean_ssim_p2,
        mean_ssim_p3,
    ],
    marker="o",
)

plt.xlabel(
    "Recursive Pass"
)

plt.ylabel(
    "SSIM"
)

plt.title(
    "Recursive Inference - SSIM"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "recursive_ssim.png",
    dpi=150,
)

plt.close()


# ============================================================
# 16. 第一批可视化
#
# Input | P1 | P2 | P3 | Target
# ============================================================

num_visuals = min(
    4,
    visual_inputs.size(0),
)


fig, axes = plt.subplots(
    num_visuals,
    5,
    figsize=(
        18,
        num_visuals * 4,
    ),
)


for i in range(
    num_visuals
):

    input_image = (
        visual_inputs[i]
        .permute(1, 2, 0)
        .numpy()
    )

    p1_image = (
        visual_p1[i]
        .squeeze(0)
        .numpy()
    )

    p2_image = (
        visual_p2[i]
        .squeeze(0)
        .numpy()
    )

    p3_image = (
        visual_p3[i]
        .squeeze(0)
        .numpy()
    )

    target_image = (
        visual_targets[i]
        .squeeze(0)
        .numpy()
    )


    # Input
    axes[i, 0].imshow(
        input_image
    )

    axes[i, 0].set_title(
        f"Input {i + 1}"
    )

    axes[i, 0].axis(
        "off"
    )


    # P1
    axes[i, 1].imshow(
        p1_image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[i, 1].set_title(
        f"P1 {i + 1}"
    )

    axes[i, 1].axis(
        "off"
    )


    # P2
    axes[i, 2].imshow(
        p2_image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[i, 2].set_title(
        f"P2 {i + 1}"
    )

    axes[i, 2].axis(
        "off"
    )


    # P3
    axes[i, 3].imshow(
        p3_image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[i, 3].set_title(
        f"P3 {i + 1}"
    )

    axes[i, 3].axis(
        "off"
    )


    # Target
    axes[i, 4].imshow(
        target_image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[i, 4].set_title(
        f"Target {i + 1}"
    )

    axes[i, 4].axis(
        "off"
    )


plt.tight_layout()

visual_path = (
    OUTPUT_DIR
    / "recursive_examples.png"
)

plt.savefig(
    visual_path,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


print(
    "可视化已保存：",
    visual_path,
)


# ============================================================
# 17. Finished
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "Recursive Inference Test Finished!"
)

print(
    "=" * 70
)

print(
    "输出目录：",
    OUTPUT_DIR,
)