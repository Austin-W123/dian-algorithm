from pathlib import Path
import argparse
import math

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt

from unet import UNet
from preprocessing import preprocess_document


# ============================================================
# 1. Config
# ============================================================

PATCH_SIZE = 256

# 256 - 192 = 64
# 即相邻 Patch 有 64 像素重叠。
STRIDE = 192

BLUR_RADIUS = 15

CHECKPOINT = Path(
    "outputs/train_weighted_loss/best_unet_weighted.pth"
)

OUTPUT_DIR = Path(
    "outputs/full_image_inference"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 2. 计算完整覆盖一张图片所需要的 Padding
#
# 我们希望最后一个 Patch 也恰好能够落在图像内部。
#
# 对某个长度 L：
#
#     Patch Size = P
#     Stride = S
#
# Patch 数量：
#
#     n = ceil((L - P) / S) + 1
#
# 覆盖后的长度：
#
#     covered = (n - 1) * S + P
#
# Padding：
#
#     pad = covered - L
#
# 如果原图本身小于 256，
# 至少 Padding 到 256。
# ============================================================

def calculate_padded_length(
    length,
    patch_size,
    stride,
):
    if length <= patch_size:
        return patch_size

    num_patches = (
        math.ceil(
            (length - patch_size)
            / stride
        )
        + 1
    )

    padded_length = (
        (num_patches - 1)
        * stride
        + patch_size
    )

    return padded_length


# ============================================================
# 3. 构造 Blending Weight
#
# 如果直接平均：
#
# Patch A:
# ████████████████
#
#             Patch B:
#             ████████████████
#
# 重叠处虽然可以平均，
# 但 Patch 边缘往往预测不如中心稳定。
#
# 因此这里使用一个平滑的二维权重：
#
# Patch 中央权重大
# Patch 边缘权重小
#
# 最后：
#
#     output =
#         sum(prediction * weight)
#         ------------------------
#             sum(weight)
#
# 这样可以减少 Patch 接缝。
# ============================================================

def create_blend_weight(
    patch_size,
):
    window_1d = np.hanning(
        patch_size
    ).astype(
        np.float32
    )

    # Hann Window 的最边缘是 0。
    #
    # 为避免图像最外边缘出现除 0，
    # 给权重设置一个很小的下限。
    window_1d = np.clip(
        window_1d,
        1e-3,
        None,
    )

    window_2d = np.outer(
        window_1d,
        window_1d,
    )

    window_2d = (
        window_2d
        / window_2d.max()
    )

    return window_2d.astype(
        np.float32
    )


# ============================================================
# 4. 加载 U-Net
# ============================================================

def load_model():
    model = UNet(
        in_channels=1,
        out_channels=1,
    ).to(
        device
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    # 兼容两种保存方式：
    #
    # 1.
    # torch.save(model.state_dict(), ...)
    #
    # 2.
    # torch.save({
    #     "model_state_dict": ...
    # }, ...)
    #
    if isinstance(
        checkpoint,
        dict,
    ) and (
        "model_state_dict"
        in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


# ============================================================
# 5. 单个 RGB Patch → 模型输入
#
# 注意：
#
# 训练时 dataset_preprocessed.py 的逻辑是：
#
#     RGB Patch
#         ↓
#     preprocess_document()
#         ↓
#     1-channel Tensor
#
# 所以完整图片推理时也保持完全相同的逻辑。
# ============================================================

def preprocess_patch(
    patch,
):
    processed = preprocess_document(
        patch,
        blur_radius=BLUR_RADIUS,
    )

    # preprocess_document() 在我们当前工程中
    # 应该返回单通道 PIL Image。
    #
    # TF.to_tensor:
    #
    # PIL L
    #     ↓
    # [1, H, W]
    #     ↓
    # float32 [0, 1]
    #
    tensor = TF.to_tensor(
        processed
    )

    return tensor


# ============================================================
# 6. 完整图片推理
# ============================================================

def infer_full_image(
    image_path,
    model,
):
    image_path = Path(
        image_path
    )

    print(
        "\n读取图片：",
        image_path,
    )

    # --------------------------------------------------------
    # 原始图片统一转换为 RGB
    # --------------------------------------------------------

    original = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    original_width, original_height = (
        original.size
    )

    print(
        "原始尺寸：",
        f"{original_width} × {original_height}",
    )

    # --------------------------------------------------------
    # 计算 Padding 后尺寸
    # --------------------------------------------------------

    padded_width = (
        calculate_padded_length(
            original_width,
            PATCH_SIZE,
            STRIDE,
        )
    )

    padded_height = (
        calculate_padded_length(
            original_height,
            PATCH_SIZE,
            STRIDE,
        )
    )

    print(
        "Padding 后尺寸：",
        f"{padded_width} × {padded_height}",
    )

    # --------------------------------------------------------
    # 建立白色画布
    #
    # 文档图片 Padding 用白色比黑色合理。
    # --------------------------------------------------------

    padded_image = Image.new(
        "RGB",
        (
            padded_width,
            padded_height,
        ),
        color=(
            255,
            255,
            255,
        ),
    )

    padded_image.paste(
        original,
        (
            0,
            0,
        ),
    )

    # --------------------------------------------------------
    # Prediction 累加器
    #
    # prediction_sum:
    #     保存所有 patch prediction × weight 的累加
    #
    # weight_sum:
    #     保存每个位置总共累积了多少 weight
    # --------------------------------------------------------

    prediction_sum = np.zeros(
        (
            padded_height,
            padded_width,
        ),
        dtype=np.float32,
    )

    weight_sum = np.zeros(
        (
            padded_height,
            padded_width,
        ),
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # 为了生成完整的 Preprocessed 可视化，
    # 同时把预处理后的 Patch 也拼起来。
    # --------------------------------------------------------

    preprocessed_sum = np.zeros(
        (
            padded_height,
            padded_width,
        ),
        dtype=np.float32,
    )

    preprocessed_weight_sum = np.zeros(
        (
            padded_height,
            padded_width,
        ),
        dtype=np.float32,
    )

    blend_weight = (
        create_blend_weight(
            PATCH_SIZE
        )
    )

    # --------------------------------------------------------
    # Patch 坐标
    # --------------------------------------------------------

    x_positions = list(
        range(
            0,
            padded_width
            - PATCH_SIZE
            + 1,
            STRIDE,
        )
    )

    y_positions = list(
        range(
            0,
            padded_height
            - PATCH_SIZE
            + 1,
            STRIDE,
        )
    )

    total_patches = (
        len(x_positions)
        * len(y_positions)
    )

    print(
        "Patch Size：",
        PATCH_SIZE,
    )

    print(
        "Stride：",
        STRIDE,
    )

    print(
        "Overlap：",
        PATCH_SIZE - STRIDE,
    )

    print(
        "X 方向 Patch：",
        len(x_positions),
    )

    print(
        "Y 方向 Patch：",
        len(y_positions),
    )

    print(
        "总 Patch 数：",
        total_patches,
    )

    # --------------------------------------------------------
    # 开始推理
    # --------------------------------------------------------

    patch_counter = 0

    with torch.no_grad():

        for y in y_positions:

            for x in x_positions:

                patch_counter += 1

                # --------------------------------------------
                # 从完整 RGB 图片裁 256×256 Patch
                # --------------------------------------------

                patch = padded_image.crop(
                    (
                        x,
                        y,
                        x + PATCH_SIZE,
                        y + PATCH_SIZE,
                    )
                )

                # --------------------------------------------
                # 与训练完全相同的 preprocessing
                # --------------------------------------------

                input_tensor = (
                    preprocess_patch(
                        patch
                    )
                    .unsqueeze(0)
                    .to(device)
                )

                # Shape:
                #
                # [1, 1, 256, 256]

                # --------------------------------------------
                # U-Net Forward
                # --------------------------------------------

                prediction = model(
                    input_tensor
                )

                # 如果当前 unet.py 最后一层本身已经 Sigmoid，
                # 这里不再额外 Sigmoid。
                #
                # 为确保保存为合法灰度范围，
                # 最后只 clamp 到 [0, 1]。
                prediction = torch.clamp(
                    prediction,
                    0.0,
                    1.0,
                )

                # --------------------------------------------
                # Tensor → numpy
                # --------------------------------------------

                prediction_np = (
                    prediction[
                        0,
                        0,
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                input_np = (
                    input_tensor[
                        0,
                        0,
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                # --------------------------------------------
                # Weighted Blending
                # --------------------------------------------

                prediction_sum[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE,
                ] += (
                    prediction_np
                    * blend_weight
                )

                weight_sum[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE,
                ] += blend_weight

                # --------------------------------------------
                # 同时拼接 Preprocessed Input
                # --------------------------------------------

                preprocessed_sum[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE,
                ] += (
                    input_np
                    * blend_weight
                )

                preprocessed_weight_sum[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE,
                ] += blend_weight

                if (
                    patch_counter % 50 == 0
                    or patch_counter
                    == total_patches
                ):
                    print(
                        "已处理："
                        f"{patch_counter}"
                        "/"
                        f"{total_patches}"
                        " patches"
                    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    prediction_full = (
        prediction_sum
        / np.maximum(
            weight_sum,
            1e-8,
        )
    )

    preprocessed_full = (
        preprocessed_sum
        / np.maximum(
            preprocessed_weight_sum,
            1e-8,
        )
    )

    # --------------------------------------------------------
    # 去掉 Padding
    # --------------------------------------------------------

    prediction_full = (
        prediction_full[
            :original_height,
            :original_width,
        ]
    )

    preprocessed_full = (
        preprocessed_full[
            :original_height,
            :original_width,
        ]
    )

    # --------------------------------------------------------
    # [0,1] → uint8
    # --------------------------------------------------------

    prediction_uint8 = (
        np.clip(
            prediction_full,
            0.0,
            1.0,
        )
        * 255.0
    ).astype(
        np.uint8
    )

    preprocessed_uint8 = (
        np.clip(
            preprocessed_full,
            0.0,
            1.0,
        )
        * 255.0
    ).astype(
        np.uint8
    )

    prediction_image = Image.fromarray(
        prediction_uint8,
        mode="L",
    )

    preprocessed_image = Image.fromarray(
        preprocessed_uint8,
        mode="L",
    )

    return (
        original,
        preprocessed_image,
        prediction_image,
    )


# ============================================================
# 7. 保存结果
# ============================================================

def save_results(
    image_path,
    original,
    preprocessed,
    prediction,
):
    image_path = Path(
        image_path
    )

    stem = image_path.stem

    prediction_path = (
        OUTPUT_DIR
        / f"{stem}_cleaned.png"
    )

    preprocessed_path = (
        OUTPUT_DIR
        / f"{stem}_preprocessed.png"
    )

    comparison_path = (
        OUTPUT_DIR
        / f"{stem}_comparison.png"
    )

    # --------------------------------------------------------
    # 保存真正的完整预测结果
    # --------------------------------------------------------

    prediction.save(
        prediction_path
    )

    preprocessed.save(
        preprocessed_path
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    plt.figure(
        figsize=(
            18,
            7,
        )
    )

    plt.subplot(
        1,
        3,
        1,
    )

    plt.imshow(
        original
    )

    plt.title(
        "Original"
    )

    plt.axis(
        "off"
    )

    plt.subplot(
        1,
        3,
        2,
    )

    plt.imshow(
        preprocessed,
        cmap="gray",
        vmin=0,
        vmax=255,
    )

    plt.title(
        "Document Enhanced"
    )

    plt.axis(
        "off"
    )

    plt.subplot(
        1,
        3,
        3,
    )

    plt.imshow(
        prediction,
        cmap="gray",
        vmin=0,
        vmax=255,
    )

    plt.title(
        "Handwriting Removed"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        comparison_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\n===== Files Saved ====="
    )

    print(
        prediction_path
    )

    print(
        preprocessed_path
    )

    print(
        comparison_path
    )


# ============================================================
# 8. Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Full-image handwriting removal "
            "using overlapping U-Net patches."
        )
    )

    parser.add_argument(
        "image",
        type=str,
        help=(
            "Path to the full document image."
        ),
    )

    args = parser.parse_args()

    print(
        "使用设备：",
        device,
    )

    print(
        "Checkpoint：",
        CHECKPOINT,
    )

    print(
        "Input Mode： "
        "RGB Patch → L → "
        "Background Normalization → "
        "Document Enhancement"
    )

    print(
        "Inference Mode： "
        "Overlapping Patch + Weighted Blending"
    )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint 不存在：{CHECKPOINT}"
        )

    image_path = Path(
        args.image
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"输入图片不存在：{image_path}"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model()

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "U-Net Parameters：",
        total_params,
    )

    print(
        "\n===== Full Image Inference Started ====="
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    (
        original,
        preprocessed,
        prediction,
    ) = infer_full_image(
        image_path,
        model,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        image_path,
        original,
        preprocessed,
        prediction,
    )

    print(
        "\n========================================"
    )

    print(
        "Full Image Inference Finished!"
    )

    print(
        "========================================"
    )

    print(
        "输出目录：",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()