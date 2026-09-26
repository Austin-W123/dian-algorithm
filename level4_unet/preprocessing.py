from PIL import Image, ImageOps, ImageFilter
import numpy as np


def to_grayscale(image):
    """
    RGB / RGBA / L
        ↓
    单通道灰度图 L

    这里只负责去掉颜色信息，不负责背景归一化。
    """
    if image.mode == "RGBA":
        # RGBA 图片先铺到白色背景上，避免透明区域变黑。
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")

    elif image.mode != "RGB":
        image = image.convert("RGB")

    return ImageOps.grayscale(image)


def background_normalization(gray_image, blur_radius=15):
    """
    背景归一化：

    原始灰度图中可能存在：
    - 光照不均
    - 阴影
    - 灰色纸张
    - 桌面 / 环境背景
    - 手机拍摄造成的亮度变化

    我们用较大范围 Gaussian Blur 估计低频背景：

        background = Blur(gray)

    然后通过：

        normalized = gray / background

    消除低频背景变化。

    注意：
    这一步的目的不是删除手写，
    而是尽量把背景变得接近白色。
    """

    gray = np.asarray(gray_image, dtype=np.float32)

    # 大尺度模糊，用来估计背景亮度。
    background_image = gray_image.filter(
        ImageFilter.GaussianBlur(radius=blur_radius)
    )
    background = np.asarray(background_image, dtype=np.float32)

    # 防止除 0。
    background = np.maximum(background, 1.0)

    # 如果一个区域只是因为光照变暗：
    #
    #     gray ≈ background
    #
    # 那么：
    #
    #     gray / background ≈ 1
    #
    # 最终就会被拉回白色附近。
    normalized = gray / background * 255.0

    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    return Image.fromarray(normalized, mode="L")


def document_enhancement(
    normalized_image,
    autocontrast_cutoff=0.5,
    threshold=210,
    strength=0.70,
):
    """
    Document Enhancement

    BackgroundNorm 后通常会出现：

        白背景 + 偏灰的文字/线条

    但 Target 更接近：

        白背景 + 深色打印内容

    因此这里继续增强文档前景。

    --------------------------------------------------

    Step 1:
        AutoContrast

    自动重新拉伸灰度范围。

    Step 2:
        对较暗像素进一步增强。

    threshold:
        小于 threshold 的区域被认为是潜在前景。

    strength:
        控制增强程度。

    重要：
        这里不是在判断“打印 / 手写”。

        打印文字和手写文字都属于前景，
        因此理论上都会被增强。

        真正判断“应该保留还是删除”的任务，
        仍然交给后面的 U-Net。
    """

    # 先做一次温和的自动对比度拉伸。
    enhanced = ImageOps.autocontrast(
        normalized_image,
        cutoff=autocontrast_cutoff,
    )

    arr = np.asarray(enhanced, dtype=np.float32)

    # 找出较暗的前景区域。
    foreground_mask = arr < threshold

    # 非线性增强：
    #
    # 越暗的像素，增强越明显；
    # 接近白色的背景基本保持不动。
    #
    # x in [0, 1]
    x = arr / 255.0

    # gamma > 1 会让中间灰度进一步变暗。
    gamma = 1.0 + strength

    darkened = np.power(x, gamma) * 255.0

    # 只对潜在前景做增强。
    result = arr.copy()
    result[foreground_mask] = darkened[foreground_mask]

    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result, mode="L")


def preprocess_document(
    image,
    blur_radius=15,
    autocontrast_cutoff=0.5,
    threshold=210,
    strength=0.70,
):
    """
    完整文档预处理流程：

        RGB / RGBA
            ↓
        Grayscale (L)
            ↓
        Background Normalization
            ↓
        Document Enhancement
            ↓
        L

    最终输出仍然是 PIL L 图像。
    """

    gray = to_grayscale(image)

    normalized = background_normalization(
        gray,
        blur_radius=blur_radius,
    )

    enhanced = document_enhancement(
        normalized,
        autocontrast_cutoff=autocontrast_cutoff,
        threshold=threshold,
        strength=strength,
    )

    return enhanced