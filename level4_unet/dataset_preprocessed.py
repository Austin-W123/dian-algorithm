from pathlib import Path
import random

from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from PIL import Image

from preprocessing import preprocess_document


class PreprocessedHandwritingDataset(Dataset):
    """
    使用文档预处理后的手写去除 Dataset。

    Input:
        原始 RGB / RGBA 图片
            ↓
        同步 Padding / Crop
            ↓
        preprocess_document()
            ↓
        Grayscale
            ↓
        Background Normalization
            ↓
        Document Enhancement
            ↓
        [1, H, W] Tensor

    Target:
        原始 Output
            ↓
        Grayscale L
            ↓
        同步 Padding / Crop
            ↓
        [1, H, W] Tensor

    注意：
        preprocess_document() 只作用于 Input。
        Target 是监督信号，绝对不做 Document Enhancement。
    """

    def __init__(
        self,
        data_root,
        split_file,
        patch_size=256,
        mode="train",
        patches_per_image=5,
        blur_radius=15,
    ):
        self.data_root = Path(data_root)
        self.split_file = Path(split_file)

        self.patch_size = patch_size
        self.mode = mode

        self.patches_per_image = patches_per_image
        self.blur_radius = blur_radius

        self.pairs = []

        # ----------------------------------------------------
        # 读取已经冻结的 split
        #
        # 每一行：
        #
        # input/path.jpg  output/path.jpg
        #
        # 后续训练、验证、测试都只读取已有 split，
        # 不在 Dataset 内重新划分数据。
        # ----------------------------------------------------

        with open(
            self.split_file,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid split line: {line}"
                    )

                input_rel, target_rel = parts

                input_path = (
                    self.data_root
                    / input_rel
                )

                target_path = (
                    self.data_root
                    / target_rel
                )

                self.pairs.append(
                    (
                        input_path,
                        target_path,
                    )
                )

        if self.mode not in {
            "train",
            "val",
            "test",
        }:
            raise ValueError(
                f"Unknown mode: {self.mode}"
            )

    def __len__(self):
        # ----------------------------------------------------
        # Train：
        #
        # 每张原图每个 epoch 随机产生一个 patch。
        #
        # 所以：
        #
        # len(train_dataset) = 原图 Pair 数量
        #
        #
        # Validation / Test：
        #
        # 每张原图固定取 patches_per_image 个 patch。
        #
        # 因此：
        #
        # len(val_dataset)
        # =
        # Pair 数量 × patches_per_image
        #
        # 这样不同 epoch 的验证位置保持一致。
        # ----------------------------------------------------

        if self.mode == "train":
            return len(self.pairs)

        return (
            len(self.pairs)
            * self.patches_per_image
        )

    # ========================================================
    # Padding
    #
    # 数据集中极少数图片可能比 patch_size 小。
    #
    # Input 和 Target 必须完全同步 Padding，
    # 否则像素监督关系会被破坏。
    # ========================================================

    def _pad_if_needed(
        self,
        input_image,
        target_image,
    ):
        width, height = input_image.size

        pad_right = max(
            0,
            self.patch_size - width,
        )

        pad_bottom = max(
            0,
            self.patch_size - height,
        )

        if (
            pad_right > 0
            or pad_bottom > 0
        ):
            padding = [
                0,
                0,
                pad_right,
                pad_bottom,
            ]

            # Input 使用白色 Padding。
            input_image = TF.pad(
                input_image,
                padding,
                fill=255,
            )

            # Target 也必须使用完全相同的位置和大小 Padding。
            target_image = TF.pad(
                target_image,
                padding,
                fill=255,
            )

        return (
            input_image,
            target_image,
        )

    # ========================================================
    # Random Crop
    #
    # 仅用于 Training。
    #
    # 每次 __getitem__ 都重新随机选择一个 256×256 区域，
    # 因此不同 epoch 可以看到同一原图的不同位置。
    #
    # Input / Target 必须使用完全相同的 Crop 坐标。
    # ========================================================

    def _random_crop(
        self,
        input_image,
        target_image,
    ):
        width, height = input_image.size

        max_left = (
            width
            - self.patch_size
        )

        max_top = (
            height
            - self.patch_size
        )

        left = random.randint(
            0,
            max_left,
        )

        top = random.randint(
            0,
            max_top,
        )

        box = (
            left,
            top,
            left + self.patch_size,
            top + self.patch_size,
        )

        input_patch = input_image.crop(
            box
        )

        target_patch = target_image.crop(
            box
        )

        return (
            input_patch,
            target_patch,
        )

    # ========================================================
    # Fixed Validation / Test Crop
    #
    # Validation 和 Test 不能每个 epoch 随机换位置，
    # 否则不同 epoch 的 Loss / PSNR / SSIM
    # 实际上是在不同图片区域上计算的，不便比较。
    #
    # 所以根据：
    #
    # image_index
    # patch_index
    #
    # 构造固定随机种子。
    #
    # 同一个样本每次都会得到完全相同的 Crop。
    # ========================================================

    def _fixed_crop(
        self,
        input_image,
        target_image,
        image_index,
        patch_index,
    ):
        width, height = input_image.size

        max_left = (
            width
            - self.patch_size
        )

        max_top = (
            height
            - self.patch_size
        )

        # 使用独立 Random，
        # 不影响 Python 全局 random 状态。
        rng = random.Random(
            image_index * 1000
            + patch_index
            + 42
        )

        left = rng.randint(
            0,
            max_left,
        )

        top = rng.randint(
            0,
            max_top,
        )

        box = (
            left,
            top,
            left + self.patch_size,
            top + self.patch_size,
        )

        input_patch = input_image.crop(
            box
        )

        target_patch = target_image.crop(
            box
        )

        return (
            input_patch,
            target_patch,
        )

    # ========================================================
    # __getitem__
    # ========================================================

    def __getitem__(
        self,
        index,
    ):
        # ----------------------------------------------------
        # 1. 确定当前是哪一张原图
        # ----------------------------------------------------

        if self.mode == "train":
            # Train:
            # 一个 index 对应一张原图。
            image_index = index
            patch_index = None

        else:
            # Validation / Test:
            #
            # 一张原图对应 patches_per_image 个固定 Patch。
            image_index = (
                index
                // self.patches_per_image
            )

            patch_index = (
                index
                % self.patches_per_image
            )

        input_path, target_path = (
            self.pairs[image_index]
        )

        # ----------------------------------------------------
        # 2. 读取 Input / Target
        #
        # Input：
        # 暂时保留 RGB。
        #
        # preprocess_document() 后面会负责：
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
        # Target：
        # 数据集 Output 本身就是灰度目标，
        # 统一 convert("L")。
        # ----------------------------------------------------

        input_image = Image.open(
            input_path
        ).convert("RGB")

        target_image = Image.open(
            target_path
        ).convert("L")

        # ----------------------------------------------------
        # 3. 防止极小图片无法 Crop
        # ----------------------------------------------------

        (
            input_image,
            target_image,
        ) = self._pad_if_needed(
            input_image,
            target_image,
        )

        # ----------------------------------------------------
        # 4. Input / Target 同步 Crop
        #
        # 注意：
        #
        # 必须先完成同步 Crop，
        # 再单独处理 Input。
        #
        # 这样 Input Patch 与 Target Patch
        # 始终代表原图完全相同的空间位置。
        # ----------------------------------------------------

        if self.mode == "train":
            (
                input_patch,
                target_patch,
            ) = self._random_crop(
                input_image,
                target_image,
            )

        else:
            (
                input_patch,
                target_patch,
            ) = self._fixed_crop(
                input_image,
                target_image,
                image_index,
                patch_index,
            )

        # ----------------------------------------------------
        # 5. Document Preprocessing
        #
        # 这里只处理 Input！
        #
        # RGB
        # ↓
        # Grayscale (L)
        # ↓
        # Background Normalization
        # ↓
        # Document Enhancement
        # ↓
        # L
        #
        #
        # Target 不做这个 preprocessing。
        #
        # Target 是监督信号：
        # 模型最终应该学习逼近 Target，
        # 而不是修改 Target 去适应 Input。
        # ----------------------------------------------------

        input_patch = preprocess_document(
            input_patch,
            blur_radius=self.blur_radius,
        )

        # ----------------------------------------------------
        # 6. PIL → Tensor
        #
        # Input:
        # [1, H, W]
        #
        # Target:
        # [1, H, W]
        #
        # Pixel Range:
        # [0, 1]
        # ----------------------------------------------------

        input_tensor = TF.to_tensor(
            input_patch
        )

        target_tensor = TF.to_tensor(
            target_patch
        )

        return (
            input_tensor,
            target_tensor,
        )