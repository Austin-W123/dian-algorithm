from pathlib import Path
import random

from PIL import Image, ImageOps

from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class HandwritingDataset(Dataset):
    """
    文档清理 / 手写擦除任务的 Paired Dataset。

    Input:
        默认 RGB
        Tensor shape = [3, H, W]

        如果 input_grayscale=True：
        先转换为灰度图，再复制成 3 通道。
        Tensor shape 仍然保持 [3, H, W]。

    Target:
        Grayscale / L
        Tensor shape = [1, H, W]

    mode="train":
        每张原图对应一个样本。
        每次访问时 Random Crop。

    mode="val":
        每张原图对应 5 个固定 Patch：
        左上、右上、中心、左下、右下。

    mode="test":
        暂时使用 Center Crop。
        最终测试阶段再实现整图 / tiled inference。

    注意：
        Input / Target 的所有几何操作必须完全同步，
        否则像素级监督关系会被破坏。
    """

    def __init__(
        self,
        data_root,
        split_file,
        patch_size=256,
        mode="train",
        input_grayscale=False,
    ):
        self.data_root = Path(data_root)
        self.split_file = Path(split_file)

        self.patch_size = patch_size
        self.mode = mode

        # ----------------------------------------------------
        # 是否移除 Input 的颜色信息。
        #
        # False:
        #     RGB Input
        #
        # True:
        #     RGB → Grayscale → 复制成 3 通道
        #
        # 这样即使做灰度实验，
        # U-Net 输入 shape 仍然是 [3,H,W]，
        # 不需要修改网络结构。
        # ----------------------------------------------------
        self.input_grayscale = input_grayscale

        if self.mode not in {
            "train",
            "val",
            "test",
        }:
            raise ValueError(
                f"未知 mode：{self.mode}"
            )

        self.pairs = self._load_split()

        # Validation 使用固定的 5 个位置。
        #
        # 这样每一个 epoch 的 Validation
        # 都会看到完全相同的 Patch，
        # 保证指标之间可以直接比较。
        self.val_positions = [
            "top_left",
            "top_right",
            "center",
            "bottom_left",
            "bottom_right",
        ]


    # ========================================================
    # 读取已经冻结的 Train / Val / Test Split
    # ========================================================

    def _load_split(self):

        pairs = []

        with open(
            self.split_file,
            "r",
            encoding="utf-8",
        ) as f:

            for line_number, line in enumerate(
                f,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                # create_splits.py 中保存格式：
                #
                # input_path<TAB>target_path
                parts = line.split("\t")

                if len(parts) != 2:
                    raise ValueError(
                        f"Split 文件格式错误："
                        f"{self.split_file}, "
                        f"第 {line_number} 行\n"
                        f"{line}"
                    )

                input_relative = parts[0]
                target_relative = parts[1]

                input_path = (
                    self.data_root
                    / input_relative
                )

                target_path = (
                    self.data_root
                    / target_relative
                )

                if not input_path.exists():
                    raise FileNotFoundError(
                        f"Input 不存在："
                        f"{input_path}"
                    )

                if not target_path.exists():
                    raise FileNotFoundError(
                        f"Target 不存在："
                        f"{target_path}"
                    )

                pairs.append(
                    (
                        input_path,
                        target_path,
                    )
                )

        return pairs


    # ========================================================
    # Dataset 长度
    # ========================================================

    def __len__(self):

        # Validation：
        #
        # 每张原图生成 5 个固定 Patch。
        #
        # 241 张图片
        # →
        # 241 × 5 = 1205 个 Validation Patch
        if self.mode == "val":

            return (
                len(self.pairs)
                * len(self.val_positions)
            )

        return len(self.pairs)


    # ========================================================
    # 如果图片小于 Patch Size，则先 Padding
    # ========================================================

    def _pad_if_needed(
        self,
        input_image,
        target_image,
    ):

        width, height = input_image.size

        pad_width = max(
            0,
            self.patch_size - width,
        )

        pad_height = max(
            0,
            self.patch_size - height,
        )

        # 图片已经足够大，不需要 Padding。
        if (
            pad_width == 0
            and pad_height == 0
        ):
            return input_image, target_image

        # 将 Padding 尽量平均放在两侧。
        left = pad_width // 2
        right = pad_width - left

        top = pad_height // 2
        bottom = pad_height - top

        padding = (
            left,
            top,
            right,
            bottom,
        )

        # Input 是 RGB，所以白色为 (255,255,255)。
        input_image = ImageOps.expand(
            input_image,
            border=padding,
            fill=(255, 255, 255),
        )

        # Target 是灰度图，所以白色为 255。
        target_image = ImageOps.expand(
            target_image,
            border=padding,
            fill=255,
        )

        return input_image, target_image


    # ========================================================
    # Validation 固定位置 Crop 坐标
    # ========================================================

    def _get_fixed_crop_position(
        self,
        width,
        height,
        position,
    ):

        max_left = (
            width - self.patch_size
        )

        max_top = (
            height - self.patch_size
        )

        if position == "top_left":

            left = 0
            top = 0

        elif position == "top_right":

            left = max_left
            top = 0

        elif position == "center":

            left = max_left // 2
            top = max_top // 2

        elif position == "bottom_left":

            left = 0
            top = max_top

        elif position == "bottom_right":

            left = max_left
            top = max_top

        else:

            raise ValueError(
                f"未知 Validation 位置："
                f"{position}"
            )

        return left, top


    # ========================================================
    # Input / Target 同步 Crop
    # ========================================================

    def _paired_crop(
        self,
        input_image,
        target_image,
        position=None,
    ):

        width, height = input_image.size


        # ----------------------------------------------------
        # Train:
        #
        # 每次访问图片都重新随机选择位置。
        # 所以不同 epoch 可以看到不同 Patch。
        # ----------------------------------------------------

        if self.mode == "train":

            left = random.randint(
                0,
                width - self.patch_size,
            )

            top = random.randint(
                0,
                height - self.patch_size,
            )


        # ----------------------------------------------------
        # Validation:
        #
        # 使用固定位置。
        # 每个 epoch 完全一致。
        # ----------------------------------------------------

        elif self.mode == "val":

            left, top = (
                self._get_fixed_crop_position(
                    width,
                    height,
                    position,
                )
            )


        # ----------------------------------------------------
        # Test:
        #
        # 当前暂时 Center Crop。
        # 最终 Test 会另外实现 tiled inference。
        # ----------------------------------------------------

        else:

            left = (
                width - self.patch_size
            ) // 2

            top = (
                height - self.patch_size
            ) // 2


        box = (
            left,
            top,
            left + self.patch_size,
            top + self.patch_size,
        )


        # 极其重要：
        #
        # Input 和 Target 必须使用完全相同的 box。
        input_image = input_image.crop(
            box
        )

        target_image = target_image.crop(
            box
        )

        return input_image, target_image


    # ========================================================
    # __getitem__
    # ========================================================

    def __getitem__(self, index):

        # ----------------------------------------------------
        # Validation Index Mapping
        #
        # index 0~4：
        #     原图 0 的五个位置
        #
        # index 5~9：
        #     原图 1 的五个位置
        #
        # ...
        # ----------------------------------------------------

        if self.mode == "val":

            num_positions = len(
                self.val_positions
            )

            pair_index = (
                index // num_positions
            )

            position_index = (
                index % num_positions
            )

            position = (
                self.val_positions[
                    position_index
                ]
            )

        else:

            pair_index = index
            position = None


        input_path, target_path = (
            self.pairs[pair_index]
        )


        # ====================================================
        # 读取 Input
        # ====================================================

        with Image.open(
            input_path
        ) as image:

            if self.input_grayscale:

                # --------------------------------------------
                # Grayscale Ablation Experiment
                #
                # RGB
                #  ↓
                # Gray
                #  ↓
                # Gray / Gray / Gray
                #
                # 三个通道数值完全相同。
                #
                # 这样删除了颜色信息，
                # 但仍然保持 [3,H,W]，
                # 所以不需要修改 U-Net。
                # --------------------------------------------

                gray = image.convert("L")

                input_image = Image.merge(
                    "RGB",
                    (
                        gray,
                        gray,
                        gray,
                    ),
                )

            else:

                # --------------------------------------------
                # RGB Baseline
                # --------------------------------------------

                input_image = (
                    image.convert("RGB")
                )


        # ====================================================
        # 读取 Target
        # ====================================================

        with Image.open(
            target_path
        ) as image:

            target_image = (
                image.convert("L")
            )


        # ====================================================
        # Pair 尺寸检查
        # ====================================================

        if (
            input_image.size
            != target_image.size
        ):
            raise ValueError(
                "Input / Target 尺寸不一致：\n"
                f"Input: {input_path} "
                f"{input_image.size}\n"
                f"Target: {target_path} "
                f"{target_image.size}"
            )


        # ====================================================
        # Padding
        # ====================================================

        input_image, target_image = (
            self._pad_if_needed(
                input_image,
                target_image,
            )
        )


        # ====================================================
        # Synchronized Crop
        # ====================================================

        input_image, target_image = (
            self._paired_crop(
                input_image,
                target_image,
                position=position,
            )
        )


        # ====================================================
        # PIL → Tensor
        #
        # uint8:
        # [0,255]
        #
        # ↓
        #
        # float32:
        # [0,1]
        # ====================================================

        input_tensor = TF.to_tensor(
            input_image
        )

        target_tensor = TF.to_tensor(
            target_image
        )


        return (
            input_tensor,
            target_tensor,
        )