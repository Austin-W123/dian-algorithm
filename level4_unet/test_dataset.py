from pathlib import Path

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset import HandwritingDataset


# ============================================================
# 1. 配置
# ============================================================

DATA_ROOT = (
    "/mnt/d/User/University/QQ/Dian/deli"
)

PATCH_SIZE = 256
BATCH_SIZE = 8

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. 从已经冻结的 Split 创建 Dataset
# ============================================================

train_dataset = HandwritingDataset(
    data_root=DATA_ROOT,
    split_file="splits/train.txt",
    patch_size=PATCH_SIZE,
    training=True,
)

val_dataset = HandwritingDataset(
    data_root=DATA_ROOT,
    split_file="splits/val.txt",
    patch_size=PATCH_SIZE,
    training=False,
)

test_dataset = HandwritingDataset(
    data_root=DATA_ROOT,
    split_file="splits/test.txt",
    patch_size=PATCH_SIZE,
    training=False,
)


print("=" * 60)
print("Dataset Information")
print("=" * 60)

print("Train：", len(train_dataset))
print("Validation：", len(val_dataset))
print("Test：", len(test_dataset))


# 再次检查冻结后的数量
assert len(train_dataset) == 1929
assert len(val_dataset) == 241
assert len(test_dataset) == 242


# ============================================================
# 3. DataLoader
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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


print("\n===== DataLoader =====")

print(
    "Train batches：",
    len(train_loader),
)

print(
    "Validation batches：",
    len(val_loader),
)

print(
    "Test batches：",
    len(test_loader),
)


# ============================================================
# 4. 获取 Training Batch
# ============================================================

input_batch, target_batch = next(
    iter(train_loader)
)


print("\n===== First Training Batch =====")

print(
    "Input shape：",
    input_batch.shape,
)

print(
    "Target shape：",
    target_batch.shape,
)

print(
    "Input dtype：",
    input_batch.dtype,
)

print(
    "Target dtype：",
    target_batch.dtype,
)

print(
    "Input range：",
    input_batch.min().item(),
    "~",
    input_batch.max().item(),
)

print(
    "Target range：",
    target_batch.min().item(),
    "~",
    target_batch.max().item(),
)


# ============================================================
# 5. Tensor Sanity Check
# ============================================================

assert input_batch.shape == (
    BATCH_SIZE,
    3,
    PATCH_SIZE,
    PATCH_SIZE,
)

assert target_batch.shape == (
    BATCH_SIZE,
    1,
    PATCH_SIZE,
    PATCH_SIZE,
)

assert input_batch.min().item() >= 0.0
assert input_batch.max().item() <= 1.0

assert target_batch.min().item() >= 0.0
assert target_batch.max().item() <= 1.0


print("\nTraining Tensor 检查通过！")


# ============================================================
# 6. 再检查 Validation
# ============================================================

val_input, val_target = next(
    iter(val_loader)
)

assert val_input.shape[1:] == (
    3,
    PATCH_SIZE,
    PATCH_SIZE,
)

assert val_target.shape[1:] == (
    1,
    PATCH_SIZE,
    PATCH_SIZE,
)


print("Validation Tensor 检查通过！")


# ============================================================
# 7. 再检查 Test
#
# 注意：
#这里只检查数据 Pipeline 能否正常工作，
# 不计算模型 Test 性能。
# ============================================================

test_input, test_target = next(
    iter(test_loader)
)

assert test_input.shape[1:] == (
    3,
    PATCH_SIZE,
    PATCH_SIZE,
)

assert test_target.shape[1:] == (
    1,
    PATCH_SIZE,
    PATCH_SIZE,
)


print("Test Tensor 检查通过！")


# ============================================================
# 8. 保存 Training Pair 可视化
# ============================================================

NUM_EXAMPLES = min(
    6,
    input_batch.size(0),
)

fig, axes = plt.subplots(
    NUM_EXAMPLES,
    2,
    figsize=(8, NUM_EXAMPLES * 4),
)


for i in range(NUM_EXAMPLES):

    # Input:
    # [3,H,W]
    # →
    # [H,W,3]
    input_image = (
        input_batch[i]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    # Target:
    # [1,H,W]
    # →
    # [H,W]
    target_image = (
        target_batch[i]
        .squeeze(0)
        .cpu()
        .numpy()
    )

    axes[i, 0].imshow(
        input_image
    )

    axes[i, 0].set_title(
        f"Input Patch {i + 1}"
    )

    axes[i, 0].axis("off")

    axes[i, 1].imshow(
        target_image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[i, 1].set_title(
        f"Target Patch {i + 1}"
    )

    axes[i, 1].axis("off")


plt.tight_layout()

train_visual_path = (
    OUTPUT_DIR
    / "train_pair_examples.png"
)

plt.savefig(
    train_visual_path,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


print(
    "\nTraining Pair 可视化：",
    train_visual_path,
)


# ============================================================
# 9. 保存 Validation Pair 可视化
# ============================================================

NUM_VAL_EXAMPLES = min(
    4,
    val_input.size(0),
)

fig, axes = plt.subplots(
    NUM_VAL_EXAMPLES,
    2,
    figsize=(8, NUM_VAL_EXAMPLES * 4),
)


for i in range(NUM_VAL_EXAMPLES):

    input_image = (
        val_input[i]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    target_image = (
        val_target[i]
        .squeeze(0)
        .cpu()
        .numpy()
    )

    axes[i, 0].imshow(
        input_image
    )

    axes[i, 0].set_title(
        f"Validation Input {i + 1}"
    )

    axes[i, 0].axis("off")

    axes[i, 1].imshow(
        target_image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[i, 1].set_title(
        f"Validation Target {i + 1}"
    )

    axes[i, 1].axis("off")


plt.tight_layout()

val_visual_path = (
    OUTPUT_DIR
    / "val_pair_examples.png"
)

plt.savefig(
    val_visual_path,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


print(
    "Validation Pair 可视化：",
    val_visual_path,
)


print("\nDataset / DataLoader 测试全部完成！")