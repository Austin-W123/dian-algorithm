from pathlib import Path
import random


DATA_ROOT = Path(
    "/mnt/d/User/University/QQ/Dian/deli"
)

SPLIT_DIR = Path("splits")

DATES = [
    "20250211",
    "20250212",
    "20250213",
]

SEED = 42

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


# ============================================================
# 1. 收集所有 Pair
# ============================================================

pairs = []

for date in DATES:

    input_dir = (
        DATA_ROOT
        / date
        / "dataset"
        / "input"
    )

    output_dir = (
        DATA_ROOT
        / date
        / "dataset"
        / "output"
    )

    for input_path in sorted(
        input_dir.glob("*.jpg")
    ):

        output_path = (
            output_dir
            / input_path.name
        )

        if not output_path.exists():

            raise FileNotFoundError(
                f"Missing target: {output_path}"
            )

        # 保存相对于 DATA_ROOT 的路径，
        # 避免 split 文件绑定绝对路径。
        input_relative = (
            input_path.relative_to(DATA_ROOT)
        )

        output_relative = (
            output_path.relative_to(DATA_ROOT)
        )

        pairs.append(
            (
                str(input_relative),
                str(output_relative),
            )
        )


print("总 Pair 数量：", len(pairs))

assert len(pairs) == 2412


# ============================================================
# 2. 固定随机种子
# ============================================================

rng = random.Random(SEED)

rng.shuffle(pairs)


# ============================================================
# 3. Train / Validation / Test Split
# ============================================================

total = len(pairs)

train_size = int(
    total * TRAIN_RATIO
)

val_size = int(
    total * VAL_RATIO
)

train_pairs = pairs[:train_size]

val_pairs = pairs[
    train_size:
    train_size + val_size
]

test_pairs = pairs[
    train_size + val_size:
]


print("\n===== Split =====")

print("Train：", len(train_pairs))
print("Validation：", len(val_pairs))
print("Test：", len(test_pairs))

assert (
    len(train_pairs)
    + len(val_pairs)
    + len(test_pairs)
    == total
)


# ============================================================
# 4. 检查三个集合绝对没有文件重叠
# ============================================================

train_set = set(train_pairs)
val_set = set(val_pairs)
test_set = set(test_pairs)

assert train_set.isdisjoint(val_set)
assert train_set.isdisjoint(test_set)
assert val_set.isdisjoint(test_set)

print("\nTrain / Val / Test 无重复 Pair。")


# ============================================================
# 5. 写入 split 文件
# ============================================================

SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def save_split(name, split_pairs):

    path = SPLIT_DIR / f"{name}.txt"

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for input_path, output_path in split_pairs:

            f.write(
                f"{input_path}\t"
                f"{output_path}\n"
            )

    print(
        f"{name}: "
        f"{len(split_pairs)} pairs "
        f"→ {path}"
    )


save_split(
    "train",
    train_pairs,
)

save_split(
    "val",
    val_pairs,
)

save_split(
    "test",
    test_pairs,
)


print("\n数据集划分已经冻结。")
print("后续训练不要重新生成 Split。")