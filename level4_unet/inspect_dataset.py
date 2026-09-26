from pathlib import Path
from collections import Counter
from statistics import median

from PIL import Image, UnidentifiedImageError


# ============================================================
# 1. 数据集路径
# ============================================================

DATA_ROOT = Path("/mnt/d/User/University/QQ/Dian/deli")

DATE_FOLDERS = [
    "20250211",
    "20250212",
    "20250213",
]


# ============================================================
# 2. 用来保存整个数据集的统计信息
# ============================================================

total_pairs = 0

all_widths = []
all_heights = []
all_areas = []

input_modes = Counter()
output_modes = Counter()

input_sizes = Counter()
output_sizes = Counter()

landscape_count = 0
portrait_count = 0
square_count = 0

missing_outputs = []
missing_inputs = []

broken_inputs = []
broken_outputs = []

size_mismatches = []


# ============================================================
# 3. 遍历三个日期的数据
# ============================================================

print("=" * 70)
print("Deli Dataset Audit")
print("=" * 70)

for date in DATE_FOLDERS:

    dataset_dir = DATA_ROOT / date / "dataset"

    input_dir = dataset_dir / "input"
    output_dir = dataset_dir / "output"

    # 找出当前目录中的所有 jpg 文件
    input_files = {
        path.name: path
        for path in input_dir.glob("*.jpg")
    }

    output_files = {
        path.name: path
        for path in output_dir.glob("*.jpg")
    }

    input_names = set(input_files.keys())
    output_names = set(output_files.keys())

    # --------------------------------------------------------
    # 检查文件名是否一一对应
    # --------------------------------------------------------

    only_input = sorted(input_names - output_names)
    only_output = sorted(output_names - input_names)

    for name in only_input:
        missing_outputs.append((date, name))

    for name in only_output:
        missing_inputs.append((date, name))

    paired_names = sorted(input_names & output_names)

    print(f"\n[{date}]")
    print(f"Input 数量：       {len(input_files)}")
    print(f"Output 数量：      {len(output_files)}")
    print(f"成功配对数量：     {len(paired_names)}")
    print(f"缺少 Output：      {len(only_input)}")
    print(f"缺少 Input：       {len(only_output)}")

    # --------------------------------------------------------
    # 检查每一对图片
    # --------------------------------------------------------

    for name in paired_names:

        input_path = input_files[name]
        output_path = output_files[name]

        # ------------------------
        # 读取 Input
        # ------------------------

        try:
            with Image.open(input_path) as img:
                img.load()

                input_size = img.size
                input_mode = img.mode

        except (UnidentifiedImageError, OSError, ValueError) as e:

            broken_inputs.append(
                (date, name, str(e))
            )

            continue

        # ------------------------
        # 读取 Output
        # ------------------------

        try:
            with Image.open(output_path) as img:
                img.load()

                output_size = img.size
                output_mode = img.mode

        except (UnidentifiedImageError, OSError, ValueError) as e:

            broken_outputs.append(
                (date, name, str(e))
            )

            continue

        # ------------------------
        # 当前图片通过基本读取检查
        # ------------------------

        total_pairs += 1

        input_modes[input_mode] += 1
        output_modes[output_mode] += 1

        input_sizes[input_size] += 1
        output_sizes[output_size] += 1

        width, height = input_size

        all_widths.append(width)
        all_heights.append(height)
        all_areas.append(width * height)

        # ------------------------
        # 横图 / 竖图 / 方图
        # ------------------------

        if width > height:
            landscape_count += 1

        elif height > width:
            portrait_count += 1

        else:
            square_count += 1

        # ------------------------
        # Input / Output 尺寸检查
        # ------------------------

        if input_size != output_size:

            size_mismatches.append(
                (
                    date,
                    name,
                    input_size,
                    output_size,
                )
            )


# ============================================================
# 4. 输出总体统计
# ============================================================

print("\n")
print("=" * 70)
print("总体统计")
print("=" * 70)

print(f"成功读取并检查的图片对： {total_pairs}")

print(f"\n缺少 Output： {len(missing_outputs)}")
print(f"缺少 Input：  {len(missing_inputs)}")

print(f"\n损坏 Input：  {len(broken_inputs)}")
print(f"损坏 Output： {len(broken_outputs)}")

print(
    f"\nInput / Output 尺寸不一致： "
    f"{len(size_mismatches)}"
)


# ============================================================
# 5. 图片颜色模式
# ============================================================

print("\n")
print("=" * 70)
print("图片颜色模式")
print("=" * 70)

print("\nInput：")

for mode, count in input_modes.most_common():
    print(f"{mode}: {count}")

print("\nOutput：")

for mode, count in output_modes.most_common():
    print(f"{mode}: {count}")


# ============================================================
# 6. 图片方向
# ============================================================

print("\n")
print("=" * 70)
print("图片方向")
print("=" * 70)

print(f"横图 Landscape： {landscape_count}")
print(f"竖图 Portrait：  {portrait_count}")
print(f"方图 Square：    {square_count}")


# ============================================================
# 7. 图片尺寸统计
# ============================================================

if all_widths and all_heights:

    print("\n")
    print("=" * 70)
    print("Input 图片尺寸统计")
    print("=" * 70)

    print(f"最小 Width：  {min(all_widths)}")
    print(f"最大 Width：  {max(all_widths)}")
    print(f"中位 Width：  {median(all_widths)}")

    print()

    print(f"最小 Height： {min(all_heights)}")
    print(f"最大 Height： {max(all_heights)}")
    print(f"中位 Height： {median(all_heights)}")

    print()

    smallest_area_index = all_areas.index(min(all_areas))
    largest_area_index = all_areas.index(max(all_areas))

    print(
        "最小像素面积： "
        f"{all_widths[smallest_area_index]}"
        " × "
        f"{all_heights[smallest_area_index]}"
        " = "
        f"{min(all_areas):,}"
    )

    print(
        "最大像素面积： "
        f"{all_widths[largest_area_index]}"
        " × "
        f"{all_heights[largest_area_index]}"
        " = "
        f"{max(all_areas):,}"
    )


# ============================================================
# 8. 最常见尺寸
# ============================================================

print("\n")
print("=" * 70)
print("最常见的 20 个 Input 尺寸")
print("=" * 70)

for (width, height), count in input_sizes.most_common(20):

    print(
        f"{width:5d} × {height:5d}"
        f"    {count:4d} 张"
    )


# ============================================================
# 9. 尺寸不一致样本
# ============================================================

if size_mismatches:

    print("\n")
    print("=" * 70)
    print("Input / Output 尺寸不一致样本（最多显示 20 个）")
    print("=" * 70)

    for item in size_mismatches[:20]:

        date, name, input_size, output_size = item

        print(
            f"{date} / {name}"
            f" | input={input_size}"
            f" | output={output_size}"
        )


# ============================================================
# 10. 缺失配对样本
# ============================================================

if missing_outputs:

    print("\n缺少 Output 的样本（最多显示 20 个）：")

    for item in missing_outputs[:20]:
        print(item)


if missing_inputs:

    print("\n缺少 Input 的样本（最多显示 20 个）：")

    for item in missing_inputs[:20]:
        print(item)


# ============================================================
# 11. 损坏图片
# ============================================================

if broken_inputs:

    print("\n损坏 Input（最多显示 20 个）：")

    for item in broken_inputs[:20]:
        print(item)


if broken_outputs:

    print("\n损坏 Output（最多显示 20 个）：")

    for item in broken_outputs[:20]:
        print(item)


print("\n")
print("=" * 70)
print("Dataset Audit Finished")
print("=" * 70)