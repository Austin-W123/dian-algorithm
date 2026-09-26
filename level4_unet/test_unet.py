import torch

from unet import UNet


# ============================================================
# 1. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("使用设备：", device)


# ============================================================
# 2. 创建 U-Net
# ============================================================

model = UNet(
    in_channels=3,
    out_channels=1,
).to(device)


# ============================================================
# 3. 参数量
# ============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print("\n===== U-Net Parameters =====")

print(
    "Total Parameters：",
    total_parameters,
)

print(
    "Trainable Parameters：",
    trainable_parameters,
)


# ============================================================
# 4. 模拟一个 Batch
#
# 与真实 Dataset Shape 一致：
#
# [B,3,256,256]
# ============================================================

x = torch.randn(
    2,
    3,
    256,
    256,
    device=device,
)


print("\n===== Shape Test =====")

print(
    "Input shape：",
    x.shape,
)


# ============================================================
# 5. Forward
# ============================================================

with torch.no_grad():

    output = model(x)


print(
    "Output shape：",
    output.shape,
)


print(
    "Output range：",
    output.min().item(),
    "~",
    output.max().item(),
)


# ============================================================
# 6. Sanity Check
# ============================================================

assert output.shape == (
    2,
    1,
    256,
    256,
)

assert output.min().item() >= 0.0
assert output.max().item() <= 1.0


print("\nU-Net Forward 测试通过！")