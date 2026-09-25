import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ============================================================
# 1. ResNet BasicBlock
# ============================================================
#
# ResNet 的核心：
#
#       x --------------------+
#       |                     |
#       ↓                     |
#     Conv                    |
#       ↓                     |
#      BN                     |
#       ↓                     |
#     ReLU                    |
#       ↓                     |
#     Conv                    |
#       ↓                     |
#      BN                     |
#       ↓                     |
#      F(x)                   |
#       |                     |
#       +------ Add <---------+
#               ↓
#             ReLU
#
# 最终：
#
# y = F(x) + x
#
# 如果 F(x) 和 x 的 shape 不一致，
# 使用 1×1 Conv 对 shortcut 进行变换。
# ============================================================


class BasicBlock(nn.Module):

    # BasicBlock 的 expansion = 1
    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):
        super().__init__()

        # ----------------------------------------------------
        # 主分支 F(x)
        # ----------------------------------------------------

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )


        # ----------------------------------------------------
        # Shortcut 分支
        # ----------------------------------------------------
        #
        # 如果：
        #
        # 1. stride != 1
        # 或
        # 2. in_channels != out_channels
        #
        # 那么 x 与 F(x) 的 shape 不一致，
        # 无法直接相加。
        #
        # 使用 1×1 Conv 调整：
        #
        # H / W
        # Channel
        #
        # 使 shortcut 和主分支 shape 相同。
        # ----------------------------------------------------

        if (
            stride != 1
            or in_channels != out_channels
        ):

            self.shortcut = nn.Sequential(

                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm2d(
                    out_channels
                )
            )

        else:

            # Shape 本身相同，
            # shortcut 不需要任何处理
            self.shortcut = nn.Identity()


    def forward(self, x):

        # 保存原始输入
        identity = self.shortcut(x)

        # -------------------------
        # 主分支 F(x)
        # -------------------------

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)


        # -------------------------
        # Residual Connection
        # -------------------------
        #
        # y = F(x) + x
        #
        # 如果 shape 不同，
        # identity 已经经过 1×1 Conv
        # 调整到了相同 shape。
        # -------------------------

        out = out + identity

        out = self.relu(out)

        return out


# ============================================================
# 2. ResNet-18-style for MNIST
# ============================================================
#
# 原始 ResNet-18：
#
# BasicBlock 数量：
#
# [2, 2, 2, 2]
#
# 这里保留这一核心结构，
# 但将输入部分适配为 28×28 灰度 MNIST。
# ============================================================


class ResNetMNIST(nn.Module):

    def __init__(self):
        super().__init__()

        # 当前 feature map 的 Channel 数
        self.in_channels = 32


        # ----------------------------------------------------
        # Stem
        # ----------------------------------------------------
        #
        # 输入：
        #
        # [B, 1, 28, 28]
        #
        # 原始 ResNet 为大尺寸 RGB 图像设计，
        # 通常使用更大的初始卷积和下采样。
        #
        # MNIST 只有 28×28，
        # 因此使用 3×3 Conv、stride=1，
        # 避免过早丢失空间信息。
        # ----------------------------------------------------

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            32
        )

        self.relu = nn.ReLU(
            inplace=True
        )


        # ----------------------------------------------------
        # Residual Stage 1
        # ----------------------------------------------------
        #
        # [B, 32, 28, 28]
        #
        # 两个 BasicBlock
        # 不进行下采样
        # ----------------------------------------------------

        self.layer1 = self._make_layer(
            out_channels=32,
            num_blocks=2,
            stride=1
        )


        # ----------------------------------------------------
        # Residual Stage 2
        # ----------------------------------------------------
        #
        # 第一个 Block stride=2
        #
        # [B, 32, 28, 28]
        #
        # →
        #
        # [B, 64, 14, 14]
        # ----------------------------------------------------

        self.layer2 = self._make_layer(
            out_channels=64,
            num_blocks=2,
            stride=2
        )


        # ----------------------------------------------------
        # Residual Stage 3
        # ----------------------------------------------------
        #
        # [B, 64, 14, 14]
        #
        # →
        #
        # [B, 128, 7, 7]
        # ----------------------------------------------------

        self.layer3 = self._make_layer(
            out_channels=128,
            num_blocks=2,
            stride=2
        )


        # ----------------------------------------------------
        # Residual Stage 4
        # ----------------------------------------------------
        #
        # [B, 128, 7, 7]
        #
        # →
        #
        # [B, 256, 4, 4]
        # ----------------------------------------------------

        self.layer4 = self._make_layer(
            out_channels=256,
            num_blocks=2,
            stride=2
        )


        # ----------------------------------------------------
        # Global Average Pooling
        # ----------------------------------------------------
        #
        # 不管输入 feature map 的 H、W 是多少，
        # 最终都压缩成：
        #
        # [B, 256, 1, 1]
        # ----------------------------------------------------

        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )


        # ----------------------------------------------------
        # 分类器
        # ----------------------------------------------------

        self.fc = nn.Linear(
            256,
            10
        )


    # ========================================================
    # 创建一个 Residual Stage
    # ========================================================

    def _make_layer(
        self,
        out_channels,
        num_blocks,
        stride
    ):

        layers = []


        # ----------------------------------------------------
        # 第一个 BasicBlock
        #
        # 可能负责：
        #
        # 1. 下采样
        # 2. Channel 改变
        #
        # 所以使用传入的 stride。
        # ----------------------------------------------------

        layers.append(

            BasicBlock(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride
            )
        )

        # 更新当前 Channel 数
        self.in_channels = out_channels


        # ----------------------------------------------------
        # 后续 BasicBlock
        #
        # H/W 和 Channel 不再改变，
        # 所以 stride=1。
        # ----------------------------------------------------

        for _ in range(
            1,
            num_blocks
        ):

            layers.append(

                BasicBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=1
                )
            )


        return nn.Sequential(
            *layers
        )


    def forward(self, x):

        # ----------------------------------------------------
        # Stem
        # ----------------------------------------------------

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        # [B, 32, 28, 28]


        # ----------------------------------------------------
        # Residual Stages
        # ----------------------------------------------------

        x = self.layer1(x)

        # [B, 32, 28, 28]


        x = self.layer2(x)

        # [B, 64, 14, 14]


        x = self.layer3(x)

        # [B, 128, 7, 7]


        x = self.layer4(x)

        # [B, 256, 4, 4]


        # ----------------------------------------------------
        # Global Average Pooling
        # ----------------------------------------------------

        x = self.avgpool(x)

        # [B, 256, 1, 1]


        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

        x = torch.flatten(
            x,
            start_dim=1
        )

        # [B, 256]


        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        x = self.fc(x)

        # [B, 10]

        return x


# ============================================================
# 3. 基本配置
# ============================================================

torch.manual_seed(42)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "使用设备：",
    device
)

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 5

OUTPUT_DIR = "outputs/resnet"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 4. MNIST 数据
# ============================================================

transform = transforms.ToTensor()

data_root = "../level1_mlp/data"

full_train_dataset = datasets.MNIST(
    root=data_root,
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root=data_root,
    train=False,
    transform=transform,
    download=True
)

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [55000, 5000],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(
    "训练集大小：",
    len(train_dataset)
)

print(
    "验证集大小：",
    len(val_dataset)
)

print(
    "测试集大小：",
    len(test_dataset)
)


# ============================================================
# 5. 创建模型
# ============================================================

model = ResNetMNIST().to(
    device
)

criterion = nn.CrossEntropyLoss()

# 为了进行控制变量比较，
# 与 Level 2 CNN / AlexNet / VGG 保持一致。

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 6. 参数量
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "\n===== ResNet 参数量 ====="
)

print(
    "Total Parameters：",
    total_params
)

print(
    "Trainable Parameters：",
    trainable_params
)


# ============================================================
# 7. 记录实验数据
# ============================================================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

epoch_times = []


# ============================================================
# 8. Train + Validation
# ============================================================

total_training_start = time.time()

for epoch in range(
    NUM_EPOCHS
):

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    epoch_start = time.time()


    for images, labels in train_loader:

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )


        # 1. 清除上一轮梯度
        optimizer.zero_grad()


        # 2. Forward
        outputs = model(
            images
        )


        # 3. Loss
        loss = criterion(
            outputs,
            labels
        )


        # 4. Backward
        loss.backward()


        # 5. Update
        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            dim=1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss
        / len(train_dataset)
    )

    train_acc = (
        correct
        / total
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    running_val_loss = 0.0

    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            running_val_loss += (
                loss.item()
                * images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                dim=1
            )


            val_total += (
                labels.size(0)
            )


            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        running_val_loss
        / len(val_dataset)
    )

    val_acc = (
        val_correct
        / val_total
    )


    epoch_time = (
        time.time()
        - epoch_start
    )


    epoch_times.append(
        epoch_time
    )

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    train_accuracies.append(
        train_acc
    )

    val_accuracies.append(
        val_acc
    )


    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_acc:.4f} "
        f"Time: {epoch_time:.2f}s"
    )


total_training_time = (
    time.time()
    - total_training_start
)


# ============================================================
# 9. Test
# ============================================================

model.eval()

test_correct = 0
test_total = 0

wrong_samples = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )


        outputs = model(
            images
        )


        _, predicted = torch.max(
            outputs,
            dim=1
        )


        test_total += (
            labels.size(0)
        )


        test_correct += (
            predicted == labels
        ).sum().item()


        # 保存最多 16 个错误样本
        wrong_mask = (
            predicted != labels
        )

        wrong_indices = (
            wrong_mask
            .nonzero(as_tuple=True)[0]
        )


        for idx in wrong_indices:

            if len(
                wrong_samples
            ) >= 16:

                break


            wrong_samples.append(
                (
                    images[idx]
                    .detach()
                    .cpu(),

                    labels[idx]
                    .item(),

                    predicted[idx]
                    .item()
                )
            )


test_accuracy = (
    test_correct
    / test_total
)


print(
    "\n===== ResNet 最终测试结果 ====="
)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Total Training Time: "
    f"{total_training_time:.2f}s"
)


# ============================================================
# 10. 保存模型
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "resnet_mnist.pth"
)

torch.save(
    model.state_dict(),
    model_path
)

print(
    "\n模型已保存：",
    model_path
)


# ============================================================
# 11. Loss Curve
# ============================================================

epochs = range(
    1,
    NUM_EPOCHS + 1
)

plt.figure()

plt.plot(
    epochs,
    train_losses,
    marker="o",
    label="Train Loss"
)

plt.plot(
    epochs,
    val_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "ResNet MNIST Loss Curve"
)

plt.legend()
plt.grid()

loss_curve_path = os.path.join(
    OUTPUT_DIR,
    "loss_curve.png"
)

plt.savefig(
    loss_curve_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    "Loss 曲线已保存：",
    loss_curve_path
)


# ============================================================
# 12. Accuracy Curve
# ============================================================

plt.figure()

plt.plot(
    epochs,
    train_accuracies,
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    epochs,
    val_accuracies,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "ResNet MNIST Accuracy Curve"
)

plt.legend()
plt.grid()

accuracy_curve_path = os.path.join(
    OUTPUT_DIR,
    "accuracy_curve.png"
)

plt.savefig(
    accuracy_curve_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    "Accuracy 曲线已保存：",
    accuracy_curve_path
)


# ============================================================
# 13. 错误样本
# ============================================================

if len(wrong_samples) > 0:

    plt.figure(
        figsize=(10, 10)
    )


    for i, (
        image,
        true_label,
        pred_label
    ) in enumerate(
        wrong_samples
    ):

        plt.subplot(
            4,
            4,
            i + 1
        )

        plt.imshow(
            image.squeeze(),
            cmap="gray"
        )

        plt.title(
            f"T:{true_label} "
            f"P:{pred_label}"
        )

        plt.axis("off")


    plt.tight_layout()


    wrong_samples_path = os.path.join(
        OUTPUT_DIR,
        "wrong_samples.png"
    )


    plt.savefig(
        wrong_samples_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "错误样本已保存：",
        wrong_samples_path
    )


# ============================================================
# 14. 单张图片推理
# ============================================================

single_image, true_label = (
    test_dataset[0]
)


input_tensor = (
    single_image
    .unsqueeze(0)
    .to(device)
)


model.eval()


with torch.no_grad():

    output = model(
        input_tensor
    )

    predicted_label = (
        output
        .argmax(dim=1)
        .item()
    )


print(
    "\n===== 单张图片推理 ====="
)

print(
    "真实标签：",
    true_label
)

print(
    "预测标签：",
    predicted_label
)


plt.figure()

plt.imshow(
    single_image.squeeze(),
    cmap="gray"
)

plt.title(
    f"True: {true_label}, "
    f"Pred: {predicted_label}"
)

plt.axis("off")


inference_path = os.path.join(
    OUTPUT_DIR,
    "inference_example.png"
)


plt.savefig(
    inference_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print(
    "推理图片已保存：",
    inference_path
)