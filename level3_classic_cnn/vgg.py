import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ============================================================
# 1. VGG-style 网络
# ============================================================
# 原始 VGG 是为 ImageNet 大尺寸 RGB 图像设计的。
# 这里使用适配 MNIST 的 VGG-style 网络。
#
# VGG 的核心思想：
# 1. 大量使用 3×3 小卷积核
# 2. 在同一空间分辨率下连续进行多次卷积
# 3. 再通过 MaxPool 进行下采样
# 4. 网络结构非常规则
#
# 与 Level 2 Baseline CNN 相比：
# Baseline:
#   Conv -> Pool -> Conv -> Pool
#
# VGG-style:
#   Conv -> Conv -> Pool
#   Conv -> Conv -> Pool
#   Conv -> Conv -> Conv -> Pool
# ============================================================


class VGGMNIST(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # =================================================
            # Block 1
            # 输入：[B, 1, 28, 28]
            # =================================================

            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            # [B, 64, 28, 28]
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            # [B, 64, 28, 28]
            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            # [B, 64, 14, 14]


            # =================================================
            # Block 2
            # =================================================

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            # [B, 128, 14, 14]
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            # [B, 128, 14, 14]
            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            # [B, 128, 7, 7]


            # =================================================
            # Block 3
            # =================================================

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            # [B, 256, 7, 7]
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            # [B, 256, 7, 7]
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            # [B, 256, 7, 7]
            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
            # [B, 256, 3, 3]
        )

        # =====================================================
        # 分类器
        # =====================================================

        self.classifier = nn.Sequential(

            # 256 × 3 × 3 = 2304
            nn.Linear(
                256 * 3 * 3,
                512
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),

            nn.Linear(
                512,
                256
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),

            nn.Linear(
                256,
                10
            )
        )

    def forward(self, x):

        # VGG 卷积特征提取
        x = self.features(x)

        # [B, 256, 3, 3]
        # →
        # [B, 2304]
        x = torch.flatten(
            x,
            start_dim=1
        )

        # 分类
        x = self.classifier(x)

        # [B, 10] logits
        return x


# ============================================================
# 2. 基本配置
# ============================================================

torch.manual_seed(42)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("使用设备：", device)

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 5

OUTPUT_DIR = "outputs/vgg"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 3. MNIST 数据
# ============================================================

transform = transforms.ToTensor()

# 与 Level 2 / AlexNet 使用完全相同的数据
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

# 与前面的实验保持相同划分：
#
# Train      55,000
# Validation  5,000
# Test       10,000

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
# 4. 创建模型
# ============================================================

model = VGGMNIST().to(device)

criterion = nn.CrossEntropyLoss()

# 与 Baseline CNN / AlexNet 保持相同：
#
# Adam
# learning rate = 0.001

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 5. 参数量统计
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

print("\n===== VGG 参数量 =====")

print(
    "Total Parameters：",
    total_params
)

print(
    "Trainable Parameters：",
    trainable_params
)


# ============================================================
# 6. 记录实验数据
# ============================================================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

epoch_times = []


# ============================================================
# 7. Train + Validation
# ============================================================

total_training_start = time.time()

for epoch in range(NUM_EPOCHS):

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    epoch_start = time.time()

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # 清空上一轮梯度
        optimizer.zero_grad()

        # Forward
        outputs = model(images)

        # Loss
        loss = criterion(
            outputs,
            labels
        )

        # Backward
        loss.backward()

        # Update Parameters
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

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

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
# 8. Test
# ============================================================

model.eval()

test_correct = 0
test_total = 0

wrong_samples = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

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

            if len(wrong_samples) >= 16:
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
    "\n===== VGG 最终测试结果 ====="
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
# 9. 保存模型
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "vgg_mnist.pth"
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
# 10. Loss Curve
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
    "VGG MNIST Loss Curve"
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
# 11. Accuracy Curve
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
    "VGG MNIST Accuracy Curve"
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
# 12. 错误样本可视化
# ============================================================

if len(wrong_samples) > 0:

    plt.figure(
        figsize=(10, 10)
    )

    for i, (
        image,
        true_label,
        pred_label
    ) in enumerate(wrong_samples):

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
# 13. 单张图片推理
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