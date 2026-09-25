import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ============================================================
# 1. AlexNet-style 网络
# ============================================================
# 原始 AlexNet 是为 ImageNet 大尺寸 RGB 图像设计的。
# 这里为了和 Level 2 的 MNIST CNN 进行对比，
# 保留 AlexNet 的核心结构思想，并适配 28×28 灰度 MNIST。
#
# 核心特点：
# 1. 比 Level 2 Baseline CNN 更深
# 2. 多层 Conv + ReLU
# 3. 使用 MaxPool 进行下采样
# 4. 分类器中使用 Dropout
# ============================================================

class AlexNetMNIST(nn.Module):
    def __init__(self):
        super().__init__()

        # -------------------------
        # 特征提取部分
        # -------------------------
        self.features = nn.Sequential(

            # 输入：[B, 1, 28, 28]
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            # 输出：[B, 64, 28, 28]
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            # 输出：[B, 64, 14, 14]

            nn.Conv2d(
                in_channels=64,
                out_channels=192,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            # 输出：[B, 192, 14, 14]
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            # 输出：[B, 192, 7, 7]

            nn.Conv2d(
                in_channels=192,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            # 输出：[B, 384, 7, 7]
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            # 输出：[B, 256, 7, 7]
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            # 输出：[B, 256, 7, 7]
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
            # 7×7 经过 2×2 Pool：
            # 输出：[B, 256, 3, 3]
        )

        # -------------------------
        # 分类部分
        # -------------------------
        self.classifier = nn.Sequential(

            # 256 × 3 × 3 = 2304
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(inplace=True),

            # AlexNet 的经典思想之一：
            # 使用 Dropout 缓解过拟合
            nn.Dropout(p=0.5),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),

            # MNIST 一共有 10 个类别
            nn.Linear(256, 10)
        )

    def forward(self, x):
        # 卷积特征提取
        x = self.features(x)

        # [B, 256, 3, 3]
        # →
        # [B, 2304]
        x = torch.flatten(x, start_dim=1)

        # 分类
        x = self.classifier(x)

        # 输出 logits：[B, 10]
        return x


# ============================================================
# 2. 基本配置
# ============================================================

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用设备：", device)

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 5

OUTPUT_DIR = "outputs/alexnet"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 3. MNIST 数据集
# ============================================================

transform = transforms.ToTensor()

# 直接复用 Level 1 / Level 2 已下载的 MNIST 数据
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

# 和 Level 2 保持一致：
# 55,000 Train
# 5,000 Validation

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

print("训练集大小：", len(train_dataset))
print("验证集大小：", len(val_dataset))
print("测试集大小：", len(test_dataset))


# ============================================================
# 4. 创建模型
# ============================================================

model = AlexNetMNIST().to(device)

criterion = nn.CrossEntropyLoss()

# 为了和 Level 2 尽量保持相同实验条件：
# Adam + lr=0.001
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

print("\n===== AlexNet 参数量 =====")
print("Total Parameters：", total_params)
print("Trainable Parameters：", trainable_params)


# ============================================================
# 6. 保存训练过程数据
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

        # 清除上一轮累积梯度
        optimizer.zero_grad()

        # Forward
        outputs = model(images)

        # Loss
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()

        # 根据梯度更新参数
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_loss = running_loss / len(train_dataset)

    train_acc = correct / total


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

            val_total += labels.size(0)

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

    epoch_times.append(epoch_time)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

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

        test_total += labels.size(0)

        test_correct += (
            predicted == labels
        ).sum().item()

        # 保存部分错误样本
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

print("\n===== AlexNet 最终测试结果 =====")

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
    "alexnet_mnist.pth"
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
plt.title("AlexNet MNIST Loss Curve")
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
plt.title("AlexNet MNIST Accuracy Curve")
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

        plt.subplot(4, 4, i + 1)

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


print("\n===== 单张图片推理 =====")

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