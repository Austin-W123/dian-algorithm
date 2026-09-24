import os
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split


# ==================== 1. 定义 CNN ====================

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # [B, 1, 28, 28] -> [B, 32, 28, 28]
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu1 = nn.ReLU()

        # [B, 32, 28, 28] -> [B, 32, 14, 14]
        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # [B, 32, 14, 14] -> [B, 64, 14, 14]
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu2 = nn.ReLU()

        # [B, 64, 14, 14] -> [B, 64, 7, 7]
        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()

        # 输出 10 个 logits
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # [B, 64, 7, 7] -> [B, 3136]
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x


# ==================== 2. 基本设置 ====================

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

batch_size = 64
learning_rate = 0.001
num_epochs = 5

os.makedirs("outputs", exist_ok=True)

print("使用设备：", device)


# ==================== 3. 加载 MNIST ====================

# 复用 Level 1 已经下载好的 MNIST 数据
full_train_data = datasets.MNIST(
    root="../level1_mlp/data",
    train=True,
    transform=ToTensor(),
    download=True
)

test_data = datasets.MNIST(
    root="../level1_mlp/data",
    train=False,
    transform=ToTensor(),
    download=True
)


# ==================== 4. Train / Validation 划分 ====================

generator = torch.Generator().manual_seed(42)

train_data, val_data = random_split(
    full_train_data,
    [55000, 5000],
    generator=generator
)


# ==================== 5. DataLoader ====================

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False
)

print("训练集大小：", len(train_data))
print("验证集大小：", len(val_data))
print("测试集大小：", len(test_data))


# ==================== 6. 创建模型 ====================

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()

# CNN 这里使用 Adam，减少调参时间并加快收敛
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# ==================== 7. 参数量统计 ====================

total_params = sum(
    p.numel() for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\n===== CNN 参数量 =====")
print("Total Parameters：", total_params)
print("Trainable Parameters：", trainable_params)


# ==================== 8. 训练 ====================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

training_start_time = time.time()


for epoch in range(num_epochs):

    # ---------- Train ----------

    epoch_start_time = time.time()

    model.train()

    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)


    train_loss = (
        running_loss
        / len(train_loader.dataset)
    )

    train_accuracy = (
        train_correct
        / train_total
    )

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)


    # ---------- Validation ----------

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

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)


    val_loss = (
        running_val_loss
        / len(val_loader.dataset)
    )

    val_accuracy = (
        val_correct
        / val_total
    )

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)


    epoch_time = (
        time.time()
        - epoch_start_time
    )

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.4f} "
        f"Time: {epoch_time:.2f}s"
    )


total_training_time = (
    time.time()
    - training_start_time
)


# ==================== 9. Test ====================

model.eval()

test_correct = 0
test_total = 0

# 保存错误样本
wrong_images = []
wrong_labels = []
wrong_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        test_correct += (
            predictions == labels
        ).sum().item()

        test_total += labels.size(0)

        # 找出当前 batch 中预测错误的图片
        wrong_mask = (
            predictions != labels
        )

        wrong_indices = (
            wrong_mask
            .nonzero(as_tuple=True)[0]
        )

        # 只保存前 16 个错误样本
        for index in wrong_indices:

            if len(wrong_images) >= 16:
                break

            wrong_images.append(
                images[index]
                .detach()
                .cpu()
            )

            wrong_labels.append(
                labels[index].item()
            )

            wrong_predictions.append(
                predictions[index].item()
            )


test_accuracy = (
    test_correct
    / test_total
)

print("\n===== CNN 最终测试结果 =====")

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


# ==================== 10. 保存模型 ====================

model_path = "outputs/cnn_mnist.pth"

torch.save(
    model.state_dict(),
    model_path
)

print(
    "\n模型已保存：",
    model_path
)


# ==================== 11. Loss 曲线 ====================

epochs = range(
    1,
    num_epochs + 1
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
plt.title("CNN MNIST Loss Curve")

plt.legend()
plt.grid()

plt.savefig(
    "outputs/loss_curve.png"
)

plt.close()

print(
    "Loss 曲线已保存："
    "outputs/loss_curve.png"
)


# ==================== 12. Accuracy 曲线 ====================

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
plt.title("CNN MNIST Accuracy Curve")

plt.legend()
plt.grid()

plt.savefig(
    "outputs/accuracy_curve.png"
)

plt.close()

print(
    "Accuracy 曲线已保存："
    "outputs/accuracy_curve.png"
)


# ==================== 13. 错误样本可视化 ====================

if len(wrong_images) > 0:

    plt.figure(
        figsize=(10, 10)
    )

    for i in range(
        len(wrong_images)
    ):

        plt.subplot(
            4,
            4,
            i + 1
        )

        plt.imshow(
            wrong_images[i]
            .squeeze(),
            cmap="gray"
        )

        plt.title(
            f"True: {wrong_labels[i]}\n"
            f"Pred: {wrong_predictions[i]}"
        )

        plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        "outputs/wrong_samples.png"
    )

    plt.close()

    print(
        "错误样本已保存："
        "outputs/wrong_samples.png"
    )


# ==================== 14. 单张图片推理 ====================

image, true_label = test_data[0]

input_image = (
    image
    .unsqueeze(0)
    .to(device)
)

model.eval()

with torch.no_grad():

    output = model(input_image)

    predicted_label = torch.argmax(
        output,
        dim=1
    ).item()


print("\n===== 单张图片推理 =====")
print("真实标签：", true_label)
print("预测标签：", predicted_label)


plt.figure()

plt.imshow(
    image.squeeze(),
    cmap="gray"
)

plt.title(
    f"True: {true_label}, "
    f"Predicted: {predicted_label}"
)

plt.axis("off")

plt.savefig(
    "outputs/inference_example.png"
)

plt.close()

print(
    "推理图片已保存："
    "outputs/inference_example.png"
)