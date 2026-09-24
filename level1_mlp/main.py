import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split


# ==================== 1. 定义 MLP ====================

class MLP(nn.Module):       # 搭建 MLP（继承 nn.Module）
    def __init__(self):     # Python 类的初始化函数
        super().__init__()

        self.fc1 = nn.Linear(784, 128)   # 第一层全连接层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):     # 前向传播计算定义
        x = torch.flatten(x, start_dim=1)     # [B, 1, 28, 28] -> [B, 784]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


# ==================== 2. 基本设置 ====================

torch.manual_seed(42)     # 固定随机种子，使实验尽量可复现

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
learning_rate = 0.1
num_epochs = 5

os.makedirs("outputs", exist_ok=True)

print("使用设备：", device)


# ==================== 3. 加载 MNIST ====================

full_train_data = datasets.MNIST(
    root="./data",
    train=True,
    transform=ToTensor(),
    download=True
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    transform=ToTensor(),
    download=True
)


# ==================== 4. 划分训练集和验证集 ====================

generator = torch.Generator().manual_seed(42)

train_data, val_data = random_split(
    full_train_data,
    [55000, 5000],
    generator=generator
)


# ==================== 5. 创建 DataLoader ====================

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

model = MLP().to(device)

criterion = nn.CrossEntropyLoss()     # 交叉熵 Loss
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate
)


# 用于之后绘制曲线
train_losses = []
val_losses = []


# ==================== 7. 正式训练 ====================

for epoch in range(num_epochs):

    # ---------- Train ----------

    model.train()

    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()     # 清空上一轮梯度
        loss.backward()           # 反向传播，计算梯度
        optimizer.step()          # 根据梯度更新模型参数

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = train_correct / train_total

    train_losses.append(train_loss)


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
            loss = criterion(outputs, labels)

            running_val_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss = running_val_loss / len(val_loader.dataset)
    val_accuracy = val_correct / val_total

    val_losses.append(val_loss)


    # ---------- 输出当前 Epoch 结果 ----------

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )


# ==================== 8. Test 最终测试 ====================

model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        test_correct += (predictions == labels).sum().item()
        test_total += labels.size(0)


test_accuracy = test_correct / test_total

print("\n===== 最终测试结果 =====")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ==================== 9. 保存模型 ====================

model_path = "outputs/mlp_mnist.pth"

torch.save(model.state_dict(), model_path)

print("\n模型已保存：", model_path)


# ==================== 10. 保存 Loss 曲线 ====================

epochs = range(1, num_epochs + 1)

plt.figure()

plt.plot(epochs, train_losses, marker="o", label="Train Loss")
plt.plot(epochs, val_losses, marker="o", label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP MNIST Loss Curve")
plt.legend()
plt.grid()

plt.savefig("outputs/loss_curve.png")
plt.close()

print("Loss 曲线已保存：outputs/loss_curve.png")


# ==================== 11. 单张图片推理 ====================

image, true_label = test_data[0]

# 原本 [1, 28, 28]
# 增加 batch 维度 -> [1, 1, 28, 28]
input_image = image.unsqueeze(0).to(device)

model.eval()

with torch.no_grad():
    output = model(input_image)
    predicted_label = torch.argmax(output, dim=1).item()


print("\n===== 单张图片推理 =====")
print("真实标签：", true_label)
print("预测标签：", predicted_label)


# 保存这张测试图片

plt.figure()
plt.imshow(image.squeeze(), cmap="gray")
plt.title(
    f"True: {true_label}, Predicted: {predicted_label}"
)
plt.axis("off")

plt.savefig("outputs/inference_example.png")
plt.close()

print("推理图片已保存：outputs/inference_example.png")