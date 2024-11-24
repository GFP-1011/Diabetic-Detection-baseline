
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.models as models
from loaddata import load_datasets
from tqdm import tqdm
import matplotlib.pyplot as plt

def test(model, test_loader, criterion, device):
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    correct = 0
    total = 0

    # 不需要计算梯度
    with torch.no_grad():
        for images, labels in test_loader:
            # 将数据移到指定设备
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 统计损失和准确率
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(test_loader)
    epoch_accuracy = correct / total

    print(f"Test Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    return epoch_loss, epoch_accuracy

def val_and_save(model, val_loader, criterion, device, epoch, best_accuracy, save_path):
    """
    验证模型并保存表现最好的模型。

    Args:
        model: PyTorch 模型。
        val_loader: 验证数据的 DataLoader。
        criterion: 损失函数。
        device: 设备（CPU 或 GPU）。
        epoch: 当前训练轮数。
        best_accuracy: 当前最佳验证准确率。
        save_path: 最佳模型保存路径。

    Returns:
        val_loss: 当前验证损失。
        val_accuracy: 当前验证准确率。
    """
    model.eval()  # 设置为评估模式
    running_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():  # 不计算梯度
        for slo_fundus, male, dr_class in tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}", leave=False):
            # 将数据移动到设备
            slo_fundus, dr_class = slo_fundus.to(device), dr_class.to(device)

            # 前向传播
            outputs = model(slo_fundus)  # 模型只接收图像输入
            loss = criterion(outputs, dr_class)

            # 统计验证损失和准确率
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_val += (predicted == dr_class).sum().item()
            total_val += dr_class.size(0)

    val_loss = running_loss / len(val_loader)
    val_accuracy = correct_val / total_val

    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # 保存最佳模型
    if val_accuracy > best_accuracy:
        torch.save(model.state_dict(), save_path)

    return val_loss, val_accuracy



def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path="best_model.pth"):
    """
    训练和验证模型，保存验证准确率最高的模型。

    Args:
        model: PyTorch 模型。
        train_loader: 训练数据的 DataLoader。
        val_loader: 验证数据的 DataLoader。
        criterion: 损失函数。
        optimizer: 优化器。
        num_epochs: 训练轮数。
        device: 设备（CPU 或 GPU）。
        save_path: 最佳模型保存路径。

    Returns:
        train_losses: 每轮训练损失列表。
        val_losses: 每轮验证损失列表。
    """
    best_accuracy = 0.0  # 记录最佳验证准确率
    train_losses = []  # 保存每轮训练损失
    val_losses = []  # 保存每轮验证损失
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()  # 设置为训练模式
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # tqdm 显示训练进度条
        train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False)

        for slo_fundus, male, dr_class in train_loader_tqdm:
            # 将数据移动到设备
            slo_fundus, dr_class = slo_fundus.to(device), dr_class.to(device)

            # 前向传播
            outputs = model(slo_fundus)  # 模型只接收图像输入
            loss = criterion(outputs, dr_class)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计训练损失和准确率
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_train += (predicted == dr_class).sum().item()
            total_train += dr_class.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Validation phase
        val_loss, val_accuracy = val_and_save(model, val_loader, criterion, device, epoch, best_accuracy, save_path)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # 更新最佳验证准确率
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            print(f"Best model saved with Val Accuracy: {best_accuracy:.4f}")

    print("Training complete.")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")

    return train_losses, val_losses ,train_accuracies, val_accuracies



if __name__ == '__main__':

    train_loader, val_loader, test_loader = load_datasets('/data/fupeiguo/ORID', batch_size=32)

    model = models.densenet121(pretrained=True)  # 使用 DenseNet-121 作为示例

    # 修改最后的分类层
    model.classifier = nn.Linear(model.classifier.in_features, 2)  # 修改为二分类
    # model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 假设二分类：DR 或非 DR

    criterion = torch.nn.CrossEntropyLoss()  # 二分类
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 设备选择
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model.to(device)

    # 调用训练方法
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model=model,  # 模型
        train_loader=train_loader,  # 训练数据加载器
        val_loader=val_loader,  # 验证数据加载器
        criterion=criterion,  # 损失函数
        optimizer=optimizer,  # 优化器
        num_epochs=50,  # 训练轮数
        device=device,  # 设备
        save_path="best_model_dense.pth"  # 保存最佳模型的路径
    )

    # 绘制训练和验证损失曲线
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()
    
       # 绘制训练和验证损失曲线
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train ACC', marker='o')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation ACC', marker='s')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('ACC', fontsize=12)
    plt.title('Training and Validation ACC', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

