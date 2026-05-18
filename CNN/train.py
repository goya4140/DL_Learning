"""
训练脚本

训练流程与 MLP 完全一致：
    for epoch in range(epochs):
        1. 前向传播：logits = model(x)
        2. 计算损失：loss = CrossEntropyLoss(logits, y)
        3. 清零梯度：optimizer.zero_grad()
        4. 反向传播：loss.backward()
        5. 参数更新：optimizer.step()
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from config import config
from model import CNN, count_parameters
from dataset import get_cifar10_loaders, show_dataset_info
from utils import ExperimentLogger


def get_device(preference: str = "auto") -> torch.device:
    """优先级：CUDA（NVIDIA）> MPS（Apple Silicon）> CPU"""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(preference)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """运行一个完整的训练 epoch，返回平均损失和准确率"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % config.log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            acc = correct / total * 100
            print(f"  Epoch {epoch:02d} [{batch_idx+1:4d}/{len(loader)}] "
                  f"Loss: {avg_loss:.4f}  Acc: {acc:.2f}%")

    return total_loss / len(loader), correct / total * 100


def evaluate(model, loader, criterion, device):
    """在测试集上评估模型，返回平均损失和准确率"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total * 100


def main():
    torch.manual_seed(config.seed)

    device = get_device(config.device)
    print(f"使用设备: {device}\n")

    pin_memory = device.type == "cuda"

    train_loader, test_loader = get_cifar10_loaders(
        config.data_dir, config.batch_size, config.num_workers, pin_memory=pin_memory
    )
    show_dataset_info(train_loader, test_loader)
    print()

    model = CNN(
        in_channels=config.in_channels,
        channels=config.channels,
        fc_hidden=config.fc_hidden,
        output_size=config.output_size,
        dropout_rate=config.dropout_rate,
    ).to(device)
    print(f"模型参数总量: {count_parameters(model):,}")
    print(model)
    print()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)

    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

    best_acc = 0.0
    logger = ExperimentLogger(log_dir="./logs", lr_step_size=config.lr_step_size)

    print("=" * 60)
    print("开始训练")
    print("=" * 60)

    for epoch in range(1, config.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n[Epoch {epoch:02d}/{config.epochs}]  学习率: {current_lr:.6f}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        print(f"  训练损失: {train_loss:.4f}  训练准确率: {train_acc:.2f}%")
        print(f"  测试损失: {val_loss:.4f}  测试准确率: {val_acc:.2f}%")

        logger.log(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.save_path)
            print(f"  ✓ 保存最优模型（准确率: {best_acc:.2f}%）")

        scheduler.step()

    print(f"\n训练完成！最优测试准确率: {best_acc:.2f}%")

    print()
    logger.save()
    logger.plot()


if __name__ == "__main__":
    main()
