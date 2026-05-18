"""
训练脚本

训练流程：
    for epoch in range(epochs):
        1. 前向传播：logits = model(x)
        2. 计算损失：loss = CrossEntropyLoss(logits, y)
        3. 清零梯度：optimizer.zero_grad()
        4. 反向传播：loss.backward()  → 计算 ∂loss/∂W
        5. 参数更新：optimizer.step() → W ← W - lr * ∂loss/∂W
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from config import config
from model import MLP, count_parameters
from dataset import get_mnist_loaders, show_dataset_info


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """运行一个完整的训练 epoch，返回平均损失和准确率"""
    model.train()  # 开启训练模式（启用 Dropout、BatchNorm 使用批统计量）
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        # --- 核心五步 ---
        logits = model(images)                    # 1. 前向传播
        loss = criterion(logits, labels)          # 2. 计算损失
        optimizer.zero_grad()                     # 3. 清零梯度
        loss.backward()                           # 4. 反向传播
        optimizer.step()                          # 5. 更新参数

        # 统计
        total_loss += loss.item()
        pred = logits.argmax(dim=1)               # 取概率最大的类别作为预测
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % config.log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            acc = correct / total * 100
            print(f"  Epoch {epoch:02d} [{batch_idx+1:4d}/{len(loader)}] "
                  f"Loss: {avg_loss:.4f}  Acc: {acc:.2f}%")

    return total_loss / len(loader), correct / total * 100


def evaluate(model, loader, criterion, device):
    """在验证/测试集上评估模型，返回平均损失和准确率"""
    model.eval()  # 关闭 Dropout，BatchNorm 使用运行时统计量
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():  # 关闭梯度计算，节省显存
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
    # 固定随机种子，保证实验可复现
    torch.manual_seed(config.seed)

    # 自动选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    # 数据加载
    train_loader, test_loader = get_mnist_loaders(config.data_dir, config.batch_size, config.num_workers)
    show_dataset_info(train_loader, test_loader)
    print()

    # 模型初始化
    model = MLP(
        input_size=config.input_size,
        hidden_sizes=config.hidden_sizes,
        output_size=config.output_size,
        dropout_rate=config.dropout_rate,
    ).to(device)
    print(f"模型参数总量: {count_parameters(model):,}")
    print(model)
    print()

    # 损失函数：CrossEntropyLoss = LogSoftmax + NLLLoss
    # 公式：L = -log( exp(z_y) / Σ exp(z_j) )
    criterion = nn.CrossEntropyLoss()

    # 优化器：Adam（带 L2 正则化）
    # W ← W - lr * (m̂_t / (√v̂_t + ε) + λW)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # 学习率调度：每 step_size 个 epoch 将 lr 乘以 gamma
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)

    # 创建 checkpoint 目录
    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

    best_acc = 0.0
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

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.save_path)
            print(f"  ✓ 保存最优模型（准确率: {best_acc:.2f}%）")

        scheduler.step()

    print(f"\n训练完成！最优测试准确率: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
