"""
训练脚本

GCN 训练与 MLP/CNN/RNN/LSTM 的核心区别：
  - 无 DataLoader：整个图是单一输入对象（N 个节点同时前向传播）
  - 无 batch 循环：每个 epoch 只有一次前向传播 + 一次反向传播
  - 掩码选择：train_mask/val_mask/test_mask 控制参与损失计算的节点
  - 损失函数：NLLLoss（配合模型输出的 log_softmax）
  - 无梯度裁剪：GCN 梯度稳定，无爆炸问题
  - 无学习率调度：原论文固定 lr=0.01，200 epoch
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from config import config
from model import GCNClassifier, count_parameters
from dataset import load_cora, show_dataset_info
from utils import ExperimentLogger


def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(preference)


def train_one_epoch(model, data, optimizer, criterion):
    """整图一次前向传播，只对 train_mask 节点计算损失"""
    model.train()
    optimizer.zero_grad()

    out = model(data["features"], data["adj"])   # (N, num_classes)

    mask   = data["train_mask"]
    loss   = criterion(out[mask], data["labels"][mask])
    loss.backward()
    optimizer.step()

    pred    = out.argmax(dim=1)
    correct = (pred[mask] == data["labels"][mask]).sum().item()
    acc     = correct / mask.sum().item() * 100

    return loss.item(), acc


def evaluate(model, data, criterion, mask):
    """在指定 mask 的节点上评估（val_mask 或 test_mask）"""
    model.eval()
    with torch.no_grad():
        out  = model(data["features"], data["adj"])   # (N, num_classes)
        loss = criterion(out[mask], data["labels"][mask])

        pred    = out.argmax(dim=1)
        correct = (pred[mask] == data["labels"][mask]).sum().item()
        acc     = correct / mask.sum().item() * 100

    return loss.item(), acc


def main():
    torch.manual_seed(config.seed)

    device = get_device(config.device)
    print(f"使用设备: {device}\n")

    # 加载 Cora 图数据
    data = load_cora(config.data_dir)
    show_dataset_info(data)
    print()

    # 将所有张量移到目标设备
    data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()}

    model = GCNClassifier(
        in_features=config.in_features,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        num_layers=config.num_layers,
        dropout_rate=config.dropout_rate,
    ).to(device)

    print(f"模型参数总量: {count_parameters(model):,}  "
          f"({config.num_layers} 层 GCN, hidden={config.hidden_dim})")
    print(model)
    print()

    # NLLLoss 配合 log_softmax（数值稳定，等价于 CrossEntropyLoss）
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,
                           weight_decay=config.weight_decay)

    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

    best_val_acc = 0.0
    logger = ExperimentLogger(log_dir="./logs")

    print("=" * 60)
    print("开始训练")
    print("=" * 60)

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, data, optimizer, criterion)
        val_loss,   val_acc   = evaluate(model, data, criterion, data["val_mask"])

        logger.log(epoch, train_loss, train_acc, val_loss, val_acc,
                   lr=optimizer.param_groups[0]["lr"])

        if (epoch) % config.log_interval == 0 or epoch == 1:
            print(f"[Epoch {epoch:03d}/{config.epochs}]  "
                  f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  |  "
                  f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.save_path)

    print(f"\n训练完成！最优验证准确率: {best_val_acc:.2f}%")

    # 用最优模型在测试集上报告最终结果
    model.load_state_dict(torch.load(config.save_path, map_location=device))
    test_loss, test_acc = evaluate(model, data, criterion, data["test_mask"])
    print(f"测试集准确率: {test_acc:.2f}%  测试损失: {test_loss:.4f}")
    print()

    logger.save()
    logger.plot()


if __name__ == "__main__":
    main()
