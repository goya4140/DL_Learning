"""
训练脚本

Transformer 训练流程与 LSTM 基本一致，两处关键差异：
  1. src_key_padding_mask：由 texts 中的 <pad> 位置自动生成，
     传给模型让 Self-Attention 忽略 padding 位置
  2. 无梯度裁剪：Transformer 配合 LayerNorm 训练稳定，不需要 clip_grad_norm_
"""

import warnings
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

import os
import torch
import torch.nn as nn
import torch.optim as optim

from config import config
from model import TransformerClassifier, count_parameters
from dataset import build_vocab, get_agnews_loaders, show_dataset_info
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


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (labels, texts) in enumerate(loader):
        labels, texts = labels.to(device), texts.to(device)

        logits = model(texts)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        # Transformer 训练稳定，无需梯度裁剪
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
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for labels, texts in loader:
            labels, texts = labels.to(device), texts.to(device)
            logits = model(texts)
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

    vocab = build_vocab(config.data_dir, config.vocab_size)
    print(f"词汇表大小: {len(vocab)}")
    pad_idx = vocab["<pad>"]
    print()

    train_loader, test_loader = get_agnews_loaders(
        config.data_dir, vocab, config.max_len,
        config.batch_size, config.num_workers, pin_memory=pin_memory,
    )
    show_dataset_info(train_loader, test_loader)
    print()

    model = TransformerClassifier(
        vocab_size=len(vocab),
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        d_ff=config.d_ff,
        num_classes=config.num_classes,
        dropout_rate=config.dropout_rate,
        pad_idx=pad_idx,
        max_len=config.max_len,
    ).to(device)

    print(f"模型参数总量: {count_parameters(model):,}  "
          f"({config.num_encoder_layers} 层, d_model={config.d_model}, nhead={config.nhead})")
    print(model)
    print()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size,
                                          gamma=config.lr_gamma)

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
        val_loss,   val_acc   = evaluate(model, test_loader, criterion, device)

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
