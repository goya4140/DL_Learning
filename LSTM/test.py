"""
测试脚本

加载训练好的最优模型，在测试集上：
  1. 计算整体准确率
  2. 每类准确率统计
  3. 绘制 4×4 混淆矩阵热力图
  4. 随机抽样预测展示（显示原始文本 + 真实标签 + 预测标签）
"""

import os
import textwrap

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import config
from model import LSTMClassifier, count_parameters
from dataset import (build_vocab, get_agnews_loaders, CLASSES, NUM_CLASSES,
                     _download_agnews, _read_csv, _tokenizer)
from train import get_device


def _setup_cjk_font() -> bool:
    cjk_fonts = ["PingFang SC", "Heiti SC", "STHeiti", "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in cjk_fonts:
        if font in available:
            matplotlib.rcParams["font.sans-serif"] = [font] + matplotlib.rcParams["font.sans-serif"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


_HAS_CJK = _setup_cjk_font()


def evaluate_per_class(model, loader, device):
    model.eval()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    with torch.no_grad():
        for labels, texts in loader:
            labels, texts = labels.to(device), texts.to(device)
            preds = model(texts).argmax(dim=1)
            for true, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion[true][pred] += 1

    return confusion


def plot_confusion_matrix(confusion: np.ndarray, save_dir: str = "./logs"):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(confusion, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASSES, rotation=20, ha="right")
    ax.set_yticklabels(CLASSES)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(confusion[i][j]),
                    ha="center", va="center",
                    color="white" if confusion[i][j] > confusion.max() * 0.5 else "black",
                    fontsize=10)

    ax.set_xlabel("Predicted" if not _HAS_CJK else "预测类别")
    ax.set_ylabel("True"      if not _HAS_CJK else "真实类别")
    ax.set_title("Confusion Matrix (AG News)" if not _HAS_CJK else "混淆矩阵（AG News）")

    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"混淆矩阵已保存至: {path}")


def print_class_accuracy(confusion: np.ndarray):
    print("\n── 各类别准确率 ──")
    for i, cls in enumerate(CLASSES):
        correct = confusion[i][i]
        total   = confusion[i].sum()
        print(f"  {cls:10s}: {correct:4d}/{total:4d}  ({100*correct/total:.1f}%)")
    total_correct = np.trace(confusion)
    total_all     = confusion.sum()
    print(f"\n  总体准确率: {total_correct}/{total_all}  ({100*total_correct/total_all:.2f}%)")


def predict_samples(model, vocab, device, data_dir: str, num_samples: int = 8):
    model.eval()
    _, test_path = _download_agnews(data_dir)
    test_data = _read_csv(test_path)

    indices = np.random.choice(len(test_data), size=num_samples, replace=False)

    print(f"\n── 随机抽样预测（{num_samples} 条）──")
    correct_count = 0
    for idx in indices:
        label_raw, text = test_data[idx]
        true_label = int(label_raw) - 1    # 1-4 → 0-3

        token_ids = vocab(_tokenizer(text))[:config.max_len]
        x = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_label = model(x).argmax(dim=1).item()

        mark = "✓" if pred_label == true_label else "✗"
        if pred_label == true_label:
            correct_count += 1

        short_text = textwrap.shorten(text, width=80, placeholder="...")
        print(f"  [{mark}] 真实: {CLASSES[true_label]:10s}  预测: {CLASSES[pred_label]:10s}")
        print(f"       {short_text}\n")

    print(f"  抽样准确率: {correct_count}/{num_samples}")


def main():
    device     = get_device(config.device)
    pin_memory = device.type == "cuda"
    print(f"使用设备: {device}")

    vocab = build_vocab(config.data_dir, config.vocab_size)
    pad_idx = vocab["<pad>"]

    _, test_loader = get_agnews_loaders(
        config.data_dir, vocab, config.max_len,
        config.batch_size, config.num_workers, pin_memory=pin_memory,
    )

    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=config.embed_dim,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        num_layers=config.num_layers,
        dropout_rate=config.dropout_rate,
        pad_idx=pad_idx,
        bidirectional=config.bidirectional,
    ).to(device)

    if not os.path.exists(config.save_path):
        print(f"错误：模型文件不存在 {config.save_path}，请先运行 train.py")
        return

    model.load_state_dict(torch.load(config.save_path, map_location=device))
    print(f"模型参数总量: {count_parameters(model):,}")
    print(f"已加载模型: {config.save_path}\n")

    os.makedirs("./logs", exist_ok=True)

    confusion = evaluate_per_class(model, test_loader, device)
    print_class_accuracy(confusion)
    plot_confusion_matrix(confusion)

    predict_samples(model, vocab, device, config.data_dir)


if __name__ == "__main__":
    main()
