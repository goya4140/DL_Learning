"""
测试脚本

加载训练好的最优模型，在测试集上：
  1. 计算整体准确率 + 各类准确率
  2. 绘制 7×7 混淆矩阵热力图
  3. 随机抽样测试节点展示（节点 ID + 真实标签 + 预测标签）
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from config import config
from model import GCNClassifier, count_parameters
from dataset import load_cora, CLASSES, NUM_CLASSES
from train import get_device, evaluate


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


def evaluate_per_class(model, data, device) -> np.ndarray:
    """在 test_mask 节点上构建 7×7 混淆矩阵"""
    model.eval()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    with torch.no_grad():
        out   = model(data["features"], data["adj"])   # (N, 7)
        preds = out.argmax(dim=1)

    mask = data["test_mask"]
    for true, pred in zip(data["labels"][mask].cpu().numpy(),
                          preds[mask].cpu().numpy()):
        confusion[true][pred] += 1

    return confusion


def plot_confusion_matrix(confusion: np.ndarray, save_dir: str = "./logs"):
    # 使用缩短的类别名以免重叠
    short_names = ["Case", "GA", "NN", "Prob", "RL", "Rule", "Theory"]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(confusion, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_yticklabels(short_names)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(confusion[i][j]),
                    ha="center", va="center",
                    color="white" if confusion[i][j] > confusion.max() * 0.5 else "black",
                    fontsize=9)

    ax.set_xlabel("Predicted" if not _HAS_CJK else "预测类别")
    ax.set_ylabel("True"      if not _HAS_CJK else "真实类别")
    ax.set_title("Confusion Matrix (Cora)" if not _HAS_CJK else "混淆矩阵（Cora）")

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
        if total > 0:
            print(f"  {cls:25s}: {correct:3d}/{total:3d}  ({100*correct/total:.1f}%)")
        else:
            print(f"  {cls:25s}: N/A")
    total_correct = np.trace(confusion)
    total_all     = confusion.sum()
    print(f"\n  总体准确率: {total_correct}/{total_all}  ({100*total_correct/total_all:.2f}%)")


def predict_samples(model, data, device, num_samples: int = 8):
    """随机抽取测试节点，展示预测结果"""
    model.eval()

    test_indices = data["test_mask"].nonzero(as_tuple=True)[0].cpu().numpy()
    sampled = np.random.choice(test_indices, size=num_samples, replace=False)

    with torch.no_grad():
        out   = model(data["features"], data["adj"])
        preds = out.argmax(dim=1)

    paper_ids = data.get("paper_ids", [])

    print(f"\n── 随机抽样预测（{num_samples} 个测试节点）──")
    correct_count = 0
    for node_idx in sampled:
        true_label = data["labels"][node_idx].item()
        pred_label = preds[node_idx].item()
        mark = "✓" if pred_label == true_label else "✗"
        if pred_label == true_label:
            correct_count += 1

        pid = paper_ids[node_idx] if paper_ids else str(node_idx)
        print(f"  [{mark}] 节点 {pid:>6s}  "
              f"真实: {CLASSES[true_label]:25s}  "
              f"预测: {CLASSES[pred_label]}")

    print(f"\n  抽样准确率: {correct_count}/{num_samples}")


def main():
    device = get_device(config.device)
    print(f"使用设备: {device}")

    data = load_cora(config.data_dir)
    data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()}

    model = GCNClassifier(
        in_features=config.in_features,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        num_layers=config.num_layers,
        dropout_rate=config.dropout_rate,
    ).to(device)

    if not os.path.exists(config.save_path):
        print(f"错误：模型文件不存在 {config.save_path}，请先运行 train.py")
        return

    model.load_state_dict(torch.load(config.save_path, map_location=device))
    print(f"模型参数总量: {count_parameters(model):,}")
    print(f"已加载模型: {config.save_path}\n")

    os.makedirs("./logs", exist_ok=True)

    confusion = evaluate_per_class(model, data, device)
    print_class_accuracy(confusion)
    plot_confusion_matrix(confusion)

    predict_samples(model, data, device)


if __name__ == "__main__":
    main()
