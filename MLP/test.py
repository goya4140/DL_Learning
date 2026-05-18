"""
测试脚本

功能：
    1. 加载已训练好的模型权重
    2. 在测试集上评估整体准确率 + 各类别准确率
    3. 生成混淆矩阵热力图（confusion_matrix.png）
    4. 对单张图片进行推理示例（含概率柱状图）
    5. 可视化 16 张预测样本（predictions.png）
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from config import config
from model import MLP
from dataset import get_mnist_loaders


# ── 中文字体检测（有则用中文，无则用英文，避免方块乱码）──────────────────────
def _setup_cjk_font() -> bool:
    cjk_fonts = ["PingFang SC", "Heiti SC", "STHeiti", "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in cjk_fonts:
        if font in available:
            matplotlib.rcParams["font.sans-serif"] = [font] + matplotlib.rcParams["font.sans-serif"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False

_HAS_CJK = _setup_cjk_font()


def get_device(preference: str = "auto") -> torch.device:
    """优先级：CUDA > MPS（Apple Silicon）> CPU"""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(preference)


# ── 1. 各类别准确率 ──────────────────────────────────────────────────────────

def evaluate_per_class(model, loader, device, num_classes=10):
    """统计并打印每个类别的预测准确率，同时返回混淆矩阵"""
    model.eval()
    # confusion[i][j]：真实类别 i 被预测为 j 的次数
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            for true, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion[true][pred] += 1

    print("\n各类别准确率：")
    print("-" * 30)
    for i in range(num_classes):
        correct = confusion[i][i]
        total   = confusion[i].sum()
        acc = correct / total * 100 if total > 0 else 0
        print(f"  数字 {i}: {acc:.2f}%  ({correct}/{total})")
    print("-" * 30)
    total_correct = np.trace(confusion)
    total_samples = confusion.sum()
    print(f"  总体准确率: {total_correct / total_samples * 100:.2f}%")

    return confusion


# ── 2. 混淆矩阵可视化 ────────────────────────────────────────────────────────

def plot_confusion_matrix(confusion: np.ndarray, save_path: str = "./confusion_matrix.png"):
    """将混淆矩阵绘制为热力图并保存"""
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(confusion, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    title = "Confusion Matrix (MLP - MNIST)" if not _HAS_CJK else "混淆矩阵（MLP - MNIST）"
    ax.set_title(title, fontsize=13, pad=12)
    xlabel = "Predicted Label" if not _HAS_CJK else "预测类别"
    ylabel = "True Label"      if not _HAS_CJK else "真实类别"
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    # 在每个格子里写数字，深色背景用白字
    thresh = confusion.max() / 2.0
    for i in range(10):
        for j in range(10):
            color = "white" if confusion[i, j] > thresh else "black"
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center",
                    fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"混淆矩阵已保存至: {save_path}")


# ── 3. 单张推理（含概率条形图）───────────────────────────────────────────────

def predict_single(model, image_tensor: torch.Tensor, device) -> int:
    """对单张图片推理，打印概率条形图，返回预测类别"""
    model.eval()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred   = int(probs.argmax())

    print(f"\n单张推理结果：预测类别 = {pred}")
    for cls, p in enumerate(probs):
        bar   = "█" * int(p * 30)
        mark  = " ←" if cls == pred else ""
        print(f"  {cls}: {p:.4f}  {bar}{mark}")

    return pred


# ── 4. 预测样本可视化（4×4，16 张）─────────────────────────────────────────

def visualize_predictions(model, loader, device, num_samples: int = 16,
                          save_path: str = "./predictions.png"):
    """可视化 16 张测试图片的预测结果（4×4 布局）"""
    model.eval()
    images_list, labels_list, preds_list = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            preds  = logits.argmax(dim=1).cpu()
            images_list.append(images.cpu())
            labels_list.append(labels.cpu())
            preds_list.append(preds)
            if sum(len(x) for x in images_list) >= num_samples:
                break

    images = torch.cat(images_list)[:num_samples]
    labels = torch.cat(labels_list)[:num_samples]
    preds  = torch.cat(preds_list)[:num_samples]

    cols, rows = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    sup_title = ("MLP Predictions  (green=correct  red=wrong)"
                 if not _HAS_CJK else
                 "MLP 预测结果（绿色=正确，红色=错误）")
    fig.suptitle(sup_title, fontsize=12)

    for i, ax in enumerate(axes.flat):
        img        = images[i].squeeze().numpy()
        true_label = labels[i].item()
        pred_label = preds[i].item()
        correct    = true_label == pred_label
        color      = "green" if correct else "red"

        ax.imshow(img, cmap="gray")
        if _HAS_CJK:
            ax.set_title(f"真: {true_label}  预: {pred_label}", color=color, fontsize=10)
        else:
            ax.set_title(f"T:{true_label}  P:{pred_label}", color=color, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"预测可视化已保存至: {save_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    device = get_device(config.device)
    print(f"使用设备: {device}")

    # 加载模型
    model = MLP(
        input_size=config.input_size,
        hidden_sizes=config.hidden_sizes,
        output_size=config.output_size,
        dropout_rate=config.dropout_rate,
    ).to(device)
    model.load_state_dict(torch.load(config.save_path, map_location=device))
    print(f"已加载模型: {config.save_path}")

    # MPS 不支持 pin_memory
    pin_memory = (device.type == "cuda")
    _, test_loader = get_mnist_loaders(
        config.data_dir, config.batch_size, config.num_workers, pin_memory=pin_memory
    )

    # 各类别准确率 + 混淆矩阵数据
    confusion = evaluate_per_class(model, test_loader, device)

    # 混淆矩阵可视化
    plot_confusion_matrix(confusion, save_path="./confusion_matrix.png")

    # 单张推理示例
    sample_images, sample_labels = next(iter(test_loader))
    print(f"\n抽取样本真实标签: {sample_labels[0].item()}")
    predict_single(model, sample_images[0], device)

    # 16 张预测可视化
    visualize_predictions(model, test_loader, device,
                          num_samples=16, save_path="./predictions.png")


if __name__ == "__main__":
    main()
