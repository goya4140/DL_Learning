"""
测试脚本

功能：
    1. 加载已训练好的模型权重
    2. 在测试集上评估整体准确率 + 各类别准确率
    3. 生成混淆矩阵热力图（confusion_matrix.png）
    4. 对单张图片进行推理示例（含概率条形图）
    5. 可视化 16 张预测样本（predictions.png，彩色图像）
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from config import config
from model import CNN
from dataset import get_cifar10_loaders, CIFAR10_CLASSES


# ── 中文字体检测 ──────────────────────────────────────────────────────────────
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


def _denormalize(images: torch.Tensor) -> np.ndarray:
    """将归一化的图像张量还原为可显示的 [0,1] 数组，形状 (N,H,W,C)"""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
    imgs = images * std + mean
    imgs = imgs.clamp(0, 1)
    # (B,C,H,W) → (B,H,W,C)
    return imgs.permute(0, 2, 3, 1).numpy()


# ── 1. 各类别准确率 ──────────────────────────────────────────────────────────

def evaluate_per_class(model, loader, device, num_classes=10):
    """统计每个类别的准确率，返回混淆矩阵"""
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            for true, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion[true][pred] += 1

    print("\n各类别准确率：")
    print("-" * 42)
    for i, cls_name in enumerate(CIFAR10_CLASSES):
        correct = confusion[i][i]
        total   = confusion[i].sum()
        acc = correct / total * 100 if total > 0 else 0
        print(f"  {cls_name:12s}: {acc:.2f}%  ({correct}/{total})")
    print("-" * 42)
    total_correct = np.trace(confusion)
    total_samples = confusion.sum()
    print(f"  总体准确率: {total_correct / total_samples * 100:.2f}%")

    return confusion


# ── 2. 混淆矩阵可视化 ────────────────────────────────────────────────────────

def plot_confusion_matrix(confusion: np.ndarray, save_path: str = "./confusion_matrix.png"):
    """将混淆矩阵绘制为热力图并保存"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    title = "Confusion Matrix (CNN - CIFAR-10)" if not _HAS_CJK else "混淆矩阵（CNN - CIFAR-10）"
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label" if not _HAS_CJK else "真实类别", fontsize=11)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CIFAR10_CLASSES, fontsize=9)

    thresh = confusion.max() / 2.0
    for i in range(10):
        for j in range(10):
            color = "white" if confusion[i, j] > thresh else "black"
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center",
                    fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"混淆矩阵已保存至: {save_path}")


# ── 3. 单张推理（含概率条形图）───────────────────────────────────────────────

def predict_single(model, image_tensor: torch.Tensor, device) -> int:
    """对单张图片推理，打印各类别概率，返回预测类别"""
    model.eval()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred   = int(probs.argmax())

    print(f"\n单张推理结果：预测类别 = {CIFAR10_CLASSES[pred]} ({pred})")
    for cls_idx, (cls_name, p) in enumerate(zip(CIFAR10_CLASSES, probs)):
        bar  = "█" * int(p * 30)
        mark = " ←" if cls_idx == pred else ""
        print(f"  {cls_name:12s}: {p:.4f}  {bar}{mark}")

    return pred


# ── 4. 预测样本可视化（4×4，16 张彩色图）────────────────────────────────────

def visualize_predictions(model, loader, device, num_samples: int = 16,
                          save_path: str = "./predictions.png"):
    """可视化 16 张测试图片的预测结果（彩色，4×4 布局）"""
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

    # 反归一化为可视化的 RGB 数组
    imgs_np = _denormalize(images)

    cols, rows = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    fig.suptitle("CNN Predictions on CIFAR-10  (green=correct  red=wrong)",
                 fontsize=12)

    for i, ax in enumerate(axes.flat):
        true_label = labels[i].item()
        pred_label = preds[i].item()
        correct    = true_label == pred_label
        color      = "green" if correct else "red"

        ax.imshow(imgs_np[i])
        true_name = CIFAR10_CLASSES[true_label]
        pred_name = CIFAR10_CLASSES[pred_label]
        ax.set_title(f"T:{true_name}\nP:{pred_name}", color=color, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"预测可视化已保存至: {save_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    device = get_device(config.device)
    print(f"使用设备: {device}")

    model = CNN(
        in_channels=config.in_channels,
        channels=config.channels,
        fc_hidden=config.fc_hidden,
        output_size=config.output_size,
        dropout_rate=config.dropout_rate,
    ).to(device)
    model.load_state_dict(torch.load(config.save_path, map_location=device))
    print(f"已加载模型: {config.save_path}")

    pin_memory = (device.type == "cuda")
    _, test_loader = get_cifar10_loaders(
        config.data_dir, config.batch_size, config.num_workers, pin_memory=pin_memory
    )

    confusion = evaluate_per_class(model, test_loader, device)
    plot_confusion_matrix(confusion, save_path="./confusion_matrix.png")

    sample_images, sample_labels = next(iter(test_loader))
    print(f"\n抽取样本真实标签: {CIFAR10_CLASSES[sample_labels[0].item()]} ({sample_labels[0].item()})")
    predict_single(model, sample_images[0], device)

    visualize_predictions(model, test_loader, device,
                          num_samples=16, save_path="./predictions.png")


if __name__ == "__main__":
    main()
