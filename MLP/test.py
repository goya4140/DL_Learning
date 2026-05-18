"""
测试脚本

功能：
    1. 加载已训练好的模型权重
    2. 在测试集上评估整体准确率
    3. 打印每个类别（0-9）的单独准确率
    4. 对单张图片进行推理示例
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无显示器环境下使用非交互后端
import matplotlib.pyplot as plt

from config import config
from model import MLP
from dataset import get_mnist_loaders


def evaluate_per_class(model, loader, device, num_classes=10):
    """统计每个类别的预测准确率"""
    model.eval()
    class_correct = [0] * num_classes
    class_total   = [0] * num_classes

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)

            for label, pred in zip(labels, preds):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

    print("\n各类别准确率：")
    print("-" * 30)
    for i in range(num_classes):
        acc = class_correct[i] / class_total[i] * 100 if class_total[i] > 0 else 0
        print(f"  数字 {i}: {acc:.2f}%  ({class_correct[i]}/{class_total[i]})")
    print("-" * 30)
    total_acc = sum(class_correct) / sum(class_total) * 100
    print(f"  总体准确率: {total_acc:.2f}%")


def predict_single(model, image_tensor: torch.Tensor, device) -> int:
    """
    对单张图片进行推理，返回预测类别。

    image_tensor: 形状 (1, 28, 28) 或 (1, 1, 28, 28)，已归一化
    """
    model.eval()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # 增加 batch 维度

    with torch.no_grad():
        logits = model(image_tensor.to(device))          # (1, 10)
        probabilities = torch.softmax(logits, dim=1)     # 转为概率分布
        pred = probabilities.argmax(dim=1).item()

    print(f"\n单张推理结果：")
    print(f"  预测类别: {pred}")
    print(f"  各类别概率：")
    for cls, prob in enumerate(probabilities[0]):
        bar = "█" * int(prob.item() * 30)
        print(f"    {cls}: {prob.item():.4f}  {bar}")

    return pred


def visualize_predictions(model, loader, device, num_samples=10, save_path="./predictions.png"):
    """可视化前 num_samples 张测试图片的预测结果，保存为图片"""
    model.eval()
    images_list, labels_list, preds_list = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu()
            images_list.append(images)
            labels_list.append(labels)
            preds_list.append(preds)
            if sum(len(x) for x in images_list) >= num_samples:
                break

    images = torch.cat(images_list)[:num_samples]
    labels = torch.cat(labels_list)[:num_samples]
    preds  = torch.cat(preds_list)[:num_samples]

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("MLP 预测结果（绿色=正确，红色=错误）", fontsize=13)

    for i, ax in enumerate(axes.flat):
        img = images[i].squeeze().numpy()
        true_label = labels[i].item()
        pred_label = preds[i].item()
        color = "green" if true_label == pred_label else "red"
        ax.imshow(img, cmap="gray")
        ax.set_title(f"真: {true_label}  预: {pred_label}", color=color, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"\n预测可视化已保存至: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # 加载测试集
    _, test_loader = get_mnist_loaders(config.data_dir, config.batch_size, config.num_workers)

    # 各类别准确率
    evaluate_per_class(model, test_loader, device)

    # 单张推理示例（取测试集第一张图）
    sample_image, sample_label = next(iter(test_loader))
    print(f"\n抽取样本真实标签: {sample_label[0].item()}")
    predict_single(model, sample_image[0], device)

    # 可视化
    visualize_predictions(model, test_loader, device, save_path="./predictions.png")


if __name__ == "__main__":
    main()
