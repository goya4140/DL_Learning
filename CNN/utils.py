"""
实验日志工具

训练过程中自动记录每个 epoch 的核心指标，训练结束后：
  1. 导出 CSV：logs/metrics.csv
  2. 生成训练曲线图：logs/training_curves.png
       左图：训练损失 vs 测试损失
       右图：训练准确率 vs 测试准确率
     图中用竖虚线标注学习率衰减的 epoch
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _setup_cjk_font():
    """尝试加载系统中文字体，成功返回 True，否则返回 False"""
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


class ExperimentLogger:
    """
    记录训练过程中每个 epoch 的指标，并在训练结束后生成可视化报告。

    用法：
        logger = ExperimentLogger(log_dir="./logs", lr_step_size=10)
        # 每个 epoch 结束后
        logger.log(epoch, train_loss, train_acc, val_loss, val_acc, lr)
        # 训练全部结束后
        logger.save()   # 保存 CSV
        logger.plot()   # 生成训练曲线图
    """

    def __init__(self, log_dir: str = "./logs", lr_step_size: int = 10):
        self.log_dir = log_dir
        self.lr_step_size = lr_step_size
        os.makedirs(log_dir, exist_ok=True)

        self.records: list[dict] = []

    def log(self, epoch: int, train_loss: float, train_acc: float,
            val_loss: float, val_acc: float, lr: float):
        self.records.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  4),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    4),
            "lr":         lr,
        })

    def save(self):
        """将所有 epoch 指标保存为 CSV 文件"""
        if not self.records:
            return
        csv_path = os.path.join(self.log_dir, "metrics.csv")
        fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
        print(f"训练日志已保存至: {csv_path}")

    def plot(self):
        """生成损失和准确率双图并保存"""
        if not self.records:
            return

        epochs     = [r["epoch"]      for r in self.records]
        train_loss = [r["train_loss"] for r in self.records]
        val_loss   = [r["val_loss"]   for r in self.records]
        train_acc  = [r["train_acc"]  for r in self.records]
        val_acc    = [r["val_acc"]    for r in self.records]

        decay_epochs = [e for e in epochs if e % self.lr_step_size == 0 and e != epochs[-1]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("CNN Training Curves (CIFAR-10)" if not _HAS_CJK else "CNN 训练曲线（CIFAR-10）",
                     fontsize=14, fontweight="bold")

        ax1.plot(epochs, train_loss, "b-o", markersize=4, label="Train Loss" if not _HAS_CJK else "训练损失")
        ax1.plot(epochs, val_loss,   "r-o", markersize=4, label="Val Loss"   if not _HAS_CJK else "测试损失")
        for de in decay_epochs:
            ax1.axvline(x=de, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curve" if not _HAS_CJK else "损失曲线")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, train_acc, "b-o", markersize=4, label="Train Acc" if not _HAS_CJK else "训练准确率")
        ax2.plot(epochs, val_acc,   "r-o", markersize=4, label="Val Acc"   if not _HAS_CJK else "测试准确率")
        for de in decay_epochs:
            ax2.axvline(x=de, color="gray", linestyle="--", alpha=0.5, linewidth=1,
                        label="LR Decay" if de == decay_epochs[0] else "")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Accuracy Curve" if not _HAS_CJK else "准确率曲线")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        best_val_acc = max(val_acc)
        best_epoch   = epochs[val_acc.index(best_val_acc)]
        ax2.annotate(f"Best: {best_val_acc:.2f}%",
                     xy=(best_epoch, best_val_acc),
                     xytext=(best_epoch + 0.5, best_val_acc - 1.0),
                     arrowprops=dict(arrowstyle="->", color="red"),
                     color="red", fontsize=9)

        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=130)
        plt.close()
        print(f"训练曲线已保存至: {plot_path}")
