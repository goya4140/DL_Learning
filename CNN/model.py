"""
CNN（卷积神经网络）模型定义

前向传播流程：
    输入 (B, 3, 32, 32)
      → [Conv → BN → ReLU → MaxPool] × 3   # 特征提取
      → Flatten                              # (B, 128*4*4) = (B, 2048)
      → Linear → ReLU → Dropout             # 分类头
      → Linear (logits)                     # (B, 10)

卷积后特征图尺寸公式：
    H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1
    使用 3×3 卷积 + padding=1 保持尺寸不变；MaxPool(2×2) 将尺寸减半。
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels: int, channels: list, fc_hidden: int,
                 output_size: int, dropout_rate: float = 0.5):
        """
        参数：
            in_channels:  输入通道数（CIFAR-10 RGB 图像为 3）
            channels:     各卷积块输出通道数列表，如 [32, 64, 128]
            fc_hidden:    全连接隐藏层维度
            output_size:  分类数（CIFAR-10 为 10）
            dropout_rate: FC 层后的 Dropout 比例
        """
        super(CNN, self).__init__()

        # ── 特征提取部分（卷积块序列）───────────────────────────────────────
        conv_blocks = []
        prev_ch = in_channels
        for out_ch in channels:
            conv_blocks += [
                nn.Conv2d(prev_ch, out_ch, kernel_size=3, padding=1, bias=False),
                # bias=False：后接 BatchNorm，BN 的 β 参数替代了 bias 的作用
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),   # 尺寸减半
            ]
            prev_ch = out_ch

        self.features = nn.Sequential(*conv_blocks)

        # 经过 3 次 MaxPool(2×2)，32×32 → 16×16 → 8×8 → 4×4
        # 最后 feature map 尺寸：channels[-1] × 4 × 4
        flatten_dim = channels[-1] * 4 * 4

        # ── 分类头（全连接）────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fc_hidden, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, 32, 32)
        返回 logits: (B, 10)
        """
        x = self.features(x)              # (B, 128, 4, 4)
        x = x.view(x.size(0), -1)        # 展平：(B, 2048)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """统计模型可训练参数总量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
