"""
MLP（多层感知机）模型定义

前向传播公式（第 l 层）：
    z^(l) = W^(l) · a^(l-1) + b^(l)       # 线性变换
    a^(l) = ReLU(z^(l))                    # 非线性激活（最后一层不加激活）
    a^(l) = Dropout(a^(l), p)              # 随机丢弃，仅在训练时生效

输出层使用 LogSoftmax，配合 NLLLoss 等价于 CrossEntropyLoss。
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, dropout_rate: float = 0.3):
        """
        参数：
            input_size:    输入特征维度（MNIST 为 784）
            hidden_sizes:  隐藏层列表，如 [512, 256, 128]
            output_size:   输出类别数（MNIST 为 10）
            dropout_rate:  Dropout 概率
        """
        super(MLP, self).__init__()

        # 动态构建网络层：输入层 → 若干隐藏层 → 输出层
        layer_sizes = [input_size] + hidden_sizes  # 拼接得到每层的输入/输出维度
        layers = []

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))  # 全连接层
            layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))              # 批归一化，稳定训练
            layers.append(nn.ReLU())                                        # 非线性激活
            layers.append(nn.Dropout(p=dropout_rate))                      # 随机丢弃

        # 输出层：线性变换，不加激活（CrossEntropyLoss 内部包含 Softmax）
        layers.append(nn.Linear(layer_sizes[-1], output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, 1, 28, 28)  ← MNIST 原始输入
        返回 logits: (batch_size, 10)
        """
        x = x.view(x.size(0), -1)  # 展平：(B, 1, 28, 28) → (B, 784)
        return self.network(x)


def count_parameters(model: nn.Module) -> int:
    """统计模型可训练参数总量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
