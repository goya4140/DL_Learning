"""
RNN 情感/文本分类模型

维度变化总览：
  输入 (B, L)
    → Embedding                       → (B, L, E)
    → nn.RNN（batch_first=True）      → output (B, L, H), hidden (num_layers, B, H)
    → 取最后一层最终隐藏状态 hidden[-1]  → (B, H)
    → Dropout
    → FC: Linear(H, output_size)      → (B, output_size)

B = batch_size, L = 序列长度, E = embed_dim, H = hidden_size
"""

import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 output_size: int, num_layers: int = 2,
                 dropout_rate: float = 0.5, pad_idx: int = 1):
        super().__init__()

        # 词嵌入层：将离散 token id 映射到稠密向量
        # padding_idx=pad_idx：<pad> 位置的梯度为 0，不更新
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # 核心 RNN 层
        # h_t = tanh(x_t @ W_ih.T + b_ih + h_{t-1} @ W_hh.T + b_hh)
        # dropout 只作用于层间（num_layers > 1 时），最后一层不加
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            nonlinearity="tanh",   # 也可以用 "relu"，但 tanh 是经典设置
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) — token id 序列（已填充到相同长度）

        embedded = self.dropout(self.embedding(x))    # (B, L, E)

        # output: 所有时间步的隐藏状态 (B, L, H)
        # hidden: 最后一个时间步的隐藏状态 (num_layers, B, H)
        _, hidden = self.rnn(embedded)

        # 取最顶层（最后一层）的最终隐藏状态作为序列表示
        last_hidden = hidden[-1]                      # (B, H)

        return self.fc(self.dropout(last_hidden))     # (B, output_size)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
