"""
LSTM 文本分类模型

维度变化总览（以双向 LSTM 为例，bidirectional=True）：
  输入 (B, L)
    → Embedding                              → (B, L, E)
    → nn.LSTM（batch_first=True, bidirec.）  → output (B, L, 2H)
    → 所有时间步均值池化 output.mean(1)       → (B, 2H)
    → Dropout
    → FC: Linear(2H, output_size)            → (B, output_size)

单向时（bidirectional=False）：
  output (B, L, H) → 均值池化 (B, H) → FC (B, output_size)

B = batch_size, L = 序列长度, E = embed_dim, H = hidden_size

【LSTM vs vanilla RNN：为什么 LSTM 能处理长序列】

vanilla RNN 的隐藏状态递推：
  h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b)
  梯度反传时：∂h_t/∂h_{t-1} = W_hh · diag(1 - tanh²(·))
  多步连乘后，该项指数收缩（梯度消失）或指数增大（梯度爆炸）

LSTM 增加了细胞状态 c_t（长期记忆）作为"高速公路"：
  遗忘门：f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  ← 决定忘记多少旧记忆
  输入门：i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  ← 决定写入多少新信息
  候选值：g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)
  细胞更新：c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t   ← 加法更新！梯度直通
  输出门：o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
  输出：  h_t = o_t ⊙ tanh(c_t)

关键：c_t 的梯度路径是加法（c_t = f_t⊙c_{t-1} + …），不像 RNN 是乘法连乘，
      梯度近似直接流过所有时间步（类似 ResNet 的残差连接），梯度消失大幅减轻。

【双向 LSTM（BiLSTM）】
  正向 LSTM：h_t^→ = LSTM(x_t, h_{t-1}^→)  （从左到右）
  反向 LSTM：h_t^← = LSTM(x_t, h_{t+1}^←)  （从右到左）
  拼接输出：h_t = [h_t^→; h_t^←]             维度从 H 变为 2H

  双向建模允许每个位置同时看到左右上下文，对分类任务通常有 1-3% 的提升。
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 output_size: int, num_layers: int = 2,
                 dropout_rate: float = 0.5, pad_idx: int = 1,
                 bidirectional: bool = True):
        super().__init__()

        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # nn.LSTM 在 num_layers > 1 时，dropout 作用于层间连接（最后一层不加）
        # 当 bidirectional=True：
        #   输出 output shape: (B, L, 2*hidden_size)
        #   hidden/cell shape: (num_layers * 2, B, hidden_size)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout_rate)
        # 双向时特征维度翻倍
        self.fc = nn.Linear(hidden_size * num_directions, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L)

        embedded = self.dropout(self.embedding(x))    # (B, L, E)

        # output: (B, L, H*num_directions) — 所有时间步的隐藏状态
        # hidden: (num_layers*num_dir, B, H) — 最后时间步
        # cell:   (num_layers*num_dir, B, H) — 最后时间步的细胞状态
        output, (hidden, cell) = self.lstm(embedded)

        # 均值池化：梯度从所有时间步均匀流回，与 RNN 版保持一致的架构选择
        pooled = output.mean(dim=1)                   # (B, H*num_dir)

        return self.fc(self.dropout(pooled))          # (B, output_size)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
