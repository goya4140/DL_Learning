"""
Transformer 文本分类模型（Encoder-only）

架构总览（以 AG News 为例，batch_size=B, seq_len=L, d_model=128）：

  输入 (B, L)
    → TokenEmbedding                  → (B, L, 128)
    → PositionalEncoding（相加）       → (B, L, 128)
    → TransformerEncoderLayer × 2     → (B, L, 128)
       每层内部：
         MultiHeadSelfAttention（4头）→ Add & LayerNorm
         FeedForward（128→256→128）   → Add & LayerNorm
    → 掩码均值池化（排除 <pad>）       → (B, 128)
    → Dropout
    → Linear(128, 4)                  → (B, 4)

【为什么用 Encoder-only 而非完整的 Encoder-Decoder】
  完整 Transformer（Encoder+Decoder）用于 seq2seq 任务（翻译、摘要）。
  文本分类只需提取全局语义向量，Encoder 已足够，是 BERT 的同类架构。

【掩码均值池化 vs 直接 .mean(dim=1)】
  直接 mean 会把 <pad> 位置的输出（值接近零但非严格零）也纳入平均，
  不同长度序列的有效信息会被不均匀稀释。
  掩码均值只对真实 token 位置取平均：
    pooled = (out * valid_mask).sum(1) / valid_mask.sum(1)

【src_key_padding_mask 的作用】
  传给 nn.TransformerEncoder，让 Self-Attention 在计算注意力权重时
  将 <pad> 位置的权重强制置为 -inf（softmax 后为 0），
  使 <pad> 不向其他位置传递信息。
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """标准正弦/余弦位置编码（Vaswani et al. 2017）

    将序列位置信息注入词嵌入，使模型感知词序。

    公式：
      PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    - 不同频率的正弦/余弦函数对应不同位置维度
    - 预先计算 (max_len, d_model) 的编码表，注册为 buffer（不参与梯度更新）
    - 加法融合：词嵌入 + 位置编码（与词义和位置各自解耦）
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预计算位置编码矩阵，shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        # 频率项：10000^(2i/d_model)，取对数再 exp 数值更稳定
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )                                                                      # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维：sin
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维：cos
        pe = pe.unsqueeze(0)                            # (1, max_len, d_model)

        # register_buffer：跟随模型 .to(device)，但不作为参数参与训练
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        x = x + self.pe[:, :x.size(1)]    # 广播相加，自动截取到当前序列长度
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int,
                 num_encoder_layers: int, d_ff: int, num_classes: int,
                 dropout_rate: float = 0.1, pad_idx: int = 1, max_len: int = 512):
        super().__init__()

        self.pad_idx = pad_idx
        self.d_model = d_model

        # 词嵌入：padding_idx=pad_idx 使 <pad> 的梯度为 0
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # 正弦位置编码
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout_rate, max_len=max_len)

        # Transformer Encoder
        # nn.TransformerEncoderLayer 内置：
        #   MultiHeadSelfAttention → Add & LayerNorm → FFN → Add & LayerNorm
        # batch_first=True：输入形状为 (B, L, d_model)，与本项目其他模型一致
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout_rate,
            activation="relu",
            batch_first=True,
            norm_first=False,   # Post-LN（原始论文），Pre-LN 训练更稳定但改变行为
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Xavier 均匀初始化线性层和嵌入层（改善收敛速度）"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) — token id 序列（已填充到相同长度）

        # 生成 padding mask：True 的位置在 Attention 中被忽略（置为 -inf）
        src_key_padding_mask = (x == self.pad_idx)   # (B, L), True = <pad>

        # 词嵌入 + 位置编码
        emb = self.pos_enc(self.embedding(x))         # (B, L, d_model)

        # Transformer Encoder（含多层 MHA + FFN）
        out = self.encoder(emb, src_key_padding_mask=src_key_padding_mask)  # (B, L, d_model)

        # 掩码均值池化：只对真实 token 取均值
        valid_mask = ~src_key_padding_mask              # (B, L), True = 真实 token
        valid_mask_f = valid_mask.unsqueeze(-1).float() # (B, L, 1)
        pooled = (out * valid_mask_f).sum(dim=1) / valid_mask_f.sum(dim=1)  # (B, d_model)

        return self.fc(self.dropout(pooled))            # (B, num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
