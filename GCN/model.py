"""
GCN 图卷积神经网络

维度变化总览（以 Cora 为例）：
  输入 X: (N, 1433)  — N=2708 个节点，每个节点 1433 维词袋特征
  Ã:      (N, N)     — 预计算的归一化邻接矩阵（D̂^{-1/2} Â D̂^{-1/2}）

  Layer 1: GraphConvLayer(1433, 64)
    support = X @ W1          → (N, 64)   先做特征线性变换
    H1      = Ã @ support     → (N, 64)   再做邻居聚合
    H1      = ReLU(H1) + Dropout

  Layer 2: GraphConvLayer(64, 7)
    support = H1 @ W2         → (N, 7)
    H2      = Ã @ support     → (N, 7)
    out     = log_softmax(H2) → (N, 7)    用于 NLLLoss

【为什么先 X@W 再 Ã@result（而非 Ã@X@W）】
  数学上等价（矩阵乘法结合律），但计算顺序不同：
  - 先 X@W：(N,1433)×(1433,64) = (N,64)，再 Ã@(N,64) → 计算量较小
  - 先 Ã@X：(N,N)×(N,1433) = (N,1433)，再 ×W → 计算量大
  特征维度 1433 >> 64，先降维再聚合效率更高。

【邻接矩阵用 dense FloatTensor（而非 sparse）】
  Cora 图仅 2708 节点，dense 矩阵约 29MB，完全可接受。
  MPS（Apple Silicon）对稀疏张量支持有限，dense 更兼容。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """单层图卷积：H_out = Ã @ (H_in @ W) + b"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # 线性变换（不在此加 bias，等聚合后再加，减少计算量）
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias   = nn.Parameter(torch.FloatTensor(out_features))
        self._reset_parameters()

    def _reset_parameters(self):
        # Glorot uniform 初始化（与 Kipf 原始实现一致）
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x:   (N, in_features)
        # adj: (N, N)  归一化邻接矩阵 Ã
        support = x @ self.weight            # (N, out_features)：特征线性变换
        out     = adj @ support + self.bias  # (N, out_features)：邻居聚合
        return out


class GCNClassifier(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 2, dropout_rate: float = 0.5):
        super().__init__()
        assert num_layers >= 2, "GCN 至少需要 2 层（输入层 + 输出层）"

        self.dropout_rate = dropout_rate

        layers = []
        # 输入层
        layers.append(GraphConvLayer(in_features, hidden_dim))
        # 中间层
        for _ in range(num_layers - 2):
            layers.append(GraphConvLayer(hidden_dim, hidden_dim))
        # 输出层
        layers.append(GraphConvLayer(hidden_dim, num_classes))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x:   (N, in_features)
        # adj: (N, N)

        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # 最后一层：无激活函数，直接输出 logits
        x = self.layers[-1](x, adj)

        # log_softmax 与 NLLLoss 搭配（等价于 CrossEntropyLoss，数值更稳定）
        return F.log_softmax(x, dim=1)   # (N, num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
