# 消息传递框架（Message Passing Neural Networks）

> 核心思想：将所有图神经网络统一为"聚合邻居消息 + 更新自身状态"的通用框架。

---

## 1. 消息传递框架（MPNN）

Gilmer et al. 2017 提出 MPNN 框架，将 GNN 的前向传播统一为两步：

```
① 消息（Message）：每条边发送消息
   m_{ij}^{l} = M^l(h_i^l, h_j^l, e_{ij})
   
   h_i, h_j：端点的特征，e_ij：边的特征（可选）
   M^l：消息函数（可学习，如 MLP）

② 聚合 + 更新（Aggregate + Update）：每个节点聚合邻居消息后更新自身
   h_i^{l+1} = U^l( h_i^l, AGG({ m_{ij}^l : j ∈ N(i) }) )
   
   AGG：聚合函数（sum / mean / max）
   U^l：更新函数（可学习，如 GRU 或 MLP）
```

绝大多数 GNN 都是 MPNN 的特例，区别仅在于消息函数 M、聚合函数 AGG、更新函数 U 的具体形式。

---

## 2. 各 GNN 模型在 MPNN 框架下的表达

### GCN（Kipf 2017）

```
消息：m_{ij} = h_j / √(d_i · d_j)        （邻接权重归一化）
聚合：AGG = 求和（含自身）
更新：h_i^{l+1} = ReLU( Σ_{j∈N(i)∪{i}} m_{ij} · W^l )

矩阵形式：H^{l+1} = σ( Ã H^l W^l )
```

特点：聚合权重由度数决定（**固定**，不依赖特征）。

### GAT（Veličković 2018）

```
注意力权重：α_{ij} = softmax_j( LeakyReLU(a^T [W h_i || W h_j]) )
消息：m_{ij} = α_{ij} · W h_j             （按注意力权重加权）
聚合：AGG = 加权求和
更新：h_i^{l+1} = σ( Σ_{j∈N(i)∪{i}} m_{ij} )

矩阵形式：H^{l+1} = σ( α ⊙ (H^l W^l) )，α 为注意力矩阵
```

特点：聚合权重由**特征决定**（可学习），允许区分重要/不重要的邻居。

### GraphSAGE（Hamilton 2017）

```
聚合：h_N(i)^l = MEAN/MAX/LSTM({ h_j^l : j ∈ SAMPLE(N(i)) })
更新：h_i^{l+1} = σ( W · [h_i^l || h_N(i)^l] )  （拼接而非求和）
```

特点：**邻居采样**（不用全图），支持归纳学习；**拼接**保留节点自身信息。

### GIN（Xu 2019）

```
更新：h_i^{l+1} = MLP^l( (1 + ε^l) · h_i^l + Σ_{j∈N(i)} h_j^l )
```

特点：理论最强（等价于 WL 图同构测试），ε 可学习。

---

## 3. 聚合函数对比

| 聚合函数 | 公式 | 特点 | 缺点 |
|---------|------|------|------|
| Sum | Σ h_j | 保留邻居数量信息 | 度数大的节点值域大 |
| Mean | Σ h_j / d | 归一化，稳定 | 丢失邻居数量信息 |
| Max | max(h_j) | 保留最显著特征 | 丢失大量信息 |
| LSTM | LSTM(h_j序列) | 强表达能力 | 需要节点有序排列 |
| Attention | Σ α_j h_j | 区分邻居重要性 | 参数量增加 |

**GCN 用的是 Mean 的变体**（对称归一化 = 加权平均）。

**GIN 用 Sum**（理论证明 Sum 聚合表达能力最强）。

---

## 4. 过平滑问题（Over-Smoothing）

### 现象

多层 GNN 后，所有节点的特征向量趋于相同：

```
层数 1：节点 v 聚合 1-hop 邻居
层数 2：节点 v 聚合 2-hop 邻居
层数 K：节点 v 聚合 K-hop 邻居

当 K 足够大时，几乎所有节点都在聚合整个连通分量的信息
→ 节点特征趋于全局均值
→ 无法区分不同节点 → 分类失败
```

### 理论解释

Li et al. 2018 证明：随层数增加，GCN 本质上在做图上的低通滤波（平滑操作），
最终收敛到 Â 的最大特征向量（不含区分性信息）。

### 缓解方案

```python
# 1. 残差连接（ResGCN）
h = gcn_layer(h, adj) + h_prev    # 保留原始特征，防止过度平滑

# 2. JumpingKnowledge（Xu 2018）
# 拼接所有层的输出，让最终分类器自己选择最有用的层
h_final = torch.cat([h1, h2, h3], dim=-1)

# 3. DropEdge（Rong 2020）
# 训练时随机丢弃部分边，相当于图上的 Dropout
mask = torch.rand(edge_index.shape[1]) > drop_prob
adj_dropped = adj * mask

# 4. PairNorm（Zhao 2020）
# 约束节点特征的总体分散度不下降
h = h - h.mean(dim=0)             # 中心化
h = h / (h.norm(dim=1, keepdim=True).mean() + 1e-8)  # 归一化
```

---

## 5. 可扩展性：大图上的挑战

| 方法 | 策略 | 适用场景 |
|------|------|---------|
| GCN（full-batch） | 全图一次前向传播 | 小图（<10万节点） |
| GraphSAGE（邻居采样） | 每节点采样固定数量邻居 | 中等图 |
| ClusterGCN（图分割） | 将图分成子图，按子图 mini-batch | 大图（百万节点） |
| GraphSAINT（子图采样） | 随机采样子图 | 大图 |

本项目使用全图（full-batch）方式，适合 Cora（2708 节点）这样的小图。

---

## 6. GCN → Transformer：统一视角

Self-Attention（Transformer）可以看作**全连接图上的 GAT**：

```
GAT：   h_i^{l+1} = Σ_{j∈N(i)} α_{ij} · W h_j
         （只聚合有边的邻居，α 由特征决定）

Self-Attention：
        h_i^{l+1} = Σ_{j=1}^{N} α_{ij} · W h_j
         （聚合所有节点，α = softmax(QK^T/√d)）
```

区别：Self-Attention 假设完全图（每个 token 与所有其他 token 相连），
     GAT/GCN 只聚合实际存在的邻居（稀疏图）。

Graph Transformer 系列（如 Graphormer）结合了两者的优势：
既利用图的稀疏结构（节省计算），又使用 Attention 的动态权重。

---

## 7. PyTorch 中的实现模式

```python
# 完整的单层消息传递（以 GCN 为例）
import torch
import torch.nn.functional as F

def gcn_layer(x, adj, weight):
    # x:      (N, in_feat)
    # adj:    (N, N) 归一化邻接矩阵 Ã
    # weight: (in_feat, out_feat)

    # Step 1: 线性变换（消息生成）
    support = x @ weight          # (N, out_feat)

    # Step 2: 邻居聚合（矩阵乘法 = 加权求和）
    out = adj @ support            # (N, out_feat)

    return out

# 多层叠加
for layer in gcn_layers[:-1]:
    x = F.relu(layer(x, adj))
    x = F.dropout(x, p=dropout, training=model.training)
x = gcn_layers[-1](x, adj)        # 最后一层不加激活
out = F.log_softmax(x, dim=1)
```
