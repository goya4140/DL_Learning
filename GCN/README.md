# GCN 图卷积神经网络

> 任务：Cora 引文网络节点分类（7 类机器学习子领域）  
> 数据集：2,708 节点，5,429 边，训练仅用 140 个节点（5.2%）  
> 目标准确率：**~81-82%**（2 层 GCN，200 epoch，Kipf 2017 基线）

---

## 目录

1. [为什么需要 GCN](#1-为什么需要-gcn)
2. [图的基本概念](#2-图的基本概念)
3. [GCN 核心数学推导](#3-gcn-核心数学推导)
4. [对称归一化的意义](#4-对称归一化的意义)
5. [架构总览](#5-架构总览)
6. [PyTorch 实现解析](#6-pytorch-实现解析)
7. [转导学习 vs 归纳学习](#7-转导学习-vs-归纳学习)
8. [项目结构](#8-项目结构)
9. [Quick Start](#9-quick-start)
10. [超参数调优指南](#10-超参数调优指南)
11. [GCN 局限性与后续发展](#11-gcn-局限性与后续发展)

---

## 1. 为什么需要 GCN

### 图数据的独特挑战

现实世界中大量数据天然具有**图结构**：
- 社交网络：人 → 节点，关注关系 → 边
- 引文网络：论文 → 节点，引用 → 边
- 分子结构：原子 → 节点，化学键 → 边
- 知识图谱：实体 → 节点，关系 → 边

这些数据的关键特点：**节点之间不独立**，邻居节点共享信息。

### 用 MLP 处理图数据的缺陷

如果忽略图结构，直接把每个节点的特征向量输入 MLP：

```
节点 v 的特征 x_v ∈ ℝ^{1433}
MLP(x_v) → 预测类别

问题：
  1. 忽略了邻居信息（引文关系包含丰富的语义信号）
  2. 节点之间完全独立，无法传递图结构信息
  3. 忽略邻居 = 放弃图数据中最有价值的部分
```

**实验对比**（Cora 数据集）：
| 模型 | 使用图结构 | 准确率 |
|------|---------|--------|
| MLP（仅节点特征） | ✗ | ~55-60% |
| GCN | ✓ | ~81-82% |
| GAT | ✓ | ~83-84% |

### GCN 的核心思想：邻居聚合

GCN 让每个节点**聚合自身及邻居的特征**：

```
更新前：节点 v 只知道自己的特征 x_v
更新后：节点 v 知道自己 + 所有邻居的特征（加权平均）
```

迭代 K 层 GCN 后，每个节点可以感知 K 跳（K-hop）邻居的信息。

---

## 2. 图的基本概念

### 基本定义

图 G = (V, E)，N = |V| 个节点，M = |E| 条边。

**邻接矩阵** A ∈ {0,1}^{N×N}：
```
A_ij = 1  若节点 i 和 j 之间有边
A_ij = 0  否则
```

无向图中 A 是对称矩阵（A = A^T）。Cora 是无向图（引用视为双向）。

**度矩阵** D ∈ ℝ^{N×N}（对角矩阵）：
```
D_ii = Σ_j A_ij  （节点 i 的度数，即邻居数量）
D_ij = 0  （i ≠ j）
```

**图拉普拉斯矩阵** L = D - A：
```
L_ii =  d_i  （度数）
L_ij = -1    （若 i,j 相邻）
L_ij =  0    （否则）
```

L 是半正定矩阵，其特征值（谱）捕获图的全局结构信息。

### Cora 数据集结构

```
论文 A（"神经网络"类）
  ├── 特征：[0,1,0,1,...,0]  ← 1433 维词袋（词汇表中的词是否出现）
  └── 引用：论文 B, 论文 D   ← 边

论文 B（"强化学习"类）
  ├── 特征：[1,0,0,0,...,1]
  └── 引用：论文 A, 论文 C

→ 相似领域的论文倾向于互相引用（图结构 = 标签信号）
```

---

## 3. GCN 核心数学推导

### 3.1 从谱图卷积到 GCN

**图上的卷积**无法像图像那样直接用滑动窗口（图没有规则网格结构），但可以在**谱域**定义：

```
谱域卷积：x * g = U · (Û ⊙ g(Λ)) · U^T · x
  U：图拉普拉斯矩阵 L 的特征向量矩阵（傅里叶基）
  Λ：对应特征值（频率）
  g：滤波器函数
```

问题：U 的计算复杂度 O(N²)，对大图不可行。

### 3.2 Chebyshev 多项式近似

Hammond et al. 2011 用 K 阶 Chebyshev 多项式近似滤波器：
```
g(Λ) ≈ Σ_{k=0}^{K} θ_k T_k(Λ̃)
其中 Λ̃ = 2Λ/λ_max - I（归一化特征值到 [-1,1]）
```

这把 O(N²) 降到 O(KM)（K 是阶数，M 是边数）。

### 3.3 Kipf & Welling 的一阶近似（2017 ICLR）

取 K=1，λ_max≈2，忽略高阶项，化简得：

```
H^{l+1} = σ( (D^{-1/2} A D^{-1/2} + I) H^l W^l )
```

进一步，将自环合并（`Â = A + I`，`D̂ = diag(Â·1)`）：

```
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   H^{l+1} = σ( D̂^{-1/2} Â D̂^{-1/2}  H^l  W^l )       ║
║                                                           ║
║   Â = A + I        （加自环）                             ║
║   D̂ = diag(Â·1)   （Â 的度矩阵）                        ║
║   W^l ∈ ℝ^{F_l × F_{l+1}}  （可学习权重）                ║
║   σ = ReLU（中间层）/ log_softmax（输出层）               ║
╚═══════════════════════════════════════════════════════════╝
```

**关键简化意义**：

预先计算 `Ã = D̂^{-1/2} Â D̂^{-1/2}`（一次性，不需梯度），
训练时每层只需一次矩阵乘法 `Ã @ H @ W`，计算高效。

### 3.4 逐节点理解

对节点 v，第 l 层更新：

```
h_v^{l+1} = σ( Σ_{u ∈ N(v) ∪ {v}} (D̂_vv · D̂_uu)^{-1/2} · h_u^l · W^l )
```

直觉：对 v 的所有邻居（含自身），用 `1/√(d_v·d_u)` 加权后求和，再线性变换。

权重 `1/√(d_v·d_u)` 的含义：
- 度数大的节点贡献权重小（防止高度数节点主导）
- 既考虑目标节点的度数（接收方平滑），也考虑源节点的度数（发送方平滑）

---

## 4. 对称归一化的意义

### 三种归一化方式对比

| 方式 | 公式 | 特点 |
|------|------|------|
| 无归一化 | A | 度数大的节点聚合量爆炸，梯度不稳定 |
| 行归一化 | D^{-1}A | 均值聚合，破坏对称性（图谱性质改变） |
| 对称归一化 | D^{-1/2}AD^{-1/2} | **保持对称**，特征值在 [-1,1]，训练稳定 |

**为什么加自环（A + I）**：
- 没有自环时，节点聚合时不包含自身特征
- 加 I 后，每个节点至少保留自身信息（防止特征被邻居完全覆盖）
- 改变了度数：原来度数为 0 的孤立节点现在有度数 1

---

## 5. 架构总览

### Cora 节点特征聚合示意

```
                    邻居聚合过程（第 1 层）
                         ↓
       论文 B  ──┐
       论文 D  ──┤  对称归一化加权  ┌──→  论文 A 新特征 (64维)
       论文 A ───┘  Σ 1/√(d_i·d_j) └──   ReLU + Dropout
       （自身）

                    第 2 层：再次聚合（2-hop 信息）
                         ↓
                    log_softmax → 7 类概率
```

### 完整前向传播维度图

```
输入: 节点特征矩阵
┌──────────────────────────────────────────────────────────────┐
│  X: (2708, 1433)  — 每行是一个论文节点的词袋特征              │
└──────────────────────┬───────────────────────────────────────┘
                       │
              预计算（不参与训练）
┌──────────────────────▼───────────────────────────────────────┐
│  Ã = D̂^{-1/2} (A+I) D̂^{-1/2}                              │
│  shape: (2708, 2708)  dense FloatTensor（约 29MB）           │
└──────────────────────┬───────────────────────────────────────┘
                       │
    ┌──────────────────▼──────────────────────────────────────┐
    │           GraphConvLayer 1  (1433 → 64)                 │
    │  support = X @ W1        (2708,1433)×(1433,64)=(2708,64)│
    │  H1      = Ã @ support   (2708,2708)×(2708,64)=(2708,64)│
    │  H1      = ReLU(H1 + b1)                                │
    │  H1      = Dropout(H1, p=0.5)                           │
    └──────────────────┬──────────────────────────────────────┘
                       │ (2708, 64)
    ┌──────────────────▼──────────────────────────────────────┐
    │           GraphConvLayer 2  (64 → 7)                    │
    │  support = H1 @ W2       (2708,64)×(64,7) = (2708,7)   │
    │  H2      = Ã @ support   (2708,2708)×(2708,7)=(2708,7) │
    │  out     = log_softmax(H2 + b2)                         │
    └──────────────────┬──────────────────────────────────────┘
                       │ (2708, 7)
                       │
    ┌──────────────────▼──────────────────────────────────────┐
    │  NLLLoss 仅计算 train_mask（140 个训练节点）的损失        │
    │  梯度通过邻居聚合传播到所有节点（含未标注节点）           │
    └─────────────────────────────────────────────────────────┘
```

---

## 6. PyTorch 实现解析

### 6.1 GraphConvLayer：为何用手写参数而非 nn.Linear

```python
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias   = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)   # Glorot 初始化
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = x @ self.weight     # 先降维（高效）
        out     = adj @ support       # 再聚合
        return out + self.bias
```

用 `nn.Parameter` 手写而非 `nn.Linear`，原因：
- GCN 的线性变换（`x @ W`）在聚合之前，而 `nn.Linear` 假设 `W @ x + b`（输入在右）
- 可以灵活控制是否在聚合前/后加 bias
- 更清楚展示 GCN 的计算逻辑

等价用法（若使用 nn.Linear）：
```python
support = self.fc(x)      # fc = nn.Linear(in_feat, out_feat, bias=False)
out = adj @ support + self.bias
```

### 6.2 邻接矩阵预计算（dataset.py）

```python
def _normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])          # Â = A + I
    degree = adj.sum(axis=1)                  # 每个节点的度（含自环）
    d_inv_sqrt = degree ** -0.5               # D̂^{-1/2} 对角元素
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0   # 孤立节点（度=0）置0
    # D̂^{-1/2} @ Â @ D̂^{-1/2}（广播实现对角矩阵乘法）
    return d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
```

`d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]` 等价于完整矩阵乘 `D@A@D`，但避免了显式构建对角矩阵，节省内存。

### 6.3 为何用 NLLLoss 而非 CrossEntropyLoss

```python
# 模型输出 log_softmax（已取对数）
out = F.log_softmax(x, dim=1)

# NLLLoss 期望输入是 log 概率
criterion = nn.NLLLoss()

# 两者等价：
# NLLLoss(log_softmax(x)) ≡ CrossEntropyLoss(x)
# 但分开写时，log_softmax 的数值稳定性更好（避免先 softmax 再 log 的精度损失）
```

### 6.4 训练循环与 DataLoader 的差异

```python
# 普通分类（MLP/CNN）：DataLoader 按 batch 迭代
for epoch in range(epochs):
    for batch_x, batch_y in loader:
        loss = criterion(model(batch_x), batch_y)
        loss.backward()

# GCN：每个 epoch 整图一次前向传播
for epoch in range(epochs):
    out  = model(data["features"], data["adj"])   # 全部 2708 个节点
    loss = criterion(out[train_mask], labels[train_mask])  # 只对 140 个节点算损失
    loss.backward()  # 梯度通过聚合传播到所有节点
```

---

## 7. 转导学习 vs 归纳学习

### 转导学习（Transductive Learning）— GCN 默认模式

```
训练时：看到所有节点的特征（含测试节点），但只用有标注节点的标签
测试时：预测已见过的（无标注）节点的标签

类比：考前看过所有题目（但只知道部分题目的答案），考试时回答全部题目
```

本项目的 GCN 是转导学习：
- `data["features"]` 包含所有 2,708 个节点的特征
- `adj` 包含所有边（含连接测试节点的边）
- 训练时 `forward(features, adj)` 对全图运行，测试节点已参与聚合

### 归纳学习（Inductive Learning）— GraphSAGE 等

```
训练时：只看训练集中的节点
测试时：泛化到从未见过的新节点（如新加入网络的用户）

类比：用部分题目训练，考试出现全新题目
```

GraphSAGE（Hamilton et al. 2017）通过邻居采样实现归纳学习，可处理不断增长的图。

---

## 8. 项目结构

```
GCN/
├── config.py      # 超参数（in_features, hidden_dim, num_layers, …）
├── model.py       # GraphConvLayer + GCNClassifier
├── dataset.py     # Cora 下载 + 解析 + 归一化 + 掩码生成
├── train.py       # 训练循环（整图前向传播，mask 掩码）
├── test.py        # 测试评估 + 混淆矩阵 + 抽样预测
├── utils.py       # ExperimentLogger（CSV + 训练曲线图）
├── data/          # Cora 原始文件 + parsed_cora.pt 缓存
├── checkpoints/   # best_model.pth
└── logs/          # metrics.csv + training_curves.png + confusion_matrix.png
```

---

## 9. Quick Start

### 版本兼容性

| 包 | 版本 | 说明 |
|----|------|------|
| Python | 3.9 – 3.11 | 推荐 3.10 |
| torch | 2.3.1 | MPS / CUDA / CPU |
| torch_geometric | >=2.4.0 | 图数据集加载（Cora/Planetoid） |
| numpy | >=1.24,<2.0 | - |
| matplotlib | >=3.7,<4.0 | 绘图 |

> **数据集说明**：Cora 由 `torch_geometric` 在首次运行时从 GitHub CDN 自动下载（约 2MB），之后从本地 `data/Cora/` 缓存加载，无需手动操作。

### macOS（Apple Silicon M 系列）

```bash
source ../.venv/bin/activate   # 复用 MLP/CNN/RNN/LSTM 的虚拟环境

# 安装 torch_geometric（仅首次）
pip install torch_geometric

cd GCN
python train.py   # 首次运行自动下载 Cora（约 2MB）
python test.py
```

### Windows（NVIDIA GPU，CUDA 12.1）

```bash
..\.venv\Scripts\activate

pip install torch_geometric

cd GCN
python train.py
python test.py
```

### Windows（CPU）

```bash
..\.venv\Scripts\activate
pip install torch_geometric
cd GCN
python train.py
```

### 预期训练输出

```
使用设备: mps

Cora 数据集已从缓存加载（2708 节点，5429 边）

节点数: 2708  边数: 5429  特征维度: 1433
训练节点: 140  验证节点: 500  测试节点: 1000

模型参数总量: 101,383  (2 层 GCN, hidden=64)

============================================================
开始训练
============================================================
[Epoch 001/200]  Train Loss: 1.9459  Train Acc: 14.29%  |  Val Loss: 1.9421  Val Acc: 26.40%
[Epoch 010/200]  Train Loss: 1.3201  Train Acc: 70.71%  |  Val Loss: 1.4522  Val Acc: 54.20%
[Epoch 050/200]  Train Loss: 0.4523  Train Acc: 97.86%  |  Val Loss: 0.8912  Val Acc: 76.20%
[Epoch 100/200]  Train Loss: 0.1834  Train Acc: 100.0%  |  Val Loss: 0.7203  Val Acc: 80.40%
[Epoch 200/200]  Train Loss: 0.0721  Train Acc: 100.0%  |  Val Loss: 0.7041  Val Acc: 81.20%

训练完成！最优验证准确率: 81.60%
测试集准确率: 81.30%
```

---

## 10. 超参数调优指南

### 10.1 核心超参数影响

| 参数 | 默认值 | 调整方向 | 说明 |
|------|--------|---------|------|
| `hidden_dim` | 64 | 16→256 | 原论文用 16，64 更稳定；>128 一般无提升 |
| `num_layers` | 2 | 1→3 | 超过 3 层出现**过平滑**（所有节点特征趋同） |
| `dropout_rate` | 0.5 | 0.3→0.7 | 对小训练集防过拟合很重要 |
| `learning_rate` | 0.01 | 5e-3→5e-2 | 比 MLP/CNN 大 10 倍，图数据更需要快速收敛 |
| `weight_decay` | 5e-4 | 1e-4→1e-3 | 正则化强度，配合小训练集 |
| `epochs` | 200 | 100→500 | GCN 收敛快，100 epoch 通常足够 |

### 10.2 层数与过平滑问题

```
1 层 GCN：每个节点聚合 1-hop 邻居  → 信息不够丰富
2 层 GCN：聚合 2-hop 邻居          → 最佳点（Cora 平均直径约 6）
3 层 GCN：聚合 3-hop 邻居          → 开始过平滑
4+ 层：  严重过平滑，准确率下降      → 所有节点特征趋于全局均值
```

**过平滑（Over-smoothing）**：
多层聚合后，所有节点的特征向量趋于相同（全图的加权平均），丧失节点的个性特征，分类能力下降。

缓解方案：
```python
# 1. JumpingKnowledge：拼接所有层的输出
# 2. ResGCN：添加残差连接
h = layer(h, adj) + h_prev   # 保留原始特征
# 3. DropEdge：随机丢弃边
# 4. PairNorm：限制节点特征的分散度
```

### 10.3 常见问题排查

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| 训练准确率 100% 但验证 < 60% | 过拟合（训练节点仅 140） | 增大 dropout，增大 weight_decay |
| 训练损失不下降 | 学习率太小 / 归一化错误 | 检查 adj 是否已归一化，lr 换为 0.01 |
| 所有节点预测同一类 | 过平滑（层数过多） | 减少到 2 层 |
| 验证准确率在 epoch 50 后下降 | 过拟合 | Early stopping，或增加正则化 |

---

## 11. GCN 局限性与后续发展

### GCN 的根本局限

| 局限 | 问题描述 |
|------|---------|
| 固定权重 | 邻居权重由度数决定（固定），无法区分不同邻居的重要性 |
| 过平滑 | 层数增加导致特征趋同 |
| 转导学习 | 无法泛化到训练时未见过的节点 |
| 全图计算 | 无法扩展到超大图（百万节点时内存溢出） |
| 无方向性 | 无法区分"引用"和"被引用" |

### 后续发展路线

```
GCN（Kipf 2017）
  │
  ├── GAT（Graph Attention Network, Veličković 2018）
  │     用注意力机制替代固定的对称归一化权重：
  │     α_ij = softmax(a · [Wh_i || Wh_j])
  │     节点可以"关注"重要邻居，忽略噪声邻居
  │
  ├── GraphSAGE（Hamilton 2017）
  │     邻居采样 → 支持归纳学习（新节点）
  │     不依赖完整邻接矩阵，可扩展到大图
  │
  ├── GIN（Graph Isomorphism Network, Xu 2019）
  │     理论上更强的表达能力（与 Weisfeiler-Lehman 图同构测试等价）
  │
  └── Graph Transformer（多种变体）
        将 Transformer Attention 引入图，克服过平滑
        如 Graphormer（微软，用于分子属性预测）
```

**为什么仍然学 GCN？**

1. 图卷积是所有 GNN 的基础，理解它等于理解整个 GNN 家族的设计原则
2. 归一化邻接矩阵的思想在所有 GNN 变体中都有体现
3. GCN 代码极其简洁（核心逻辑 2 行），清晰展示"聚合 + 变换"的本质
4. Cora 数据集小（<3K 节点），调试快，适合理解图数据的处理流程
