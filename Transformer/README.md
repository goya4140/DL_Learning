# Transformer 文本分类

> 任务：AG News 新闻标题 4 分类（World / Sports / Business / Sci/Tech）  
> 数据集：120,000 训练 / 7,600 测试（与 RNN/LSTM 相同，便于直接对比）  
> 目标准确率：**~88-90%**（2 层 Transformer Encoder，10 epoch）

---

## 目录

1. [为什么需要 Transformer](#1-为什么需要-transformer)
2. [Attention 核心数学](#2-attention-核心数学)
3. [位置编码](#3-位置编码)
4. [Transformer Encoder Block](#4-transformer-encoder-block)
5. [架构总览](#5-架构总览)
6. [nn.TransformerEncoder 源码解析](#6-nntransformerencoder-源码解析)
7. [src_key_padding_mask 详解](#7-src_key_padding_mask-详解)
8. [项目结构](#8-项目结构)
9. [Quick Start](#9-quick-start)
10. [超参数调优指南](#10-超参数调优指南)
11. [RNN / LSTM / Transformer 对比](#11-rnn--lstm--transformer-对比)

---

## 1. 为什么需要 Transformer

### RNN/LSTM 的瓶颈

| 问题 | RNN/LSTM | Transformer |
|------|---------|-------------|
| **串行计算** | h_t 依赖 h_{t-1}，必须逐步计算，无法并行 | 所有位置同时计算，GPU 利用率高 |
| **长距离依赖** | 梯度路径 = 序列长度，仍有衰减 | 任意两个位置路径长度 = 1（直接 Attention） |
| **瓶颈表示** | 序列信息压缩到固定大小的隐藏状态 | 每个位置都保留完整的上下文向量 |
| **计算复杂度** | O(L) 时间步 × O(d²) 每步 = O(Ld²) | O(L²d)（自注意力）但可完全并行 |

### Attention 的核心洞察

LSTM 通过"门控"决定传递什么信息，但仍受限于序列的顺序结构。

Transformer 直接问：**"这个词与句中哪些词最相关？"**，用相似度权重直接聚合：

```
对于每个词 i，计算它与所有词 j 的相关性得分，
按得分加权汇聚所有词的信息 → 词 i 的新表示
```

无论词距多远，只要相关性高，信息就能直接流动。

---

## 2. Attention 核心数学

### 2.1 Scaled Dot-Product Attention

输入：
- **Q**（Query，查询）：当前词"想要什么信息"
- **K**（Key，键）：每个词"能提供什么信息"
- **V**（Value，值）：每个词"实际包含的信息"

在 Self-Attention 中，Q / K / V 都来自同一序列（用不同线性变换得到）。

```
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   Attention(Q, K, V) = softmax( QK^T / √d_k ) · V      ║
║                                                          ║
║   Q ∈ ℝ^{L×d_k}，K ∈ ℝ^{L×d_k}，V ∈ ℝ^{L×d_v}        ║
║   输出 ∈ ℝ^{L×d_v}                                      ║
╚══════════════════════════════════════════════════════════╝
```

**步骤**：

1. **相似度矩阵**：`S = QK^T ∈ ℝ^{L×L}`，`S_ij` 表示词 i 对词 j 的关注程度
2. **缩放**：除以 `√d_k`，防止 softmax 饱和（d_k 大时点积方差增大，softmax 梯度趋零）
3. **Softmax**：对每行归一化，得到注意力权重 α（每行和为 1）
4. **加权求和**：`output = α · V`，每个词的新表示 = 所有词的值的加权平均

**为什么除以 √d_k？**

设 q, k 均为零均值单位方差的随机向量，则：
```
Var(q · k) = d_k   →   Std(q · k) = √d_k
```
不缩放时，d_k 越大，点积方差越大，softmax 输入差异越大，梯度极小（饱和）。
除以 √d_k 后方差归一为 1，softmax 输出平滑，梯度正常。

### 2.2 Multi-Head Attention

单头注意力只能学习一种"关注模式"。多头注意力用 h 个不同的线性投影并行学习多种模式：

```
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)   i = 1,...,h
MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
```

每个头可以专注于不同类型的关系：
- 头 1：关注语法依存（主语↔谓语）
- 头 2：关注语义相似（同义词）
- 头 3：关注局部位置（相邻词）
- 头 4：关注长距离指代（代词↔先行词）

维度关系（`d_model=128, h=4`）：
```
每头维度 d_k = d_v = d_model / h = 32
Q W_i^Q：(B,L,128) × (128,32) → (B,L,32)
h 个头 → Concat → (B,L,128)
× W^O → (B,L,128)    输出维度不变
```

---

## 3. 位置编码

### 3.1 为什么需要位置编码

Self-Attention 是**置换不变的**（Permutation Invariant）：
```
Attention([猫,吃,鱼]) = Attention([鱼,吃,猫])  ← 同样的输出！
```
打乱词序不影响注意力计算，模型无法感知词的位置信息。

解决方案：在词嵌入中加入位置信息。

### 3.2 正弦位置编码（Vaswani et al. 2017）

```
PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
```

- `pos`：词在序列中的位置（0, 1, 2, ...）
- `i`：维度索引（0, 1, ..., d_model/2 - 1）
- 不同维度对应不同频率（低维高频，高维低频）

**优点**：
- 不增加可学习参数
- 可外推到比训练时更长的序列
- 相对位置信息可从 PE(pos) 和 PE(pos+k) 的线性组合中提取

**与词嵌入融合**：
```python
x = word_embedding(tokens) + positional_encoding   # 直接相加
```

---

## 4. Transformer Encoder Block

每个 Encoder 层包含两个子层，每个子层都有**残差连接 + 层归一化（Post-LN）**：

```
子层 1（多头自注意力）：
  x = LayerNorm(x + MultiHeadSelfAttention(x))

子层 2（前馈网络 FFN）：
  x = LayerNorm(x + FFN(x))

FFN(x) = Linear(ReLU(Linear(x)))   # 两层 MLP
         (d_model → d_ff → d_model)
```

### 残差连接的作用

```
output = LayerNorm(x + Sublayer(x))
                  ↑
          跳跃连接（恒等映射）
```

- 避免深层网络梯度消失（梯度可通过跳跃连接直接回传）
- 让每个子层只需学习"残差"（相对于恒等映射的修正量）
- 与 ResNet 的核心思想完全一致

### LayerNorm vs BatchNorm

在 Transformer 中使用 **LayerNorm** 而非 BatchNorm：
- BatchNorm：对 batch 维度归一化（batch 内每个特征维度的均值/方差）
- LayerNorm：对特征维度归一化（每个样本内所有特征的均值/方差）

Transformer 用 LayerNorm 的原因：
- 序列长度可变（不同 batch 内序列长短不一，BatchNorm 的统计量不稳定）
- NLP 中 batch_size 通常较小，BatchNorm 统计噪声大
- LayerNorm 与 batch 大小无关，推理时行为与训练时完全一致

---

## 5. 架构总览

```
AG News 输入（最大 64 个 token）
┌─────────────────────────────────────────────────────────────┐
│  token ids: [42, 87, 3, ..., 1, 1]  shape: (B, 64)         │
└───────────────────────┬─────────────────────────────────────┘
                        │
          ┌─────────────▼──────────────┐
          │  TokenEmbedding            │
          │  (B,64) → (B,64,128)       │
          │  vocab=20000, d_model=128  │
          └─────────────┬──────────────┘
                        │ (B,64,128)
          ┌─────────────▼──────────────┐
          │  PositionalEncoding        │
          │  + PE(pos) 正弦编码        │
          │  Dropout(0.1)              │
          └─────────────┬──────────────┘
                        │ (B,64,128)
    ┌───────────────────▼──────────────────────────────────────┐
    │           TransformerEncoderLayer × 2                    │
    │                                                          │
    │  ┌─────────────────────────────────────────────────┐    │
    │  │  MultiHeadSelfAttention (nhead=4, d_k=32/head)  │    │
    │  │  → Add & LayerNorm                              │    │
    │  └────────────────────┬────────────────────────────┘    │
    │                       │                                  │
    │  ┌─────────────────────▼───────────────────────────┐    │
    │  │  FeedForward: Linear(128→256) → ReLU → Linear(256→128)│
    │  │  → Add & LayerNorm                              │    │
    │  └─────────────────────────────────────────────────┘    │
    │                                                          │
    │  （第 2 层结构相同）                                     │
    └───────────────────┬──────────────────────────────────────┘
                        │ (B,64,128)
          ┌─────────────▼──────────────┐
          │  掩码均值池化               │
          │  排除 <pad> 位置           │
          │  (B,64,128) → (B,128)      │
          └─────────────┬──────────────┘
                        │
          ┌─────────────▼──────────────┐
          │  Dropout(0.1)              │
          │  Linear(128 → 4)          │
          └─────────────┬──────────────┘
                        │ (B,4) logits
          ┌─────────────▼──────────────┐
          │  CrossEntropyLoss          │
          │  argmax → 类别预测         │
          └────────────────────────────┘
```

---

## 6. nn.TransformerEncoder 源码解析

### 6.1 初始化参数

```python
# 单个 Encoder 层
encoder_layer = nn.TransformerEncoderLayer(
    d_model=128,          # 模型维度（输入/输出维度）
    nhead=4,              # 注意力头数（d_model 必须能被 nhead 整除）
    dim_feedforward=256,  # FFN 中间层维度
    dropout=0.1,          # Attention + FFN 内部的 Dropout
    activation="relu",    # FFN 激活函数（"relu" 或 "gelu"）
    batch_first=True,     # True: (B,L,d)；False: (L,B,d)
    norm_first=False,     # False: Post-LN（原论文）；True: Pre-LN（更稳定）
)

# 堆叠多层
encoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=2,         # 层数
    norm=None,            # 可选：最后一层后加额外 LayerNorm
)
```

### 6.2 前向传播

```python
output = encoder(
    src,                         # (B, L, d_model)
    mask=None,                   # (L, L) Attention Mask（用于因果掩码等）
    src_key_padding_mask=mask,   # (B, L) Padding Mask（True = 忽略该位置）
)
# output: (B, L, d_model)
```

**两种 mask 的区别**：
- `mask`：控制哪些**位置对**（i,j）可以互相注意（如 GPT 的因果掩码）
- `src_key_padding_mask`：控制哪些**位置**是 padding，应被忽略

### 6.3 内部 Q/K/V 线性层

```python
# TransformerEncoderLayer 内部等价于：
W_Q = nn.Linear(d_model, d_model)   # 实际实现合并为一个线性层
W_K = nn.Linear(d_model, d_model)
W_V = nn.Linear(d_model, d_model)
W_O = nn.Linear(d_model, d_model)

# 多头通过 reshape 实现：
# (B, L, d_model) → (B, L, nhead, d_k) → (B, nhead, L, d_k)
```

---

## 7. src_key_padding_mask 详解

### 生成方法

```python
# texts: (B, L)，pad_idx = vocab["<pad>"] = 1
src_key_padding_mask = (texts == pad_idx)   # (B, L), dtype=bool
# True 的位置在 Attention 时会被置为 -inf，softmax 后为 0
```

### 作用流程

```
texts = [[42, 87, 3, 1, 1],   # 序列 1：3 个真实词 + 2 个 <pad>
         [5,  12, 8, 9, 1]]   # 序列 2：4 个真实词 + 1 个 <pad>

mask  = [[F,  F,  F, T, T],   # T = 被屏蔽
         [F,  F,  F, F, T]]

Attention 计算时：
  score_ij += mask_j ? -inf : 0
  softmax(-inf) = 0           ← <pad> 位置权重为零，不向其他词传递信息
```

### 为什么对分类任务重要

如果不屏蔽 padding：
- <pad> 位置输出的向量会参与均值池化，稀释真实内容的信息
- 更严重的是：真实词会"关注"padding 位置，引入噪声

---

## 8. 项目结构

```
Transformer/
├── config.py      # 超参数（d_model, nhead, num_encoder_layers, d_ff, …）
├── model.py       # PositionalEncoding + TransformerClassifier
├── dataset.py     # AG News 数据加载（复用 LSTM/dataset.py）
├── train.py       # 训练循环（含 src_key_padding_mask）
├── test.py        # 测试评估 + 混淆矩阵 + 抽样预测
├── utils.py       # ExperimentLogger（CSV + 训练曲线图）
├── data/          # AG News CSV + vocab 缓存（与 LSTM 共享）
├── checkpoints/   # best_model.pth
└── logs/          # metrics.csv + training_curves.png + confusion_matrix.png
```

---

## 9. Quick Start

### 版本兼容性

| 包 | 版本 | 说明 |
|----|------|------|
| Python | 3.9 – 3.11 | 推荐 3.10 |
| torch | 2.3.1 | `nn.TransformerEncoderLayer(batch_first=True)` 需 ≥1.9 |
| torchtext | 0.18.0 | AG News 分词 + 词汇表 |
| numpy | >=1.24,<2.0 | - |
| matplotlib | >=3.7,<4.0 | 绘图 |

### macOS（Apple Silicon M 系列）

```bash
source ../.venv/bin/activate   # 复用 MLP/CNN/RNN/LSTM 的虚拟环境

cd Transformer
python train.py   # 首次运行自动下载 AG News（若 data/ 目录已有则跳过）
python test.py
```

> **提示**：若已训练过 LSTM，可将 `LSTM/data/` 软链接或复制到 `Transformer/data/`，
> 跳过重新下载和词汇表构建：
> ```bash
> ln -s ../LSTM/data ./data
> ```

### Windows（NVIDIA GPU，CUDA 12.1）

```bash
..\.venv\Scripts\activate
cd Transformer
python train.py
python test.py
```

### Windows（CPU）

```bash
..\.venv\Scripts\activate
cd Transformer
python train.py
```

### 预期训练输出

```
使用设备: mps

词汇表大小: 20000

训练集: 120,000 条  测试集: 7,600 条
类别: ['World', 'Sports', 'Business', 'Sci/Tech']

模型参数总量: 2,371,332  (2 层, d_model=128, nhead=4)

============================================================
开始训练
============================================================

[Epoch 01/10]  学习率: 0.001000
  ...
  训练损失: 0.5213  训练准确率: 80.34%
  测试损失: 0.4102  测试准确率: 85.92%
  ✓ 保存最优模型（准确率: 85.92%）
...
训练完成！最优测试准确率: 89.XX%
```

---

## 10. 超参数调优指南

### 10.1 核心超参数影响

| 参数 | 默认值 | 调整建议 | 说明 |
|------|--------|---------|------|
| `d_model` | 128 | 64 ~ 256 | 必须能被 nhead 整除；增大提升容量 |
| `nhead` | 4 | 2, 4, 8 | 每头维度 = d_model/nhead，建议 ≥16 |
| `num_encoder_layers` | 2 | 1 ~ 4 | 层数增加边际效益递减（文本分类） |
| `d_ff` | 256 | 128 ~ 512 | FFN 中间维度，通常 2×~4× d_model |
| `dropout_rate` | 0.1 | 0.1 ~ 0.3 | Transformer 标准值，不宜过大 |
| `max_len` | 64 | 32 ~ 128 | AG News 平均约 40 词，64 已足够 |
| `learning_rate` | 1e-3 | 5e-4 ~ 5e-3 | Transformer 对 lr 较敏感 |

### 10.2 Warmup 学习率（进阶）

原始 Transformer 论文使用带 Warmup 的学习率调度：

```python
# 前 warmup_steps 步线性增大，之后按 1/√step 衰减
def lr_lambda(step):
    warmup = 4000
    if step == 0:
        return 0
    return min(step ** -0.5, step * warmup ** -1.5) * (d_model ** -0.5)

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
# 注意：每个 batch 调用一次 scheduler.step()，而非每个 epoch
```

对于 AG News 这样的小数据集，StepLR 已足够，Warmup 在大模型/大数据时更必要。

### 10.3 norm_first（Pre-LN vs Post-LN）

```python
# Post-LN（默认，原论文）
x = LayerNorm(x + Sublayer(x))

# Pre-LN（更稳定，现代大模型偏好）
x = x + Sublayer(LayerNorm(x))
```

Post-LN 训练有时不稳定（梯度方差大），Pre-LN 收敛更平滑。
在 `nn.TransformerEncoderLayer` 中用 `norm_first=True` 切换。

### 10.4 常见问题排查

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| 训练损失不下降 | d_model 不能被 nhead 整除 | 确保 d_model % nhead == 0 |
| NaN 损失 | 学习率太大 / Attention 溢出 | 降低 lr，添加梯度裁剪 |
| 训练准确率高但验证低 | 过拟合 | 增大 dropout，减小 num_layers |
| 比 LSTM 慢很多 | batch_first=False（低效）| 确认 batch_first=True |
| MPS 报错 | 某些 Attention 操作 MPS 不支持 | 升级 torch 或使用 device="cpu" |

---

## 11. RNN / LSTM / Transformer 对比

| 特性 | vanilla RNN | LSTM | Transformer |
|------|-------------|------|-------------|
| 序列建模方式 | 逐步隐状态传递 | 门控 + 细胞状态 | 全局 Self-Attention |
| 并行计算 | ✗（串行） | ✗（串行） | ✓（所有位置同时） |
| 长距离依赖 | 差（指数衰减） | 较好（门控） | 优（O(1)路径） |
| 梯度消失 | 严重 | 大幅缓解 | 几乎无（残差+LayerNorm） |
| 位置信息 | 隐式（序列顺序） | 隐式 | 需显式位置编码 |
| 参数量（本项目） | ~2.5M | ~3.2M | ~2.4M |
| AG News 准确率 | ~78% | ~88% | ~89% |
| 计算复杂度 | O(L·d²) | O(L·d²) | O(L²·d)（注意力） |
| 适合序列长度 | 短（<50） | 中（<500） | 可变（有位置编码上限） |
| 现代应用 | 教学/嵌入式 | 时序预测 | BERT/GPT 等大模型基础 |
