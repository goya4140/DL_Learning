# Attention 机制 / Self-Attention / Multi-Head Attention

> 本文档对应 `Transformer/` 项目，覆盖 Scaled Dot-Product Attention、Multi-Head Attention、位置编码、残差连接 + LayerNorm、复杂度分析。

---

## 1. 为什么需要 Attention：RNN 的局限

RNN/LSTM 处理序列时存在两个结构性瓶颈：

| 问题 | 根因 | 后果 |
|------|------|------|
| **串行计算** | 第 t 步依赖第 t-1 步的隐藏状态 | 无法并行，长序列训练慢 |
| **梯度消失** | 长程梯度经过多次矩阵乘法衰减 | 难以捕捉 100+ 步的依赖 |
| **信息瓶颈** | 整个序列压缩进固定长度向量 | seq2seq 任务中长句信息丢失 |

Attention 的核心思想：**让每个位置直接与序列中所有其他位置交互**，信息传递路径长度从 O(L) 缩短为 O(1)。

---

## 2. Scaled Dot-Product Attention

### 2.1 基本公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

- **Q**（Query）：当前位置"想查询什么"，shape `(L, d_k)`
- **K**（Key）：每个位置"提供什么索引"，shape `(L, d_k)`
- **V**（Value）：每个位置"真正携带的信息"，shape `(L, d_v)`

计算步骤：

```
1. 相似度矩阵：scores = Q @ K^T           # (L, L)
2. 缩放：       scores = scores / sqrt(d_k) # 防止 softmax 饱和
3. 归一化：     weights = softmax(scores)   # (L, L)，每行和为 1
4. 加权聚合：   output  = weights @ V       # (L, d_v)
```

### 2.2 为什么除以 √d_k

当 `d_k` 较大时，`QK^T` 中每个元素是 `d_k` 个随机变量之积的和，方差约为 `d_k`。  
标准差为 `√d_k`，导致 dot-product 绝对值很大，落入 softmax 梯度近零的饱和区。

除以 `√d_k` 后方差回归 1，梯度流正常：

```python
# PyTorch 中的等价实现
scale = math.sqrt(d_k)
scores = torch.matmul(q, k.transpose(-2, -1)) / scale
weights = torch.softmax(scores, dim=-1)
output = torch.matmul(weights, v)
```

### 2.3 Padding Mask（src_key_padding_mask）

对填充位置，在 softmax 前将对应分数置为 `-inf`，softmax 后权重趋近 0，`<pad>` 不向任何位置传递信息：

```python
# 生成 mask：True = pad 位置，需要屏蔽
mask = (x == pad_idx)                    # (B, L)

# nn.TransformerEncoder 内部等价操作
scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))
weights = softmax(scores, dim=-1)        # pad 列权重 → 0
```

---

## 3. Multi-Head Attention

### 3.1 直觉：多个"关注角度"

单头 Attention 每次只能关注一种语义关系（如句法依赖）。  
Multi-Head Attention 学习 h 组不同的 Q/K/V 投影，让模型**同时关注多种关系**（指代、语法、语义相似度等）。

### 3.2 公式

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)
$$

其中：
- 每头维度：`d_head = d_model / h`（保证总计算量不变）
- `W_i^Q, W_i^K, W_i^V ∈ R^{d_model × d_head}`：每头独立学习投影
- `W^O ∈ R^{d_model × d_model}`：拼接后的输出投影

### 3.3 PyTorch 实现（nn.MultiheadAttention 等价逻辑）

```python
# nn.TransformerEncoderLayer 内部
# d_model=128, nhead=4, d_head=32

# 实际实现把 h 个头打包成一个大矩阵运算，等价于：
for i in range(nhead):
    q_i = q @ W_Q[i]   # (B, L, 32)
    k_i = k @ W_K[i]   # (B, L, 32)
    v_i = v @ W_V[i]   # (B, L, 32)
    head_i = attention(q_i, k_i, v_i)   # (B, L, 32)
output = concat(head_0, ..., head_3) @ W_O   # (B, L, 128)
```

---

## 4. 位置编码（Positional Encoding）

Attention 本身**对位置不感知**（打乱词序不影响注意力权重计算的结果）。  
需要手动将位置信息注入词嵌入。

### 4.1 正弦/余弦编码（Vaswani et al. 2017）

$$
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)
$$

$$
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)
$$

```python
pe = torch.zeros(max_len, d_model)
position = torch.arange(0, max_len).unsqueeze(1).float()    # (L, 1)
div_term = torch.exp(
    torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
)                                                             # (d_model/2,)
pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
```

**关键性质**：
- 不同位置的编码向量点积随距离衰减 → 模型能"感知"相对距离
- 使用 `register_buffer`：随模型 `.to(device)` 移动，不参与梯度更新
- 加法融合：`x = embedding(tokens) + pe[:, :L]`

### 4.2 可学习位置编码（BERT 方式）

```python
self.pos_embedding = nn.Embedding(max_len, d_model)
pos_ids = torch.arange(seq_len).unsqueeze(0)
x = token_emb + self.pos_embedding(pos_ids)
```

对比：

| | 正弦编码 | 可学习编码 |
|--|---------|----------|
| 参数量 | 0（固定） | max_len × d_model |
| 泛化到更长序列 | ✅ 可外推 | ❌ 超出训练长度失效 |
| 性能 | 相当 | 通常略优（任务相关） |

---

## 5. Transformer Encoder Block

每个 Encoder 层包含两个子层，均使用**残差连接 + LayerNorm**：

```
输入 x
  ├─ Multi-Head Self-Attention(x, x, x)
  │       ↓
  │  + x  （残差连接）
  │       ↓
  │  LayerNorm
  │       ↓ x'
  ├─ FeedForward(x')
  │       ↓
  │  + x' （残差连接）
  │       ↓
  └─ LayerNorm
        ↓ 输出
```

### 5.1 残差连接（Residual Connection）

$$
\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))
$$

**作用**：
- 梯度直接通过恒等路径反传，缓解深层网络梯度消失
- 网络最差情况退化为恒等映射（学习增量），训练更稳定

### 5.2 Layer Normalization

对**单个样本的特征维度**做归一化（BatchNorm 对 batch 维度归一化）：

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta
$$

其中 `μ, σ` 在特征维度（d_model）上计算，`γ, β` 是可学习参数。

**为什么 Transformer 用 LayerNorm 而非 BatchNorm**：

| | BatchNorm | LayerNorm |
|--|----------|----------|
| 归一化轴 | batch 维 | feature 维 |
| 变长序列 | ❌ 不同长度统计量不一致 | ✅ 每个位置独立归一化 |
| 小 batch | 统计量不稳定 | 不受影响 |
| 推理时 | 需要运行时均值/方差 | 不需要（直接对当前输入计算） |

### 5.3 Pre-LN vs Post-LN

```python
# Post-LN（原始论文，nn.TransformerEncoderLayer norm_first=False）
x = LayerNorm(x + Attention(x))

# Pre-LN（更稳定，norm_first=True）
x = x + Attention(LayerNorm(x))
```

Pre-LN 训练更稳定、不需要 warmup，但表示能力略有不同，两者在实践中都被广泛使用。

---

## 6. FeedForward Network（FFN）

每层 Attention 之后跟一个 2 层 MLP，在特征空间做非线性变换：

$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

- 通常 `d_ff = 4 × d_model`（本项目 `d_ff = 2 × d_model = 256`）
- 每个位置**独立**通过同一个 FFN（权重共享，但不跨位置交互）
- 作用：增加网络非线性表达能力，相当于每层的"逐位置特征变换"

---

## 7. Self-Attention 复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 计算 QK^T | O(L² · d_k) | O(L²) |
| 计算 Attention·V | O(L² · d_v) | O(L²) |
| 整体（单头） | **O(L² · d)** | O(L²) |

与 RNN 的对比：

| | RNN/LSTM | Transformer Self-Attention |
|--|---------|--------------------------|
| 时间复杂度 | O(L · d²)（串行） | O(L² · d)（并行） |
| 最大路径长度 | O(L) | **O(1)** |
| 并行度 | ❌ 串行 | ✅ 完全并行 |
| 长程依赖 | 弱（梯度消失） | 强（直接交互） |

**L² 问题**：序列长度超过 1000 后，L² 的注意力矩阵会变得很大（4096 token → 16M 个注意力权重）。  
解决方案：Sparse Attention（Longformer）、线性 Attention（Performer）、Flash Attention（重计算减少 HBM 访问）。

---

## 8. 分类头设计：均值池化 vs [CLS] token

Encoder 输出 shape `(B, L, d_model)`，需要压缩为 `(B, d_model)` 送入分类层。

### 8.1 掩码均值池化（本项目采用）

```python
valid_mask = ~src_key_padding_mask           # True = 真实 token
valid_mask_f = valid_mask.unsqueeze(-1).float()
pooled = (out * valid_mask_f).sum(dim=1) / valid_mask_f.sum(dim=1)
```

- 只对真实 token 取均值，排除 `<pad>` 位置的干扰
- 简单有效，适合分类任务

### 8.2 [CLS] token（BERT 方式）

在序列开头插入特殊 `[CLS]` token，训练后该位置的输出向量聚合全局语义：

```python
cls_token = self.cls_embedding.expand(B, 1, d_model)
x = torch.cat([cls_token, token_emb], dim=1)   # (B, L+1, d_model)
out = encoder(x)
pooled = out[:, 0, :]                           # 取 [CLS] 位置
```

- 更符合 BERT 的预训练范式
- [CLS] 通过全局 Self-Attention 自然地聚合所有位置信息
- 需要专门初始化 cls token 嵌入

---

## 9. 总结：Transformer 的核心贡献

```
传统 RNN 瓶颈：
  位置 1 → 位置 2 → ... → 位置 L   （信息必须逐步传递）

Self-Attention：
  任意两个位置直接计算相似度，1步完成全局信息聚合

配合：
  多头（Multi-Head）     → 同时捕捉多种语义关系
  位置编码               → 补充位置信息（Self-Attention 本身无序）
  残差 + LayerNorm       → 训练稳定，支持更深的网络
  FFN                    → 逐位置非线性变换
```

这套机制使 Transformer 成为 BERT、GPT、T5 等预训练语言模型的统一底层架构。
