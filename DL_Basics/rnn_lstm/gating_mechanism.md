# 门控机制（Gating Mechanism）

> 核心思想：用可学习的"阀门"控制信息的流通与遗忘，解决梯度消失问题。

---

## 1. 为什么需要门控

vanilla RNN 梯度反传路径（T 步序列）：

```
∂L/∂h_t = ∂L/∂h_T · ∏_{k=t}^{T-1} W_hh · diag(1 - tanh²(s_k))
```

连乘 `(T-t)` 次后：
- 特征值 < 1 → **梯度消失**（远距离依赖无法学习）
- 特征值 > 1 → **梯度爆炸**（训练不稳定，用梯度裁剪缓解）

门控机制通过引入**加法更新路径**绕过这一问题。

---

## 2. LSTM 门控

### 四个门

| 门 | 公式 | 作用 |
|----|------|------|
| 遗忘门 f | `σ(W_f·[h,x]+b_f)` | 决定清除多少旧细胞状态 |
| 输入门 i | `σ(W_i·[h,x]+b_i)` | 决定写入多少新信息 |
| 候选值 g | `tanh(W_g·[h,x]+b_g)` | 新信息的候选内容 |
| 输出门 o | `σ(W_o·[h,x]+b_o)` | 决定从细胞状态输出多少 |

### 状态更新

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t   ← 加法！梯度高速公路
h_t = o_t ⊙ tanh(c_t)
```

### 梯度分析

```
∂c_t/∂c_{t-1} = f_t   （仅逐元素乘，不是矩阵连乘）
```

若遗忘门 `f_t ≈ 1`，细胞状态梯度近似为 1，可穿越任意长序列。

---

## 3. GRU 门控（轻量替代方案）

GRU（Gated Recurrent Unit，Cho et al. 2014）将 LSTM 的 3 个门简化为 2 个：

```
重置门：r_t = σ(W_r·[h_{t-1}, x_t])   ← 控制过去信息的"重置"程度
更新门：z_t = σ(W_z·[h_{t-1}, x_t])   ← 控制新旧信息的混合比例
候选值：h̃_t = tanh(W·[r_t⊙h_{t-1}, x_t])
输出：  h_t = (1-z_t)⊙h_{t-1} + z_t⊙h̃_t
```

GRU 合并了 LSTM 的细胞状态和隐藏状态为单一 `h_t`，更新门兼任遗忘+输入功能。

### LSTM vs GRU 选择

| 对比维度 | LSTM | GRU |
|---------|------|-----|
| 参数量 | 4H(H+E) | 3H(H+E) |
| 表达能力 | 略强（分离 c 和 h） | 略弱 |
| 训练速度 | 较慢 | 快 ~25% |
| 实践效果 | 通常持平 | 通常持平 |
| 推荐场景 | 需要精细控制记忆 | 参数/速度受限 |

**经验法则**：两者效果在大多数任务上无显著差异，优先 GRU 节省计算，必要时换 LSTM 微调。

---

## 4. 注意力机制（Attention）—— 门控的进化

Attention（Bahdanau et al. 2014）可视为**软性、动态的门控**：

```
传统门控：h_t = gate_t ⊙ h_{t-1}  （固定权重模式，gate 由局部信息决定）
Attention： z = Σ_t α_t · h_t     （α_t 由全局信息动态计算）
```

区别：
- LSTM/GRU 门：由当前时间步局部信息决定，权重沿时间步变化
- Self-Attention：直接建模所有位置对之间的关系，无需逐步传递

Transformer 完全抛弃 RNN/LSTM，改用纯 Attention，从根本上解决了长程依赖问题（路径长度从 O(T) 降为 O(1)）。

---

## 5. PyTorch 中的使用

```python
import torch.nn as nn

# LSTM
lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2,
               batch_first=True, dropout=0.5, bidirectional=True)
output, (hidden, cell) = lstm(x)   # x: (B, L, 128)
# output: (B, L, 512)，hidden/cell: (4, B, 256)

# GRU（接口几乎相同，少了 cell）
gru = nn.GRU(input_size=128, hidden_size=256, num_layers=2,
             batch_first=True, dropout=0.5, bidirectional=True)
output, hidden = gru(x)
# output: (B, L, 512)，hidden: (4, B, 256)
```

**常见陷阱**：
- `dropout` 参数在 `num_layers=1` 时无效（单层无"层间"可加 Dropout）
- 双向时 `hidden` 维度为 `(num_layers*2, B, H)`，取最后一层需用 `hidden[-2:]`
- 不提供 `(h_0, c_0)` 时默认用全零初始化（通常足够）
