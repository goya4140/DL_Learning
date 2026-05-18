# LSTM 文本分类

> 任务：AG News 新闻标题 4 分类（World / Sports / Business / Sci/Tech）  
> 数据集：120,000 训练 / 7,600 测试  
> 目标准确率：**~88%**（双向 2 层 LSTM，10 epoch）

---

## 目录

1. [为什么需要 LSTM](#1-为什么需要-lstm)
2. [LSTM 核心数学推导](#2-lstm-核心数学推导)
3. [双向 LSTM（BiLSTM）](#3-双向-lstmbilstm)
4. [架构总览](#4-架构总览)
5. [nn.LSTM 源码解析](#5-nnlstm-源码解析)
6. [项目结构](#6-项目结构)
7. [Quick Start](#7-quick-start)
8. [超参数调优指南](#8-超参数调优指南)
9. [RNN vs LSTM 对比](#9-rnn-vs-lstm-对比)

---

## 1. 为什么需要 LSTM

### vanilla RNN 的根本缺陷

vanilla RNN 的隐藏状态递推公式：

```
h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b)
```

反向传播时，梯度从时间步 T 传回时间步 t 需要连乘 (T-t) 个雅可比矩阵：

```
∂L/∂h_t = ∂L/∂h_T · ∏_{k=t}^{T-1} (W_hh · diag(1 - tanh²(s_k)))
           \_____________________/
              (T-t) 次矩阵连乘
```

- `|λ_max(W_hh)| < 1`：梯度指数级 **消失** → 远距离依赖无法学习
- `|λ_max(W_hh)| > 1`：梯度指数级 **爆炸** → 训练不稳定

**实验证据**：RNN 项目中 max_len=128, num_layers=2 时，10 epoch 后准确率仍在 25%（随机基线），降至 max_len=32, num_layers=1 才正常工作。

### LSTM 的解决思路

LSTM（Long Short-Term Memory，Hochreiter & Schmidhuber, 1997）的核心创新：
引入**细胞状态** `c_t` 作为独立的长期记忆通道，通过**加法更新**保持梯度流动。

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
         ↑ 遗忘旧记忆      ↑ 写入新记忆
```

细胞状态的梯度：
```
∂c_t/∂c_{t-1} = f_t   （逐元素乘法，非矩阵连乘！）
```

只要遗忘门 `f_t ≈ 1`（模型学会"记住"），梯度可以不衰减地穿越任意长的序列。

---

## 2. LSTM 核心数学推导

### 2.1 门控机制

LSTM 在每个时间步 t 维护两个状态：
- `h_t` ∈ ℝ^H：隐藏状态（短期输出）
- `c_t` ∈ ℝ^H：细胞状态（长期记忆）

四个门（Gates）共享相同的输入 `[h_{t-1}; x_t]` ∈ ℝ^{H+E}：

```
遗忘门：f_t = σ(W_f · [h_{t-1}; x_t] + b_f)    f_t ∈ [0,1]^H
输入门：i_t = σ(W_i · [h_{t-1}; x_t] + b_i)    i_t ∈ [0,1]^H
候选值：g_t = tanh(W_g · [h_{t-1}; x_t] + b_g)  g_t ∈ (-1,1)^H
输出门：o_t = σ(W_o · [h_{t-1}; x_t] + b_o)    o_t ∈ [0,1]^H
```

其中 `σ` 为 Sigmoid（将值压缩到 0-1，充当"阀门"），`tanh` 提供候选新信息。

### 2.2 状态更新

```
细胞更新：c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
隐藏输出：h_t = o_t ⊙ tanh(c_t)
```

**直觉理解**：
| 门 | 作用 | 开（≈1）时 | 关（≈0）时 |
|----|------|-----------|-----------|
| `f_t` 遗忘门 | 保留多少旧细胞状态 | 保留历史记忆 | 清空历史 |
| `i_t` 输入门 | 写入多少新信息 | 更新记忆 | 忽略当前输入 |
| `o_t` 输出门 | 从细胞状态输出多少 | 暴露记忆 | 隐藏记忆 |

### 2.3 参数量计算

每个门的参数：`W ∈ ℝ^{H×(H+E)}`，`b ∈ ℝ^H`

LSTM 总参数（单向单层）：
```
4 × (H × (H + E) + H) = 4H(H + E + 1)
```

对比 vanilla RNN：
```
RNN 参数：H(H + E) + H = H(H + E + 1)
LSTM/RNN 参数比 ≈ 4
```

**本项目参数量估算**（H=128, E=128, 双向 2 层）：
```
单层单向：4 × 128 × (128 + 128 + 1) = 131,584
双向 = ×2 = 263,168
第 2 层输入是 2H：4 × 128 × (256 + 128 + 1) = 196,608 × 2 = 393,216（双向）
Embedding：20000 × 128 = 2,560,000
FC：256 × 4 + 4 = 1,028
总计 ≈ 3.2M 参数
```

### 2.4 BPTT 中的梯度分析

损失对细胞状态的梯度（简化版）：

```
∂L/∂c_t = ∂L/∂c_{t+1} · f_{t+1} + (局部贡献)
```

这是一个**加法递推**，不再是矩阵乘法链。在 `f_t ≈ 1` 时，梯度几乎原样传递，有效解决梯度消失。

但注意：`f_t` 是 sigmoid 输出，**不是常数**；当 `f_t ≈ 0` 时，遗忘门关闭，当前时间步之前的梯度被截断（这正是"遗忘"的语义，但也意味着极长序列仍可能有问题）。

---

## 3. 双向 LSTM（BiLSTM）

### 3.1 原理

文本理解需要同时利用**左侧上下文**和**右侧上下文**：
- 正向 LSTM：`h_t^→ = LSTM_fwd(x_t, h_{t-1}^→)`（从左到右）
- 反向 LSTM：`h_t^← = LSTM_bwd(x_t, h_{t+1}^←)`（从右到左）
- 拼接：`h_t = [h_t^→ ; h_t^←] ∈ ℝ^{2H}`

正向 LSTM 在位置 t 能看到 `x_1, …, x_t`，反向能看到 `x_t, …, x_T`，
拼接后 `h_t` 编码了**完整的双向上下文信息**。

### 3.2 维度变化

```
输入 (B, L, E)
  → 正向 LSTM → (B, L, H)
  → 反向 LSTM → (B, L, H)
  → 拼接     → (B, L, 2H)   ← nn.LSTM(bidirectional=True) 自动完成
  → 均值池化 → (B, 2H)
  → FC(2H, C)→ (B, C)
```

nn.LSTM 返回的 hidden/cell shape 变为 `(num_layers * 2, B, H)`，
其中偶数层索引 [0,2,4,...] 是正向，奇数 [1,3,5,...] 是反向。

### 3.3 何时用双向

| 任务 | 推荐 | 原因 |
|------|------|------|
| 文本分类 | BiLSTM ✓ | 全局理解，右侧信息同样重要 |
| 序列标注（NER/POS） | BiLSTM ✓ | 每个位置都需要左右上下文 |
| 语言模型（文本生成） | 单向 ✗ | 预测下一词时不能"看"到未来 |
| 机器翻译（编码器） | BiLSTM ✓ | 编码器可见完整源句 |
| 机器翻译（解码器） | 单向 ✗ | 自回归生成，不能看未来 |

---

## 4. 架构总览

```
输入序列（已填充到 max_len=64）
┌──────────────────────────────────────────────────────┐
│  token ids: [42, 87, 3, ..., 1, 1]  shape: (B, 64)  │
└──────────────────┬───────────────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │  Embedding (+ Dropout)    │
         │  (B, 64) → (B, 64, 128)  │
         │  vocab=20000, dim=128     │
         └─────────┬──────────┘
                   │ (B, 64, 128)
    ┌──────────────▼───────────────────────────────┐
    │           BiLSTM Layer 1                      │
    │  正向: → → → → → → → → → → → → → →          │
    │  反向: ← ← ← ← ← ← ← ← ← ← ← ← ← ←         │
    │  输出: (B, 64, 256)  [128 正向 + 128 反向]   │
    │           层间 Dropout(0.5)                   │
    └──────────────┬───────────────────────────────┘
                   │ (B, 64, 256)
    ┌──────────────▼───────────────────────────────┐
    │           BiLSTM Layer 2                      │
    │  正向: → → → → → → → → → → → → → →          │
    │  反向: ← ← ← ← ← ← ← ← ← ← ← ← ← ←         │
    │  输出: (B, 64, 256)                           │
    └──────────────┬───────────────────────────────┘
                   │ (B, 64, 256)
         ┌─────────▼──────────┐
         │    均值池化         │
         │  .mean(dim=1)      │
         │  (B, 64, 256)      │
         │       ↓            │
         │    (B, 256)        │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │  Dropout(0.5)      │
         │  FC: 256 → 4       │
         └─────────┬──────────┘
                   │ (B, 4)
         ┌─────────▼──────────┐
         │   CrossEntropyLoss │
         │   argmax → 类别    │
         └────────────────────┘
```

---

## 5. nn.LSTM 源码解析

### 5.1 初始化参数

```python
nn.LSTM(
    input_size,      # 每个时间步的输入维度（词向量维度 E）
    hidden_size,     # 隐藏状态 h 的维度（H）
    num_layers=1,    # 堆叠层数
    batch_first=True,# True: 输入形状 (B,L,E)；False: (L,B,E)
    dropout=0.0,     # 层间 Dropout（仅在 num_layers>1 时有效，最后一层无 Dropout）
    bidirectional=False,  # 是否双向
    proj_size=0,     # >0 时启用 LSTM with projection（降维，少见）
)
```

**参数存储**（以 `num_layers=2, bidirectional=True` 为例）：
- `lstm.weight_ih_l0`：第 0 层正向，形状 `(4H, E)`，对应 4 个门的输入权重
- `lstm.weight_hh_l0`：第 0 层正向，形状 `(4H, H)`，对应 4 个门的隐状态权重
- `lstm.bias_ih_l0`：`(4H,)`
- `lstm.bias_hh_l0`：`(4H,)`
- `lstm.weight_ih_l0_reverse`：第 0 层反向（bidirectional=True 时存在）
- 以此类推到 `l1`, `l1_reverse`

```python
# 访问遗忘门权重（切片 H:2H）
W_f = lstm.weight_ih_l0[H:2*H, :]   # 输入→遗忘门
W_hf = lstm.weight_hh_l0[H:2*H, :]  # 隐状态→遗忘门
```

### 5.2 前向传播返回值

```python
output, (hidden, cell) = lstm(input, (h_0, c_0))
# input:  (B, L, E)  (batch_first=True)
# output: (B, L, H*num_directions)  — 所有时间步的 h_t
# hidden: (num_layers*num_dir, B, H) — 最后时间步的 h_t
# cell:   (num_layers*num_dir, B, H) — 最后时间步的 c_t

# 若不提供初始状态，默认 h_0=c_0=zeros
output, (hidden, cell) = lstm(input)
```

取最后一层的双向隐状态（用于 sequence-to-vector 任务）：
```python
# 最后一层正向的最终隐状态：hidden[-2] （num_layers=2 时索引为 2）
# 最后一层反向的最终隐状态：hidden[-1]
last_fwd = hidden[-2]  # (B, H)
last_bwd = hidden[-1]  # (B, H)
last_combined = torch.cat([last_fwd, last_bwd], dim=-1)  # (B, 2H)

# 本项目用均值池化，等价但梯度更均匀：
pooled = output.mean(dim=1)  # (B, 2H)
```

### 5.3 pack_padded_sequence（进阶）

标准做法对填充位置也运行 LSTM（浪费计算，且填充 token 会影响 hidden state）。
更精确的方式：

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

lengths = (texts != pad_idx).sum(dim=1).cpu()  # 每个样本真实长度
packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
output_packed, (hidden, cell) = lstm(packed)
output, _ = pad_packed_sequence(output_packed, batch_first=True)
```

本项目为简洁性不使用 pack（影响约 1%），但在实际项目中建议加入。

---

## 6. 项目结构

```
LSTM/
├── config.py      # 超参数（hidden_size, num_layers, bidirectional, …）
├── model.py       # LSTMClassifier
├── dataset.py     # AG News 下载 + 词汇表 + DataLoader
├── train.py       # 训练循环（含梯度裁剪）
├── test.py        # 测试评估 + 混淆矩阵 + 抽样预测
├── utils.py       # ExperimentLogger（CSV + 训练曲线图）
├── data/          # AG News CSV（首次运行自动下载）
├── checkpoints/   # best_model.pth
└── logs/          # metrics.csv + training_curves.png + confusion_matrix.png
```

---

## 7. Quick Start

### 版本兼容性

| 包 | 版本 | 说明 |
|----|------|------|
| Python | 3.9 – 3.11 | 推荐 3.10 |
| torch | 2.3.1 | MPS / CUDA / CPU |
| torchvision | 0.18.1 | 与 torch 2.3.1 配套 |
| torchtext | 0.18.0 | 仅用 tokenizer 和 vocab |
| numpy | >=1.24,<2.0 | torch 2.3.1 要求 <2.0 |
| matplotlib | >=3.7,<4.0 | 绘图 |

### macOS（Apple Silicon M 系列）

```bash
# 在项目根目录创建虚拟环境（与 MLP/CNN/RNN 共享）
python3 -m venv ../.venv
source ../.venv/bin/activate

pip install torch==2.3.1 torchvision==0.18.1
pip install torchtext==0.18.0
pip install numpy==1.26.4 matplotlib

# 验证 MPS 可用
python3 -c "import torch; print(torch.backends.mps.is_available())"  # True

cd LSTM
python train.py   # 首次运行自动下载 AG News（约 30MB）
python test.py
```

### Windows（NVIDIA GPU，CUDA 12.1）

```bash
python -m venv ..\.venv
..\.venv\Scripts\activate

# 必须先安装 CUDA 版 torch（pip install -r requirements.txt 会装 CPU 版）
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install torchtext==0.18.0
pip install numpy==1.26.4 matplotlib

# 验证 CUDA 可用
python -c "import torch; print(torch.cuda.is_available())"  # True

cd LSTM
python train.py
python test.py
```

### Windows（CPU）

```bash
python -m venv ..\.venv
..\.venv\Scripts\activate
pip install torch==2.3.1 torchvision==0.18.1 torchtext==0.18.0 numpy==1.26.4 matplotlib

cd LSTM
python train.py
```

### 预期训练输出

```
使用设备: mps

词汇表已从缓存加载（大小: 20000）

训练集: 120,000 条  测试集: 7,600 条
类别: ['World', 'Sports', 'Business', 'Sci/Tech']
训练 batch 数: 1875  测试 batch 数: 119

模型参数总量: 3,217,412  (双向 LSTM, 2 层)

============================================================
开始训练
============================================================

[Epoch 01/10]  学习率: 0.001000
  ...
  训练损失: 0.4521  训练准确率: 83.21%
  测试损失: 0.3892  测试准确率: 86.45%
  ✓ 保存最优模型（准确率: 86.45%）
...
训练完成！最优测试准确率: 88.XX%
```

---

## 8. 超参数调优指南

### 8.1 核心超参数影响

| 参数 | 默认值 | 增大效果 | 减小效果 |
|------|--------|---------|---------|
| `hidden_size` | 128 | 模型容量↑，过拟合风险↑ | 欠拟合风险↑，速度↑ |
| `num_layers` | 2 | 提取更抽象特征，训练变慢 | 更快，但表达能力有限 |
| `max_len` | 64 | 捕获更长依赖，计算量↑ | 截断长文本信息丢失 |
| `dropout_rate` | 0.5 | 正则化更强，欠拟合风险↑ | 过拟合风险↑ |
| `bidirectional` | True | 准确率通常提升 1-3% | 速度快 1.5-2x |
| `batch_size` | 64 | 训练稳定性↑，内存占用↑ | 梯度噪声↑（正则效果） |

### 8.2 双向 vs 单向

```python
# 关闭双向（速度快，内存省）
bidirectional = False
# hidden_size 可相应增大补偿容量损失：
hidden_size = 256  # 单向 256 ≈ 双向 128 参数量接近
```

### 8.3 LSTM 层数选择

```
1 层：适合短文本（max_len ≤ 64），训练最快
2 层：大多数 NLP 分类任务的最佳点
3+ 层：一般不会继续提升，且容易过拟合；Transformer 完全替代了这个需求
```

### 8.4 学习率调度

当前使用 StepLR（每 3 epoch 学习率减半）。可以尝试：

```python
# ReduceLROnPlateau：验证损失不再下降时自动调整
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2, verbose=True
)
# 注意：step() 需要传入验证损失
scheduler.step(val_loss)
```

### 8.5 常见问题排查

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| 训练损失不下降（~1.386） | 梯度消失 / 学习率太小 | 检查 grad_clip，增大 lr |
| 训练准确率高但测试低 5%+ | 过拟合 | 增大 dropout，减小 hidden_size |
| 训练/测试损失都高 | 欠拟合 | 增大 hidden_size 或 num_layers |
| GPU 内存溢出 | batch_size 太大 | 减小 batch_size（32 或 16）|
| 训练曲线剧烈震荡 | 学习率太大 | 降低 lr（从 1e-3 到 5e-4） |

---

## 9. RNN vs LSTM 对比

| 特性 | vanilla RNN | LSTM |
|------|-------------|------|
| 隐藏状态 | h_t（单一） | h_t + c_t（分离短/长期） |
| 门控机制 | 无 | 遗忘门 / 输入门 / 输出门 |
| 梯度消失 | 严重（指数级） | 大幅缓解（加法更新） |
| 梯度爆炸 | 有（需 clip） | 有（需 clip，但较轻） |
| 参数量 | H(H+E+1) | 4H(H+E+1) |
| 长序列能力 | max_len ≤ 50 实用 | max_len ≤ 500+ 可用 |
| 适用场景 | 教学、简单短序列 | 大多数实际 NLP 任务 |
| 现代替代品 | Transformer | Transformer |

**为什么仍然学 LSTM？**

1. LSTM 是 Transformer Attention 机制的重要前驱——理解门控即理解 Attention 的前身
2. 许多嵌入式/边缘设备场景中，LSTM 比 Transformer 更轻量高效
3. 时间序列预测（非 NLP）中，LSTM 仍然广泛使用
4. Transformer 的残差连接（Residual）和 LayerNorm 解决了与 LSTM 相同的问题，但路径不同——对比学习加深理解
