# RNN：循环神经网络 — AG News 新闻文本分类（~91%）

**任务**：将新闻标题+摘要分为 4 类（World / Sports / Business / Sci/Tech）  
**核心目标**：理解 RNN 如何用隐藏状态 $h_t$ 建模序列的时序依赖关系

---

## 1. 为什么需要 RNN？MLP/CNN 的局限

| 模型 | 输入假设 | 局限 |
|------|---------|------|
| MLP | 固定长度向量，特征无序 | 无法处理变长序列，忽略词序 |
| CNN | 固定尺寸网格，局部感受野 | 感受野有限，难以捕获长距离依赖 |
| **RNN** | **变长序列，按时间步处理** | **天然适合文本、时间序列** |

MLP 把一句话的所有词压成一个向量再处理，丢失了"词序"这一关键信息。  
RNN 则按时间顺序逐词处理，将之前的信息"记在"隐藏状态 $h_t$ 中传递下去。

---

## 2. RNN 数学公式推导

### 2.1 单步递推

给定时间步 $t$ 的输入 $x_t \in \mathbb{R}^{d}$ 和上一时刻隐藏状态 $h_{t-1} \in \mathbb{R}^{H}$：

$$\boxed{h_t = \tanh\!\left(x_t W_{ih}^{\top} + b_{ih} + h_{t-1} W_{hh}^{\top} + b_{hh}\right)}$$

| 参数 | 形状 | 含义 |
|------|------|------|
| $W_{ih}$ | $(H, d)$ | 输入到隐藏的权重 |
| $W_{hh}$ | $(H, H)$ | 隐藏到隐藏的权重（时序传递的关键） |
| $b_{ih}, b_{hh}$ | $(H,)$ | 偏置 |

$h_0 = \mathbf{0}$（初始隐藏状态为全零）

### 2.2 展开示意（序列长度 T）

```
x_1  →  [RNN cell]  →  h_1
             ↑ h_0
x_2  →  [RNN cell]  →  h_2
             ↑ h_1
x_3  →  [RNN cell]  →  h_3
             ↑ h_2
  ...
x_T  →  [RNN cell]  →  h_T  →  FC  →  ŷ
             ↑ h_{T-1}
```

**分类任务**：只取最后时刻的隐藏状态 $h_T$ 作为整个序列的表示，再接全连接层分类。

### 2.3 多层堆叠 RNN

第 $\ell$ 层使用上一层的输出作为输入：

$$h_t^{(\ell)} = \tanh\!\left(h_t^{(\ell-1)} W_{ih}^{(\ell)\top} + b_{ih}^{(\ell)} + h_{t-1}^{(\ell)} W_{hh}^{(\ell)\top} + b_{hh}^{(\ell)}\right)$$

本项目使用 **2 层 RNN**，最终取 $h_T^{(2)}$ 用于分类。

---

## 3. 反向传播：BPTT（时间反向传播）

RNN 的反向传播需要沿时间轴展开计算，称为 **Backpropagation Through Time（BPTT）**。

### 梯度链式法则

损失对 $h_t$ 的梯度：

$$\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_{t+1}} \cdot \frac{\partial h_{t+1}}{\partial h_t} + \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h_t}$$

展开到第 1 步：

$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \cdot \prod_{t=2}^{T} W_{hh} \cdot \text{diag}(\tanh'(h_t))$$

### 梯度消失 / 梯度爆炸

连乘 $\prod_{t=2}^{T} W_{hh}$ 是问题的根源：

| 情形 | $\|W_{hh}\|$ | 后果 | 解决方案 |
|------|-------------|------|---------|
| 梯度消失 | $< 1$ | 早期时间步梯度趋近于 0，无法学习长距离依赖 | **LSTM / GRU**（下一项目） |
| 梯度爆炸 | $> 1$ | 梯度无限增大，参数更新不稳定 | **梯度裁剪（Gradient Clipping）** |

**梯度裁剪**（本项目使用）：

$$\text{if } \|\mathbf{g}\|_2 > \text{max\_norm}: \quad \mathbf{g} \leftarrow \mathbf{g} \cdot \frac{\text{max\_norm}}{\|\mathbf{g}\|_2}$$

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

---

## 4. 网络结构图

```
输入文本: "Stocks fall on rate hike fears"
         ↓
┌─────────────────────────────────────────────────────────┐
│  Tokenizer（分词）                                        │
│  → ["stocks", "fall", "on", "rate", "hike", "fears"]    │
└─────────────────────────────────────────────────────────┘
         ↓  token id 序列 (B, L=128)
┌─────────────────────────────────────────────────────────┐
│  Embedding  nn.Embedding(20000, 128)                     │
│  每个 id → 128 维向量                                     │
└─────────────────────────────────────────────────────────┘
         ↓  (B, L, E=128)
┌─────────────────────────────────────────────────────────┐
│  RNN Layer 1  nn.RNN(128→256, batch_first=True)          │
│  h_t = tanh(x_t W_ih + h_{t-1} W_hh + b)               │
└─────────────────────────────────────────────────────────┘
         ↓  (B, L, H=256)      Dropout(0.5)（层间）
┌─────────────────────────────────────────────────────────┐
│  RNN Layer 2  nn.RNN(256→256, batch_first=True)          │
│  最终隐藏状态 h_T^(2): (B, H=256)                         │
└─────────────────────────────────────────────────────────┘
         ↓  取 hidden[-1]: (B, 256)
         ↓  Dropout(0.5)
┌─────────────────────────────────────────────────────────┐
│  FC  nn.Linear(256, 4)                                   │
└─────────────────────────────────────────────────────────┘
         ↓  logits (B, 4)
    CrossEntropyLoss → 反向传播 + 梯度裁剪
```

**参数量估算**：

| 层 | 参数量 |
|----|--------|
| Embedding(20000, 128) | 2,560,000 |
| RNN Layer1(128→256, 2层 combined) | 128×256 + 256×256 + 2×256 ≈ 99K |
| RNN Layer2(256→256) | 256×256 + 256×256 + 2×256 ≈ 131K |
| FC(256→4) | 1,028 |
| **总计** | **~2.8M** |

---

## 5. PyTorch 源码解析

### 5.1 `nn.Embedding`

```python
class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        # weight 形状：(num_embeddings, embedding_dim)
        # 每一行对应一个 token 的向量表示
        self.weight = Parameter(torch.empty(num_embeddings, embedding_dim))

    def forward(self, input):
        # input: (B, L)  → 查表 → (B, L, embedding_dim)
        return F.embedding(input, self.weight, self.padding_idx)
```

`padding_idx`：该位置对应的行向量梯度为 0（`<pad>` 位不参与学习）。

### 5.2 `nn.RNN`

```python
nn.RNN(
    input_size,           # 输入维度 d（= embed_dim）
    hidden_size,          # 隐藏层维度 H
    num_layers=1,         # 堆叠层数
    nonlinearity='tanh',  # 激活函数（tanh 或 relu）
    batch_first=False,    # True → 输入形状 (B, L, d)，False → (L, B, d)
    dropout=0,            # 层间 dropout（只对非最后层生效）
    bidirectional=False,  # 双向 RNN
)
```

**前向输出**：
```python
output, hidden = rnn(x)
# output: (B, L, H)         — 所有时间步的隐藏状态
# hidden: (num_layers, B, H) — 最后时间步的隐藏状态（用于分类）
```

### 5.3 `pad_sequence`（批处理变长序列）

```python
from torch.nn.utils.rnn import pad_sequence

# 将一批长度不同的序列填充为相同长度
texts_padded = pad_sequence(texts, batch_first=True, padding_value=1)
# padding_value=1 对应 <pad> 的词汇表索引
```

### 5.4 梯度裁剪

```python
# 在 loss.backward() 之后、optimizer.step() 之前调用
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
# 等价操作：将所有参数梯度的 L2 范数缩放到 <= 5.0
```

---

## 6. 快速开始

### 环境依赖

```bash
pip install torch torchvision torchtext
```

> torchtext 版本需与 PyTorch 匹配，参考 https://github.com/pytorch/text#installation

### macOS（Apple Silicon，MPS 加速）

```bash
cd RNN
python train.py
```

首次运行会自动下载 AG News 数据集（~30MB）并构建词汇表缓存。  
后续运行直接从缓存加载，无需重复下载。

预期输出（M4 Pro）：

```
使用设备: mps

词汇表已从缓存加载（大小: 20000）
训练集: 120,000 条  测试集: 7,600 条
...
[Epoch 01/10]  学习率: 0.001000
  ...
  测试准确率: 88.xx%
...
训练完成！最优测试准确率: ~91.xx%
```

训练结束后运行测试：

```bash
python test.py
```

### Windows（NVIDIA GPU）

```bat
cd RNN
python train.py
```

代码会自动检测 CUDA 并使用 GPU 加速。

### Windows（CPU）

```bat
cd RNN
python train.py
```

CPU 训练约 3-5 分钟/epoch（词汇表较小，AG News 数据量适中）。

---

## 7. 调参指南

### 7.1 提升准确率

| 调整项 | 当前值 | 建议方向 |
|--------|-------|---------|
| `embed_dim` | 128 | 增大到 256（更丰富的词表示） |
| `hidden_size` | 256 | 增大到 512 |
| `num_layers` | 2 | 增加到 3（收益递减，注意梯度消失） |
| `max_len` | 128 | 增大到 256（捕获更多上下文） |
| `vocab_size` | 20000 | 增大到 30000 |
| 模型架构 | 单向 RNN | 改用 **LSTM**（下一项目）可提升 2-3% |

### 7.2 梯度裁剪调参

`grad_clip = 5.0` 是经验值。观察训练损失曲线：

- 如果损失震荡很大 → 减小 `grad_clip`（如 1.0）
- 如果损失下降很慢 → 适当增大 `grad_clip`（如 10.0）
- 监控梯度范数：在 `clip_grad_norm_` 前加一行 `print(grad_norm)` 观察典型值

### 7.3 RNN 的固有局限（为什么 LSTM 更好）

Vanilla RNN 即使有梯度裁剪，**梯度消失**问题依然存在：

- 短文本（AG News 标题 ~20 词）：RNN 表现良好
- 长文本（影评 ~500 词）：RNN 无法记住开头的关键词，准确率显著下降
- LSTM 通过"遗忘门 + 输入门 + 输出门"解决了梯度消失，是 RNN 的标准升级版

---

## 8. RNN vs LSTM vs CNN（AG News 基准对比）

| 模型 | AG News 准确率 | 训练速度 | 长文本能力 |
|------|--------------|---------|----------|
| FastText（词袋 MLP） | ~92% | 极快 | 无（忽略词序） |
| **Vanilla RNN（本项目）** | **~91%** | 中等 | 弱（梯度消失） |
| LSTM | ~93% | 较慢 | 强 |
| TextCNN | ~93% | 快 | 中等 |
| BERT fine-tune | ~95%+ | 慢 | 极强 |

AG News 是相对简单的任务（标题短、类别明确），因此 RNN 已接近上限。  
**LSTM 的优势在长序列任务上更为明显。**
