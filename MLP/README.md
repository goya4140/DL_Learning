# MLP：多层感知机（Multi-Layer Perceptron）

**任务**：MNIST 手写数字识别（10 分类）  
**框架**：PyTorch  
**准确率**：~98%（20 epoch）

---

## 目录结构

```
MLP/
├── config.py       # 超参数配置
├── model.py        # 模型定义
├── dataset.py      # 数据集加载
├── train.py        # 训练脚本
├── test.py         # 测试与可视化
└── README.md       # 本文档
```

---

## 1. 数学公式推导

### 1.1 网络结构

本项目的 MLP 结构（以 3 个隐藏层为例）：

```
输入层 (784) → 隐藏层1 (512) → 隐藏层2 (256) → 隐藏层3 (128) → 输出层 (10)
```

每层之间的变换：线性变换 → 批归一化 → ReLU 激活 → Dropout

### 1.2 前向传播（Forward Pass）

设第 $l$ 层的权重矩阵为 $W^{(l)}$，偏置为 $b^{(l)}$，上一层输出为 $a^{(l-1)}$。

**第 1 步：线性变换**

$$z^{(l)} = W^{(l)} \cdot a^{(l-1)} + b^{(l)}$$

- $W^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}$，$b^{(l)} \in \mathbb{R}^{d_l}$
- $d_l$ 是第 $l$ 层的神经元数

**第 2 步：批归一化（Batch Normalization）**

$$\hat{z}^{(l)} = \frac{z^{(l)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \cdot \gamma + \beta$$

- $\mu_B$，$\sigma_B^2$ 是当前 batch 的均值和方差
- $\gamma$，$\beta$ 是可学习的缩放和偏移参数
- 作用：加速收敛，减少对初始化的敏感性

**第 3 步：ReLU 激活**

$$a^{(l)} = \text{ReLU}(\hat{z}^{(l)}) = \max(0, \hat{z}^{(l)})$$

ReLU 解决了 Sigmoid/Tanh 的梯度消失问题，计算高效。

**第 4 步：Dropout（仅训练时）**

$$a^{(l)} = \frac{1}{1-p} \cdot \text{mask} \odot a^{(l)}, \quad \text{mask}_i \sim \text{Bernoulli}(1-p)$$

- $p$ 是丢弃概率，除以 $(1-p)$ 保持期望值不变（inverted dropout）
- 推理时关闭 Dropout，直接输出 $a^{(l)}$

**输出层（无激活）**

$$z^{(L)} = W^{(L)} \cdot a^{(L-1)} + b^{(L)} \in \mathbb{R}^{10}$$

这10个值称为 **logits**（对数几率）。

### 1.3 Softmax 与交叉熵损失

**Softmax**（将 logits 转为概率）：

$$p_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \quad k = 1, \ldots, K$$

**交叉熵损失**（Cross-Entropy Loss）：

$$\mathcal{L} = -\sum_{k=1}^{K} y_k \log p_k = -\log p_y$$

- $y_k$ 是 one-hot 标签（真实类别为1，其余为0）
- 合并后：$\mathcal{L} = -z_y + \log\sum_{j=1}^{K} e^{z_j}$（数值稳定版本）

> PyTorch 的 `nn.CrossEntropyLoss` 已将 Softmax 和交叉熵合并，输入 logits 即可，无需手动 Softmax。

### 1.4 反向传播（Backpropagation）

通过链式法则计算每层参数的梯度：

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial z^{(l)}} \cdot (a^{(l-1)})^\top$$

$$\frac{\partial \mathcal{L}}{\partial a^{(l-1)}} = (W^{(l)})^\top \cdot \frac{\partial \mathcal{L}}{\partial z^{(l)}}$$

其中误差信号 $\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial z^{(l)}}$ 从输出层向输入层逐层传递。

### 1.5 Adam 优化器

比 SGD 更鲁棒，自适应调整每个参数的学习率：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{（一阶矩，梯度均值）}$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{（二阶矩，梯度方差）}$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{（偏差修正）}$$

$$W \leftarrow W - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

默认参数：$\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$

---

## 2. 网络结构图（文字版）

```
输入图片 (28×28 灰度图)
        │
        ▼
   ┌─────────────────────────────────────────┐
   │  展平 (Flatten)                         │
   │  (B, 1, 28, 28) → (B, 784)             │
   └─────────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────────┐
   │  全连接层 1 (Linear 784 → 512)          │
   │  批归一化 (BatchNorm1d 512)             │
   │  激活函数 (ReLU)                        │
   │  正则化  (Dropout p=0.3)               │
   └─────────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────────┐
   │  全连接层 2 (Linear 512 → 256)          │
   │  批归一化 (BatchNorm1d 256)             │
   │  激活函数 (ReLU)                        │
   │  正则化  (Dropout p=0.3)               │
   └─────────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────────┐
   │  全连接层 3 (Linear 256 → 128)          │
   │  批归一化 (BatchNorm1d 128)             │
   │  激活函数 (ReLU)                        │
   │  正则化  (Dropout p=0.3)               │
   └─────────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────────┐
   │  输出层 (Linear 128 → 10)              │
   │  （logits，不加激活函数）               │
   └─────────────────────────────────────────┘
        │
        ▼
   CrossEntropyLoss（内含 Softmax）
        │
        ▼
   预测类别 argmax(logits)
```

---

## 3. PyTorch 源码解析

### 3.1 `nn.Linear` — 全连接层

```python
# PyTorch 源码（简化版）
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.weight = Parameter(torch.empty(out_features, in_features))
        self.bias   = Parameter(torch.empty(out_features)) if bias else None
        # 权重初始化：kaiming_uniform_（专为 ReLU 设计）
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        # 等价于 input @ weight.T + bias
        return F.linear(input, self.weight, self.bias)
```

**关键理解**：
- `weight` 形状为 `(out_features, in_features)`，实际计算时转置
- 默认使用 Kaiming 初始化，适配 ReLU 激活，防止梯度消失/爆炸

### 3.2 `nn.BatchNorm1d` — 批归一化

```python
class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.weight = Parameter(torch.ones(num_features))   # γ
        self.bias   = Parameter(torch.zeros(num_features))  # β
        self.running_mean = torch.zeros(num_features)       # 推理时使用
        self.running_var  = torch.ones(num_features)

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)                      # 当前 batch 均值
            var  = x.var(dim=0, unbiased=False)       # 当前 batch 方差
            # 更新运行统计量（用于推理）
            self.running_mean = (1 - momentum) * self.running_mean + momentum * mean
            self.running_var  = (1 - momentum) * self.running_var  + momentum * var
        else:
            mean, var = self.running_mean, self.running_var

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias       # γ * x_norm + β
```

### 3.3 `nn.ReLU` — 激活函数

```python
class ReLU(Module):
    def forward(self, input):
        return F.relu(input)  # 底层调用 torch.clamp(input, min=0)
```

反向传播时的梯度：$\frac{d\text{ReLU}}{dx} = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$

### 3.4 `nn.Dropout` — 随机丢弃

```python
class Dropout(Module):
    def __init__(self, p=0.5):
        self.p = p

    def forward(self, input):
        if self.training:
            # 生成伯努利掩码，并做 inverted dropout 缩放
            mask = torch.bernoulli(torch.full_like(input, 1 - self.p))
            return input * mask / (1 - self.p)
        return input  # 推理时直接返回，不做任何操作
```

### 3.5 `nn.CrossEntropyLoss`

```python
# 内部等价于：
def cross_entropy(logits, target):
    log_softmax = logits - logits.logsumexp(dim=1, keepdim=True)  # 数值稳定版 log(softmax)
    return -log_softmax[range(len(target)), target].mean()
```

> 注意：`CrossEntropyLoss` 的输入是 **logits**（未经 Softmax 的原始输出），不要在模型输出后手动加 Softmax。

### 3.6 `model.eval()` vs `model.train()`

| 方法 | BatchNorm 行为 | Dropout 行为 |
|------|--------------|-------------|
| `model.train()` | 使用当前 batch 统计量 | 随机丢弃神经元 |
| `model.eval()` | 使用运行时统计量 | 不丢弃，直接输出 |

> 训练时调用 `model.train()`，评估/推理时必须调用 `model.eval()`。

---

## 4. 快速开始

```bash
# 安装依赖
pip install torch torchvision matplotlib

# 训练（自动下载 MNIST 数据集）
python train.py

# 测试（需先完成训练）
python test.py
```

---

## 5. 调参指南

### 5.1 学习率（最重要）

| 学习率 | 效果 |
|--------|------|
| > 1e-2 | 损失震荡，难以收敛 |
| **1e-3（默认）** | 收敛稳定，推荐起点 |
| 1e-4 | 收敛慢，但最终结果可能更好 |
| < 1e-5 | 几乎不更新，欠拟合 |

**建议**：从 `1e-3` 开始，配合学习率调度（StepLR / CosineAnnealingLR）逐步降低。

### 5.2 隐藏层结构

```python
# 较小网络（快速验证）
hidden_sizes = [256, 128]

# 默认网络（均衡）
hidden_sizes = [512, 256, 128]

# 较大网络（可能过拟合，需增大 dropout）
hidden_sizes = [1024, 512, 256, 128]
```

层数越深、宽度越大，拟合能力越强，但过深会导致训练困难（梯度消失），应配合 BatchNorm 使用。

### 5.3 Dropout 比例

| Dropout 值 | 适用场景 |
|-----------|---------|
| 0.0 | 数据充足或模型较小时 |
| **0.3（默认）** | 通用场景 |
| 0.5 | 模型较大、过拟合明显时 |
| > 0.7 | 通常过度正则化，不推荐 |

### 5.4 Batch Size 影响

| Batch Size | 显存占用 | 训练速度 | 泛化性 |
|-----------|---------|---------|--------|
| 16~32 | 低 | 慢 | 较好（引入噪声） |
| **64（默认）** | 中 | 中 | 均衡 |
| 256~512 | 高 | 快 | 可能略差 |

### 5.5 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 训练损失不下降 | 学习率过大/过小 | 尝试 `1e-3`，观察前5轮趋势 |
| 训练准确高，测试准确低 | 过拟合 | 增大 dropout，减少层数 |
| 训练和测试准确都低 | 欠拟合 | 增加网络宽度/深度，减小 dropout |
| 损失出现 NaN | 学习率过大或数值溢出 | 降低学习率，检查数据预处理 |
| BatchNorm 报错 | batch_size=1 | BatchNorm 需要 batch_size ≥ 2 |

---

## 6. 关键 API 速查

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义模型
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
)

# 损失函数
criterion = nn.CrossEntropyLoss()   # 分类任务
# criterion = nn.MSELoss()          # 回归任务

# 优化器
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# 训练一步
logits = model(x)                   # 前向传播
loss = criterion(logits, y)         # 计算损失
optimizer.zero_grad()               # 清零梯度（重要！否则梯度累积）
loss.backward()                     # 反向传播
optimizer.step()                    # 更新参数

# 推理
model.eval()
with torch.no_grad():               # 关闭梯度，节省显存
    pred = model(x).argmax(dim=1)

# 保存/加载模型
torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth"))
```
