# CNN：卷积神经网络（Convolutional Neural Network）

**任务**：CIFAR-10 图像分类（10 分类，彩色图像）  
**框架**：PyTorch  
**准确率**：~85%（30 epoch）  
**对比 MLP**：MLP 在 CIFAR-10 上约 55%，CNN 利用空间结构达到 85%+

---

## 目录结构

```
CNN/
├── config.py       # 超参数配置
├── model.py        # CNN 模型定义
├── dataset.py      # CIFAR-10 数据集加载（含数据增强）
├── train.py        # 训练脚本
├── test.py         # 测试与可视化
├── utils.py        # 实验日志（ExperimentLogger）
└── README.md       # 本文档
```

---

## 1. 数学公式推导

### 1.1 为什么需要 CNN？MLP 的局限性

对于 32×32×3 的 CIFAR-10 图片，如果用 MLP：
- 输入维度：$32 \times 32 \times 3 = 3072$
- 第一个全连接层（3072→512）的参数量：$3072 \times 512 + 512 = \mathbf{1{,}573{,}376}$

**两个根本问题**：
1. **参数爆炸**：输入尺寸增大，参数量以 $O(n^2)$ 增长
2. **忽略空间结构**：全连接层将图像视为一维向量，丢失了像素间的空间关联（相邻像素构成边缘、纹理等特征）

卷积的解决方案：**局部连接 + 权重共享**，用 $k \times k$ 的卷积核提取局部特征，卷积核在整张图上**滑动共享**同一组参数。

### 1.2 2D 卷积公式

**单输入通道，单输出通道**（最简单情形）：

$$O[i,j] = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I[i+m,\ j+n] \cdot K[m,n] + b$$

- $I$：输入特征图，$K$：卷积核（kernel），$b$：偏置
- $k$：卷积核大小（本项目用 $k=3$）
- $O[i,j]$：输出特征图在 $(i,j)$ 位置的值

**多输入通道，多输出通道**（实际使用）：

$$O_f[i,j] = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I_c[i+m,\ j+n] \cdot K_{f,c}[m,n] + b_f$$

- $C_{in}$：输入通道数（CIFAR-10 RGB 为 3）
- $f$：第 $f$ 个输出特征图（输出通道）
- 每个输出通道有一组独立的卷积核 $K_{f,:,:,:}$，形状为 $(C_{in}, k, k)$

### 1.3 特征图尺寸计算

经过卷积（padding=$p$，stride=$s$，kernel=$k$）后的输出尺寸：

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1$$

**本项目各层尺寸变化**（使用 $k=3, p=1, s=1$，尺寸保持不变；MaxPool 2×2 减半）：

```
输入:         (3,  32, 32)
Conv1(3→32):  (32, 32, 32)   # floor((32+2*1-3)/1)+1 = 32，保持不变
MaxPool:      (32, 16, 16)   # 减半
Conv2(32→64): (64, 16, 16)   # 保持不变
MaxPool:      (64,  8,  8)
Conv3(64→128):(128, 8,  8)
MaxPool:      (128, 4,  4)
Flatten:      2048           # 128 × 4 × 4
```

### 1.4 感受野（Receptive Field）

感受野指输出特征图上的一个像素对应输入图像的区域大小。

每加一层 $3 \times 3$ 卷积（stride=1），感受野增加 2：

$$RF_l = RF_{l-1} + 2 \times \prod_{i=1}^{l-1} s_i$$

**本项目（含 MaxPool 的等效感受野）**：

| 层 | 等效感受野 |
|----|---------|
| Conv Block 1 后 | 3×3 |
| Conv Block 2 后 | 10×10（含 MaxPool 的扩大） |
| Conv Block 3 后 | 22×22 |

深层神经元能"看到"输入图像中 22×22 的区域，足以捕获复杂的图像特征。

### 1.5 MaxPool 公式

$$O[i,j] = \max_{0 \leq m < k,\ 0 \leq n < k} I[s \cdot i + m,\ s \cdot j + n]$$

取 $k \times k$ 区域内的最大值。本项目 $k=2, s=2$：每个 $2 \times 2$ 区域取最大值，尺寸减半。

**反向传播**：梯度只流向取最大值的那个像素，其余置 0（Max-Routing）。

### 1.6 卷积的反向传播

对卷积核的梯度（用于更新参数）：

$$\frac{\partial \mathcal{L}}{\partial K[m,n]} = \sum_{i,j} \frac{\partial \mathcal{L}}{\partial O[i,j]} \cdot I[i+m,\ j+n]$$

对输入的梯度（传递给上一层）：

$$\frac{\partial \mathcal{L}}{\partial I[i,j]} = \sum_{m,n} \frac{\partial \mathcal{L}}{\partial O[i-m,\ j-n]} \cdot K[m,n]$$

这等价于对梯度做**全卷积（full convolution）**。

---

## 2. 网络结构图（文字版）

```
输入图片 (32×32 RGB 彩色图)
        │
        ▼
   ┌────────────────────────────────────────────────┐
   │  卷积块 1                                       │
   │  Conv2d(3  → 32,  3×3, pad=1)  → (32, 32, 32) │
   │  BatchNorm2d(32)                                │
   │  ReLU                                           │
   │  MaxPool2d(2×2)                → (32, 16, 16)  │
   └────────────────────────────────────────────────┘
        │
        ▼
   ┌────────────────────────────────────────────────┐
   │  卷积块 2                                       │
   │  Conv2d(32 → 64,  3×3, pad=1)  → (64, 16, 16) │
   │  BatchNorm2d(64)                                │
   │  ReLU                                           │
   │  MaxPool2d(2×2)                → (64,  8,  8)  │
   └────────────────────────────────────────────────┘
        │
        ▼
   ┌────────────────────────────────────────────────┐
   │  卷积块 3                                       │
   │  Conv2d(64 → 128, 3×3, pad=1)  → (128, 8,  8) │
   │  BatchNorm2d(128)                               │
   │  ReLU                                           │
   │  MaxPool2d(2×2)                → (128, 4,  4)  │
   └────────────────────────────────────────────────┘
        │
        ▼
   ┌────────────────────────────────────────────────┐
   │  展平 (Flatten)                                 │
   │  (128, 4, 4) → 2048                            │
   └────────────────────────────────────────────────┘
        │
        ▼
   ┌────────────────────────────────────────────────┐
   │  全连接层 (Linear 2048 → 256)                   │
   │  ReLU                                           │
   │  Dropout(p=0.5)                                 │
   └────────────────────────────────────────────────┘
        │
        ▼
   ┌────────────────────────────────────────────────┐
   │  输出层 (Linear 256 → 10)                       │
   │  （logits，不加激活函数）                         │
   └────────────────────────────────────────────────┘
        │
        ▼
   CrossEntropyLoss（内含 Softmax）
        │
        ▼
   预测类别 argmax(logits)
```

---

## 3. PyTorch 源码解析

### 3.1 `nn.Conv2d` — 2D 卷积层

```python
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        # weight 形状：(out_channels, in_channels, kernel_H, kernel_W)
        # 例如 Conv2d(3, 32, 3) 的 weight 形状为 (32, 3, 3, 3)
        self.weight = Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        self.bias   = Parameter(torch.empty(out_channels)) if bias else None

        # Kaiming Uniform 初始化（专为 ReLU 设计）
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        # 底层调用高度优化的 CUDA/MPS 卷积实现
        return F.conv2d(input, self.weight, self.bias,
                        self.stride, self.padding)
```

**关键理解**：
- `weight` 形状为 `(C_out, C_in, H, W)`，参数量 = $C_{out} \times C_{in} \times k^2$
- 本项目 Conv1 参数量：$32 \times 3 \times 3 \times 3 = 864$（远少于等效全连接的 $3072 \times 32 = 98304$）

### 3.2 `nn.MaxPool2d` — 最大池化

```python
class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        # stride 默认等于 kernel_size（不重叠池化）

    def forward(self, input):
        return F.max_pool2d(input, self.kernel_size, self.stride)
        # 反向传播：梯度只传给取最大值的位置（max-routing）
```

### 3.3 `nn.BatchNorm2d` — 2D 批归一化

```python
# 与 BatchNorm1d 的区别：对每个 channel（feature map）独立归一化
# 计算维度：对 (N, H, W) 三个维度求均值和方差，channel 维度保持独立
class BatchNorm2d(Module):
    def forward(self, x):   # x: (N, C, H, W)
        # 对每个 channel c 计算：
        # mean_c = x[:, c, :, :].mean()  ← 跨样本+空间位置求均值
        # var_c  = x[:, c, :, :].var()
        # x_norm = (x - mean_c) / sqrt(var_c + eps)
        # output = gamma_c * x_norm + beta_c
        ...
```

### 3.4 `bias=False` 与 BatchNorm 的配合

```python
# 错误写法（浪费参数）：
nn.Conv2d(3, 32, 3, padding=1, bias=True)   # bias 会被 BN 的减均值操作抵消
nn.BatchNorm2d(32)

# 正确写法（本项目采用）：
nn.Conv2d(3, 32, 3, padding=1, bias=False)  # 省略 bias
nn.BatchNorm2d(32)                           # BN 的 β 参数承担偏置的作用
```

---

## 4. 快速开始

### 版本兼容表

| 库 | 版本 | 说明 |
|----|------|------|
| Python | 3.10 ~ 3.12 | torch 2.3.x 官方支持范围，3.9 不支持 |
| torch | 2.3.1 | MPS 后端稳定，支持 CUDA 11.8 / 12.1 |
| torchvision | 0.18.1 | 严格对应 torch 2.3.x；CIFAR-10 数据集加载依赖此库 |
| matplotlib | >=3.7.0,<4.0 | 训练曲线可视化 |
| numpy | >=1.24.0,<2.0 | numpy 2.0 与 torch 2.3.x 存在兼容问题 |

### 环境验证

安装完成后运行以下命令，确认各库版本正确：

```bash
python -c "
import torch, torchvision, matplotlib, numpy
print(f'torch       : {torch.__version__}')
print(f'torchvision : {torchvision.__version__}')
print(f'matplotlib  : {matplotlib.__version__}')
print(f'numpy       : {numpy.__version__}')
if torch.cuda.is_available():
    print('加速设备     : CUDA', torch.version.cuda)
elif torch.backends.mps.is_available():
    print('加速设备     : MPS (Apple Silicon)')
else:
    print('加速设备     : CPU only')
"
```

### macOS（Apple Silicon M1/M2/M3/M4，默认）

```bash
# 1. 在项目根目录创建共享虚拟环境（MLP/CNN/RNN 三个子项目共用）
cd DL_Learning
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装依赖（版本锁定，避免兼容性问题）
pip install torch==2.3.1 torchvision==0.18.1 "matplotlib>=3.7.0,<4.0" "numpy>=1.24.0,<2.0"

# 3. 进入子项目并训练（自动检测 MPS，首次运行自动下载 CIFAR-10 约 170MB）
cd CNN
python train.py

# 4. 测试
python test.py
```

> MPS 是 Apple Silicon 的 GPU 加速框架，torch 2.3.x 对 MPS 的支持已非常成熟，速度比纯 CPU 快 3-5 倍。

### Windows（NVIDIA GPU，CUDA 12.1）

```powershell
# 1. 在项目根目录创建共享虚拟环境
cd DL_Learning
python -m venv .venv
.venv\Scripts\activate

# 2. 安装 PyTorch（指定 CUDA 12.1 索引，版本号固定）
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install "matplotlib>=3.7.0,<4.0" "numpy>=1.24.0,<2.0"

# 3. 训练（自动检测 CUDA，输出"使用设备: cuda"）
cd CNN
python train.py

# 4. 测试
python test.py
```

> 若使用 CUDA 11.8，将 `cu121` 替换为 `cu118`。可通过 `nvidia-smi` 查看 CUDA 版本。

### Windows（无 GPU，仅 CPU）

```powershell
# 1. 在项目根目录创建共享虚拟环境
cd DL_Learning
python -m venv .venv
.venv\Scripts\activate

# 2. 安装 CPU 版 PyTorch（无需 --index-url）
pip install torch==2.3.1 torchvision==0.18.1 "matplotlib>=3.7.0,<4.0" "numpy>=1.24.0,<2.0"

# 3. 训练（输出"使用设备: cpu"，速度较慢，建议减少 epochs 至 10 先验证）
cd CNN
python train.py
```

### 使用 requirements.txt 安装（推荐）

```bash
# macOS / Windows CPU（激活虚拟环境后）
pip install -r requirements.txt

# Windows NVIDIA GPU（需先单独安装带 CUDA 的 torch/torchvision）
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install "matplotlib>=3.7.0,<4.0" "numpy>=1.24.0,<2.0" portalocker
```

`requirements.txt` 位于项目根目录，覆盖所有子项目的完整依赖。

### 手动指定设备

修改 `config.py` 中的 `device` 字段：

```python
device = "auto"   # 自动检测（CUDA > MPS > CPU）
device = "mps"    # 强制 Apple Silicon GPU
device = "cuda"   # 强制 NVIDIA GPU
device = "cpu"    # 强制 CPU（调试用）
```

### 注意事项

- 虚拟环境建议统一创建在**项目根目录**（`DL_Learning/.venv`），MLP/CNN/RNN 三个子项目共享，避免重复安装和版本冲突。
- torch 2.3.x 要求 Python 3.10 ~ 3.12，可用 `python --version` 确认。
- Windows 路径如含中文或空格，建议将项目放在 `C:\Projects\` 下。

---

## 5. 调参指南

### 5.1 通道数（最影响模型容量）

```python
# 轻量版（训练快，适合验证代码）
channels = [16, 32, 64]    # 参数量约 50 万

# 默认版（均衡）
channels = [32, 64, 128]   # 参数量约 210 万

# 加深版（准确率更高，训练更慢）
channels = [64, 128, 256]  # 参数量约 850 万
```

### 5.2 数据增强强度

| 场景 | 建议增强 |
|------|---------|
| 数据充足、模型小 | 轻度增强（只用 RandomFlip） |
| **默认（本项目）** | 中度（RandomCrop + RandomFlip + ColorJitter） |
| 过拟合明显 | 加入 AutoAugment 或 Cutout |
| 训练集极小 | 激进增强 + Mixup |

### 5.3 学习率调度

| 配置 | 效果 |
|------|------|
| StepLR step=10, γ=0.5（默认） | 稳定、可预期 |
| CosineAnnealingLR | 通常比 StepLR 高 0.5-1% |
| OneCycleLR | 训练快，适合 epoch 数少的场景 |

### 5.4 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 第 1 epoch 准确率 < 20% | 正常，CIFAR-10 比 MNIST 难 | 继续训练 |
| 训练准确率 95%+，测试 65% | 严重过拟合 | 增大 Dropout、增强数据增强 |
| 训练和测试都停在 60% | 欠拟合 | 增加通道数或添加更多卷积层 |
| MPS 运行比 CPU 还慢 | batch_size 太小 | 调大 batch_size ≥ 64 |
| OOM 显存溢出 | batch_size 太大 | 减半 batch_size |

---

## 6. CNN vs MLP 对比

| 维度 | MLP | CNN |
|------|-----|-----|
| 输入处理 | 展平为一维向量 | 保留二维空间结构 |
| 连接方式 | 全连接（每个神经元连所有输入） | 局部连接（每个神经元只看 k×k 区域） |
| 权重共享 | 无（每对连接独立参数） | 有（同一卷积核在全图共享） |
| 平移不变性 | 无（换位置就认不出） | 有（卷积核在任何位置识别同一特征） |
| 参数效率 | 低（参数随图像尺寸平方增长） | 高（参数只与卷积核大小有关） |
| MNIST 准确率 | ~98.7% | ~99.5% |
| CIFAR-10 准确率 | ~55% | ~85% |

---

## 7. 关键 API 速查

```python
import torch.nn as nn

# 卷积层
nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False)

# 池化层
nn.MaxPool2d(kernel_size=2, stride=2)   # 最大池化，尺寸减半
nn.AvgPool2d(kernel_size=2, stride=2)   # 平均池化
nn.AdaptiveAvgPool2d((1, 1))            # 全局平均池化，输出 1×1

# 批归一化（2D，用于卷积后）
nn.BatchNorm2d(num_features=32)

# 展平
x = x.view(x.size(0), -1)   # 手动展平
nn.Flatten()                  # 作为 Sequential 层使用

# 特征图尺寸计算（验证用）
import torch
dummy = torch.zeros(1, 3, 32, 32)
out = model.features(dummy)
print(out.shape)  # 确认各层输出尺寸
```
