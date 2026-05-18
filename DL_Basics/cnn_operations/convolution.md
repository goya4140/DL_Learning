# CNN 基础：2D 卷积

**首次出现**：CNN  
**解决的问题**：全连接层对图像的两大缺陷：参数爆炸 + 忽略空间结构

---

## 1. 问题背景：全连接处理图像的局限

设输入图像为 32×32×3 = 3072 维：
- 第一个 FC 层（3072→256）就需要 **786,944** 个参数
- 图像分辨率翻倍 → 参数量变 4 倍
- 更致命：FC 层把像素当作独立特征，完全忽略"相邻像素构成边缘"这一事实

**卷积的两个核心创新**：

| 特性 | 含义 | 效果 |
|------|------|------|
| **局部连接** | 每个输出神经元只连接输入的 $k \times k$ 区域 | 大幅减少参数 |
| **权重共享** | 同一个卷积核在整张图上滑动，使用相同参数 | 参数量与图像尺寸无关 |

---

## 2. 卷积计算原理

### 单通道卷积（最简情形）

卷积核 $K$（大小 $k \times k$）在输入 $I$ 上滑动：

$$O[i,j] = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I[i+m,\ j+n] \cdot K[m,n] + b$$

**数字示例**（3×3 输入，2×2 卷积核，stride=1，padding=0）：

```
输入 I:          卷积核 K:      输出 O（2×2）:
┌─────────┐      ┌─────┐        ┌─────────┐
│ 1  2  3 │   ×  │ 1 0 │   =    │ 5   7   │
│ 4  5  6 │      │ 0 1 │        │ 11  13  │
│ 7  8  9 │      └─────┘        └─────────┘
└─────────┘
O[0,0] = 1×1 + 2×0 + 4×0 + 5×1 = 6  （无偏置）
O[0,1] = 2×1 + 3×0 + 5×0 + 6×1 = 8
...（实际结果以具体数值为准）
```

### 多通道卷积（实际使用）

输入 $C_{in}$ 个通道，输出 $C_{out}$ 个通道（feature map）：

$$O_f[i,j] = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I_c[i+m,\ j+n] \cdot K_{f,c}[m,n] + b_f$$

- 共有 $C_{out}$ 组卷积核，每组负责提取一种特征模式
- 总参数量：$C_{out} \times C_{in} \times k \times k + C_{out}$（加偏置）

**CNN 第一层（Conv2d(3, 32, 3)）参数量**：
$$32 \times 3 \times 3 \times 3 = 864$$

**等效全连接（3×32×32 → 32×32×32）的参数量**：
$$3072 \times 32768 \approx \mathbf{1亿}$$

---

## 3. 特征图尺寸公式

经过卷积（padding=$p$，stride=$s$，kernel=$k$）：

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1$$

**常见配置**：

| 配置 | 公式结果 | 效果 |
|------|---------|------|
| $k=3, p=1, s=1$ | $H_{out} = H_{in}$ | 尺寸不变（最常用） |
| $k=3, p=0, s=1$ | $H_{out} = H_{in} - 2$ | 尺寸缩小 2 |
| $k=3, p=1, s=2$ | $H_{out} = H_{in}/2$ | 尺寸减半（步长卷积） |
| MaxPool $k=2, s=2$ | $H_{out} = H_{in}/2$ | 尺寸减半（池化） |

---

## 4. 感受野（Receptive Field）

感受野 = 输出特征图上一个像素所对应的**输入图像区域大小**。

**逐层累积公式（stride=1）**：
$$RF_l = RF_{l-1} + (k-1)$$

**含步长（MaxPool 等）时**：
$$RF_l = RF_{l-1} + (k-1) \times \prod_{i < l} s_i$$

**示例（3层 3×3 卷积 + 2层 MaxPool）**：

```
层           感受野（输入图像上的区域）
─────────────────────────────────────
Conv1 (3×3)         3×3
MaxPool (2×2)       4×4
Conv2 (3×3)         8×8
MaxPool (2×2)       10×10
Conv3 (3×3)         22×22
```

**意义**：深层神经元能"看到"输入图像 22×22 的区域，足以捕获复杂的轮廓和纹理。

---

## 5. PyTorch 源码解析：`nn.Conv2d`

```python
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        # weight 形状：(out_channels, in_channels/groups, kH, kW)
        self.weight = Parameter(torch.empty(out_channels, in_channels // groups, *pair(kernel_size)))
        if bias:
            self.bias = Parameter(torch.empty(out_channels))

        # 默认初始化：Kaiming Uniform（专为 ReLU 激活设计）
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        # 底层调用 cuDNN（CUDA）或 Metal（MPS）的高效卷积实现
        return F.conv2d(input, self.weight, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)
```

**关键参数理解**：

| 参数 | 默认值 | 作用 |
|------|-------|------|
| `padding=1` | 0 | 3×3 卷积时设 1，保持特征图尺寸不变 |
| `bias=False` | True | 接 BN 时应设 False（BN 的 β 参数替代 bias） |
| `groups` | 1 | Depthwise 卷积（MobileNet）时设为 `in_channels` |
| `dilation` | 1 | 空洞卷积（扩大感受野而不增加参数） |

---

## 6. 卷积 vs 全连接对比

| 特性 | 全连接（Linear） | 卷积（Conv2d） |
|------|---------------|--------------|
| 连接方式 | 每个输出连所有输入 | 每个输出只连局部 $k \times k$ |
| 权重共享 | 无 | 同一卷积核在全图共享 |
| 参数量 | $C_{in} \times H \times W \times C_{out}$ | $C_{out} \times C_{in} \times k^2$ |
| 平移不变性 | 无 | 有（同一特征在任意位置都能被识别） |
| 适用数据 | 表格/文本等一维数据 | 图像/音频等具有空间/时序结构的数据 |

---

## 7. 常见误用

| 误用 | 后果 | 正确做法 |
|------|------|---------|
| 接 BN 时保留 bias | bias 被 BN 均值减法抵消，白白占参数 | `Conv2d(..., bias=False)` |
| 忘记设 padding | 3×3 卷积每次缩小 2，深网络特征图变 0 | 3×3 卷积统一设 `padding=1` |
| kernel_size 设太大 | 参数增多但感受野增大有限，不如堆叠小卷积 | 优先用 3×3，极少用 7×7 以上 |
| 网络太深但无 BN | 梯度消失，训练不收敛 | 每个卷积块加 BN |

```python
# 验证各层输出尺寸的标准方法
import torch
dummy_input = torch.zeros(1, 3, 32, 32)  # batch=1, C=3, H=W=32
output = model.features(dummy_input)
print(output.shape)  # 应输出 torch.Size([1, 128, 4, 4])
```
