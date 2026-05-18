# CNN 基础：池化（Pooling）

**首次出现**：CNN  
**解决的问题**：特征图尺寸过大导致参数过多；使特征具有局部平移不变性

---

## 1. 问题背景

经过卷积后，特征图仍然很大（如 32×32×128）：
- 直接展平送 FC 层：$32 \times 32 \times 128 = 131072$ 维，参数爆炸
- 特征对位置过于敏感：图像平移 1 个像素，特征向量就完全不同

**池化的两个作用**：
1. **降维**：减小特征图尺寸，降低后续计算量
2. **平移不变性**：对输入的小平移/形变不敏感（在 $k \times k$ 区域内的轻微移动不影响输出）

---

## 2. MaxPool（最大池化）

### 公式

在 $k \times k$ 窗口内取最大值：

$$O[i,j] = \max_{0 \leq m < k,\ 0 \leq n < k} I[s \cdot i + m,\ s \cdot j + n]$$

- $s$：stride，通常与 $k$ 相等（不重叠池化，本项目 $k=s=2$）

### 可视化

```
输入（4×4）         MaxPool(2×2, stride=2)      输出（2×2）
┌──┬──┬──┬──┐                                   ┌──┬──┐
│1 │3 │2 │4 │      左上 2×2 取 max=3            │3 │4 │
├──┼──┼──┼──┤   →                          →   ├──┼──┤
│5 │2 │3 │1 │      右上 2×2 取 max=4            │6 │8 │
├──┼──┼──┼──┤      左下 2×2 取 max=6            └──┴──┘
│2 │6 │1 │3 │      右下 2×2 取 max=8
├──┼──┼──┼──┤
│1 │4 │5 │8 │
└──┴──┴──┴──┘
```

### 反向传播（Max-Routing）

梯度只流向**取得最大值的那个位置**，其他位置梯度为 0：

```python
# 前向：记录每个窗口中 max 的位置索引
# 反向：将梯度放回 max 位置，其他位置梯度=0
```

**后果**：池化层本身没有可学习参数，但会影响梯度流动路径。

### PyTorch 使用

```python
nn.MaxPool2d(kernel_size=2, stride=2)     # 最常用：2×2，步长2，尺寸减半
nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 3×3，尺寸近似减半
```

---

## 3. AvgPool（平均池化）

### 公式

在 $k \times k$ 窗口内取平均值：

$$O[i,j] = \frac{1}{k^2} \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I[s \cdot i + m,\ s \cdot j + n]$$

### MaxPool vs AvgPool

| | MaxPool | AvgPool |
|--|---------|---------|
| 保留信息 | 最显著的特征（边缘、角点） | 平均特征（整体纹理） |
| 对噪声 | 不敏感（噪声通常不是最大值） | 较敏感（噪声会拉低均值） |
| 主要用途 | 特征提取中间层（本项目） | 分类网络末端替代全连接 |
| 反向梯度 | 只流向 max 位置 | 均匀分配给窗口内所有位置 |

**经验法则**：中间卷积层用 MaxPool，网络末端（替代全连接）用 Global Average Pool。

```python
nn.AvgPool2d(kernel_size=2, stride=2)
```

---

## 4. Global Average Pooling（全局平均池化）

### 概念

将整个特征图（$H \times W$）压缩为一个标量，对每个 channel 独立计算：

$$O_c = \frac{1}{H \times W} \sum_{i=0}^{H-1} \sum_{j=0}^{W-1} F_c[i,j]$$

输入 $(B, C, H, W)$ → 输出 $(B, C, 1, 1)$，展平后为 $(B, C)$。

### 优势（替代最后的全连接层）

| | 全连接头（传统） | Global Average Pool |
|--|--------------|-------------------|
| 参数量 | $C \times H \times W \times \text{num\_class}$ | 0（无可学习参数） |
| 过拟合风险 | 高 | 低（天然正则化） |
| 输入尺寸要求 | 固定 | 任意（支持不同分辨率输入） |
| 用于 | AlexNet、VGG | ResNet、MobileNet、EfficientNet |

### PyTorch 使用

```python
# 方法一：AdaptiveAvgPool2d（推荐，自动计算 kernel_size）
nn.AdaptiveAvgPool2d((1, 1))   # 任意输入尺寸都输出 1×1

# 方法二：手动 mean
x = x.mean(dim=[2, 3])   # (B, C, H, W) → (B, C)

# 典型用法（ResNet-style 分类头）
self.gap = nn.AdaptiveAvgPool2d((1, 1))
self.fc  = nn.Linear(512, num_classes)

def forward(self, x):
    x = self.features(x)          # (B, 512, H, W)
    x = self.gap(x)               # (B, 512, 1, 1)
    x = x.view(x.size(0), -1)     # (B, 512)
    return self.fc(x)
```

---

## 5. 池化层的位置与频率

```
常见网络中的池化策略：

VGG 风格：每 2-3 个卷积层后接一个 MaxPool
    Conv → Conv → MaxPool → Conv → Conv → MaxPool → ...

ResNet 风格：仅在 stem 和最后用池化，中间用步长卷积降维
    Conv(stride=2) → ResBlock → ResBlock → GAP → FC

MobileNet 风格：Depthwise Conv + PointwiseConv，最后用 GAP
    ...→ DepthwiseConv → PointwiseConv → GAP → FC
```

---

## 6. 常见误用

| 误用 | 后果 | 正确做法 |
|------|------|---------|
| 池化前特征图太小（<4×4） | 池化后丢失信息 | 确保池化前特征图 ≥ 4×4 |
| 对 1D 序列用 MaxPool2d | 报错或维度错误 | 用 `nn.MaxPool1d` |
| 在 GAP 后仍用大 FC 层 | 失去 GAP 的参数效率优势 | GAP 后直接接分类层 |
| AdaptiveAvgPool 输出设 (4,4) | 不是"全局"平均池化 | 输出设 (1,1) |
