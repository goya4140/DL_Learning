# 正则化：数据增强（Data Augmentation）

**首次出现**：CNN（CIFAR-10）  
**解决的问题**：训练数据量不足导致的过拟合

---

## 1. 问题背景

深度学习模型的泛化能力很大程度上取决于训练数据的多样性。当训练数据不足时：
- 模型会"记住"训练集的特定角度、光照、位置
- 测试集中稍微不同的图片就会预测错误

**数据增强的思路**：在每次训练时，对原始图像做随机变换，生成"新"的训练样本，等价于人工扩充了数据集。

**关键原则**：
- 增强只在**训练集**上施加，测试集只做标准化
- 变换必须**保持语义不变**（水平翻转一只猫还是猫，上下翻转一只猫可能就不合理了）

---

## 2. CIFAR-10 标准数据增强流程

```python
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),       # 1. 随机裁剪
    transforms.RandomHorizontalFlip(),           # 2. 随机水平翻转
    transforms.ColorJitter(brightness=0.2,       # 3. 色彩抖动
                           contrast=0.2,
                           saturation=0.2),
    transforms.ToTensor(),                       # 4. 转为 Tensor
    transforms.Normalize(mean, std),             # 5. 标准化
])
```

---

## 3. 常用增强方法详解

### 3.1 RandomCrop（随机裁剪）

```python
transforms.RandomCrop(size=32, padding=4)
```

**操作**：先在图像四周填充 4 个像素（用 0 或边界值），再随机裁剪回 32×32。

**效果**：训练时图像内容可能略微偏移，模型学会对位置变化不敏感（平移不变性）。

**公式**：最多可产生 $(2p+1)^2 = 81$ 种裁剪位置（$p=4$）。

### 3.2 RandomHorizontalFlip（随机水平翻转）

```python
transforms.RandomHorizontalFlip(p=0.5)  # 默认概率 0.5
```

**操作**：以概率 $p$ 将图像左右镜像翻转。

**适用条件**：左右对称的类别（猫、狗、汽车等都适用，文字类别不适用）。

### 3.3 ColorJitter（色彩抖动）

```python
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
```

**操作**：随机调整亮度、对比度、饱和度、色调，每个参数代表最大变化幅度。

- `brightness=0.2`：亮度在 $[1-0.2, 1+0.2] = [0.8, 1.2]$ 范围内随机缩放
- 模拟不同光照条件下的同一物体

### 3.4 RandomRotation（随机旋转）

```python
transforms.RandomRotation(degrees=15)  # 在 [-15°, 15°] 范围内随机旋转
```

**注意**：CIFAR-10 不常用大角度旋转（汽车旋转 90° 就不正常了）。

### 3.5 Normalize（标准化，必须有）

```python
# CIFAR-10 官方统计值
transforms.Normalize(
    mean=(0.4914, 0.4822, 0.4465),   # RGB 三通道各自的均值
    std =(0.2023, 0.1994, 0.2010)    # RGB 三通道各自的标准差
)
```

**公式**：$x_{norm} = (x - \mu) / \sigma$，将像素值分布变为均值 0、标准差 1。

这不是"增强"而是必须的预处理，训练集和测试集都要做。

---

## 4. 增强的数学效果

原始数据集 $N$ 张图片，每个 batch 通过随机增强，理论上：

$$\text{等效数据量} = N \times \underbrace{81}_{\text{RandomCrop}} \times \underbrace{2}_{\text{Flip}} \times \underbrace{\infty}_{\text{ColorJitter（连续）}}$$

实际训练中每张图片在不同 epoch 会呈现不同形态，帮助模型见过更多变化。

---

## 5. PyTorch `transforms` 速查

```python
from torchvision import transforms

# 几何变换
transforms.RandomCrop(size, padding)             # 随机裁剪
transforms.CenterCrop(size)                      # 中心裁剪（测试时常用）
transforms.RandomHorizontalFlip(p=0.5)           # 随机水平翻转
transforms.RandomVerticalFlip(p=0.5)             # 随机垂直翻转
transforms.RandomRotation(degrees)               # 随机旋转
transforms.RandomResizedCrop(size, scale, ratio) # 随机缩放裁剪（ImageNet 常用）
transforms.Resize(size)                          # 固定缩放

# 色彩变换
transforms.ColorJitter(brightness, contrast, saturation, hue)
transforms.Grayscale(num_output_channels=1)      # 转灰度图
transforms.RandomGrayscale(p=0.1)               # 随机转灰度

# Tensor 操作
transforms.ToTensor()                            # PIL/ndarray → Tensor [0,1]
transforms.Normalize(mean, std)                  # 标准化
transforms.RandomErasing(p=0.5, scale=(0.02,0.33))  # 随机擦除（Cutout 的简单版）

# 组合
transforms.Compose([...])                        # 顺序执行
transforms.RandomApply([...], p=0.5)             # 随机应用某组变换
transforms.RandomChoice([...])                   # 从列表中随机选一个

# 高级增强（torchvision 0.9+）
transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10)  # AutoAugment
transforms.TrivialAugmentWide()                  # 简单有效，来自 SOTA 论文
transforms.RandAugment()                         # 随机选择增强策略
```

---

## 6. 常见增强策略对比

| 策略 | 适用场景 | CIFAR-10 效果 |
|------|---------|--------------|
| 无增强 | 数据充足 | 基准 |
| RandomCrop + Flip（本项目） | 通用 | +2-3% |
| + ColorJitter | 光照变化场景 | +0.5-1% |
| AutoAugment | 充足计算资源 | +1-2% |
| Cutout | 遮挡场景 | +0.5-1% |
| Mixup | 标签平滑 | +0.5-1% |
| CutMix | 区域级混合 | +1% |

---

## 7. 常见误用

| 误用 | 后果 | 正确做法 |
|------|------|---------|
| 测试集也做随机增强 | 每次评估结果不同，无法复现 | 测试集只做 Resize + CenterCrop + Normalize |
| 对文字/医学图像做随机翻转 | 语义改变（文字翻转后不可读） | 根据任务选择合适的增强 |
| Normalize 使用 ImageNet 参数 | CIFAR-10 数据分布不同，归一化不准 | 使用对应数据集的统计值 |
| 增强强度过大 | 图像失真，训练损失不下降 | 逐步增加增强强度，观察训练曲线 |
| 忘记 `.eval()` 时关闭 BatchNorm | 非增强问题，但常一起出现 | 评估前务必 `model.eval()` |

```python
# 正确的测试集 transform（CIFAR-10）
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
    # ← 不要加任何随机变换！
])
```
