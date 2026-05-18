# 深度学习基础：优化与训练技巧

本文件夹记录深度学习训练中的**工程技巧**，与网络架构（MLP/CNN/Transformer 等）学习并行推进。

> **学习建议**：先理解"不用这个技巧会出什么问题"，再学"它怎么解决的"。带着问题读，比从头死记公式效率高 10 倍。

---

## 技巧总览

| 技术 | 分类 | 解决的问题 | 首次出现 | 文档 |
|------|------|-----------|---------|------|
| Adam 优化器 | 优化器 | SGD 调参难、对稀疏梯度不友好 | MLP | [optimizer/adam.md](optimizer/adam.md) |
| SGD + Momentum | 优化器 | 纯 SGD 收敛慢、震荡 | MLP | [optimizer/adam.md](optimizer/adam.md) |
| Batch Normalization | 归一化 | 训练不稳定、梯度消失/爆炸 | MLP | [normalization/batch_norm.md](normalization/batch_norm.md) |
| Dropout | 正则化 | 过拟合（训练好测试差） | MLP | [regularization/dropout.md](regularization/dropout.md) |
| L2 正则化（Weight Decay） | 正则化 | 过拟合、权重无限增大 | MLP | [regularization/weight_decay.md](regularization/weight_decay.md) |
| StepLR / 学习率调度 | 学习率 | 固定学习率难以兼顾收敛速度和精度 | MLP | [lr_schedule/lr_schedule.md](lr_schedule/lr_schedule.md) |
| 2D 卷积（Conv2d） | CNN 基础 | FC 层忽略空间结构、参数爆炸 | CNN | [cnn_operations/convolution.md](cnn_operations/convolution.md) |
| 池化（MaxPool/AvgPool/GAP） | CNN 基础 | 特征图尺寸过大、对位置过敏感 | CNN | [cnn_operations/pooling.md](cnn_operations/pooling.md) |
| 数据增强（Data Augmentation） | 正则化 | 训练数据不足导致过拟合 | CNN | [regularization/data_augmentation.md](regularization/data_augmentation.md) |

---

## 文件结构

```
DL_Basics/
├── README.md                        # 本文件（总目录）
├── optimizer/
│   └── adam.md                      # Adam、SGD+Momentum、AdamW 对比
├── normalization/
│   └── batch_norm.md                # BatchNorm、LayerNorm、GroupNorm
├── regularization/
│   ├── dropout.md                   # Dropout 原理与实践
│   ├── weight_decay.md              # L1/L2 正则化
│   └── data_augmentation.md         # 数据增强（RandomCrop/Flip/ColorJitter 等）
├── lr_schedule/
│   └── lr_schedule.md               # 学习率调度策略
└── cnn_operations/
    ├── convolution.md               # 2D 卷积原理、感受野、参数量分析
    └── pooling.md                   # MaxPool、AvgPool、Global Average Pooling
```

---

## 后续更新计划

随着学习新的网络架构，遇到新技巧时会追加到本文件夹：

| 预期技巧 | 来源网络 | 状态 |
|---------|---------|------|
| 2D 卷积 / 感受野 | CNN | ✅ 已完成 |
| MaxPool / GAP 池化 | CNN | ✅ 已完成 |
| Data Augmentation（数据增强） | CNN | ✅ 已完成 |
| Layer Normalization | Transformer | 待更新 |
| Gradient Clipping（梯度裁剪） | RNN/LSTM | 待更新 |
| Weight Initialization（权重初始化） | CNN/ResNet | 待更新 |
| Residual Connection（残差连接） | ResNet/Transformer | 待更新 |
| Attention Mechanism | Transformer | 待更新 |
