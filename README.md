# DL_Learning

深度学习经典神经网络实现（MLP / CNN / RNN / LSTM / GCN / Transformer），每个网络独立成项目，配套数学公式推导、PyTorch 源码解析和调参指南。

## 项目结构

```
DL_Learning/
├── MLP/          # 多层感知机 - MNIST 手写数字分类（~98.7%）
├── CNN/          # 卷积神经网络 - CIFAR-10 图像分类（~85%）
├── RNN/          # 循环神经网络 - AG News 新闻文本分类（~91%）
├── DL_Basics/    # 深度学习基础技巧（优化器/归一化/正则化/学习率调度）
└── README.md
```

## 运行方式

每个子项目独立运行，进入对应目录后执行：

```bash
# macOS（Apple Silicon，自动使用 MPS 加速）
cd MLP && python train.py

# Windows（NVIDIA GPU）
cd CNN && python train.py
```

详细环境配置见各子项目的 README.md。
