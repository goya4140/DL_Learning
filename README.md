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

## 环境配置

推荐在**项目根目录**创建统一虚拟环境，MLP / CNN / RNN 三个子项目共享，避免重复安装和版本冲突。

| 库 | 版本 | 说明 |
|----|------|------|
| Python | 3.10 ~ 3.12 | torch 2.3.x 官方支持范围 |
| torch | 2.3.1 | MPS / CUDA 11.8 / CUDA 12.1 均支持 |
| torchvision | 0.18.1 | 严格对应 torch 2.3.x |
| torchtext | 0.18.0 | 严格对应 torch 2.3.x（RNN 需要） |
| numpy | >=1.24.0,<2.0 | numpy 2.0 与 torch 2.3.x 存在兼容问题 |
| matplotlib | >=3.7.0,<4.0 | 训练曲线可视化 |

```bash
# macOS / Windows CPU — 直接安装
pip install -r requirements.txt

# Windows NVIDIA GPU（CUDA 12.1）
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install torchtext==0.18.0 "matplotlib>=3.7.0,<4.0" "numpy>=1.24.0,<2.0" portalocker
```

> 详细步骤（含 venv 创建、环境验证命令、CUDA 11.8 替换方式）见各子项目 README.md。

## 运行方式

创建并激活虚拟环境后，进入对应子目录执行：

```bash
# macOS（Apple Silicon，自动使用 MPS 加速）
cd MLP && python train.py

# Windows（NVIDIA GPU）
cd CNN && python train.py
```
