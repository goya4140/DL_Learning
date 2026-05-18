"""
MNIST 数据集加载模块

MNIST 数据集包含：
    - 训练集：60,000 张 28x28 灰度手写数字图片
    - 测试集：10,000 张

数据预处理流程：
    1. ToTensor()：将 PIL Image [0,255] 转为 FloatTensor [0.0,1.0]，形状 (C,H,W)
    2. Normalize(mean, std)：标准化为均值 0.1307、标准差 0.3081（MNIST 全局统计值）
       公式：x_norm = (x - mean) / std
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(data_dir: str, batch_size: int, num_workers: int = 2):
    """
    返回 MNIST 训练集和测试集的 DataLoader。

    返回：
        train_loader: 训练集加载器（shuffle=True 保证每轮顺序不同）
        test_loader:  测试集加载器
    """
    # MNIST 数据集全局均值和标准差（由官方统计得出）
    mean, std = 0.1307, 0.3081

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset  = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,   # 将数据固定在内存中，加速 GPU 传输
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def show_dataset_info(train_loader: DataLoader, test_loader: DataLoader):
    """打印数据集基本信息"""
    train_size = len(train_loader.dataset)
    test_size  = len(test_loader.dataset)
    print(f"训练集大小: {train_size} 张")
    print(f"测试集大小: {test_size} 张")
    print(f"批次大小:   {train_loader.batch_size}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}")

    # 打印一批数据的形状
    images, labels = next(iter(train_loader))
    print(f"单批输入形状: {images.shape}  →  (batch, channel, height, width)")
    print(f"单批标签形状: {labels.shape}")
