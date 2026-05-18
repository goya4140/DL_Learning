"""
CIFAR-10 数据集加载模块

CIFAR-10 数据集：
    - 训练集：50,000 张 32×32 RGB 彩色图片，10 个类别各 5000 张
    - 测试集：10,000 张
    - 类别：airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

数据预处理：
    训练集（含数据增强）：
        1. RandomCrop(32, padding=4)     随机裁剪：填充4像素后随机裁回32×32
        2. RandomHorizontalFlip()        随机水平翻转（概率0.5）
        3. ColorJitter(...)              亮度/对比度/饱和度随机抖动
        4. ToTensor()                    [0,255] → [0.0,1.0]
        5. Normalize(mean, std)          CIFAR-10 全局统计值（RGB 三通道各自归一化）
    测试集：只做 ToTensor + Normalize，不做随机增强（保证评估可重复）

CIFAR-10 全局均值和标准差（从训练集统计）：
    mean = (0.4914, 0.4822, 0.4465)  # R, G, B
    std  = (0.2023, 0.1994, 0.2010)
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CIFAR-10 的 10 个类别名称（供可视化使用）
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# CIFAR-10 官方统计的归一化参数
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def get_cifar10_loaders(data_dir: str, batch_size: int,
                        num_workers: int = 2, pin_memory: bool = True):
    """
    返回 CIFAR-10 训练集和测试集的 DataLoader。

    训练集使用数据增强，测试集只做归一化。
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),          # 先填充 4 像素再随机裁到 32×32
        transforms.RandomHorizontalFlip(),              # 以 0.5 概率水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 色彩抖动
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=train_transform)
    test_dataset  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


def show_dataset_info(train_loader: DataLoader, test_loader: DataLoader):
    """打印数据集基本信息"""
    print(f"训练集大小: {len(train_loader.dataset)} 张")
    print(f"测试集大小: {len(test_loader.dataset)} 张")
    print(f"批次大小:   {train_loader.batch_size}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    print(f"类别列表:   {CIFAR10_CLASSES}")

    images, labels = next(iter(train_loader))
    print(f"单批输入形状: {images.shape}  →  (batch, channel, height, width)")
    print(f"单批标签形状: {labels.shape}")
