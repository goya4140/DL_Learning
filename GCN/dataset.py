"""
Cora 引文网络数据集（通过 torch_geometric.datasets.Planetoid 加载）

Cora 是图神经网络研究中最经典的节点分类基准数据集：
  - 2,708 个节点（科学论文）
  - 5,429 条边（引用关系，无向）
  - 节点特征：1,433 维词袋向量（0/1 表示对应词是否出现）
  - 7 类标签：Case_Based / Genetic_Algorithms / Neural_Networks /
               Probabilistic_Methods / Reinforcement_Learning / Rule_Learning / Theory

标准半监督分割（Kipf & Welling 2017 使用的划分）：
  - 训练：每类 20 个节点 = 140 个（占总量 5.2%，极少标注）
  - 验证：500 个节点
  - 测试：1,000 个节点

数据由 torch_geometric 自动从 GitHub CDN 下载（首次运行约 2MB），之后从本地缓存加载。
依赖：pip install torch_geometric
"""

import os
import numpy as np
import torch
from torch_geometric.datasets import Planetoid


# Cora 7 类标签名
CLASSES = [
    "Case_Based",
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
    "Theory",
]
NUM_CLASSES = 7


# ──────────────────────────────────────────────
# 邻接矩阵归一化
# ──────────────────────────────────────────────

def _normalize_adj(adj: np.ndarray) -> np.ndarray:
    """对称归一化邻接矩阵：Ã = D̂^{-1/2} Â D̂^{-1/2}，其中 Â = A + I

    加自环（+I）确保每个节点在聚合时也包含自身特征。
    对称归一化保留图的无向性，防止度数大的节点主导聚合。
    """
    adj = adj + np.eye(adj.shape[0])                     # Â = A + I
    degree = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    # D̂^{-1/2} @ Â @ D̂^{-1/2}（广播实现对角矩阵乘法）
    return d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]


# ──────────────────────────────────────────────
# 公共接口
# ──────────────────────────────────────────────

def load_cora(data_dir: str) -> dict:
    """通过 torch_geometric 加载 Cora 数据集。

    首次运行时从 GitHub CDN 自动下载（约 2MB）并缓存到 data_dir/Cora/。
    后续直接从缓存加载，无需联网。
    """
    # Planetoid 下载路径：
    #   https://github.com/kimiyoung/planetoid/raw/master/data/
    # 使用 GitHub CDN，比 linqs 服务器更可靠
    dataset = Planetoid(root=data_dir, name="Cora")
    pyg_data = dataset[0]   # Cora 是单图数据集

    n = pyg_data.num_nodes

    # edge_index (2, E) → dense 邻接矩阵 (N, N)
    edge_index = pyg_data.edge_index.numpy()
    adj_np = np.zeros((n, n), dtype=np.float32)
    adj_np[edge_index[0], edge_index[1]] = 1.0
    # PyG 的 Cora 已是无向图（每条边双向存储），adj_np 已对称

    adj_norm = _normalize_adj(adj_np)

    return {
        "features":   pyg_data.x,                         # FloatTensor (N, 1433)
        "adj":        torch.FloatTensor(adj_norm),         # FloatTensor (N, N)
        "labels":     pyg_data.y,                          # LongTensor  (N,)
        "train_mask": pyg_data.train_mask,                 # BoolTensor  (N,)
        "val_mask":   pyg_data.val_mask,                   # BoolTensor  (N,)
        "test_mask":  pyg_data.test_mask,                  # BoolTensor  (N,)
        "n_nodes":    n,
        "n_edges":    pyg_data.num_edges // 2,             # 无向边数
        "paper_ids":  list(range(n)),
    }


def show_dataset_info(data: dict):
    """打印图数据集基本信息"""
    labels  = data["labels"].numpy()
    train_n = data["train_mask"].sum().item()
    val_n   = data["val_mask"].sum().item()
    test_n  = data["test_mask"].sum().item()

    print(f"节点数: {data['n_nodes']}  边数: {data['n_edges']}  特征维度: {data['features'].shape[1]}")
    print(f"训练节点: {train_n}  验证节点: {val_n}  测试节点: {test_n}")
    print(f"类别: {CLASSES}")
    print("各类别节点分布:")
    for i, cls in enumerate(CLASSES):
        cnt = (labels == i).sum()
        print(f"  {cls:25s}: {cnt:4d}")
