"""
Cora 引文网络数据集

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

数据集需要手动下载，详见 README.md「数据集准备」一节。
"""

import os
import pickle
import tarfile

import numpy as np
import torch


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

_MANUAL_DOWNLOAD_MSG = """
Cora 数据集文件缺失，请手动下载：

  方法一（推荐）：浏览器下载 tgz
    1. 用浏览器访问：https://linqs-data.scu.edu/public/datasets/cora/cora.tgz
    2. 下载 cora.tgz（约 2MB）
    3. 解压：双击 tgz 文件，或在终端运行：
         tar -xzf cora.tgz -C GCN/data/
    4. 确认以下两个文件存在：
         GCN/data/cora/cora.content
         GCN/data/cora/cora.cites

  方法二：终端一键解压（已下载 tgz 后）
    cd GCN
    tar -xzf ~/Downloads/cora.tgz -C data/

准备好后重新运行 python train.py 即可。
"""


def _check_cora_files(data_dir: str):
    """检查 Cora 原始文件是否存在，不存在则给出手动下载指引"""
    content_path = os.path.join(data_dir, "cora", "cora.content")
    cites_path   = os.path.join(data_dir, "cora", "cora.cites")

    if not os.path.exists(content_path) or not os.path.exists(cites_path):
        raise FileNotFoundError(_MANUAL_DOWNLOAD_MSG)

    return content_path, cites_path


def _normalize_features(features: np.ndarray) -> np.ndarray:
    """行归一化：每个节点的特征向量除以其 L1 范数（使节点特征可比）"""
    row_sums = features.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1   # 避免除零
    return features / row_sums


def _normalize_adj(adj: np.ndarray) -> np.ndarray:
    """对称归一化邻接矩阵：Ã = D̂^{-1/2} Â D̂^{-1/2}，其中 Â = A + I

    加自环（+I）确保每个节点在聚合时也包含自身特征。
    对称归一化（而非行归一化 D^{-1}A）保留了图的无向性，
    防止度数大的节点主导聚合，同时在谱域保持对称性。
    """
    # 加自环
    adj = adj + np.eye(adj.shape[0])

    # 计算度矩阵的逆平方根
    degree = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    # D^{-1/2} @ A_hat @ D^{-1/2}（逐列乘再逐行乘等价于完整矩阵乘法）
    adj_norm = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
    return adj_norm


def _parse_cora(content_path: str, cites_path: str) -> dict:
    """解析 Cora 原始文件，返回图数据 dict"""
    # ── 读取节点特征与标签 ──
    paper_ids  = []
    features   = []
    labels_str = []

    with open(content_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            paper_ids.append(parts[0])
            features.append([int(v) for v in parts[1:-1]])
            labels_str.append(parts[-1])

    # paper_id → 排序后的连续整数 index
    # 排序保证相同环境下结果可复现（原始文件行序不固定）
    paper_ids_sorted = sorted(paper_ids)
    id_to_idx = {pid: idx for idx, pid in enumerate(paper_ids_sorted)}

    # 重排 features 和 labels 按排序后的 index
    n = len(paper_ids)
    feat_sorted  = [None] * n
    label_sorted = [None] * n
    for i, pid in enumerate(paper_ids):
        idx = id_to_idx[pid]
        feat_sorted[idx]  = features[i]
        label_sorted[idx] = labels_str[i]

    features_np = np.array(feat_sorted, dtype=np.float32)
    features_np = _normalize_features(features_np)

    # 标签编码
    class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
    labels_np = np.array([class_to_idx[l] for l in label_sorted], dtype=np.int64)

    # ── 读取边 ──
    adj_np = np.zeros((n, n), dtype=np.float32)
    with open(cites_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            src, dst = parts[0], parts[1]
            if src not in id_to_idx or dst not in id_to_idx:
                continue
            i, j = id_to_idx[src], id_to_idx[dst]
            adj_np[i][j] = 1.0
            adj_np[j][i] = 1.0   # 无向图，对称填充

    adj_norm = _normalize_adj(adj_np)

    # ── 生成 train/val/test mask（Kipf 标准划分：每类 20 训练节点）──
    train_mask = _build_train_mask(labels_np, n_per_class=20)
    val_mask   = _build_val_mask(labels_np, train_mask, n_val=500)
    test_mask  = _build_test_mask(labels_np, train_mask, val_mask, n_test=1000)

    return {
        "features":   torch.FloatTensor(features_np),   # (N, 1433)
        "adj":        torch.FloatTensor(adj_norm),       # (N, N)
        "labels":     torch.LongTensor(labels_np),       # (N,)
        "train_mask": train_mask,
        "val_mask":   val_mask,
        "test_mask":  test_mask,
        "n_nodes":    n,
        "n_edges":    int(adj_np.sum() / 2),             # 无向边数
        "paper_ids":  paper_ids_sorted,                  # 用于 test.py 展示
    }


def _build_train_mask(labels: np.ndarray, n_per_class: int = 20) -> torch.BoolTensor:
    """每类取前 n_per_class 个节点作为训练集（节点按 label 排序后，每类前20个）"""
    n = len(labels)
    mask = torch.zeros(n, dtype=torch.bool)
    for cls in range(NUM_CLASSES):
        cls_indices = np.where(labels == cls)[0]
        selected = cls_indices[:n_per_class]
        mask[selected] = True
    return mask


def _build_val_mask(labels: np.ndarray, train_mask: torch.BoolTensor,
                    n_val: int = 500) -> torch.BoolTensor:
    """从非训练节点中取前 n_val 个作为验证集"""
    n = len(labels)
    mask = torch.zeros(n, dtype=torch.bool)
    non_train = [i for i in range(n) if not train_mask[i]]
    for i in non_train[:n_val]:
        mask[i] = True
    return mask


def _build_test_mask(labels: np.ndarray, train_mask: torch.BoolTensor,
                     val_mask: torch.BoolTensor, n_test: int = 1000) -> torch.BoolTensor:
    """从非训练非验证节点中取前 n_test 个作为测试集"""
    n = len(labels)
    mask = torch.zeros(n, dtype=torch.bool)
    remaining = [i for i in range(n) if not train_mask[i] and not val_mask[i]]
    for i in remaining[:n_test]:
        mask[i] = True
    return mask


# ──────────────────────────────────────────────
# 公共接口
# ──────────────────────────────────────────────

def load_cora(data_dir: str) -> dict:
    """加载 Cora 数据集，首次运行后缓存解析结果以加速后续加载"""
    cache_path = os.path.join(data_dir, "parsed_cora.pt")
    if os.path.exists(cache_path):
        data = torch.load(cache_path, weights_only=False)
        print(f"Cora 数据集已从缓存加载（{data['n_nodes']} 节点，{data['n_edges']} 边）")
        return data

    content_path, cites_path = _check_cora_files(data_dir)
    print("解析 Cora 数据集...")
    data = _parse_cora(content_path, cites_path)
    torch.save(data, cache_path)
    print(f"解析完成，已缓存至 {cache_path}\n")
    return data


def show_dataset_info(data: dict):
    """打印图数据集基本信息"""
    n         = data["n_nodes"]
    e         = data["n_edges"]
    labels    = data["labels"].numpy()
    train_n   = data["train_mask"].sum().item()
    val_n     = data["val_mask"].sum().item()
    test_n    = data["test_mask"].sum().item()

    print(f"节点数: {n}  边数: {e}  特征维度: {data['features'].shape[1]}")
    print(f"训练节点: {train_n}  验证节点: {val_n}  测试节点: {test_n}")
    print(f"类别: {CLASSES}")
    print("各类别节点分布:")
    for i, cls in enumerate(CLASSES):
        cnt = (labels == i).sum()
        print(f"  {cls:25s}: {cnt:4d}")
