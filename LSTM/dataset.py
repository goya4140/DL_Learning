"""
AG News 数据集加载与预处理

AG News 是一个新闻标题+摘要的 4 分类数据集：
  1 → World（世界新闻）
  2 → Sports（体育）
  3 → Business（商业）
  4 → Sci/Tech（科技）

训练集 120,000 条，测试集 7,600 条，文本长度适中，适合学习 RNN 序列建模。

数据集在首次运行时从互联网自动下载（约 30MB），之后从本地缓存加载。
需要安装：pip install torchtext
"""

import os
import csv
import pickle
import tarfile
import urllib.request

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 消除 torchtext 弃用警告（必须在 torchtext 子模块 import 之前调用）
import torchtext
torchtext.disable_torchtext_deprecation_warning()

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


# AG News 类别（CSV 标签为 1-indexed，内部统一转为 0-indexed）
CLASSES = ["World", "Sports", "Business", "Sci/Tech"]
NUM_CLASSES = 4

# 分词器（基础英文：小写 + 标点切分，无需额外模型）
_tokenizer = get_tokenizer("basic_english")

# AG News tgz 下载地址（fast.ai 公共镜像）
_AG_NEWS_URL = "https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz"


# ──────────────────────────────────────────────
# 数据集下载与解析
# ──────────────────────────────────────────────

def _download_agnews(data_dir: str):
    """首次运行时下载并解压 AG News CSV（约 30MB），后续直接读本地文件"""
    train_path = os.path.join(data_dir, "ag_news_csv", "train.csv")
    test_path  = os.path.join(data_dir, "ag_news_csv", "test.csv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        return train_path, test_path

    os.makedirs(data_dir, exist_ok=True)
    print(f"下载 AG News 数据集（约 30MB）: {_AG_NEWS_URL}")
    tgz_path = os.path.join(data_dir, "ag_news_csv.tgz")
    urllib.request.urlretrieve(_AG_NEWS_URL, tgz_path)

    print("解压数据集...")
    with tarfile.open(tgz_path, "r:gz") as f:
        f.extractall(data_dir)
    os.remove(tgz_path)
    print("AG News 数据集准备完成\n")

    return train_path, test_path


def _read_csv(csv_path: str):
    """读取 AG News CSV，返回 list of (label_1indexed, text)

    标签保持 1-4（与原始 torchtext AG_NEWS() 返回格式一致），
    由 AGNewsDataset 在构建时统一转为 0-3。
    """
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            label = int(row[0])          # 保留 1-4，AGNewsDataset 负责减 1
            text  = row[1] + " " + row[2]  # title + description 拼接
            data.append((label, text))
    return data


# ──────────────────────────────────────────────
# 词汇表构建
# ──────────────────────────────────────────────

def _yield_tokens(data_list):
    for _, text in data_list:
        yield _tokenizer(text)


def build_vocab(data_dir: str, vocab_size: int):
    """构建词汇表，第一次运行后缓存到 data_dir/vocab.pkl 加速后续加载"""
    vocab_path = os.path.join(data_dir, "vocab.pkl")
    if os.path.exists(vocab_path):
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        print(f"词汇表已从缓存加载（大小: {len(vocab)}）")
        return vocab

    print("首次运行：构建词汇表...")
    train_path, _ = _download_agnews(data_dir)
    train_data = _read_csv(train_path)

    vocab = build_vocab_from_iterator(
        _yield_tokens(train_data),
        max_tokens=vocab_size,
        specials=["<unk>", "<pad>"],
    )
    vocab.set_default_index(vocab["<unk>"])

    os.makedirs(data_dir, exist_ok=True)
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"词汇表构建完成（大小: {len(vocab)}），已缓存至 {vocab_path}")
    return vocab


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class AGNewsDataset(Dataset):
    def __init__(self, data_list: list, vocab, max_len: int):
        self.samples = []
        for label, text in data_list:
            label_idx = int(label) - 1          # 1-4 → 0-3
            token_ids = vocab(_tokenizer(text))[:max_len]
            self.samples.append((label_idx, torch.tensor(token_ids, dtype=torch.long)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _collate_fn(batch):
    """将一个 batch 的变长序列填充为相同长度"""
    labels, texts = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    # padding_value=1 对应词汇表中 <pad> 的索引
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=1)
    return labels, texts_padded


# ──────────────────────────────────────────────
# DataLoader
# ──────────────────────────────────────────────

def get_agnews_loaders(data_dir: str, vocab, max_len: int,
                       batch_size: int, num_workers: int = 2,
                       pin_memory: bool = True):
    train_path, test_path = _download_agnews(data_dir)

    train_dataset = AGNewsDataset(_read_csv(train_path), vocab, max_len)
    test_dataset  = AGNewsDataset(_read_csv(test_path),  vocab, max_len)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=_collate_fn,
    )
    return train_loader, test_loader


def show_dataset_info(train_loader, test_loader):
    train_n = len(train_loader.dataset)
    test_n  = len(test_loader.dataset)
    print(f"训练集: {train_n:,} 条  测试集: {test_n:,} 条")
    print(f"类别: {CLASSES}")
    print(f"训练 batch 数: {len(train_loader)}  测试 batch 数: {len(test_loader)}")
