class Config:
    # 数据
    data_dir    = "./data"
    batch_size  = 64
    num_workers = 2

    # 文本处理
    vocab_size = 20000   # 取频率最高的 20000 词构建词汇表
    embed_dim  = 128     # 词向量（Embedding）维度
    max_len    = 32      # 序列最大长度：AG News 标题约 10 词 + 描述约 30 词，32 已足够
                         # 更短序列 → 梯度反传路径更短 → 梯度消失更轻

    # RNN 结构
    hidden_size   = 256  # RNN 隐藏层维度 h
    num_layers    = 1    # 单层 RNN：梯度路径 = max_len × num_layers，减层降路径
    dropout_rate  = 0.3  # 模型简化后降低 dropout（原 0.5 对单层 RNN 过强）

    output_size = 4      # AG News 4 分类：World / Sports / Business / Sci&Tech

    # 训练
    epochs        = 10
    learning_rate = 1e-3
    weight_decay  = 1e-4
    lr_step_size  = 3    # 每 3 个 epoch 学习率 ×gamma
    lr_gamma      = 0.5
    grad_clip     = 5.0  # 梯度裁剪阈值（防止 RNN 梯度爆炸）

    device    = "auto"   # "auto" | "mps" | "cuda" | "cpu"
    seed      = 42
    save_path = "./checkpoints/best_model.pth"
    log_interval = 100   # 每隔多少 batch 打印一次日志


config = Config()
