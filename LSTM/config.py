class Config:
    # 数据
    data_dir    = "./data"
    batch_size  = 64
    num_workers = 2

    # 文本处理
    vocab_size = 20000   # 取频率最高的 20000 词构建词汇表
    embed_dim  = 128     # 词向量维度
    max_len    = 64      # LSTM 能处理比 vanilla RNN 更长的序列（门控机制缓解梯度消失）

    # LSTM 结构
    hidden_size   = 128  # LSTM 每层隐藏维度（LSTM 每个单元有 4 组参数，比 RNN 参数量多 4 倍）
    num_layers    = 2    # 双层 LSTM：LSTM 的门控机制使多层堆叠仍能正常训练
    dropout_rate  = 0.5  # 层间 Dropout（nn.LSTM dropout 参数 + 额外 Dropout 层）
    bidirectional = True # 双向 LSTM：同时建模正向和反向序列

    output_size = 4      # AG News 4 分类：World / Sports / Business / Sci&Tech

    # 训练
    epochs        = 10
    learning_rate = 1e-3
    weight_decay  = 1e-4
    lr_step_size  = 3    # 每 3 个 epoch 学习率 ×gamma
    lr_gamma      = 0.5
    grad_clip     = 5.0  # 梯度裁剪（LSTM 已大幅缓解梯度爆炸，但保留作为安全保障）

    device    = "auto"   # "auto" | "mps" | "cuda" | "cpu"
    seed      = 42
    save_path = "./checkpoints/best_model.pth"
    log_interval = 100   # 每隔多少 batch 打印一次日志


config = Config()
