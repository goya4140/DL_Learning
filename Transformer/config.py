class Config:
    # 数据（复用 AG News，与 RNN/LSTM 相同任务）
    data_dir    = "./data"
    batch_size  = 64
    num_workers = 2
    vocab_size  = 20000
    max_len     = 64      # Transformer 处理变长序列比 RNN 更高效，可适当增长

    # Transformer 结构
    d_model            = 128  # 词嵌入维度（必须能被 nhead 整除）
    nhead              = 4    # 多头注意力头数（每头维度 = d_model/nhead = 32）
    num_encoder_layers = 2    # Encoder 层数（每层含 MHA + FFN）
    d_ff               = 256  # FFN 中间层维度（通常 2×~4× d_model）
    dropout_rate       = 0.1  # Transformer 标准 dropout（比 RNN/LSTM 小）

    num_classes = 4   # AG News：World / Sports / Business / Sci&Tech

    # 训练
    epochs        = 10
    learning_rate = 1e-3
    weight_decay  = 1e-4
    lr_step_size  = 3    # 每 3 个 epoch 学习率 ×gamma
    lr_gamma      = 0.5

    device    = "auto"   # "auto" | "mps" | "cuda" | "cpu"
    seed      = 42
    save_path = "./checkpoints/best_model.pth"
    log_interval = 100


config = Config()
