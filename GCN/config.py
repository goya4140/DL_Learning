class Config:
    # 数据
    data_dir = "./data"
    dataset  = "cora"

    # 模型结构（Cora 数据集固定参数）
    in_features  = 1433  # Cora 节点特征维度（词袋向量）
    hidden_dim   = 64    # 隐藏层维度（原论文用 16，64 更稳定）
    num_classes  = 7     # Cora 7 类机器学习子领域
    num_layers   = 2     # GCN 层数（超过 3 层效果通常因过平滑下降）
    dropout_rate = 0.5   # Dropout（只在训练时应用于层间）

    # 训练（遵循原论文超参数）
    epochs        = 200
    learning_rate = 1e-2   # 原论文 lr=0.01（比 MLP/CNN/RNN 大10倍）
    weight_decay  = 5e-4   # 原论文 weight_decay=5e-4（只对第一层参数应用，此处简化为全局）

    device    = "auto"   # "auto" | "mps" | "cuda" | "cpu"
    seed      = 42
    save_path = "./checkpoints/best_model.pth"
    log_interval = 10    # 每隔多少 epoch 打印一次日志（GCN 无 batch，按 epoch 打印）


config = Config()
