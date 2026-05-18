# 超参数配置文件，所有可调参数集中在此处管理

class Config:
    # 数据相关
    data_dir = "./data"          # 数据集下载路径
    batch_size = 64              # 每批训练样本数
    num_workers = 2              # 数据加载并行线程数

    # 模型结构
    input_size = 784             # 28x28 像素展平后的维度
    hidden_sizes = [512, 256, 128]  # 各隐藏层神经元数，可自由增减层数
    output_size = 10             # 分类数（0-9 共10类）
    dropout_rate = 0.3           # Dropout 比例，防止过拟合

    # 训练相关
    epochs = 20                  # 训练轮数
    learning_rate = 1e-3         # 初始学习率
    weight_decay = 1e-4          # L2 正则化系数
    lr_step_size = 5             # 每隔多少 epoch 降低学习率
    lr_gamma = 0.5               # 学习率衰减倍率

    # 其他
    seed = 42                    # 随机种子，保证实验可复现
    save_path = "./checkpoints/best_model.pth"  # 最优模型保存路径
    log_interval = 100           # 每隔多少 batch 打印一次训练日志


config = Config()
