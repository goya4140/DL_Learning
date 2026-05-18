# 超参数配置文件，所有可调参数集中在此处管理

class Config:
    # 数据相关
    data_dir = "./data"          # 数据集下载路径
    batch_size = 128             # 较大 batch 利用 GPU/MPS 的并行计算能力
    num_workers = 2              # 数据加载并行线程数

    # 模型结构
    in_channels = 3              # RGB 三通道输入
    channels = [32, 64, 128]    # 三个卷积块的输出通道数，可自由增减
    fc_hidden = 256              # 全连接隐藏层神经元数
    output_size = 10             # CIFAR-10 共 10 类
    dropout_rate = 0.5           # FC 层后的 Dropout 比例

    # 训练相关
    epochs = 30                  # CIFAR-10 比 MNIST 难，训练更多轮
    learning_rate = 1e-3         # 初始学习率
    weight_decay = 1e-4          # L2 正则化系数
    lr_step_size = 10            # 每隔多少 epoch 降低学习率
    lr_gamma = 0.5               # 学习率衰减倍率

    # 设备配置
    # "auto" 自动检测（优先级：CUDA > MPS > CPU）
    # "mps"  Mac M 系列芯片（M1/M2/M3/M4）
    # "cuda" NVIDIA GPU（Windows/Linux）
    # "cpu"  仅 CPU（无 GPU 时的兜底）
    device = "auto"

    # 其他
    seed = 42                    # 随机种子，保证实验可复现
    save_path = "./checkpoints/best_model.pth"  # 最优模型保存路径
    log_interval = 50            # 每隔多少 batch 打印一次训练日志（CIFAR-10 batch 更大）


config = Config()
