class Config:
    # 数据集配置
    train_data_path = 'dataset/DIV2K_train_HR'
    valid_data_path = 'dataset/DIV2K_valid_HR'
    
    # 模型配置
    scale_factor = 3  # 放大倍数
    num_channels = 3  # 图像通道数
    learning_rate = 1e-4
    batch_size = 16
    num_epochs = 100
    
    # 训练配置
    device = 'cuda'  # 或 'cpu'
    save_interval = 5  # 每多少个epoch保存一次模型
    checkpoint_dir = 'checkpoints'
    
    # 模型结构配置
    f1 = 9  # 第一层卷积核大小
    f2 = 1  # 第二层卷积核大小
    f3 = 5  # 第三层卷积核大小
    n1 = 64  # 第一层特征图数量
    n2 = 32  # 第二层特征图数量 