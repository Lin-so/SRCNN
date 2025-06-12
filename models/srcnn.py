import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, config):
        super(SRCNN, self).__init__()
        
        # 特征提取层
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(config.num_channels, config.n1, kernel_size=config.f1, padding=config.f1//2),
            nn.ReLU(inplace=True)
        )
        
        # 非线性映射层
        self.non_linear_mapping = nn.Sequential(
            nn.Conv2d(config.n1, config.n2, kernel_size=config.f2, padding=config.f2//2),
            nn.ReLU(inplace=True)
        )
        
        # 重建层
        self.reconstruction = nn.Conv2d(config.n2, config.num_channels, kernel_size=config.f3, padding=config.f3//2)
        
    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.non_linear_mapping(x)
        x = self.reconstruction(x)
        return x 