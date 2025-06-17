import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class ImprovedSRCNN(nn.Module):
    def __init__(self, upscale=3):
        super().__init__()
        # 特征提取
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 9, padding=4),
            nn.ReLU(),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.ReLU(),
        )
        # 残差块
        self.resblocks = nn.Sequential(
            *[ResBlock(32) for _ in range(3)]
        )

        # 重建层
        self.reconstruction = nn.Conv2d(32, 3, kernel_size=3, padding=1)


    def forward(self, x):
        x = self.features(x)

        x = self.resblocks(x)

        x = self.reconstruction(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
    
    def forward(self, x):
        return x + self.conv(x)