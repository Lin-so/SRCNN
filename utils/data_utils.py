import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

class SRDataset(Dataset):
    def __init__(self, data_path, scale_factor, patch_size=33, is_train=True):
        self.data_path = data_path
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.is_train = is_train
        
        self.image_files = [f for f in os.listdir(data_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(patch_size),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        
        img = self.transform(img)
        # 生成低分辨率图像
        lr_img = F.interpolate(img.unsqueeze(0), 
                                scale_factor=1/self.scale_factor, 
                                mode='bicubic', 
                                align_corners=False).squeeze(0)
        # 上采样到原始大小
        lr_img = F.interpolate(lr_img.unsqueeze(0), 
                                size=img.shape[1:], 
                                mode='bicubic', 
                                align_corners=False).squeeze(0)
        return lr_img, img

def calculate_psnr(img1, img2):
    """计算PSNR"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    """计算SSIM"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    img1 = img1 * 255
    img2 = img2 * 255
    
    mu1 = F.avg_pool2d(img1, 3, stride=1, padding=1)
    mu2 = F.avg_pool2d(img2, 3, stride=1, padding=1)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, 3, stride=1, padding=1) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, 3, stride=1, padding=1) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, 3, stride=1, padding=1) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() 