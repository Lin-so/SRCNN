import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm

from models.srcnn import SRCNN
from utils.data_utils import SRDataset, calculate_psnr, calculate_ssim
from config import Config

def test():
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = SRCNN(config).to(device)
    model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, 'best_model.pth')))
    model.eval()
    
    # 加载测试数据
    test_dataset = SRDataset(config.valid_data_path, config.scale_factor, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 创建结果保存目录
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 测试
    total_psnr = 0
    total_ssim = 0
    
    with torch.no_grad():
        for i, (lr_imgs, hr_imgs, img_names) in enumerate(tqdm(test_loader)):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # 生成超分辨率图像
            sr_imgs = model(lr_imgs)
            
            # 计算评估指标
            psnr = calculate_psnr(sr_imgs, hr_imgs)
            ssim = calculate_ssim(sr_imgs, hr_imgs)
            
            total_psnr += psnr.item()
            total_ssim += ssim.item()
            
            # 保存结果图像
            if i < 5:  # 只保存前5张图片的结果
                # 将图像转换为numpy数组
                lr_img = lr_imgs[0].cpu().numpy().transpose(1, 2, 0)
                sr_img = sr_imgs[0].cpu().numpy().transpose(1, 2, 0)
                hr_img = hr_imgs[0].cpu().numpy().transpose(1, 2, 0)
                
                # 创建对比图
                plt.figure(figsize=(15, 5))
                
                plt.subplot(131)
                plt.imshow(np.clip(lr_img, 0, 1))
                plt.title('Low Resolution')
                plt.axis('off')
                
                plt.subplot(132)
                plt.imshow(np.clip(sr_img, 0, 1))
                plt.title('Super Resolution')
                plt.axis('off')
                
                plt.subplot(133)
                plt.imshow(np.clip(hr_img, 0, 1))
                plt.title('High Resolution')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'comparison_{img_names[0]}'))
                plt.close()
    
    # 计算平均指标
    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    
    print(f'Average PSNR: {avg_psnr:.2f} dB')
    print(f'Average SSIM: {avg_ssim:.4f}')
    
    # 保存评估结果
    with open(os.path.join(results_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f'Average PSNR: {avg_psnr:.2f} dB\n')
        f.write(f'Average SSIM: {avg_ssim:.4f}\n')

if __name__ == '__main__':
    test() 