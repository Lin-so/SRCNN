import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.srcnn import SRCNN
from models.srcnn import ImprovedSRCNN
from utils.data_utils import SRDataset, calculate_psnr, calculate_ssim
from config import Config

def train():
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    # 创建保存目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # 初始化模型
    # model = SRCNN(config).to(device)
    model = ImprovedSRCNN().to(device)
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 加载数据
    train_dataset = SRDataset(config.train_data_path, config.scale_factor, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    
    valid_dataset = SRDataset(config.valid_data_path, config.scale_factor, is_train=False)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # 初始化tensorboard
    writer = SummaryWriter('runs/srcnn')
    
    # 训练循环
    best_psnr = 0
    train_losses = []
    valid_losses = []
    valid_psnrs = []
    valid_ssims = []
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}') as pbar:
            for lr_imgs, hr_imgs in pbar:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                
                optimizer.zero_grad()
                outputs = model(lr_imgs)
                loss = criterion(outputs, hr_imgs)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(train_loader)
        print(f'Training Loss: {avg_loss:.4f}')
        train_losses.append(avg_loss)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        
        # 验证
        model.eval()
        avg_psnr = 0
        avg_ssim = 0
        valid_loss = 0

        with torch.no_grad():
            for lr_imgs, hr_imgs in valid_loader:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                
                outputs = model(lr_imgs)
                loss = criterion(outputs, hr_imgs)

                psnr = calculate_psnr(outputs, hr_imgs)
                ssim = calculate_ssim(outputs, hr_imgs)
                
                valid_loss += loss.item()
                avg_psnr += psnr.item()
                avg_ssim += ssim.item()

        avg_loss = valid_loss / len(valid_loader)
        avg_psnr /= len(valid_loader)
        avg_ssim /= len(valid_loader)
        
        valid_losses.append(avg_loss)
        valid_psnrs.append(avg_psnr)
        valid_ssims.append(avg_ssim)
        
        writer.add_scalar('PSNR/valid', avg_psnr, epoch)
        writer.add_scalar('SSIM/valid', avg_ssim, epoch)
        
        print(f'Validate Loss: {avg_loss:.4f}')
        print(f'Average PSNR: {avg_psnr:.2f} dB')
        print(f'Average SSIM: {avg_ssim:.4f}')
        
        # 保存最佳模型
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, 'best_model.pth'))
        
        # 定期保存模型
        # if (epoch + 1) % config.save_interval == 0:
        #     torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, f'model_epoch_{epoch+1}.pth'))
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(2, 2, 2)
    plt.plot(valid_losses)
    plt.title('Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 3)
    plt.plot(valid_psnrs)
    plt.title('Validation PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    
    plt.subplot(2, 2, 4)
    plt.plot(valid_ssims)
    plt.title('Validation SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

if __name__ == '__main__':
    train() 