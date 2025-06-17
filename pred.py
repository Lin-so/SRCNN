import os
import torch
import cv2
from torchvision import transforms
from models.srcnn import SRCNN
from models.srcnn import ImprovedSRCNN
from config import Config

def pred():
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    # model = SRCNN(config).to(device)
    model = ImprovedSRCNN().to(device)
    model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, 'best_model1.pth')))
    model.eval()
    
    # 创建结果保存目录
    results_dir = 'predicted'
    os.makedirs(results_dir, exist_ok=True)

    transform = transforms.ToTensor()

    img_path = r'D:\Project\SRCNN\test_images\4D088E5AF5265A2C7EDDEC4A6621C273.png'
    img_name = os.path.basename(img_path)[:-4]

    img = cv2.imread(img_path)
    cv2.imwrite(f"{results_dir}/{img_name}.png",img)
    with torch.no_grad():
        img = transform(img).to(device)
        # 生成超分辨率图像
        sr_img = model(img)
        # 将图像转换为numpy数组
        sr_img = sr_img.cpu().numpy().transpose(1, 2, 0)
        # 保存结果图像
        cv2.imwrite(f"{results_dir}/{img_name}_pred.png",255*sr_img)
    

if __name__ == '__main__':
    pred() 