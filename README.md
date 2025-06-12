# SRCNN 超分辨率重建项目

一个基于PyTorch实现的图像超分辨率卷积神经网络（SRCNN）

## 环境要求

- Python 3.6+
- PyTorch 1.7+
- torchvision
- numpy
- matplotlib

安装依赖：
```bash
.venv/bin/pip install torch torchvision numpy matplotlib
```

## 项目结构
```
SRCNN/
├── Srcnn/
│   ├── train.py          # 训练脚本
│   ├── model.py          # 网络模型定义
│   ├── datasets.py       # 数据加载器
│   ├── utils.py          # 工具函数
│   ├── best.pkl          # 最佳模型权重
│   └── data_img/
│       └── data_figure/  # 训练数据集目录
```

## 快速开始

### 数据准备
1. 将LR-HR图像对放入`data_img/data_figure/`目录
2. 图像命名格式：`[编号]_m.png` (低分辨率) 和 `[编号].png` (高分辨率)
3. 支持.png格式图像，尺寸需保持一致

### 训练模型
```bash
cd SRCNN/Srcnn
.venv/bin/python train.py
```
根据提示输入参数：
- 训练轮数（推荐50+）
- 批处理大小（根据显存调整）
- 学习率（默认0.001）

### 输出文件
- `model.pkl`: 每轮训练的模型权重
- `best.pkl`: 最佳PSNR指标对应的模型权重

## 结果可视化
训练结束后会自动显示PSNR随训练轮次的变化曲线：
![PSNR可视化示例](data_figure/example_psnr.png)

## 许可证
[MIT License](LICENSE)
