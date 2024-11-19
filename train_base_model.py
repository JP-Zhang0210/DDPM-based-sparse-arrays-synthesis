
import sys
import os

# 获取脚本所在目录的父目录（项目根目录）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
from tqdm import tqdm
import logging

# 导入基础DDPM模型
from models.ddpm_base import DDPM_Base

# 启用异常检测
torch.autograd.set_detect_anomaly(True)

# 设置日志
def setup_logger(log_file):
    """
    配置日志记录器，将日志信息输出到控制台和文件中。

    参数：
    - log_file: 日志文件的路径。

    返回：
    - logger: 配置好的日志记录器。
    """
    logger = logging.getLogger('TrainBaseDDPM')
    logger.setLevel(logging.INFO)

    # 创建文件处理器
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 避免重复日志
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

# 定义自定义Dataset
class AntennaDataset(Dataset):
    def __init__(self, data_path):
        """
        初始化数据集。

        参数：
        - data_path: 预处理后的数据文件路径（Pickle格式）。
        """
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取数据样本。

        返回：
        - x0: 天线单元属性列表，形状为 (input_dim,)
        """
        sample = self.data[idx]
        x0 = torch.tensor(sample['elements'], dtype=torch.float32).flatten()  # 形状: (input_dim,)
        return x0

def train_base_ddpm(model, train_loader, val_loader, optimizer, device, num_epochs=1000, num_timesteps=1000,
                   logger=None):
    """
    训练基础DDPM模型。

    参数：
    - model: 基础DDPM模型实例
    - train_loader: 训练数据加载器
    - val_loader: 验证数据加载器
    - optimizer: 优化器
    - device: 设备（'cuda' 或 'cpu'）
    - num_epochs: 训练轮数
    - num_timesteps: 扩散过程的时间步数
    - logger: 日志记录器
    """
    model.train()

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        logger.info(f"开始训练第 {epoch} 个 epoch")
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)

        for batch_idx, x0 in enumerate(progress_bar):
            x0 = x0.to(device)
            optimizer.zero_grad()

            # 随机采样时间步
            t = torch.randint(0, num_timesteps, (x0.size(0),), device=device).long()

            # 生成噪声
            noise = torch.randn_like(x0).to(device)

            # 正向扩散过程
            xt = model.q_sample(x0, t, noise)

            # 预测噪声
            pred_noise = model.forward(xt, t)

            # 计算损失
            loss = model.loss_fn(pred_noise, noise)

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})

            # 记录模型参数的范数（示例）
            if batch_idx % 100 == 0:
                param_norm = sum(p.data.norm(2).item() for p in model.parameters())
                logger.info(f"Batch {batch_idx} - 参数范数: {param_norm:.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch [{epoch}/{num_epochs}] - Training Loss: {avg_train_loss:.6f}")

        # 验证
        val_loss = evaluate(model, val_loader, device, num_timesteps, logger)
        logger.info(f"Epoch [{epoch}/{num_epochs}] - Validation Loss: {val_loss:.6f}")

    logger.info("训练完成！")

def evaluate(model, val_loader, device, num_timesteps=1000, logger=None):
    """
    在验证集上评估模型。

    参数：
    - model: 基础DDPM模型实例
    - val_loader: 验证数据加载器
    - device: 设备（'cuda' 或 'cpu'）
    - num_timesteps: 扩散过程的时间步数
    - logger: 日志记录器

    返回：
    - avg_val_loss: 平均验证损失
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch_idx, x0 in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
            x0 = x0.to(device)

            # 随机采样时间步
            t = torch.randint(0, num_timesteps, (x0.size(0),), device=device).long()

            # 生成噪声
            noise = torch.randn_like(x0).to(device)

            # 正向扩散过程
            xt = model.q_sample(x0, t, noise)

            # 预测噪声
            pred_noise = model.forward(xt, t)

            # 计算损失
            loss = model.loss_fn(pred_noise, noise)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    model.train()
    return avg_val_loss

def main():
    # 超参数
    input_dim = 324  # 108天线单元，每个单元x, y, z三个属性
    hidden_dim = 256
    embedding_dim = 128
    num_timesteps = 1000
    num_epochs = 1000
    batch_size = 128
    learning_rate = 1e-4
    beta_schedule = 'linear' 

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 设置日志
    log_file = os.path.join('..', 'results', 'logs', 'train_base_model.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = setup_logger(log_file)
    logger.info("开始训练基础DDPM模型")

    # 数据路径
    train_data_path = os.path.join('..', 'data', 'processed', 'train_dataset.pkl')
    val_data_path = os.path.join('..', 'data', 'processed', 'val_dataset.pkl')

    # 创建数据集和数据加载器
    train_dataset = AntennaDataset(train_data_path)
    val_dataset = AntennaDataset(val_data_path)

    # 打印数据集信息
    logger.info(f"训练集样本数量: {len(train_dataset)}")
    if len(train_dataset) > 0:
        sample_x0 = train_dataset[0]
        logger.info(f"第一个训练样本元素: {sample_x0}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型
    model = DDPM_Base(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_timesteps=num_timesteps,
        beta_schedule=beta_schedule,
        embedding_dim=embedding_dim
    ).to(device)
    logger.info(f"模型初始化完成: {model}")

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logger.info(f"优化器初始化完成: {optimizer}")

    # 训练模型
    train_base_ddpm(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs,
                   num_timesteps=num_timesteps, logger=logger)

    # 保存模型
    model_save_path = os.path.join('..', 'results', 'models', 'ddpm_base.pth')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"模型已保存至 {model_save_path}")
    print(f"模型已保存至 {model_save_path}")

if __name__ == '__main__':
    main()
