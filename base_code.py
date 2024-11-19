

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_timesteps=1000):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_timesteps = max_timesteps

        # Create sinusoidal embeddings
        position = torch.arange(0, max_timesteps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_timesteps, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, t):
        """
        参数：
        - t: 时间步张量，形状为 (batch_size,)

        返回：
        - 嵌入后的时间步张量，形状为 (batch_size, embedding_dim)
        """
        return self.pe[t].squeeze(1)


class ConditionEmbedding(nn.Module):
    def __init__(self, condition_dim, embedding_dim):
        """
        条件嵌入模块，将条件信息（S和P）嵌入到高维空间。

        参数：
        - condition_dim: 条件信息的维度（例如，S和P，dim=2）。
        - embedding_dim: 嵌入后的维度。
        """
        super(ConditionEmbedding, self).__init__()
        self.fc1 = nn.Linear(condition_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.ReLU()

    def forward(self, c):
        """
        前向传播，将条件信息嵌入到高维空间。

        参数：
        - c: 条件信息，形状为 (batch_size, condition_dim)

        返回：
        - 条件嵌入，形状为 (batch_size, embedding_dim)
        """
        x = self.activation(self.fc1(c))
        x = self.activation(self.fc2(x))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, condition_emb_dim):
        super(ResidualBlock, self).__init__()
        self.fc_time = nn.Linear(time_emb_dim, out_channels)
        self.fc_condition = nn.Linear(condition_emb_dim, out_channels)
        self.net = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU()
        )
        self.residual = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, x, t_emb, c_emb):
        """
        前向传播，结合时间步嵌入和条件嵌入。

        参数：
        - x: 输入特征，形状为 (batch_size, in_channels)
        - t_emb: 时间步嵌入，形状为 (batch_size, time_emb_dim)
        - c_emb: 条件嵌入，形状为 (batch_size, condition_emb_dim)

        返回：
        - 输出特征，形状为 (batch_size, out_channels)
        """
        out = self.net(x)
        out += self.fc_time(t_emb)
        out += self.fc_condition(c_emb)
        out = F.relu(out)
        return out + self.residual(x)


class MultiObjectiveLoss(nn.Module):
    def __init__(self, weight_s=1.0, weight_p=1.0):
        """
        多目标损失函数，结合S和P的损失。

        参数：
        - weight_s: S损失的权重。
        - weight_p: P损失的权重。
        """
        super(MultiObjectiveLoss, self).__init__()
        self.weight_s = weight_s
        self.weight_p = weight_p
        self.mse = nn.MSELoss()

    def forward(self, pred_noise, true_noise, S_pred, S_true, P_pred, P_true):
        """
        计算多目标损失。

        参数：
        - pred_noise: 预测的噪声，形状为 (batch_size, input_dim)
        - true_noise: 真实的噪声，形状为 (batch_size, input_dim)
        - Q_pred: 预测的Q值，形状为 (batch_size,)
        - Q_true: 真实的Q值，形状为 (batch_size,)
        - P_pred: 预测的P值，形状为 (batch_size,)
        - P_true: 真实的P值，形状为 (batch_size,)

        返回：
        - loss: 综合损失值
        """
        loss_noise = self.mse(pred_noise, true_noise)
        loss_s = self.mse(S_pred, S_true)
        loss_p = self.mse(P_pred, P_true)
        loss = self.weight_s * loss_s + self.weight_p * loss_p + loss_noise
        return loss


class DDPM_MultiObjective(nn.Module):
    def __init__(self, input_dim=108, hidden_dim=256, num_timesteps=1000, beta_schedule='linear', embedding_dim=128,
                 condition_dim=2):
        """
        多目标优化DDPM模型实现。

        """
        super(DDPM_MultiObjective, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.condition_dim = condition_dim

        # Beta调度
        self.betas = self.get_beta_schedule(beta_schedule)
        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

        # 时间步嵌入
        self.time_emb = SinusoidalPositionalEmbedding(embedding_dim, max_timesteps=num_timesteps)
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 条件嵌入
        self.condition_emb = ConditionEmbedding(condition_dim, hidden_dim)

        # 网络架构
        self.model = nn.Sequential(
            ResidualBlock(input_dim, hidden_dim, hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )

        # 额外的网络用于预测S和P
        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.p_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def get_beta_schedule(self, schedule_type):
        """
        获取beta参数的调度方式。

        参数：
        - schedule_type: 调度类型（'linear', 'cosine'）。

        返回：
        - betas: beta参数的张量。
        """
        if schedule_type == 'linear':
            return torch.linspace(1e-4, 0.02, self.num_timesteps)
        elif schedule_type == 'cosine':
            return self.cosine_beta_schedule()
        else:
            raise NotImplementedError(f"Beta schedule '{schedule_type}' not implemented.")

    def cosine_beta_schedule(self):
        """
        使用余弦调度方式计算beta参数。

        返回：
        - betas: beta参数的张量。
        """
        steps = self.num_timesteps + 1
        x = torch.linspace(0, math.pi / 2, steps)
        alphas_cumprod = torch.cos((x / math.pi) * (math.pi / 2)) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, min=1e-4, max=0.999)
        return betas

    def forward(self, x, t, c):
        """
        前向传播，预测噪声、S和P。

        参数：
        - x: 含噪声的数据，形状为 (batch_size, input_dim)
        - t: 时间步，形状为 (batch_size,)
        - c: 条件信息，形状为 (batch_size, condition_dim)

        返回：
        - pred_noise: 预测的噪声，形状为 (batch_size, input_dim)
        - S_pred: 预测的S值，形状为 (batch_size, 1)
        - P_pred: 预测的P值，形状为 (batch_size, 1)
        """
        t_emb = self.time_emb(t).to(x.device)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.condition_emb(c).to(x.device)
        hidden = self.model(x + t_emb + c_emb)
        pred_noise = hidden
        S_pred = self.s_network(hidden)
        P_pred = self.p_network(hidden)
        return pred_noise, S_pred, P_pred

    def q_sample(self, x0, t, c, noise=None):
        """
        根据扩散过程生成噪声图像。

        参数：
        - x0: 原始数据，形状为 (batch_size, input_dim)
        - t: 时间步，形状为 (batch_size,)
        - c: 条件信息，形状为 (batch_size, condition_dim)
        - noise: 可选的噪声，形状为 (batch_size, input_dim)

        返回：
        - xt: 含噪声的数据，形状为 (batch_size, input_dim)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).unsqueeze(1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t]).unsqueeze(1)
        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise

    def p_sample(self, xt, t, c):
        """
        反向扩散过程，预测x_{t-1}。

        参数：
        - xt: 当前时间步的数据，形状为 (batch_size, input_dim)
        - t: 当前时间步，形状为 (batch_size,)
        - c: 条件信息，形状为 (batch_size, condition_dim)

        返回：
        - x_prev: 预测的x_{t-1}，形状为 (batch_size, input_dim)
        - Q_prev: 预测的Q值，形状为 (batch_size, 1)
        - P_prev: 预测的P值，形状为 (batch_size, 1)
        """
        pred_noise, S_pred, P_pred = self.forward(xt, t, c)
        model_mean = self.s_posterior_mean(xt, t, c, pred_noise)
        if t.max() == 0:
            return model_mean, S_pred, P_pred
        else:
            noise = torch.randn_like(xt)
            posterior_variance = self.posterior_variance(t)
            return model_mean + torch.sqrt(posterior_variance) * noise, S_pred, P_pred


    def sample(self, batch_size, c, device='cpu'):
        """
        生成样本。

        参数：
        - batch_size: 批次大小
        - c: 条件信息，形状为 (batch_size, condition_dim)
        - device: 设备（'cuda' 或 'cpu'）

        返回：
        - x0: 生成的样本，形状为 (batch_size, input_dim)
        """
        x = torch.randn(batch_size, self.input_dim).to(device)
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, dtype=torch.long).to(device)
            x, Q_pred, P_pred = self.p_sample(x, t_batch, c)
        return x


