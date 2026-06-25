"""
model.py — 脉冲神经网络（基于同学成功架构）
==========================================
架构：Conv2d(5x5, 32 filters) + IF神经元 + STDP无监督训练
      用STDP训练卷积层提取特征，再用监督MLP分类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, layer, learning
from spikingjelly.activation_based.learning import STDPLearner


class STDP_ConvLayer(nn.Module):
    """使用STDP训练的卷积层 + IF脉冲神经元"""

    def __init__(self, in_channels=1, n_filters=32, kernel_size=5,
                 tau_pre=20.0, tau_post=20.0, a_plus=0.004, lr=0.02,
                 T=10):
        super().__init__()
        self.n_filters = n_filters
        self.T = T

        # 卷积层：5x5, padding=2保持28x28尺寸
        self.conv = nn.Conv2d(in_channels, n_filters, kernel_size,
                              padding=kernel_size // 2, bias=False)
        nn.init.uniform_(self.conv.weight, 0.0, 0.3)

        # IF神经元：阈值=1.0
        self.if_node = neuron.IFNode(v_threshold=1.0, v_reset=0.0,
                                     step_mode='s', detach_reset=True)

        # STDP学习器
        self.stdp = STDPLearner(
            step_mode='s',
            synapse=self.conv,
            sn=self.if_node,
            tau_pre=tau_pre,
            tau_post=tau_post,
            f_pre=lambda w: a_plus * w,      # 权重依赖LTD
            f_post=lambda w: a_plus * (1.0 - w),  # 权重依赖LTP
        )
        self.lr = lr

    def forward(self, spikes, train_stdp=False):
        """
        Args:
            spikes: [T, batch, 1, 28, 28] 泊松编码
            train_stdp: 是否进行STDP更新
        Returns:
            out_spikes: [T, batch, n_filters, 28, 28]
        """
        T = spikes.shape[0]
        self.if_node.reset()

        if train_stdp:
            self.stdp.reset()
            self.stdp.enable()
        else:
            self.stdp.disable()

        out_list = []
        for t in range(T):
            out = self.conv(spikes[t])    # [batch, 32, 28, 28]
            out = self.if_node(out)       # [batch, 32, 28, 28]
            out_list.append(out)

        if train_stdp:
            # 累积T步的STDP更新，一次性应用
            total_dw = None
            n_records = len(self.stdp.in_spike_monitor.records)
            for _ in range(n_records):
                dw = self.stdp.step(on_grad=False, scale=self.lr)
                if dw is not None:
                    total_dw = dw if total_dw is None else total_dw + dw
            if total_dw is not None:
                with torch.no_grad():
                    self.conv.weight.data += total_dw
            self.stdp.disable()

        return torch.stack(out_list)

    def normalize_weights(self, target_norm=5.0):
        """归一化每个滤波器的权重范数到target_norm"""
        with torch.no_grad():
            # [n_filters, 1, 5, 5] → 按滤波器归一化
            w = self.conv.weight.data  # [32, 1, 5, 5]
            w_flat = w.view(w.shape[0], -1)  # [32, 25]
            w_norm = w_flat.norm(p=2, dim=1, keepdim=True)  # [32, 1]
            scale = target_norm / (w_norm + 1e-8)
            w_flat *= scale

    @torch.no_grad()
    def extract_features(self, images, T=10):
        """
        提取图像的特征向量（用于训练MLP分类器）
        Args:
            images: [batch, 1, 28, 28]
            T: 时间步数
        Returns:
            features: [batch, 6272] (32*14*14)
        """
        self.eval()
        batch = images.shape[0]
        spikes = poisson_encode(images, T)

        out = self.forward(spikes, train_stdp=False)  # [T, batch, 32, 28, 28]
        # 平均发放率 → [batch, 32, 28, 28]
        spike_rate = out.mean(dim=0)
        # 2x2平均池化 → [batch, 32, 14, 14]
        pooled = F.avg_pool2d(spike_rate, kernel_size=2, stride=2)
        # 展平 → [batch, 6272]
        features = pooled.view(batch, -1)
        return features


class MLP_Classifier(nn.Module):
    """监督MLP分类器：6272 → 512 → 10"""

    def __init__(self, input_dim=6272, hidden_dim=512, n_classes=11, dropout=0.3):
        """n_classes=11: 0-9数字 + 10=undefined"""
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def poisson_encode(images, T=10):
    """泊松编码：[batch, 1, 28, 28] → [T, batch, 1, 28, 28]"""
    if images.max() > 1.0:
        images = images / 255.0
    rand = torch.rand(T, *images.shape, device=images.device)
    return (rand < images.unsqueeze(0)).float()
