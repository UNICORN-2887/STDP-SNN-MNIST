"""
model_unsupervised.py — 纯无监督STDP+卷积+神经元投票
=====================================================
无MLP分类器。STDP训练卷积层后，每个滤波器分配到响应最强的数字类别，
分类时各滤波器投票得出最终预测。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.learning import STDPLearner


class Unsupervised_STDP_Conv(nn.Module):
    """纯无监督：STDP训练卷积层 + 滤波器投票分类"""

    def __init__(self, in_channels=1, n_filters=32, kernel_size=5,
                 tau_pre=20.0, tau_post=20.0, a_plus=0.004, lr=0.02,
                 T=10, pool_size=2):
        super().__init__()
        self.n_filters = n_filters
        self.T = T
        self.pool_size = pool_size
        self.lr = lr

        # 卷积层
        self.conv = nn.Conv2d(in_channels, n_filters, kernel_size,
                              padding=kernel_size // 2, bias=False)
        nn.init.uniform_(self.conv.weight, 0.0, 0.3)

        # IF神经元
        self.if_node = neuron.IFNode(v_threshold=1.0, v_reset=0.0,
                                     step_mode='s', detach_reset=True)

        # STDP学习器
        self.stdp = STDPLearner(
            step_mode='s', synapse=self.conv, sn=self.if_node,
            tau_pre=tau_pre, tau_post=tau_post,
            f_pre=lambda w: a_plus * w,
            f_post=lambda w: a_plus * (1.0 - w),
        )

        # 滤波器标签（训练后分配）
        self.filter_labels = None  # [n_filters]
        # 每个滤波器的空间位置标签（可选，更细粒度）
        self.spatial_labels = None  # [n_filters, H, W]

    def forward(self, spikes, train_stdp=False):
        """spikes: [T, batch, 1, 28, 28] → [T, batch, n_filters, 28, 28]"""
        T = spikes.shape[0]
        self.if_node.reset()

        if train_stdp:
            self.stdp.reset()
            self.stdp.enable()
        else:
            self.stdp.disable()

        out_list = []
        for t in range(T):
            out = self.conv(spikes[t])
            out = self.if_node(out)
            out_list.append(out)

        if train_stdp:
            total_dw = None
            n = len(self.stdp.in_spike_monitor.records)
            for _ in range(n):
                dw = self.stdp.step(on_grad=False, scale=self.lr)
                if dw is not None:
                    total_dw = dw if total_dw is None else total_dw + dw
            if total_dw is not None:
                with torch.no_grad():
                    self.conv.weight.data += total_dw
            self.stdp.disable()

        return torch.stack(out_list)

    def normalize_weights(self, target_norm=5.0):
        with torch.no_grad():
            w = self.conv.weight.data
            w_flat = w.view(w.shape[0], -1)
            w_norm = w_flat.norm(p=2, dim=1, keepdim=True)
            scale = target_norm / (w_norm + 1e-8)
            w_flat *= scale

    @torch.no_grad()
    def get_spike_rates(self, images, T=None):
        """获取池化后的发放率 [batch, n_filters, H, W]"""
        if T is None:
            T = self.T
        self.eval()
        batch = images.shape[0]
        spikes = poisson_encode(images, T)
        out = self.forward(spikes, train_stdp=False)  # [T, B, F, 28, 28]
        rates = out.mean(dim=0)  # [B, F, 28, 28]
        # 池化降维
        if self.pool_size > 1:
            rates = F.avg_pool2d(rates, self.pool_size, self.pool_size)
        return rates  # [B, F, H, W]

    @torch.no_grad()
    def assign_labels(self, train_loader, n_samples_per_class=100,
                      device='cpu', use_spatial=False):
        """
        无监督标签分配：
        - use_spatial=False: 每个滤波器整体投票（32个投票者）
        - use_spatial=True: 每个空间位置独立投票（32×H×W个投票者）
        """
        self.eval()
        H = 28 // self.pool_size
        W = 28 // self.pool_size

        if use_spatial:
            # 细粒度：每个空间位置独立分配
            total_response = torch.zeros(10, self.n_filters, H, W, device=device)
            class_counts = [0] * 10
        else:
            # 粗粒度：每个滤波器整体分配
            total_response = torch.zeros(10, self.n_filters, device=device)
            class_counts = [0] * 10

        for images, labels in tqdm(train_loader, desc='标签分配'):
            images = images.to(device)
            labels = labels.to(device)
            rates = self.get_spike_rates(images)  # [B, F, H, W]

            for i in range(len(labels)):
                cls = labels[i].item()
                if class_counts[cls] >= n_samples_per_class:
                    continue
                class_counts[cls] += 1

                if use_spatial:
                    total_response[cls] += rates[i]  # [F, H, W]
                else:
                    total_response[cls] += rates[i].mean(dim=(1, 2))  # [F]

        # 分配到最大响应的类别
        if use_spatial:
            # [10, F, H, W] → argmax → [F, H, W]
            self.spatial_labels = total_response.argmax(dim=0).cpu()
            # 同时计算滤波器级标签（用于快速统计）
            # mode over spatial dimensions: most common class at each filter
            flat = self.spatial_labels.view(self.n_filters, -1)  # [F, H*W]
            self.filter_labels = flat.mode(dim=1)[0]  # [F]
        else:
            # [10, F] → argmax → [F]
            self.filter_labels = total_response.argmax(dim=0).cpu()

        print(f"标签分配完成（每类{min(class_counts)}个样本）")
        return self.filter_labels

    @torch.no_grad()
    def classify(self, images, T=None, use_spatial=False):
        """
        投票分类。
        - 滤波器级：32票
        - 空间级：32×H×W票（更细粒度）
        """
        self.eval()
        rates = self.get_spike_rates(images, T)  # [B, F, H, W]
        batch = rates.shape[0]
        scores = torch.zeros(batch, 10, device=rates.device)

        if use_spatial and self.spatial_labels is not None:
            # 空间级投票
            labels = self.spatial_labels.to(rates.device)  # [F, H, W]
            for cls in range(10):
                mask = (labels == cls).float()  # [F, H, W]
                # 按空间位置投票，权重=发放率
                cls_rates = rates * mask.unsqueeze(0)  # [B, F, H, W]
                scores[:, cls] = cls_rates.sum(dim=(1, 2, 3))
        else:
            # 滤波器级投票
            filter_rates = rates.mean(dim=(2, 3))  # [B, F]
            labels = self.filter_labels.to(rates.device)  # [F]
            for cls in range(10):
                mask = (labels == cls).float()  # [F]
                scores[:, cls] = (filter_rates * mask.unsqueeze(0)).sum(dim=1)

        return scores.argmax(dim=1), scores

    @torch.no_grad()
    def evaluate(self, test_loader, device='cpu', use_spatial=False):
        """评估准确率"""
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for images, labels in tqdm(test_loader, desc='评估'):
            images = images.to(device)
            preds, _ = self.classify(images, use_spatial=use_spatial)
            correct += (preds.cpu() == labels).sum().item()
            total += len(labels)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

        acc = correct / total
        return acc, np.array(all_preds), np.array(all_labels)


def poisson_encode(images, T=10):
    """泊松编码：[B,1,28,28] → [T,B,1,28,28]"""
    if images.max() > 1.0:
        images = images / 255.0
    rand = torch.rand(T, *images.shape, device=images.device)
    return (rand < images.unsqueeze(0)).float()


def train_stdp_unsupervised(model, train_loader, n_epochs=5, device='cpu', verbose=True):
    """纯无监督STDP训练"""
    model.train()
    for epoch in range(1, n_epochs + 1):
        total_spikes, total_pixels = 0, 0
        pbar = tqdm(train_loader, desc=f'STDP E{epoch}/{n_epochs}') if verbose else train_loader
        for images, _ in pbar:
            images = images.to(device)
            b = images.shape[0]
            spikes = poisson_encode(images, model.T)
            out = model(spikes, train_stdp=True)
            total_spikes += out.sum().item()
            total_pixels += b * model.n_filters * 28 * 28
            if verbose:
                pbar.set_postfix({'rate': f'{total_spikes/(total_pixels*model.T):.4f}'})
        model.normalize_weights(target_norm=5.0)
        if verbose:
            print(f'  Epoch {epoch}: rate={total_spikes/(total_pixels*model.T):.4f}')
