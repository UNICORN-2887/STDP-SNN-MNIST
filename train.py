"""
train.py — 两阶段训练（含断点续训 + 最佳模型保存 + undefined检测）
=============================================================
阶段1: STDP无监督训练卷积层
阶段2: 监督训练MLP分类器（11类，含undefined）
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time
import os
import json

from model import STDP_ConvLayer, MLP_Classifier, poisson_encode

# 全局 TensorBoard writer
_writer = None


def get_writer(log_dir='./runs'):
    global _writer
    if _writer is None:
        _writer = SummaryWriter(log_dir=log_dir)
    return _writer


def close_writer():
    global _writer
    if _writer is not None:
        _writer.close()
        _writer = None

CKPT_DIR = './checkpoints'
CKPT_FILE = os.path.join(CKPT_DIR, 'checkpoint.pth')
FEAT_TRAIN = os.path.join(CKPT_DIR, 'features_train.pt')
FEAT_TEST  = os.path.join(CKPT_DIR, 'features_test.pt')
FEAT_TRAIN_L = os.path.join(CKPT_DIR, 'features_train_labels.pt')
FEAT_TEST_L  = os.path.join(CKPT_DIR, 'features_test_labels.pt')
BEST_MODEL  = os.path.join(CKPT_DIR, 'best_model.pth')
CONFIG_FILE = os.path.join(CKPT_DIR, 'config.json')
UNDEFINED_CLASS = 10
CONFIDENCE_THRESHOLD = 0.3  # 低于此置信度的预测判为 undefined


# ============================================================
# 断点管理
# ============================================================

def save_checkpoint(stage, stdp_model=None, stdp_epoch=0,
                    mlp_classifier=None, mlp_epoch=0,
                    mlp_optimizer=None, best_test_acc=0.0, features_done=False):
    """保存训练断点"""
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt = {
        'stage': stage,           # 'stdp' | 'features_done' | 'mlp'
        'stdp_epoch': stdp_epoch,
        'mlp_epoch': mlp_epoch,
        'best_test_acc': best_test_acc,
        'features_done': features_done,
        'threshold': CONFIDENCE_THRESHOLD,
    }
    if stdp_model:
        ckpt['stdp_weights'] = stdp_model.state_dict()
    if mlp_classifier:
        ckpt['mlp_weights'] = mlp_classifier.state_dict()
    if mlp_optimizer:
        ckpt['mlp_optimizer'] = mlp_optimizer.state_dict()

    torch.save(ckpt, CKPT_FILE)
    # 保存配置，便于恢复时重建
    cfg = {
        'n_filters': getattr(stdp_model, 'n_filters', 32),
        'T': getattr(stdp_model, 'T', 10),
        'threshold': CONFIDENCE_THRESHOLD,
        'n_classes': 11,  # 0-9 + undefined
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f)


def load_checkpoint(device='cpu'):
    """加载断点，若不存在返回 None"""
    if not os.path.exists(CKPT_FILE):
        return None
    ckpt = torch.load(CKPT_FILE, map_location=device, weights_only=False)
    return ckpt


def save_features(train_feat, train_lbl, test_feat, test_lbl):
    """保存提取的特征到磁盘（大文件，独立保存）"""
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(train_feat, FEAT_TRAIN)
    torch.save(train_lbl, FEAT_TRAIN_L)
    torch.save(test_feat, FEAT_TEST)
    torch.save(test_lbl, FEAT_TEST_L)


def load_features():
    """加载已保存的特征"""
    if not os.path.exists(FEAT_TRAIN):
        return None
    return (torch.load(FEAT_TRAIN, weights_only=True),
            torch.load(FEAT_TRAIN_L, weights_only=True),
            torch.load(FEAT_TEST, weights_only=True),
            torch.load(FEAT_TEST_L, weights_only=True))


def save_best_model(classifier, acc):
    """保存最佳模型"""
    torch.save({'weights': classifier.state_dict(), 'accuracy': acc}, BEST_MODEL)
    print(f'  📌 最佳模型已保存 (acc={acc:.4f})')


def load_best_model(classifier, device='cpu'):
    """加载最佳模型权重"""
    if not os.path.exists(BEST_MODEL):
        return classifier, 0.0
    data = torch.load(BEST_MODEL, map_location=device, weights_only=False)
    classifier.load_state_dict(data['weights'])
    return classifier, data.get('accuracy', 0.0)


# ============================================================
# 阶段1: STDP 训练（含断点续训）
# ============================================================

def train_stdp_conv(model, train_loader, T=10, n_epochs=5,
                    device='cpu', start_epoch=1,
                    tb_log_dir='./runs'):
    """STDP训练，支持从 start_epoch 恢复 + TensorBoard"""
    model.train()
    writer = get_writer(tb_log_dir)
    if start_epoch > 1:
        print(f"阶段1: 从 Epoch {start_epoch} 恢复 STDP 训练")

    for epoch in range(start_epoch, n_epochs + 1):
        total_spikes, total_pixels = 0, 0
        t0 = time.time()
        global_step = epoch

        pbar = tqdm(train_loader, desc=f'STDP Epoch {epoch}/{n_epochs}')
        for images, _ in pbar:
            images = images.to(device)
            b = images.shape[0]
            spikes = poisson_encode(images, T=T)
            out = model(spikes, train_stdp=True)
            total_spikes += out.sum().item()
            total_pixels += b * model.n_filters * 28 * 28
            rate = total_spikes / (total_pixels * T)
            pbar.set_postfix({'spike_rate': f'{rate:.4f}'})

        model.normalize_weights(target_norm=5.0)
        avg_rate = total_spikes / (total_pixels * T)
        elapsed = time.time() - t0
        print(f'  Epoch {epoch}: 发放率={avg_rate:.4f}, 耗时={elapsed:.0f}s')

        # TensorBoard 日志
        writer.add_scalar('STDP/Spike_Rate', avg_rate, global_step)
        writer.add_scalar('STDP/Epoch_Time_s', elapsed, global_step)

        # 断点保存
        save_checkpoint('stdp', stdp_model=model, stdp_epoch=epoch)
        print(f'  💾 断点已保存 (epoch {epoch})')


# ============================================================
# 特征提取
# ============================================================

@torch.no_grad()
def extract_features(model, loader, T=10, device='cpu'):
    """从训练好的STDP卷积层提取特征"""
    model.eval()
    all_features, all_labels = [], []
    for images, labels in tqdm(loader, desc='提取特征'):
        images = images.to(device)
        features = model.extract_features(images, T=T)
        all_features.append(features.cpu())
        all_labels.append(labels)
    return torch.cat(all_features), torch.cat(all_labels)


# ============================================================
# 阶段2: MLP 训练（含最佳模型保存 + 断点续训）
# ============================================================

def train_classifier(classifier, train_features, train_labels,
                     test_features, test_labels,
                     n_epochs=50, batch_size=256, lr=0.001,
                     weight_decay=1e-4, device='cpu',
                     start_epoch=1, best_acc=0.0,
                     checkpoint_callback=None):
    """训练MLP，支持续训和最佳模型保存"""
    classifier = classifier.to(device)
    # 用 11 类分类（第10类=undefined）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)

    # 如果有之前的最佳acc，先恢复
    best_test_acc = best_acc

    n_train = len(train_features)
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)

    if start_epoch > 1:
        print(f"阶段2: 从 Epoch {start_epoch} 恢复 MLP 训练")
    else:
        print(f"阶段2: 训练MLP分类器 ({n_epochs} epochs, batch={batch_size}, 11类含undefined)")

    history = {'epochs': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(start_epoch, n_epochs + 1):
        classifier.train()
        perm = torch.randperm(n_train)
        total_loss, n_batches = 0, 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            x, y = train_features[idx], train_labels[idx]
            optimizer.zero_grad()
            out = classifier(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # 每N轮评估 + TensorBoard + 保存最佳模型
        if epoch % 5 == 0 or epoch == 1 or epoch == n_epochs:
            train_acc = compute_accuracy(classifier, train_features, train_labels, device)
            test_acc = compute_accuracy(classifier, test_features, test_labels, device)
            history['epochs'].append(epoch)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            print(f'  Epoch {epoch:3d}: loss={total_loss/n_batches:.4f}, '
                  f'train_acc={train_acc:.4f}, test_acc={test_acc:.4f}')

            # TensorBoard
            writer = get_writer()
            writer.add_scalar('MLP/Loss', total_loss / n_batches, epoch)
            writer.add_scalar('MLP/Train_Acc', train_acc, epoch)
            writer.add_scalar('MLP/Test_Acc', test_acc, epoch)

            # 保存最佳
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                save_best_model(classifier, best_test_acc)

            # 断点保存
            if checkpoint_callback:
                checkpoint_callback(classifier, optimizer, epoch, best_test_acc)

        # 每epoch记录loss到TB
        writer = get_writer()
        writer.add_scalar('MLP/Loss_per_epoch', total_loss / n_batches, epoch)

    return classifier, history, best_test_acc


@torch.no_grad()
def compute_accuracy(classifier, features, labels, device='cpu',
                     confidence_threshold=None):
    """计算准确率（含undefined检测，仅在指定阈值且包含undefined标签时）"""
    classifier.eval()
    features = features.to(device)
    labels = labels.to(device)
    bs = 256
    correct, total = 0, 0

    for i in range(0, len(features), bs):
        x = features[i:i + bs].to(device)
        y = labels[i:i + bs].to(device)
        out = classifier(x)  # [B, 11] — 11类含undefined

        if confidence_threshold and out.shape[1] == 11:
            probs = torch.softmax(out, dim=1)
            max_prob, pred = probs.max(dim=1)
            # 低置信度 → 强制预测为 undefined (10)
            pred[max_prob < confidence_threshold] = UNDEFINED_CLASS
        else:
            pred = out.argmax(dim=1)

        correct += (pred.cpu() == y.cpu()).sum().item()
        total += len(y)

    return correct / total


# ============================================================
# 推理：支持 undefined 检测
# ============================================================

@torch.no_grad()
def predict_with_confidence(classifier, features, device='cpu',
                            threshold=CONFIDENCE_THRESHOLD):
    """
    带置信度阈值的前向推理。低于阈值的预测归类为 undefined (10)。
    返回 (predictions, confidence_scores)
    """
    classifier.eval()
    features = features.to(device)
    out = classifier(features)
    probs = torch.softmax(out, dim=1)
    max_prob, pred = probs.max(dim=1)
    pred[max_prob < threshold] = UNDEFINED_CLASS
    return pred.cpu(), max_prob.cpu()


# ============================================================
# MLP 断点回调
# ============================================================

def make_mlp_checkpoint_callback(stdp_model, mlp_epoch, features_done=True):
    """返回一个闭包用于MLP训练中的断点保存"""
    def callback(classifier, optimizer, epoch, best_acc):
        save_checkpoint('mlp', stdp_model=stdp_model, stdp_epoch=5,
                        mlp_classifier=classifier, mlp_epoch=epoch,
                        mlp_optimizer=optimizer, best_test_acc=best_acc,
                        features_done=features_done)
    return callback
