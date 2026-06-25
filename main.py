"""
main.py — STDP+MLP MNIST 识别（含断点续训 + undefined检测）
============================================================
阶段1: STDP无监督训练卷积层
阶段2: 监督训练MLP分类器（11类：0-9 + undefined）
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from model import STDP_ConvLayer, MLP_Classifier
from train import (
    train_stdp_conv, extract_features, train_classifier,
    compute_accuracy, predict_with_confidence,
    save_checkpoint, load_checkpoint, save_features, load_features,
    load_best_model, make_mlp_checkpoint_callback, close_writer,
    CKPT_DIR, CKPT_FILE, FEAT_TRAIN, FEAT_TEST,
    UNDEFINED_CLASS, CONFIDENCE_THRESHOLD
)

# ============================================================
# 创建/恢复模型
# ============================================================

def create_or_resume_model(device, args):
    """创建新模型或从断点恢复"""
    ckpt = load_checkpoint(device)
    if ckpt is None:
        print("🆕 首次训练，创建新模型")
        conv_layer = STDP_ConvLayer(
            n_filters=args.n_filters, T=args.T,
            lr=args.stdp_lr, a_plus=args.a_plus,
            tau_pre=args.tau_pre, tau_post=args.tau_post,
        ).to(device)
        classifier = MLP_Classifier(
            hidden_dim=args.hidden_dim,
            n_classes=11, dropout=args.dropout
        )
        return conv_layer, classifier, 0, 0, 0.0, 1, 1

    # 恢复
    stage = ckpt.get('stage', 'stdp')
    print(f"📂 检测到断点 (stage={stage}, stdp_epoch={ckpt.get('stdp_epoch', 0)})")

    # 重建 conv 层并加载权重
    conv_layer = STDP_ConvLayer(n_filters=32, T=args.T).to(device)
    if 'stdp_weights' in ckpt:
        conv_layer.load_state_dict(ckpt['stdp_weights'])

    # 重建 MLP 并加载权重（如果有）
    classifier = MLP_Classifier(hidden_dim=args.hidden_dim, n_classes=11, dropout=args.dropout)
    if ckpt.get('mlp_weights'):
        classifier.load_state_dict(ckpt['mlp_weights'])

    best_acc = ckpt.get('best_test_acc', 0.0)
    stdp_done = ckpt.get('stdp_epoch', 0)
    mlp_done = ckpt.get('mlp_epoch', 0)

    if stdp_done >= args.stdp_epochs:
        next_stdp = args.stdp_epochs + 1  # 已完成, 跳过
    else:
        next_stdp = stdp_done + 1          # 从下一epoch继续

    if mlp_done >= args.mlp_epochs:
        next_mlp = args.mlp_epochs + 1  # 已完成, 跳过
    else:
        next_mlp = mlp_done + 1          # 从下一epoch继续

    return conv_layer, classifier, stdp_done, mlp_done, best_acc, next_stdp, next_mlp


# ============================================================
# 数据加载（含 undefined 合成样本）
# ============================================================

def add_undefined_samples(train_features, train_labels, n_undefined=500):
    """
    添加 undefined 合成样本：随机混合两个不同数字的特征。
    这教 MLP 识别"不像任何数字"的输入。
    """
    n_samples = len(train_features)
    undefined_feats = []
    undefined_labels = []

    for _ in range(n_undefined):
        i, j = torch.randint(0, n_samples, (2,))
        while train_labels[i] == train_labels[j]:  # 确保不同类
            j = torch.randint(0, n_samples, (1,)).item()
        # 混合特征（50-50）
        alpha = torch.rand(1).item()
        blended = alpha * train_features[i] + (1 - alpha) * train_features[j]
        undefined_feats.append(blended)
        undefined_labels.append(UNDEFINED_CLASS)

    if n_undefined > 0:
        train_features = torch.cat([train_features, torch.stack(undefined_feats)])
        train_labels = torch.cat([train_labels, torch.tensor(undefined_labels, dtype=train_labels.dtype)])
        print(f'  添加 {n_undefined} 个 undefined 合成样本（混合不同数字特征）')

    return train_features, train_labels


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='STDP+MLP MNIST（断点续训 + undefined检测）')
    parser.add_argument('--n-filters', type=int, default=32)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--stdp-epochs', type=int, default=5)
    parser.add_argument('--stdp-lr', type=float, default=0.02)
    parser.add_argument('--a-plus', type=float, default=0.004)
    parser.add_argument('--tau-pre', type=float, default=20.0)
    parser.add_argument('--tau-post', type=float, default=20.0)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--mlp-epochs', type=int, default=50)
    parser.add_argument('--mlp-lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--confidence-threshold', type=float, default=0.3,
                       help='低于此置信度的预测判为undefined')
    parser.add_argument('--n-undefined', type=int, default=500,
                       help='合成undefined训练样本数')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--train-samples', type=int, default=None)
    parser.add_argument('--save-dir', type=str, default='./results')
    parser.add_argument('--force-restart', action='store_true',
                       help='忽略已有断点，从头训练')
    parser.add_argument('--resume-only', action='store_true',
                       help='仅从断点恢复，若没有断点则退出')
    parser.add_argument('--demo-undefined', action='store_true',
                       help='演示undefined检测：输入非数字样本来测试')
    parser.add_argument('--tb-log-dir', type=str, default='./runs',
                       help='TensorBoard日志目录 (默认 ./runs)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 强制重启
    if args.force_restart and os.path.exists(CKPT_FILE):
        import shutil
        shutil.rmtree(CKPT_DIR)
        print('🗑️  已清除旧断点，从头训练')

    # 仅恢复模式
    if args.resume_only and not os.path.exists(CKPT_FILE):
        print('❌ 没有检测到断点文件，退出。')
        sys.exit(1)

    # 创建或恢复模型
    conv_layer, classifier, stdp_done, mlp_done, best_acc, next_stdp, next_mlp = \
        create_or_resume_model(device, args)

    # === 加载数据 ===
    transform = transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(root=args.data_dir, train=True,
                                           transform=transform, download=True)
    test_set  = torchvision.datasets.MNIST(root=args.data_dir, train=False,
                                           transform=transform, download=True)

    if args.train_samples and args.train_samples < len(train_set):
        idx = torch.randperm(len(train_set))[:args.train_samples]
        train_set = Subset(train_set, idx)

    train_loader_1 = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    full_train_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

    print(f'设备: {device} | 训练: {len(train_set)} | 测试: {len(test_set)} | '
          f'undefined阈值: {args.confidence_threshold}')
    print(f'STDP进度: {stdp_done}/{args.stdp_epochs} | MLP进度: {mlp_done}/{args.mlp_epochs}')
    print(f'TensorBoard: tensorboard --logdir {args.tb_log_dir}')
    print('=' * 60)

    # ================================================================
    # 阶段1: STDP 训练
    # ================================================================
    if next_stdp <= args.stdp_epochs:
        print(f'\n=== 阶段1: STDP训练 (Epoch {next_stdp}-{args.stdp_epochs}) ===')
        train_stdp_conv(conv_layer, train_loader_1, T=args.T,
                        n_epochs=args.stdp_epochs, device=device,
                        start_epoch=next_stdp,
                        tb_log_dir=args.tb_log_dir)
        stdp_done = args.stdp_epochs

    # ================================================================
    # 特征提取（如果还没做）
    # ================================================================
    features = load_features()
    if features is None or next_mlp == 1:
        print('\n=== 提取特征 ===')
        train_feat, train_lbl = extract_features(conv_layer, full_train_loader,
                                                  T=args.T, device=device)
        test_feat, test_lbl = extract_features(conv_layer, test_loader,
                                                T=args.T, device=device)
        save_features(train_feat, train_lbl, test_feat, test_lbl)
        save_checkpoint('features_done', stdp_model=conv_layer,
                        stdp_epoch=stdp_done, features_done=True)
    else:
        print('\n=== 加载已保存的特征 ===')
        train_feat, train_lbl, test_feat, test_lbl = features

    print(f'训练特征: {train_feat.shape}, 测试特征: {test_feat.shape}')

    # 添加 undefined 合成样本
    train_feat, train_lbl = add_undefined_samples(
        train_feat, train_lbl, n_undefined=args.n_undefined)

    # ================================================================
    # 阶段2: MLP 训练
    # ================================================================
    if next_mlp <= args.mlp_epochs:
        print(f'\n=== 阶段2: MLP训练 (Epoch {next_mlp}-{args.mlp_epochs}, '
              f'{classifier.fc2.out_features}类含undefined) ===')

        # 如果之前有最佳模型，先恢复
        if next_mlp == 1 and os.path.exists(os.path.join(CKPT_DIR, 'best_model.pth')):
            classifier, best_acc = load_best_model(classifier, device)
            print(f'  加载历史最佳模型 (acc={best_acc:.4f})')

        checkpoint_cb = make_mlp_checkpoint_callback(conv_layer, mlp_done)

        classifier, history, best_acc = train_classifier(
            classifier, train_feat, train_lbl, test_feat, test_lbl,
            n_epochs=args.mlp_epochs, batch_size=args.batch_size,
            lr=args.mlp_lr, weight_decay=args.weight_decay, device=device,
            start_epoch=next_mlp, best_acc=best_acc,
            checkpoint_callback=checkpoint_cb
        )
    else:
        # MLP 已完成，加载最佳模型
        classifier, best_acc = load_best_model(classifier, device)
        print(f'\n=== MLP训练已完成 ===')
        print(f'最佳准确率: {best_acc:.4f}')

    # ================================================================
    # 最终评估
    # ================================================================
    print(f'\n{"="*60}')
    print(f'最终评估 (undefined阈值: {args.confidence_threshold})')

    # 标准准确率
    final_acc = compute_accuracy(classifier, test_feat, test_lbl, device)
    print(f'标准准确率 (10类): {final_acc:.4f}')

    # 带 undefined 阈值的推理
    preds, confs = predict_with_confidence(classifier, test_feat, device,
                                            threshold=args.confidence_threshold)
    undefined_count = (preds == UNDEFINED_CLASS).sum().item()
    correct_no_undef = ((preds == test_lbl) & (preds != UNDEFINED_CLASS)).sum().item()
    print(f'带阈值推理: 正确={correct_no_undef}, undefined={undefined_count}, '
          f'总={len(preds)}')

    best_info = f', 最佳={best_acc:.4f}' if best_acc > 0 else ''
    print(f'最佳测试准确率: {best_acc:.4f}' if best_acc > 0 else '')
    print(f'TensorBoard: tensorboard --logdir {args.tb_log_dir}')
    print(f'断点目录: {CKPT_DIR}/')
    print(f'{"="*60}')
    close_writer()

    # ================================================================
    # 混淆矩阵（含 undefined）
    # ================================================================
    os.makedirs(args.save_dir, exist_ok=True)
    n_classes = 11
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for p, t in zip(preds.numpy(), test_lbl.numpy()):
        cm[t, p] += 1

    if not args.no_plots:
        labels = [str(i) for i in range(10)] + ['undef']
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix (Acc: {final_acc:.3f}, Undef Thresh: {args.confidence_threshold})')
        fig.savefig(f'{args.save_dir}/confusion_matrix.png', dpi=150)
        print(f'混淆矩阵已保存: {args.save_dir}/confusion_matrix.png')
        plt.show()

    # ================================================================
    # undefined 演示
    # ================================================================
    if args.demo_undefined:
        print('\n=== Undefined 检测演示 ===')
        print('（对MNIST测试集样本随机加噪声，观察是否被识别为undefined）')
        rng = np.random.RandomState(args.seed)
        n_demo = 20
        demo_idx = rng.choice(len(test_feat), n_demo, replace=False)

        for i in demo_idx[:10]:
            feat = test_feat[i:i+1]
            # 加噪声破坏特征
            noise_levels = [0, 0.3, 0.6, 1.0]
            for nl in noise_levels:
                noisy = feat + nl * feat.std() * torch.randn_like(feat)
                pred, conf = predict_with_confidence(classifier, noisy.to(device),
                                                     device=device,
                                                     threshold=args.confidence_threshold)
                label = 'UNDEFINED' if pred[0] == UNDEFINED_CLASS else str(pred[0].item())
                print(f'  True={test_lbl[i].item()}, noise={nl:.1f}, '
                      f'pred={label}, conf={conf[0]:.3f}')


if __name__ == '__main__':
    main()
