"""
草稿.py — 训练 & 推理 完整流程调用链
=======================================
按函数调用顺序追踪，从主入口到最终预测/保存。
"""

# ============================================================
# 训练流程
# ============================================================
print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    训练全流程 (main.py 入口)                         ║
╚══════════════════════════════════════════════════════════════════════╝

main.py                                     model.py / train.py
──────                                      ──────────────────

main()
  │
  ├── [1] 加载数据 ──────────────────────────────────────────────────────
  │   │
  │   │   train_set = MNIST(root='./data', train=True)    ← torchvision
  │   │   test_set  = MNIST(root='./data', train=False)   ← torchvision
  │   │   train_loader = DataLoader(train_set, batch=1)
  │   │   test_loader  = DataLoader(test_set,  batch=256)
  │   │
  │   │   维度: X=[1,28,28], y=label(int)
  │   │
  │
  ├── [2] 创建模型 ──────────────────────────────────────────────────────
  │   │
  │   │   conv_layer = STDP_ConvLayer(        ──→ __init__()
  │   │       n_filters=32, T=10,                  │
  │   │       a_plus=0.004, lr=0.02)               ├── Conv2d(1,32,5,padding=2)
  │   │                                             │   权重: U(0, 0.3), bias=False
  │   │                                             │   参数量: 800
  │   │                                             │
  │   │                                             ├── IFNode(threshold=1.0, step_mode='s')
  │   │                                             │
  │   │                                             └── STDPLearner(
  │   │                                                    step_mode='s',
  │   │                                                    synapse=self.conv,
  │   │                                                    sn=self.if_node,
  │   │                                                    f_pre =λw: 0.004×w,
  │   │                                                    f_post=λw: 0.004×(1-w))
  │   │
  │   │   classifier = MLP_Classifier(         ──→ __init__()
  │   │       hidden=512, n_classes=11)             ├── fc1: 6272→512
  │   │                                             ├── Dropout(0.3)
  │   │                                             └── fc2: 512→11
  │   │
  │   │   # 断点恢复 (如果有)
  │   │   create_or_resume_model()                 → load_checkpoint()
  │   │                                             → 加载权重, 恢复epoch
  │   │
  │
  ├── [3] 阶段1: STDP 无监督训练 ─────────────────────────────────────
  │   │
  │   │   train_stdp_conv(                   ──→ train_stdp_conv()
  │   │       model=conv_layer,                    │
  │   │       train_loader, T=10, n_epochs=5)     │
  │   │                                           │
  │   │                                           │   for epoch in 1..5:
  │   │                                           │       for X, _ in train_loader:
  │   │                                           │
  │   │                                           │   ╔══════════ 单个样本处理 ══════════╗
  │   │                                           │   ║                                  ║
  │   │                                           │   ║ ① 泊松编码                       ║
  │   │                                           │   ║                                   ║
  │   │                                           │   ║ spikes = poisson_encode(X, T=10) ║
  │   │                                           │   ║   │                              ║
  │   │                                           │   ║   │  rand = torch.rand(10,1,28,28)║
  │   │                                           │   ║   │  spikes = rand < X.unsqueeze(0)║
  │   │                                           │   ║   │  [1,28,28] → [10,1,28,28]    ║
  │   │                                           │   ║   │  值=0/1, 平均=像素/255        ║
  │   │                                           │   ║   └──→ spikes                    ║
  │   │                                           │   ║                                  ║
  │   │                                           │   ║ ② 前向传播 + STDP                 ║
  │   │                                           │   ║                                   ║
  │   │                                           │   ║ out = model(spikes, train=True)  ║
  │   │                                           │   ║   └──→ forward()                ║
  │   │                                           │   ║        │                         ║
  │   │                                           │   ║        ├── if_node.reset()       ║
  │   │                                           │   ║        ├── stdp.reset()          ║
  │   │                                           │   ║        ├── stdp.enable()  ← 监视器║
  │   │                                           │   ║        │                         ║
  │   │                                           │   ║        │ for t in range(10):     ║
  │   │                                           │   ║        │   inp = spikes[t]       ║
  │   │                                           │   ║        │     [1,28,28]            ║
  │   │                                           │   ║        │                         ║
  │   │                                           │   ║        │   cur = self.conv(inp)  ║
  │   │                                           │   ║        │     Conv2d: 输入1通道     ║
  │   │                                           │   ║        │     32个5×5核滑动784次   ║
  │   │                                           │   ║        │     [1,28,28]→[1,32,28,28]║
  │   │                                           │   ║        │     [监视器记录inp]       ║
  │   │                                           │   ║        │                         ║
  │   │                                           │   ║        │   out = self.if_node(cur)║
  │   │                                           │   ║        │     IF: V+=cur           ║
  │   │                                           │   ║        │     if V>=1.0 → spike+reset║
  │   │                                           │   ║        │     [1,32,28,28] 脉冲     ║
  │   │                                           │   ║        │     [监视器记录out]       ║
  │   │                                           │   ║        │                         ║
  │   │                                           │   ║        │   out_list.append(out)   ║
  │   │                                           │   ║        │                         ║
  │   │                                           │   ║        ├── torch.stack(out_list) ║
  │   │                                           │   ║        │     [10,1,32,28,28]      ║
  │   │                                           │   ║        │                         ║
  │   │                                           │   ║        ├── # STDP 累积更新       ║
  │   │                                           │   ║        │   total_dw = None        ║
  │   │                                           │   ║        │   for _ in range(10):    ║
  │   │                                           │   ║        │     dw = stdp.step(       ║
  │   │                                           │   ║        │       on_grad=False,      ║
  │   │                                           │   ║        │       scale=0.02)         ║
  │   │                                           │   ║        │     ↑                    ║
  │   │                                           │   ║        │     │ 内部调:             ║
  │   │                                           │   ║        │     │ stdp_linear_single_  ║
  │   │                                           │   ║        │     │   step()            ║
  │   │                                           │   ║        │     │   trace_pre更新      ║
  │   │                                           │   ║        │     │   trace_post更新     ║
  │   │                                           │   ║        │     │   Δw_pre  = -f_pre×  ║
  │   │                                           │   ║        │     │            trace_post║
  │   │                                           │   ║        │     │            ×in_spike ║
  │   │                                           │   ║        │     │   Δw_post = f_post×  ║
  │   │                                           │   ║        │     │            trace_pre ║
  │   │                                           │   ║        │     │            ×out_spike║
  │   │                                           │   ║        │     │                     ║
  │   │                                           │   ║        │     total_dw += dw       ║
  │   │                                           │   ║        │   conv.weight += total_dw║
  │   │                                           │   ║        │     ↑ 权重被修改!         ║
  │   │                                           │   ║        │                         ║
  │   │                                           │   ║        ├── stdp.disable()        ║
  │   │                                           │   ║        └── return [10,1,32,28,28]║
  │   │                                           │   ║                                  ║
  │   │                                           │   ║ ③ 统计发放率                     ║
  │   │                                           │   ║                                   ║
  │   │                                           │   ║ total_spikes += out.sum()        ║
  │   │                                           │   ║ rate = total_spikes /             ║
  │   │                                           │   ║        (samples×32×28×28×10)     ║
  │   │                                           │   ╚══════════════════════════════════╝
  │   │                                           │
  │   │                                           │   # 一个 epoch 结束
  │   │                                           │   model.normalize_weights(5.0)
  │   │                                           │   │  w_flat = w.view(32, -1)
  │   │                                           │   │  w_norm = w_flat.norm(dim=1)
  │   │                                           │   │  w_flat *= 5.0 / w_norm
  │   │                                           │   │  ↑ L2范数统一=5.0
  │   │                                           │
  │   │                                           │   save_checkpoint('stdp', ...)
  │   │                                           │   │  torch.save({weights, epoch})
  │   │                                           │   │  → ./checkpoints/checkpoint.pth
  │   │                                           │
  │   │                                           │   → 下一个 epoch
  │   │
  │   │   输出: conv层的权重已被STDP更新 (无监督)
  │   │
  │
  ├── [4] 特征提取 ─────────────────────────────────────────────────────
  │   │
  │   │   train_feat, train_lbl =             ──→ extract_features()
  │   │       extract_features(                    │
  │   │           model=conv_layer,                │   for X, y in train_loader:
  │   │           loader=full_train_loader,        │
  │   │           T=10)                            │     spikes = poisson_encode(X, 10)
  │   │                                            │     │   [1,28,28] → [10,1,28,28]
  │   │                                            │     │
  │   │                                            │     out = model(spikes, train=False)
  │   │                                            │     │   → forward(train=False)
  │   │                                            │     │     for t in range(10):
  │   │                                            │     │       conv(spikes[t])
  │   │                                            │     │       if_node(conv_out)
  │   │                                            │     │     return [10,1,32,28,28]
  │   │                                            │     │     ↑ 不更新STDP!
  │   │                                            │     │
  │   │                                            │     spike_rate = out.mean(dim=0)
  │   │                                            │     │   [10,1,32,28,28] → [1,32,28,28]
  │   │                                            │     │   ↑ 每个值 = 10步中有几步发了脉冲
  │   │                                            │     │
  │   │                                            │     pooled = avg_pool2d(rate, 2, 2)
  │   │                                            │     │   [32,28,28] → [32,14,14]
  │   │                                            │     │
  │   │                                            │     feat = pooled.view(1, -1)
  │   │                                            │     │   [32,14,14] → [1, 6272]
  │   │                                            │     │
  │   │                                            │     收集 feat + label
  │   │                                            │
  │   │                                            └── return (所有特征), (所有标签)
  │   │
  │   │   维度: train_feat=[5000,6272], test_feat=[10000,6272]
  │   │
  │   │   save_features(train_feat, train_lbl,
  │   │                  test_feat, test_lbl)       → ./checkpoints/features_*.pt
  │   │
  │   │   save_checkpoint('features_done', ...)
  │   │
  │
  ├── [5] 合成 undefined 样本 ─────────────────────────────────────────
  │   │
  │   │   add_undefined_samples(
  │   │       train_feat, train_lbl, n=500)
  │   │   │
  │   │   │   随机选两个不同类的特征向量
  │   │   │   blended = α×feat[i] + (1-α)×feat[j]   ← 50:50 混合
  │   │   │   标签 = 10 (undefined)
  │   │   │
  │   │   │   追加 500 个合成样本到训练集
  │   │
  │   │   维度: train_feat=[5500,6272]  (原5000 + 500未定义)
  │   │
  │
  ├── [6] 阶段2: MLP 监督训练 ───────────────────────────────────────
  │   │
  │   │   classifier, history =             ──→ train_classifier()
  │   │       train_classifier(                  │
  │   │           classifier,                    │   for epoch in 1..50:
  │   │           train_feat, train_lbl,         │
  │   │           test_feat, test_lbl,           │       # Shuffle
  │   │           n_epochs=50,                   │       perm = randperm(N)
  │   │           batch_size=256,                │
  │   │           lr=0.001, wd=1e-4)             │       for batch in 0..N step 256:
  │   │                                          │
  │   │                                          │         # 前向
  │   │                                          │         out = classifier(x)
  │   │                                          │         │  fc1(x) → [256,512]
  │   │                                          │         │  ReLU()
  │   │                                          │         │  dropout(x)
  │   │                                          │         │  fc2(x) → [256,11]
  │   │                                          │         │  ↑ logits, 不包含softmax
  │   │                                          │
  │   │                                          │         # 损失 + 反向
  │   │                                          │         loss = CrossEntropy(out, y)
  │   │                                          │         │  = -log(softmax(out)[y])
  │   │                                          │         │
  │   │                                          │         loss.backward()
  │   │                                          │         │  ↑ 反向传播(MLP部分有梯度)
  │   │                                          │         │  ↑ conv层不参与!
  │   │                                          │
  │   │                                          │         optimizer.step()
  │   │                                          │         │  w -= lr × w.grad
  │   │                                          │         │  ↑ Adam更新MLP权重
  │   │                                          │
  │   │                                          │   # 每5个epoch评估
  │   │                                          │   if epoch%5 == 0:
  │   │                                          │
  │   │                                          │     train_acc = compute_acc(
  │   │                                          │         classifier, train_feat,
  │   │                                          │         train_lbl)
  │   │                                          │     │  ↓
  │   │                                          │     │  out = classifier(feat)
  │   │                                          │     │  pred = out.argmax(dim=1)
  │   │                                          │     │  acc = (pred==label).mean()
  │   │                                          │
  │   │                                          │     test_acc = compute_acc(
  │   │                                          │         classifier, test_feat,
  │   │                                          │         test_lbl)
  │   │                                          │
  │   │                                          │     # TensorBoard
  │   │                                          │     writer.add_scalar('MLP/Test', acc, epoch)
  │   │                                          │
  │   │                                          │     # 保存最佳模型
  │   │                                          │     if test_acc > best_acc:
  │   │                                          │         best_acc = test_acc
  │   │                                          │         torch.save({
  │   │                                          │           weights: classifier.state_dict(),
  │   │                                          │           acc: best_acc
  │   │                                          │         }, './checkpoints/best_model.pth')
  │   │                                          │         print('📌 最佳模型已保存')
  │   │                                          │
  │   │                                          │     # 保存断点
  │   │                                          │     save_checkpoint('mlp',
  │   │                                          │         mlp_classifier=classifier,
  │   │                                          │         mlp_epoch=epoch,
  │   │                                          │         best_test_acc=best_acc)
  │   │                                          │
  │   │                                          └── return (classifier, history, best_acc)
  │   │
  │   │   输出: 训练好的MLP, 最佳acc, 训练历史
  │   │
  │
  └── [7] 最终评估 ─────────────────────────────────────────────────────
      │
      │   final_acc = compute_accuracy(classifier, test_feat, test_lbl)
      │   preds, confs = predict_with_confidence(classifier, test_feat,
      │                                          threshold=0.3)
      │   │
      │   │   for each image:
      │   │     probs = softmax(classifier(features))
      │   │     max_prob, pred = probs.max()
      │   │     if max_prob < 0.3:
      │   │         pred = 10  ← undefined
      │   │     return pred
      │   │
      │
      │   混淆矩阵 → results/confusion_matrix.png
      │   TensorBoard → python -m tensorboard.main --logdir ./runs/
      │
      └── 训练完成 ✅


╔══════════════════════════════════════════════════════════════════════╗
║                    推理全流程 (单张图片)                             ║
╚══════════════════════════════════════════════════════════════════════╝

app.py / main.py                                model.py / train.py
───────────────                                 ──────────────────

  用户输入 (MNIST图片 / 手写照片 / 截图)
      │
      ▼
  [1] 预处理 (仅上传图片)
      │
      │   preprocess_upload_image(file)
      │   │  Image.open → 灰度 → 28×28
      │   │  判断背景色 → 反色(如需要) → 二值化
      │   │  输出: [1, 28, 28] tensor, 值域[0,1], 黑底白字
      │
      ▼
  [2] 泊松编码
      │
      │   spikes = poisson_encode(img, T=10)
      │   │  torch.rand(10,1,28,28) < img
      │   │  [1,28,28] → [10,1,28,28], 值=0/1
      │
      ▼
  [3] Conv + IF 前向 (不更新STDP)
      │
      │   out = conv_layer.forward(spikes, train_stdp=False)
      │   │
      │   │   conv_layer.if_node.reset()
      │   │   conv_layer.stdp.disable()           ← 不记录,不更新
      │   │
      │   │   for t in range(10):
      │   │     cur = conv_layer.conv(spikes[t])
      │   │       [1,1,28,28] → [1,32,28,28]      ← 32个5×5核
      │   │     out = conv_layer.if_node(cur)
      │   │       [1,32,28,28] 脉冲 0/1
      │   │     out_list.append(out)
      │   │
      │   │   return torch.stack(out_list)         ← [10,1,32,28,28]
      │
      ▼
  [4] 特征提取
      │
      │   spike_rate = out.mean(dim=0)
      │   │   [10,1,32,28,28] → [1,32,28,28]
      │   │   每个值 = 10步该位置发了多少脉冲
      │   │
      │   pooled = F.avg_pool2d(spike_rate, 2, 2)
      │   │   [1,32,28,28] → [1,32,14,14]
      │   │
      │   features = pooled.view(1, -1)
      │   │   [1,32,14,14] → [1, 6272]
      │
      ▼
  [5] MLP 分类
      │
      │   pred, conf = predict_with_confidence(
      │       classifier, features, threshold=0.3)
      │   │
      │   │   # classifier.forward():
      │   │   out = classifier.fc1(features)     ← [1,6272]×[6272,512] → [1,512]
      │   │   out = F.relu(out)                  ← max(0, x)
      │   │   out = classifier.dropout(out)      ← 推理时自动=1.0,不起作用
      │   │   out = classifier.fc2(out)          ← [1,512]×[512,11] → [1,11]
      │   │
      │   │   # predict_with_confidence():
      │   │   probs = F.softmax(out, dim=1)       ← [1,11] 概率分布
      │   │   max_prob, pred = probs.max(dim=1)
      │   │   if max_prob < 0.3:
      │   │       pred = 10                       ← undefined
      │   │   return (pred, max_prob)
      │
      ▼
  [6] 输出结果
      │
      ├── pred = 0~9  → "预测为数字 X"
      ├── pred = 10   → "UNDEFINED (不是数字)"
      └── conf ~ 预测置信度


╔══════════════════════════════════════════════════════════════════════╗
║              ［关键数据流维度变化　速查］                              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  训练时:                                                             ║
║    [1,28,28] ──泊松──→ [10,1,28,28] ──Conv──→ [10,32,28,28]        ║
║                        ↑ T=10           ↑ 32个5×5核                  ║
║                                                                      ║
║    [10,32,28,28] ──IF──→ [10,32,28,28] ──STDP──→ Δw→conv            ║
║                        ↑ 0/1脉冲         ↑ 改权重!                   ║
║                                                                      ║
║  推理时:                                                             ║
║    [10,32,28,28] ──mean──→ [32,28,28] ──pool──→ [32,14,14]          ║
║                            ↑时间平均      ↑ 2×2                       ║
║                                                                      ║
║    [32,14,14] ──flatten──→ [6272] ──fc1──→ [512] ──fc2──→ [11]      ║
║                             ↑ 特征        ↑ ReLU      ↑ logits       ║
║                                                                      ║
║    [11] ──softmax──→ 概率 ──threshold──→ 0~9 或 undefined(10)       ║
╚══════════════════════════════════════════════════════════════════════╝
""")
