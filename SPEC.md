# 项目规格说明书 — 无监督 STDP MNIST 识别

> 最后更新：2026-06-18

---

## 一、任务目标

实现一个脉冲神经网络（SNN），使用**无监督 STDP** 训练卷积层提取特征，配合**监督 MLP 分类器**完成 MNIST 手写数字识别。训练后准确率目标 **≥ 92%**。

---

## 二、架构设计

```
输入图像 (28×28) → 泊松编码 (T=10) → Conv2d(1→32, 5×5, padding=2)
    → IF 神经元 (阈值=1.0) → 平均池化 (2×2) → 6272 维特征向量
    → MLP(6272 → 512 → 11) → 分类输出 (0-9 + undefined)
```

### 2.1 卷积层（STDP 训练）

| 参数 | 值 | 说明 |
|------|-----|------|
| 输入通道 | 1 | 灰度图 |
| 滤波器数量 | 32 | |
| 卷积核大小 | 5×5，padding=2 | 保持 28×28 尺寸 |
| 偏置 | 无 | bias=False |
| 权重初始化 | U(0, 0.3) | 非负初始化 |
| 神经元 | IFNode，阈值 1.0，step_mode='s' | 单步模式（关键！m 模式有 bug） |
| 池化 | AvgPool2d(2×2, stride=2) | 32×28×28 → 32×14×14 = 6272 |

### 2.2 MLP 分类器（监督训练）

| 层 | 输入 | 输出 | 激活 | Dropout |
|-----|------|------|------|---------|
| fc1 | 6272 | 512 | ReLU | 0.3 |
| fc2 | 512 | 11 | — | — |

> 11 类 = 数字 0-9 + **类 10 = undefined**（用于拒识非数字输入）

### 2.3 编码参数

| 参数 | 值 |
|------|-----|
| 编码方式 | 泊松编码（发放率 = 像素值 / 255） |
| 时间步数 T | 10 |
| 每时间步 | 独立以概率 p 发放脉冲 |

---

## 三、学习规则

### 3.1 STDP（权重依赖 / 乘性 STDP）

使用 SpikingJelly 的 `STDPLearner`（`activation_based.learning`）：

```python
f_pre(w)  = a_plus * w           # LTD ∝ 权重，自然下界
f_post(w) = a_plus * (1.0 - w)   # LTP ∝ (1-权重)，自然上界
```

| 参数 | 值 | 说明 |
|------|-----|------|
| a_plus | 0.004 | STDP 幅度 |
| tau_pre | 20.0 | 突触前迹时间常数 |
| tau_post | 20.0 | 突触后迹时间常数 |
| lr | 0.02 | 学习率缩放因子 |
| step_mode | 's' | 单步在线迹更新 |
| 更新时机 | T 步累积后一次性应用 | 不在每步更新，避免权重振荡 |

### 3.2 权重归一化

每个 STDP epoch 后，每个滤波器的权重向量缩放到 L2 范数 = **5.0**：

```python
w_flat = w.view(n_filters, -1)
w_norm = w_flat.norm(p=2, dim=1, keepdim=True)
w_flat *= 5.0 / (w_norm + 1e-8)
```

### 3.3 MLP 分类器（Adam 优化）

| 参数 | 值 |
|------|-----|
| 优化器 | Adam |
| 学习率 | 0.001 |
| weight_decay | 1e-4 |
| batch_size | 256 |
| epochs | 50 |
| 损失函数 | CrossEntropyLoss |

---

## 四、训练流程

### 4.1 阶段 1：STDP 无监督训练（5 epochs）

```
for epoch in 1..5:
    for each image in train_set (batch_size=1):
        spikes = poisson_encode(image, T=10)
        out = STDP_Forward(spikes, train=True)
        # T步累积，最后一次性应用权重更新
    normalize_weights(target_norm=5.0)
    save_checkpoint()
```

### 4.2 特征提取

STDP 训练完成后，一次遍历训练集和测试集，提取池化后的发放率作为特征向量（6272 维）。

> 特征数据保存到 `checkpoints/features_*.pt`，后续多次训练 MLP 无需重复提取。

### 4.3 合成 undefined 样本

随机混合两个不同数字类别的特征向量（50:50 比例），生成 N 个 undefined 训练样本。这教会分类器：当特征混合不确定时，输出 undefined。

### 4.4 阶段 2：MLP 监督训练（50 epochs）

```
for epoch in 1..50:
    前向：6272 维特征 → MLP → 11 类 logits
    损失：CrossEntropyLoss
    每 5 轮评估：计算 train/test 准确率
    保存最佳模型 (按 test_acc)
    保存断点
```

---

## 五、容错与监控

### 5.1 断点续训

训练在任何阶段中断后，再次运行 `python main.py` 自动恢复：

| 中断时机 | 恢复行为 |
|---------|---------|
| STDP 训练中 | 加载 STDP 权重，从下一 epoch 继续 |
| 特征提取中 | 重新提取（特征提取不可续） |
| MLP 训练中 | 加载 MLP 权重 + optimizer 状态，从下一 epoch 继续 |

**断点文件结构：**

```
checkpoints/
├── checkpoint.pth        # 训练状态（stage, epoch, 权重）
├── features_train.pt     # 训练集特征
├── features_test.pt      # 测试集特征
├── best_model.pth        # 最佳 MLP 模型
└── config.json           # 配置记录
```

**手动控制：**

```bash
python main.py                    # 自动检测断点，有则恢复
python main.py --force-restart    # 清除断点，从头开始
python main.py --resume-only      # 仅恢复，无断点则退出
```

### 5.2 TensorBoard 监控

```bash
python -m tensorboard.main --logdir ./runs/mnist
# 浏览器: http://localhost:6006
```

**监控指标：**

| 阶段 | 指标 | 更新频率 |
|------|------|---------|
| STDP | Spike_Rate（发放率） | 每 epoch |
| STDP | Epoch_Time（耗时） | 每 epoch |
| MLP | Loss | 每 epoch |
| MLP | Train_Acc | 每 5 epochs |
| MLP | Test_Acc | 每 5 epochs |

---

## 六、推理接口

### 6.1 标准分类

```python
preds, confidences = predict_with_confidence(
    classifier, features, threshold=0.3
)
# preds: 0-9 或 10 (undefined)
# 置信度 < 0.3 的预测强制标记为 undefined
```

### 6.2 Undefined 检测

当输入不属于 0-9（如字母 A、噪声），模型输出类 10（undefined），而非强制分类为一个数字。

---

## 七、使用方法

```bash
# 完整训练（5000 样本推荐，约 2 小时）
python main.py --train-samples 5000 --stdp-epochs 5 --mlp-epochs 50

# 快速验证（500 样本，约 10 分钟）
python main.py --train-samples 500 --stdp-epochs 3 --mlp-epochs 20

# 使用全量数据（60000 样本，约 15 小时）
python main.py --stdp-epochs 5 --mlp-epochs 50

# 启动 TensorBoard
python -m tensorboard.main --logdir ./runs/mnist
```

---

## 八、项目结构

```
MNIST1/
├── main.py                 # 主入口（训练 + 评估 + 断点管理）
├── model.py                # 模型定义（STDP_ConvLayer + MLP_Classifier）
├── train.py                # 训练函数（STDP + MLP + 断点 + TB）
├── model_unsupervised.py   # 纯无监督版本（实验用，不推荐）
├── utils.py                # 旧版工具（全连接架构，已废弃）
├── eval.py                 # 旧版评估（全连接架构，已废弃）
├── SPEC.md                 # 本文件 — 项目规格说明书
├── plan.md                 # 项目执行计划（历史记录）
├── project.md              # 项目早期说明（历史记录）
├── requirements.txt        # 依赖列表
├── references/
│   ├── related_projects_and_papers.md  # 相关项目与论文汇总
│   └── spikingjelly_guide.md          # SpikingJelly API 使用指南
├── checkpoints/            # 断点文件（自动生成）
├── runs/                   # TensorBoard 日志（自动生成）
├── results/                # 评估结果输出
└── data/                   # MNIST 数据集（自动下载）
```

---

## 九、已知限制与注意事项

| 项目 | 说明 |
|------|------|
| STDP_Optimizer | SpikingJelly 0.0.0.0.14 中**不存在**，用 `STDPLearner` 替代 |
| step_mode='m' | LIF/IF 单步调用时有 bug，必须用 `step_mode='s'` |
| 纯无监督投票 | 在 FC 和 Conv 架构上均失败（<10%），不推荐 |
| TensorFlow | 项目用 PyTorch，不要安装 tensorflow（会冲突） |
| protobuf | 必须 < 3.21（否则 tensorboard 报 `_ARRAY_API not found`） |
| NumPy | 项目中不需要 tensorflow，NumPy 2.x 无影响 |
| ESP32 部署 | 当前 MLP(6272→512→11) 约 12MB，需缩减到 32KB 以下才能部署 |

---

## 十、实验结果记录

| 日期 | 架构 | 样本数 | 准确率 | 备注 |
|------|------|--------|--------|------|
| 06-17 | FC + WTA（纯无监督） | 2000 | <10% | 失败 |
| 06-17 | Conv + 纯无监督投票 | 3000 | 9.80% | 失败 |
| 06-17 | Conv + STDP + MLP | 500 | 80.65% | ✅ |
| 06-17 | Conv + STDP + MLP | 3000 | 89.81% | ✅ |
| 06-18 | Conv + STDP + MLP + 断点 + TB + undefined | 5000 | 🔄 训练中 | — |
