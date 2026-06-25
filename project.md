# 无监督 STDP MNIST 识别 — 项目说明

> 最后更新：2026-06-18 | 当前最佳准确率：**90.03%** (5000 样本)

---

## 项目概述

实现脉冲神经网络（SNN），使用**无监督 STDP** 训练卷积层提取视觉特征，配合**监督 MLP** 完成 MNIST 手写数字识别（11 类，含 undefined）。

### 方案演进

| 阶段 | 架构 | 分类方式 | 准确率 | 结论 |
|------|------|---------|--------|------|
| V1 | FC 784→120 + LIF + 侧向抑制 | 纯无监督神经元投票 | <10% | ❌ 全连接+WTA 失败 |
| V2 | Conv 5×5×32 + IF | 纯无监督滤波器投票 | 9.80% | ❌ 无分类器不可行 |
| **V3** | **Conv 5×5×32 + IF + MLP** | **STDP特征 + MLP分类** | **92.03%** | ✅ 当前方案 |

> V3 中 STDP 部分仍然是**无监督**的——卷积核不看标签，纯靠脉冲时序规则自学边缘/形状特征。MLP 只是"读取器"。

---

## 网络架构

```
输入 (28×28) → 泊松编码 (T=10) → Conv2d(1→32, 5×5, pad=2)
  → IFNeuron (v_thr=1.0) → AvgPool(2×2) → 6272 维特征
  → MLP(6272 → 512 → 11) → 0-9 或 undefined(10)
```

### 卷积层

| 参数 | 值 |
|------|-----|
| 滤波器 | 32 个，5×5，padding=2 |
| 神经元 | IFNode，阈值 1.0，step_mode='s' |
| 权重初始化 | U(0, 0.3) |
| 池化 | AvgPool 2×2 → 14×14×32 = 6272 |

### MLP

| 层 | 规格 |
|-----|------|
| fc1 | 6272 → 512, ReLU, Dropout 0.3 |
| fc2 | 512 → 11（0-9 + undefined） |

---

## STDP 学习规则

使用 SpikingJelly `STDPLearner`（`activation_based.learning`），权重依赖（乘性）STDP：

```python
f_pre(w)  = a_plus * w          # LTD ∝ 权重
f_post(w) = a_plus * (1.0 - w)  # LTP ∝ (1-权重)
```

| 参数 | 值 |
|------|-----|
| a_plus | 0.004 |
| tau_pre / tau_post | 20.0 |
| lr | 0.02 |
| T（时间步） | 10 |
| STDP epochs | 5 |
| 权重归一化 | L2 范数 = 5.0（每 epoch 后） |
| 更新方式 | T 步累积后一次性应用 |

---

## 训练流程

### 阶段 1：STDP 无监督训练（5 epochs）

```
for each image (batch_size=1):
    poisson_encode → T=10 步脉冲
    forward through Conv+IF (STDP 监视器记录)
    累积 T 步 delta_w → 一次性应用
normalize_weights(target_norm=5.0)
save_checkpoint()
```

### 阶段 2：MLP 监督训练（50 epochs）

```
特征提取: Conv+IF+Pool → 6272 维发放率向量
合成 undefined 样本: 随机混合两个不同类特征
Adam(0.001) + CrossEntropyLoss, batch=256
每 5 epoch 评估 → 保存最佳模型
```

---

## 容错与监控

### 断点续训

```
checkpoints/
├── checkpoint.pth        # stage, epoch, 权重
├── features_train.pt     # 训练特征
├── features_test.pt      # 测试特征
├── best_model.pth        # 最佳 MLP
└── config.json
```

```bash
python main.py                    # 自动断点恢复
python main.py --force-restart    # 清除重来
```

### TensorBoard

```bash
python -m tensorboard.main --logdir ./runs/mnist
# http://localhost:6006
```

---

## Web 验证平台

```bash
python app.py
# http://localhost:5000
```

功能：实时验证图片流、32 滤波器特征图可视化、动态准确率、近 50 次趋势、图片上传推理。

> ⚠ 上传图片识别效果有限：模型在 MNIST 标准格式（28×28 黑底白字居中）上训练，手写照片需严格预处理（二值化+反色+居中）才能达到与测试集相当的准确率。

---

## 目录结构

```
MNIST1/
├── main.py                 # 主入口（训练 + 断点管理）
├── model.py                # STDP_ConvLayer + MLP_Classifier
├── train.py                # 训练函数 + TB 日志 + 断点管理
├── app.py                  # Flask Web 验证平台
├── templates/index.html    # Web 前端
├── model_unsupervised.py   # 纯无监督实验版（已弃用）
├── utils.py / eval.py      # V1 全连接架构工具（已弃用）
├── SPEC.md                 # 最新规格说明书
├── project.md              # 本文件
├── plan.md                 # 早期执行计划（历史）
├── requirements.txt
├── references/
│   ├── related_projects_and_papers.md
│   └── spikingjelly_guide.md
├── memory/                 # 设计想法记录
├── checkpoints/            # 断点文件
├── runs/                   # TensorBoard 日志
├── results/                # 评估输出
└── data/                   # MNIST（自动下载）
```

---

## 使用方法

```bash
# 训练（推荐 5000 样本，约 2 小时 GPU）
python main.py --train-samples 5000 --stdp-epochs 5 --mlp-epochs 50

# 快速验证
python main.py --train-samples 500 --stdp-epochs 3 --mlp-epochs 20

# Web 验证
python app.py

# TensorBoard
python -m tensorboard.main --logdir ./runs/mnist
```

---

## 实验结果

| 样本数 | 准确率 | 备注 |
|--------|--------|------|
| 500 | 80.65% | 快速验证 |
| 3,000 | 89.81% | 接近论文基线 |
| **5,000** | **90.03%** | 当前最佳 ✅ |

---

## 技术栈

Python 3.10 · PyTorch 2.5.1 (CUDA 12.1) · SpikingJelly 0.0.0.0.14 · torchvision 0.20.1 · Flask · TensorBoard · NumPy · Matplotlib · Seaborn · tqdm

## 已知限制

| 项目 | 说明 |
|------|------|
| STDP_Optimizer | SpikingJelly 中不存在，用 `STDPLearner` |
| step_mode='m' | 单步调用有 bug，必须 `'s'` |
| 纯无监督投票 | FC/Conv 架构均失败 |
| 上传图片识别 | 需严格黑底白字 28×28 居中预处理 |
| TensorFlow | 不要安装（与 torch TB 冲突） |
| ESP32 部署 | 当前 12MB，需压缩至 32KB |
