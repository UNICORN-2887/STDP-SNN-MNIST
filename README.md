# STDP-SNN MNIST 手写数字识别

> 无监督 STDP 脉冲神经网络 + MLP 分类器 | 最佳准确率 **94.75%**

---

## 🚀 快速运行（老师看这里）

```bash
# 1. 安装依赖（仅首次）
pip install -r requirements.txt

# 2. 启动 Web 验证平台
python app.py

# 3. 浏览器打开 → http://localhost:5000
```

> MNIST 数据集首次运行自动下载。预训练模型已打包在 `checkpoints/` 中，无需额外下载。

### 界面说明

| 区域 | 功能 |
|------|------|
| ▶ 开始 / ⏸ 暂停 | 空格键切换，实时流式验证 MNIST 测试集 |
| 张/次 | 控制每批处理的样本数（1~50） |
| 📊 实时统计 | 准确率、已测试、正确数、Undefined 计数 |
| 📈 近50次趋势 | 绿色=正确，红色=错误，黄色=Undefined |
| 🔬 特征图 | 32 个滤波器的发放率可视化（4×8 网格） |
| 📤 上传测试 | 拖拽图片到上传区，自动 28×28 预处理后推理 |
| ↺ 重置 | 清零统计，重新开始 |

### 训练新模型

```bash
# 快速训练（5000 样本，约 2 小时）
python main.py --train-samples 5000 --stdp-epochs 5 --mlp-epochs 50

# 完整训练（60000 样本，约 14 小时）
python main.py --stdp-epochs 5 --mlp-epochs 50

# 中断后恢复（断点续训，自动检测）
python main.py

# 实时监控
python -m tensorboard.main --logdir ./runs
# → http://localhost:6006
```

### 环境问题排查

```bash
# TensorBoard 报错 → 确保没装 tensorflow
pip uninstall tensorflow tensorflow-intel -y

# protobuf 冲突
pip install "protobuf>=3.19,<3.21"

# pkg_resources 缺失
pip install "setuptools<70"
```

---

## 技术栈

| 类别 | 技术 | 版本 | 用途 |
|------|------|------|------|
| 语言 | Python | 3.10 | — |
| 深度学习框架 | PyTorch | 2.5.1 (CUDA 12.1) | 张量计算、MLP 训练 |
| SNN 框架 | SpikingJelly | 0.0.0.0.14 | STDP 学习、IF 神经元 |
| 数据集 | torchvision (MNIST) | 0.20.1 | 自动下载 MNIST |
| Web 服务 | Flask | ≥2.0 | 实时验证平台 |
| 可视化 | TensorBoard | ≥2.10 | 训练曲线监控 |
| 绘图 | Matplotlib + Seaborn | — | 混淆矩阵、权重可视化 |
| 进度条 | tqdm | — | 训练进度显示 |
| 数值计算 | NumPy | — | 数据处理 |

---

## 一键安装

```bash
# 1. 创建 conda 环境
conda create -n snn python=3.10 -y
conda activate snn

# 2. 安装 PyTorch (CUDA 版本，CPU 则用 conda install pytorch cpuonly)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. 安装其余依赖
pip install spikingjelly matplotlib seaborn tqdm flask tensorboard

# 4. 克隆项目 (如果还没有)
git clone <repo-url>
cd MNIST1
```

> ⚠ 不要安装 `tensorflow` — 会与 PyTorch 的 TensorBoard 冲突。

---

## 快速开始

```bash
# 训练 (5000 样本，约 2 小时)
python main.py --train-samples 5000 --stdp-epochs 5 --mlp-epochs 50

# 完整训练 (60000 样本，约 14 小时)
python main.py --stdp-epochs 5 --mlp-epochs 50

# 恢复训练 (断点续训，自动检测)
python main.py

# Web 验证平台
python app.py
# → http://localhost:5000

# TensorBoard 监控
python -m tensorboard.main --logdir ./runs
# → http://localhost:6006
```

---

## 文件结构

```
MNIST1/
│
├── main.py                     # 主入口：训练流程 + 断点管理 + 评估
├── model.py                    # 模型定义：STDP_ConvLayer + MLP_Classifier
├── train.py                    # 训练函数：STDP + MLP + 断点 + TensorBoard
├── app.py                      # Flask Web 验证平台 (实时推理 + 上传)
├── templates/index.html        # Web 前端界面
│
├── model_unsupervised.py       # [实验] 纯无监督投票版 (不推荐使用)
│
├── 草稿.py                     # [笔记] 代码演示草稿：STDP原理、LIF/IF对比等
│
├── README.md                   # 本文件
├── project.md                  # 项目说明 (架构、参数、实验结果)
├── SPEC.md                     # 详细规格说明书 (训练流程、API文档)
├── ARCHITECTURE.md             # 架构全景图 + 数据流维度变化
├── PPT_OUTLINE.md              # PPT 大纲 (14页，面向老师汇报)
├── plan.md                     # [历史] 早期执行计划
├── CROSS_TERMINAL_PROMPT.md    # [工具] BindsNET 对比实验提示词
├── requirements.txt            # Python 依赖列表
│
├── references/                 # 参考文献目录
│   ├── related_projects_and_papers.md   # 相关项目与论文汇总
│   └── spikingjelly_guide.md           # SpikingJelly API 使用指南
│
├── memory/                     # 设计想法记录
│   └── uniform-neuron-assignment-idea.md
│
├── bindsnet_results/           # BindsNET 对比实验结果
│   ├── results.md              # 实验报告
│   ├── accuracy_curve.png      # 准确率曲线
│   ├── confusion_matrix.png    # 混淆矩阵
│   ├── weights.png             # 权重可视化
│   └── spike_rate.png          # 发放率曲线
│
├── checkpoints/                # [自动生成] 训练断点 + 最佳模型
├── runs/                       # [自动生成] TensorBoard 日志
├── results/                    # [自动生成] 评估结果 (混淆矩阵)
└── data/                       # [自动下载] MNIST 数据集
```

### 核心文件说明

| 文件 | 核心类/函数 | 功能 |
|------|-----------|------|
| `model.py` | `STDP_ConvLayer` | 卷积+IF+STDP，`forward()` 训练/推理，`extract_features()` 特征提取 |
| | `MLP_Classifier` | 两层 MLP (6272→512→11)，含 undefined 类 |
| | `poisson_encode()` | 泊松编码 [1,28,28] → [T,1,28,28] |
| `train.py` | `train_stdp_conv()` | 阶段 1：无监督 STDP 训练 |
| | `extract_features()` | 特征提取：发放率 + 池化 → 6272 维 |
| | `train_classifier()` | 阶段 2：监督 MLP 训练 |
| | `predict_with_confidence()` | 推理：含 undefined 阈值判断 |
| | `save_checkpoint()` / `load_checkpoint()` | 断点管理 |
| | `save_best_model()` / `load_best_model()` | 最佳模型保存 |
| `main.py` | `main()` | 训练主流程 + CLI 参数 |
| | `create_or_resume_model()` | 断点续训逻辑 |
| | `add_undefined_samples()` | 合成 undefined 训练样本 |
| `app.py` | Flask app | Web 验证：实时推理流 + 上传 + 特征图可视化 |

---

## 参数速查

```bash
python main.py \
    --train-samples 5000    # 训练样本数 (None=全量60000)
    --stdp-epochs 5         # STDP 训练轮数
    --mlp-epochs 50         # MLP 训练轮数
    --n-filters 32          # 卷积滤波器数
    --T 10                  # 泊松编码时间步
    --stdp-lr 0.02          # STDP 学习率
    --a-plus 0.004          # STDP 幅度
    --hidden-dim 512        # MLP 隐藏层维度
    --mlp-lr 0.001          # MLP 学习率
    --dropout 0.3           # MLP Dropout
    --confidence-threshold 0.3  # undefined 阈值
    --seed 42               # 随机种子
    --force-restart         # 清除断点从头训练
    --tb-log-dir ./runs     # TensorBoard 日志目录
```

---

## 实验结果

| 训练样本 | 准确率 |
|---------|--------|
| 500 | 80.65% |
| 3,000 | 89.81% |
| 5,000 | 92.03% |
| **60,000** | **94.75%** |
| BindsNET (对比) | 93.12% |
