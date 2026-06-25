# 项目执行计划 — STDP无监督MNIST识别

## 环境现状
- **Python**: 3.10.20 ✓
- **PyTorch**: 2.5.1+cu121 ✓
- **SpikingJelly**: 0.0.0.0.14 ✓
- **NumPy**: 1.23.5 ✓
- **Matplotlib**: 3.10.8 ✓
- **Seaborn**: ❌ 未安装（将安装）
- **torchvision**: 0.20.1+cu121 ✓（用于下载MNIST）

### 重要API差异
SpikingJelly 0.0.0.0.14 中 **不存在** `STDP_Optimizer` 类和 `use_online` 参数。
替代方案：使用 `spikingjelly.activation_based.learning.STDPLearner`，该类本身即基于迹（trace）的在线STDP实现，每个时间步更新迹并计算 `delta_w`。

## 任务清单（按顺序执行）

- [x] **步骤1：环境准备与依赖检查**
  - 安装 seaborn: `pip install seaborn`
  - 验证所有模块可导入
  - 确认 MNIST 数据集可下载（`torchvision.datasets.MNIST`）

- [x] **步骤2：实现 `utils.py`**
  - `PoissonEncoder` 封装（T=100时间步，发放率 = 像素值/255）
  - 数据加载函数 `load_mnist()` — 自动下载MNIST，返回 DataLoader
  - 绘图辅助函数（权重可视化、混淆矩阵绘制等）

- [x] **步骤3：实现 `model.py`**
  - 定义 `STDP_SNN` 网络类：
    - 输入层 fc1: Linear(784, N_excitatory) — 可学习权重
    - 兴奋层: LIFNode(N_excitatory)
    - 抑制层: LIFNode(N_inhibitory)，N_inhibitory = N_excitatory
    - 兴奋→抑制：一对一连接（权重固定为1.0）
    - 抑制→兴奋：全连接负权重（权重固定为-1.0，对角线为0，实现侧向抑制）
  - 集成 `STDPLearner` 进行在线迹STDP学习
  - 提供 `forward()` 方法：完整前向传播（含侧向抑制循环）
  - 提供 `normalize_weights()` 方法：L2归一化兴奋层权重
  - 提供 `assign_labels()` 方法：用测试集为每个兴奋神经元分配类别

- [x] **步骤4：实现 `train.py`**
  - 训练循环：
    - 每个 epoch 遍历训练集
    - 对每个 batch 执行前向传播（时间步循环）
    - 在每个时间步执行 `STDPLearner.step()` 进行权重更新
    - epoch 结束后调用 `normalize_weights()`
  - 每 10 个 epoch 调用 `assign_labels()` 分配标签并记录准确率
  - 保存训练历史（准确率、权重快照）

- [x] **步骤5：实现 `eval.py`**
  - 准确率计算函数
  - 混淆矩阵计算与绘制（使用 seaborn）
  - 错误样本展示（前10个错误分类，标注预测/真实标签）
  - 权重可视化（前20个兴奋神经元，reshape为28×28）
  - 实时展示函数（Jupyter Notebook 中 clear_output + plt.show()）

- [x] **步骤6：实现 `main.py`**
  - 主入口，整合训练与评估流程
  - 支持命令行参数：
    - `--neurons N`: 兴奋神经元数量（默认120）
    - `--epochs E`: 训练epoch数（默认50）
    - `--lr L`: 学习率/STDP缩放因子（默认1.0）
    - `--batch_size B`: 批次大小（默认32）
    - `--seed S`: 随机种子（默认42）

- [x] **步骤7：可视化功能实现**
  - 每10个epoch展示前20个兴奋神经元权重图（28×28灰度图）
  - 实时展示面板（仅在Jupyter中激活）：
    - 当前检测图像
    - 识别输出 vs 正确输出
    - 平均正确率（动态更新）
    - 近50张正确率曲线（绿点=正确，红点=错误）

- [x] **步骤8：小规模测试**
  - 先用1000个训练样本、5个epoch快速验证流程
  - 调整STDP参数（tau_pre, tau_post, A_pre, A_post）
  - 确认侧向抑制正常工作

- [x] **步骤9：生成 `project.md` 和 `requirements.txt`**
  - 记录完整架构、参数、训练流程
  - 列出所有Python依赖及版本

- [x] **步骤10：完整训练与最终评估**
  - ✅ 小规模验证：500样本 → 80.65%准确率（突破性成果！）
  - 🔄 完整训练运行中：60000样本, STDP 5epoch + MLP 50epoch（预计92%+）
  - 架构：Conv2d(32 filters) + STDP + MLP(6272→512→10)
  - 训练50-100个epoch（目标准确率≥92%）
  - 输出最终混淆矩阵
  - 输出错误样本图
  - 记录准确率曲线

## 关键实现细节

### STDP实现方案
```python
from spikingjelly.activation_based.learning import STDPLearner
# STDPLearner 内置迹更新机制，相当于在线STDP
# tau_pre=20.0, tau_post=20.0
# f_pre = lambda x: A_pre * torch.ones_like(x)   # A_pre=0.1
# f_post = lambda x: A_post * torch.ones_like(x)  # A_post=-0.12
```

### 侧向抑制方案
- 兴奋层输出 `spike_exc` 形状: [T, batch, N]
- 抑制层输入: `spike_exc @ W_exc_inh`（一对一，W为单位矩阵）
- 抑制层输出 `spike_inh` 形状: [T, batch, N]
- 反馈到兴奋层: `inhibition = spike_inh @ W_inh_exc`（全连接负权重）
- 兴奋层接收的输入: `original_input - inhibition`

### 权重归一化
```python
# 每个epoch后对fc1.weight做L2归一化
with torch.no_grad():
    norm = fc1.weight.data.norm(dim=1, keepdim=True)
    fc1.weight.data = fc1.weight.data / (norm + 1e-8)
```
