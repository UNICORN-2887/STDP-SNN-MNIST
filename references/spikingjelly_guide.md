# SpikingJelly STDP 使用指南

> 基于官方文档和源码分析，记录关键API用法和常见陷阱

## 一、核心API

### 1. STDPLearner（在线迹STDP）

```python
from spikingjelly.activation_based.learning import STDPLearner
from spikingjelly.activation_based.neuron import IFNode, LIFNode

# 创建STDP学习器
stdp = STDPLearner(
    step_mode='s',        # 's'=单步(每时间步更新), 'm'=多步(累积后一次更新)
    synapse=conv,         # nn.Conv2d 或 nn.Linear
    sn=if_node,           # IFNode 或 LIFNode
    tau_pre=20.0,         # 突触前迹时间常数
    tau_post=20.0,        # 突触后迹时间常数
    f_pre=lambda w: A_pre * w,         # LTD权重函数
    f_post=lambda w: A_post * (1-w),   # LTP权重函数
)
```

### 2. STDP更新公式

```
delta_w_pre  = -f_pre(w)  * trace_post * in_spike   (LTD: post→pre)
delta_w_post =  f_post(w) * trace_pre  * out_spike   (LTP: pre→post)
```

### 3. 权重依赖 vs 加法STDP

```python
# 加法STDP（我们初版用的，导致权重发散）
f_pre  = lambda w: A_pre * torch.ones_like(w)    # 常数
f_post = lambda w: A_post * torch.ones_like(w)   # 常数

# 权重依赖STDP（同学方案用的，自稳定）
f_pre  = lambda w: A_pre * w                      # ∝权重
f_post = lambda w: A_post * (1.0 - w)             # ∝(1-权重)
```

**关键发现**：权重依赖STDP是收敛的关键。加法STDP会导致所有权重同向漂移。

### 4. 监视器机制

```python
# STDPLearner内部使用InputMonitor和OutputMonitor
# InputMonitor记录synapse的输入（in_spike）
# OutputMonitor记录neuron的输出（out_spike）

# 训练循环
stdp.reset()      # 清空迹和记录
stdp.enable()     # 启用监视
# ... 前向传播 ...
stdp.step(on_grad=False, scale=lr)  # 处理记录，返回delta_w
stdp.disable()    # 禁用监视
```

### 5. step_mode陷阱（重要！）

```python
# ❌ 错误：LIF/IF用step_mode='m'但在时间循环中单步调用
lif = LIFNode(step_mode='m')  # v形状会变成[N]而不是[batch,N]
lif(x_single_step)            # BUG!

# ✅ 正确：时间循环中用step_mode='s'
lif = LIFNode(step_mode='s')
for t in range(T):
    out = lif(x[t])
```

## 二、已知Issue和解决方案

### Issue #345: STDP文档示例可能有误
- 链接：https://github.com/fangwei123456/spikingjelly/issues/345
- 问题：官方文档中STDP+梯度下降的示例代码可能不正确
- 影响：说明SpikingJelly的STDP API仍在迭代中

### 我们的实际经验

| 问题 | 原因 | 解决 |
|------|------|------|
| 发放率为0 | LIF step_mode='m'在单步调用时v形状错误 | 改用 step_mode='s' |
| 权重爆炸 | 加法STDP + 高学习率 | 改用权重依赖STDP |
| 发放率过高 | 归一化后权重范数=1→驱动不足→过度补偿 | target_norm=3~5 |
| 全连接不收敛 | 94K参数+稀疏STDP信号 | 改用卷积(800参数) |
| 纯无监督投票失败 | WTA正反馈→所有神经元归一类 | 改用MLP分类器 |

## 三、最佳实践（基于同学方案+论文）

```python
# 1. 网络结构：Conv + IF (不用LIF)
conv = nn.Conv2d(1, 32, 5, padding=2, bias=False)
nn.init.uniform_(conv.weight, 0.0, 0.3)
if_node = IFNode(v_threshold=1.0, step_mode='s')

# 2. STDP: 权重依赖 + 低a_plus
stdp = STDPLearner(
    step_mode='s', synapse=conv, sn=if_node,
    tau_pre=20.0, tau_post=20.0,
    f_pre=lambda w: 0.004 * w,
    f_post=lambda w: 0.004 * (1.0 - w),
)

# 3. 训练: T=10, 逐样本
lr = 0.02  # 比我们之前用的0.0001高200倍！
for epoch in range(5):
    for img, _ in loader:
        spikes = poisson_encode(img, T=10)
        out = conv_forward(spikes, train_stdp=True)
    normalize_weights(target_norm=5.0)  # 关键！不是1.0

# 4. 特征提取 + MLP分类
features = avg_pool(spike_rates, 2)  # 6272维
mlp = MLP(6272, 512, 10)
train_mlp(features, labels)  # 监督
```

## 四、SpikingJelly版本差异

| 功能 | 0.0.0.0.14 (当前) | 规范中的假设 |
|------|-------------------|-------------|
| STDP_Optimizer | ❌ 不存在 | ✅ 假设存在 |
| use_online参数 | ❌ STDPLearner替代 | ✅ 假设存在 |
| learning模块位置 | `activation_based.learning` | 规范写 `clock_driven.learning` |
