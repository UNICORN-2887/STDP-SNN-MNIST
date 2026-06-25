# 相关项目与论文汇总

> 搜索范围：GitHub、Google Scholar、SpikingJelly官方文档
> 搜索日期：2026-06-18

---

## 一、GitHub 相关项目

### 1. 直接可复现的项目

| 项目 | Stars | 框架 | 特点 |
|------|-------|------|------|
| [cowolff/Simple-Spiking-Neural-Network-STDP](https://github.com/cowolff/Simple-Spiking-Neural-Network-STDP) | 28 | 纯Python | 从零实现SNN+STDP训练MNIST，含论文、预训练权重、可视化notebook |
| [BindsNET/bindsnet](https://github.com/BindsNET/bindsnet) | 1.4k+ | PyTorch | 最成熟的SNN框架，内置Diehl&Cook(2015)网络，支持多种神经元模型 |
| [peteru-diehl/stdp-mnist](https://github.com/peteru-diehl/stdp-mnist) | 100+ | Brian2 | Diehl&Cook原版实现，使用Brian模拟器 |
| [RTC-research-group/stdp-nmnist](https://github.com/RTC-research-group/stdp-nmnist) | 21 | PyTorch+BindsNET | 硕士论文项目，STDP+N-MNIST，含侧向抑制和WTA |

### 2. SpikingJelly 官方资源

| 资源 | 链接 |
|------|------|
| STDP官方教程 | https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based/stdp.html |
| STDP教程(英文) | https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/en/stdp.html |
| STDP源码文档 | https://github.com/fangwei123456/spikingjelly/blob/0.0.0.0.14/docs/source/activation_based_en/stdp.rst |
| 学习模块源码 | `spikingjelly.activation_based.learning` （STDPLearner、MSTDPLearner等） |
| Issue #345 | STDP+梯度下降文档代码可能有误的讨论 |

### 3. 高度相关项目（不同框架）

| 项目 | 框架 | 特点 |
|------|------|------|
| [jcmharry/stdp-mnist-brian2](https://github.com/jcmharry/stdp-mnist-brian2) | Brian2 | Diehl&Cook的Brian2版本 |
| [Shikhargupta/conv-stdp-snn](https://github.com/topics/stdp?l=python) | PyTorch | 卷积STDP SNN |
| [haoyz/sym-STDP-SNN](https://github.com/haoyz/sym-STDP-SNN) | PyTorch | 对称STDP规则的监督学习 |
| [AllenYolk/dendritic-spiking-neuron](https://github.com/AllenYolk/dendritic-spiking-neuron) | SpikingJelly | 基于SpikingJelly的树突脉冲神经元 |

---

## 二、学术论文

### 核心论文

#### Diehl & Cook (2015) — 领域奠基论文 ⭐⭐⭐⭐⭐
```
标题: Unsupervised learning of digit recognition using spike-timing-dependent plasticity
作者: Peter U. Diehl & Matthew Cook (ETH Zürich)
期刊: Frontiers in Computational Neuroscience, 9:99, 2015
DOI: 10.3389/fncom.2015.00099
```
- **架构**：输入层(784) → 兴奋层(N) + 抑制层(N)，LIF神经元，侧向抑制
- **学习**：纯无监督STDP，无标签、无误差信号
- **分类**：训练后神经元分配到最高响应的类别
- **准确率**：100神经元→82.9%，1600→91.9%，6400→**95.0%**
- **与我们项目的关联**：这是本项目最初架构的直接参考

#### Mozafari et al. (2019) — 深度卷积STDP ⭐⭐⭐⭐
```
标题: Bio-inspired digit recognition using reward-modulated STDP in deep convolutional networks
期刊: Pattern Recognition, 94, 2019
```
- 深度卷积SNN + 延时编码（每个神经元最多发放一次）
- 低层用STDP，高层用**奖励调制STDP (R-STDP)**
- 无需外部分类器，由最终层最早脉冲决定
- 准确率：**97.2%**（纯无监督+无分类器）

#### Lee et al. (2019) — SpiCNN ⭐⭐⭐⭐
```
标题: Deep Spiking Convolutional Neural Network Trained With Unsupervised STDP
期刊: IEEE Trans. Cognitive and Developmental Systems, 2019
```
- 两个3×3卷积层LIF神经元
- 逐层无监督卷积STDP + mini-batch权重更新
- 纯无监督：**91.1%**；STDP预训练+监督微调：更高
- **与我们同学方案的相似度最高**

#### Kheradpisheh et al. (2018) ⭐⭐⭐⭐
```
标题: STDP-based spiking deep convolutional neural networks for object recognition
期刊: Neural Networks, 99, 2018
```
- 深度卷积SNN (30C5-2P-100C5-2P) + 时间编码
- 逐层无监督STDP + **SVM分类器**
- 准确率：**98.4%**
- **与我们当前方案最接近**（STDP特征提取+SVM/MLP分类）

### 论文准确率总览

| 论文 | 年份 | 方法 | 准确率 |
|------|------|------|--------|
| Ensemble SNN | 2021 | STDP + 集成投票 + 迁移学习 | **99.27%** |
| Tavanaei et al. | 2018 | STDP + SVM | **98.61%** |
| ReStoCNet | 2019 | 概率混合STDP + 二值核 | **98.54%** |
| Ferré et al. | 2018 | STDP + WTA + 监督FC | **98.49%** |
| Kheradpisheh et al. | 2018 | 卷积STDP + SVM | **98.40%** |
| Mozafari et al. | 2019 | STDP + R-STDP | **97.20%** |
| Diehl & Cook | 2015 | 纯无监督FC+STDP | **95.00%** |
| Lee et al. (SpiCNN) | 2019 | 无监督卷积STDP | **91.10%** |
| Tang et al. | 2020 | ROC + STDP (硬件优化) | **90.20%** |

---

## 三、与本项目的对照分析

### 本项目当前状态

| 阶段 | 结果 |
|------|------|
| 纯无监督FC + WTA（初版） | <10%，失败 |
| 纯无监督Conv + 投票 | 9.80%，失败 |
| STDP Conv + 监督MLP（3000样本） | **89.81%** ✅ |
| STDP Conv + 监督MLP（60000样本） | 🔄 训练中 |

### 我们的位置

我们的方案最接近 **Lee et al. (2019) SpiCNN** 和 **Kheradpisheh et al. (2018)**：
- 卷积STDP做无监督特征提取 ✓
- 监督分类器（MLP/SVM）做最终分类 ✓
- 但我们的卷积层更浅（单层32滤波器 vs 多层）

### 为什么Diehl&Cook(2015)的纯无监督能到95%而我们不行？

| Diehl & Cook | 我们 |
|--------------|------|
| 6400个兴奋神经元 | 120个（FC版）/ 32滤波器（Conv版） |
| 自适应发放阈值（homeostasis） | 固定阈值 |
| 电导型突触 | 电流型 |
| 侧向抑制 + WTA | 侧向抑制（FC版）/ 无（Conv版） |
| Brian2模拟器 | SpikingJelly |

**关键差距**：Diehl&Cook用了6400神经元 + 自适应阈值 + 电导突触，这些都是生物细节但被证明对收敛至关重要。

---

## 四、可供一比一复现的项目

| 项目 | 难度 | 预期准确率 | 说明 |
|------|------|-----------|------|
| [cowolff/Simple-STDP](https://github.com/cowolff/Simple-Spiking-Neural-Network-STDP) | ⭐ 低 | ~85% | 纯Python，代码极简，适合理解原理 |
| [BindsNET eth_mnist.py](https://github.com/BindsNET/bindsnet) | ⭐⭐ 中 | ~82% | 完整框架，改几行参数就能跑 |
| [peteru-diehl/stdp-mnist](https://github.com/peteru-diehl/stdp-mnist) | ⭐⭐ 中 | ~83-95% | 原版实现，需要Brian2 |
| SpikingJelly STDP教程 | ⭐ 低 | 教程级 | 官方示例代码，可直接运行 |

### 推荐复现路线

1. **先跑通** `cowolff/Simple-STDP` 理解全流程（30分钟）
2. **再跑通** SpikingJelly官方STDP教程理解API（1小时）
3. **然后对比** BindsNET的Diehl&Cook实现，理解和我们FC版本的区别
4. **最终验证** 我们当前的Conv+MLP方案已达到89.81%，与SpiCNN的91.1%论文基线接近
