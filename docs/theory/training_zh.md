# 训练方法 {#theory_training}

[English](training.md)

@tableofcontents

## 选择训练入口

训练通过调整模型参数，使预测的能量、力和应力接近参考值。普通 MTP 包含径向系数
和线性系数；固定径向部分后，问题可转化为线性最小二乘。神经网络和长程模型还
可能包含其他可训练参数。

| 任务 | 入口 |
| --- | --- |
| 固定径向基，仅拟合普通 MTP 的线性系数 | `imrmlp trainmtp train.ini --linear-fit` |
| 优化普通 MTP 的径向和线性系数 | `imrmlp trainmtp train.ini --do-lbfgs` |
| 拟合提供通用线性接口的模型，如 `descriptor_linear` | `imrmlp train_linear train.ini` |
| 训练 MTP-M、NN-MTP 或其他受支持的非线性势 | `imrmlp trainnlpot train.ini --do-lbfgs` |

这些命令假定已有初始模型，且设置文件已指定训练数据和模型路径。`trainmtp`
必须显式选择训练或诊断操作。需要完整可运行示例时，先运行
[快速入门](../../examples/quickstart/README_zh.md)，再阅读
[训练工作流](../guides/workflows_zh.md)。

## MTP_M 的参数链与条件线性

MTP_M 依次计算共享不变量、通道投影和能量头。拟合力时，损失梯度必须经过头的
一阶、二阶导数与描述符的混合导数，不能把非线性头当作固定线性系数。
其通道定义、五种头、Hessian 链式法则及参数分组见 [MTP_M 理论](mtp-m_zh.md)。

通常用 `initmtpm` 准备初始模型，再通过 `trainnlpot` 优化。
如果只想固定隐藏特征改善输出系数，可使用 `fit:linear_refit output`，
见[线性精修](../advanced/linear-refit_zh.md)。普通 MTP 的
[原生 VarPro](../advanced/mtp-native-varpro_zh.md)则在径向试探中重解线性问题，
两者的支持范围和优化变量不同。

## 损失函数与权重

普通 MTP 的能量、力、应力（E/F/S）数据损失，按训练构型数 @f$M@f$ 取平均：

@f[
\mathcal L_{\mathrm{data}} = \frac{1}{M}\sum_{s=1}^{M}
\left[W_{E,s}(E_s-E_s^{\mathrm{ref}})^2
+ W_{F,s}\sum_{i,a}(F_{s,i,a}-F_{s,i,a}^{\mathrm{ref}})^2
+ W_{S,s}\|S_s-S_s^{\mathrm{ref}}\|_F^2\right].
@f]

这里 @f$E_s@f$ 是总能量，@f$N_s@f$ 是原子数，@f$a@f$ 遍历三个笛卡尔分量，
@f$S_s@f$ 是内部以 eV 存储的广延应力张量（应力乘体积）。张量范数对**全部九个
分量**求平方和，即一对非对角分量分别计入。只有标签存在且相应权重大于零时，
该观测才参与拟合。

可选的 fitness 和随参考力变化的重加权采用默认值时，有效权重为
@f$W_{X,s}=w_X/N_s^{p_X}@f$：

| 设置 | 默认值 | 含义 |
| --- | --- | --- |
| `fit:energy_weight` | `1.0` | @f$w_E@f$，总能量平方误差权重 |
| `fit:force_weight` | `0.01` | @f$w_F@f$，力分量平方误差权重 |
| `fit:stress_weight` | `0.0001` | @f$w_S@f$，广延应力平方误差权重 |
| `fit:energy_pow` | `1.0` | @f$p_E@f$，能量的原子数归一化指数 |
| `fit:force_pow` | `0.0` | @f$p_F@f$，力的原子数归一化指数 |
| `fit:stress_pow` | `1.0` | @f$p_S@f$，应力的原子数归一化指数 |

因此，默认能量损失使用 @f$(\Delta E_s)^2/N_s@f$，不同于先计算每原子能量误差
再平方得到的 @f$(\Delta E_s)^2/N_s^2@f$。权重用于平衡单位和观测数量不同的
数据。增大力权重会提高力标签对拟合的影响，并不表示指定了力误差上限。
修改权重后，应比较独立验证集上的各项误差。

构型 `fitness` 缩放能量观测；逐原子 `atomic_fitness` 缩放力观测。
可选的 `fit:force_wgt_*` 设置还会按参考力大小重加权，详见
[配置参考](../reference/configuration_zh.md)。其他训练入口有各自的目标函数
选项；上式对应普通 MTP 的 E/F/S 路径。

## 梯度计算

优化器需要损失对各可训练系数的导数。能量导数通过径向函数和张量缩并计算；
拟合力还需要位置与参数的混合导数，因为力是能量梯度的**负值**：

@f[
F_{i,a}=-\frac{\partial E}{\partial x_{i,a}},\qquad
\frac{\partial F_{i,a}}{\partial c_k}
=-\frac{\partial^2 E}{\partial x_{i,a}\partial c_k}.
@f]

例如，一个加权力残差对损失梯度的贡献是
@f$2W_F(F_{i,a}-F_{i,a}^{\mathrm{ref}})\,\partial F_{i,a}/\partial c_k@f$。
程序会累加这些导数，用户无需构造完整的坐标 Hessian。
使用 Hessian 或 Hessian 向量积标签训练是另一项可选功能，见
[HVP 训练](../advanced/hvp-training_zh.md)。

## 正则化

正则化用于抑制过大的系数或不规则的径向函数。它需要配合验证使用，不能保证模型
一定准确。

对普通 MTP，`fit:weight_L1` 和 `fit:weight_L2` 控制线性**矩系数**的惩罚项，
不包括逐元素的常数能量偏移：

@f[
\mathcal L_{\mathrm{coeff}}=
\sum_{k\in\mathrm{moments}}\left(\lambda_{1,k}|c_k|+\lambda_{2,k}c_k^2\right).
@f]

默认值为 `fit:weight_L1 0` 和 `fit:weight_L2 1e-10`。线性求解可根据设计矩阵
缩放 L2 对角项，因此同一个正则化数值在不同训练器中不一定产生相同效果。

径向惩罚取决于 `fit:radial_reg_type`：模式 `0` 惩罚径向系数归一化与正交性的
偏离；模式 `1` 中，`fit:weight_rad_w0`、`fit:weight_rad_w1` 和
`fit:weight_rad_w2` 分别惩罚 @f$r^2|f(r)|@f$、@f$r^2|f'(r)|@f$ 和
@f$r^2|f''(r)|@f$ 的采样积分。这些是绝对值惩罚，并非一、二、三阶导数的平方惩罚。

## 优化算法

### L-BFGS

L-BFGS 根据梯度和有限步历史信息确定搜索方向，再进行线搜索。
它是 MTP 拟合常用的起点，通过 `--do-lbfgs` 选择。普通 MTP 命令的该标志使用
ALGLIB 后端；其他训练入口提供各自的优化器控制。

结构弛豫与参数训练是不同的任务。训练失败时，不会自动转用结构优化器 FIRE。
重新启动前，应先检查报错、输入几何、初始参数和损失权重。

### SGD 与学习率调度

`trainmtp --do-sgd` 使用小批次，可减少大数据集每次更新所需的工作量。
以下设置演示指数学习率衰减：

```text
fit:learning_rate 0.0001
fit:batch_size 100
fit:epoch_limit 100
fit:lr_scheduler exponential
fit:lr_decay_rate 0.001
```

支持的调度器名称为 `none`、`step`、`exponential` 和 `cosine`。
`step` 使用 `lr = lr0 * rate^floor(t / decay_steps)`，`exponential` 使用
`lr = lr0 * exp(-rate * t)`；这里 `t` 是小批次更新数，`rate` 为
`fit:lr_decay_rate`。因此指数模式的 `0.95` 不是“每步乘 0.95”，而是很快的衰减。
`fit:lr_decay_steps` 只控制 `step` 的间隔；预热仅在启用调度器时生效。
调度与停止条件详见
[配置参考](../reference/configuration_zh.md)。

## 理解报告的误差

训练损失经过加权；监控器报告的 RMS 则描述未加权的预测误差。
普通 MTP 的训练监控含义如下：

| 指标 | 含义 |
| --- | --- |
| `EATRMS` | 各带能量标签构型的**每原子**能量误差的均方根 |
| `FRMS` | 按带力标签原子平均的力矢量误差均方根，并非每个笛卡尔分量的均方根 |
| `SRMS` | 按带应力标签构型平均的广延应力张量 Frobenius 范数误差均方根 |

对于同一组观测，力分量 RMS 等于 `FRMS / sqrt(3)`；九分量应力 RMS 等于
`SRMS / 3`。普通 MTP 进度行将内部 eV/atom 和 eV/Å 单位的 `EATRMS`、`FRMS`
乘以 1000 后输出，`SRMS` 仍以 eV 输出。与 `calc_err`、其他训练器或外部工具
比较时，应先核对输出表头。

## 验证与部署

1. 拟合前划分数据；相关轨迹帧尽量放在同一子集中，减少训练与验证之间的信息泄漏。
2. 使用训练标签拟合；使用验证集选择基组、权重、正则化和停止点。
3. 分别检查能量、力和应力，包括较大误差及其对应构型。
4. 模型选择结束后评估最终测试集，再检查预期模拟条件下的行为并部署。

普通 `trainmtp` 的 `--validset` 用于误差监控，不会启用基于验证集的自动早停。
最终测试集应与选择超参数时使用的数据分开。主动学习可以把陌生结构加入候选池，
但这些结构需要取得参考标签后，才能改善有监督拟合。

实现细节见 @ref train "训练模块"；线性最小二乘见 @ref theory_linear "线性训练"；
主动集选择见 @ref theory_maxvol "MaxVol"。
