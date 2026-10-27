# 线性势训练 {#theory_linear}

[English](linear.md)


@tableofcontents

## 概述

对于能量是回归系数线性函数的势，训练问题转化为求解线性最小二乘系统。
这适用于固定径向系数的矩张量势（MTP），其中位点能量表示为：

@f[
E_i = \sum_{\alpha} \xi_\alpha B_\alpha(\mathbf{r}_i)
@f]

其中 @f$\xi_\alpha@f$ 为线性系数，@f$B_\alpha@f$ 为矩张量基函数。

线性训练通常比非线性优化快得多，只需一次组装与求解即可解出带正则化的最小二乘问题，
无需迭代梯度下降。数值精度仍取决于数据的条件数和求解器。

下列方程描述一般线性最小二乘问题，实现细节对应 `TrainerLinearPotential`：
它用于 `train_linear` 和 Python `LinearTrainer`，支持 `descriptor_linear`
等模型。普通 MTP 使用专用的 `trainmtp --linear-fit` 入口，未实现通用的
`supports_linear_fit()` 接口。

MTP_M 中，固定径向参数仍不足以使非线性头与投影的联合拟合成为线性问题。
固定其余参数后，`li/q2/fs/n1/n2` 各有受支持的仿射输出块，可通过 `trainnlpot`
执行[条件线性精修](../advanced/linear-refit_zh.md)。各头的具体系数与推导见
[MTP_M 理论](mtp-m_zh.md)。

## 正规方程

训练目标是最小化带 L2 正则化的加权平方误差：

@f[
\mathcal{L}(\mathbf{w}) = \frac{1}{2}\|\mathbf{A}\mathbf{w} - \mathbf{y}\|_W^2
+ \frac{\lambda}{2}\|\mathbf{w}\|^2
@f]

其中 @f$\mathbf{A} \in \mathbb{R}^{d \times p}@f$ 是设计矩阵
（@f$d@f$ 数据行，@f$p@f$ 参数），@f$\mathbf{y}@f$ 是目标向量，
@f$W@f$ 是对角损失权重矩阵，@f$\lambda@f$ 是正则化强度。

@f$\mathbf{A}@f$ 和 @f$\mathbf{y}@f$ 均未加权。每个观测的行向量
@f$\mathbf{a}@f$、目标 @f$y@f$ 和权重 @f$w@f$ 分别为 Gram 矩阵贡献
@f$w\mathbf{a}^T\mathbf{a}@f$、为右侧向量贡献 @f$w\mathbf{a}^Ty@f$。
等价做法是将行和目标同时乘以 @f$\sqrt{w}@f$；不要再次将行乘以 @f$w@f$。

最小化解满足正规方程：

@f[
(\mathbf{A}^T \mathbf{W} \mathbf{A} + \lambda \mathbf{I}) \mathbf{w} = \mathbf{A}^T \mathbf{W} \mathbf{y}
@f]

记 @f$\mathbf{G} = \mathbf{A}^T \mathbf{W} \mathbf{A}@f$（Gram 矩阵）和
@f$\mathbf{b} = \mathbf{A}^T \mathbf{W} \mathbf{y}@f$，解为：

@f[
\mathbf{w}^* = (\mathbf{G} + \lambda \mathbf{I})^{-1} \mathbf{b}
@f]

## 设计矩阵组装

设计矩阵 @f$\mathbf{A}@f$ 从训练构型逐行组装。每个构型贡献三种行：

### 能量行（结构级）

对每个训练构型，总能量为设计矩阵贡献一行。对于线性模型 @f$E = \sum_\alpha \xi_\alpha B_\alpha@f$，
系数 @f$\xi_k@f$ 的偏导数即为基函数值本身：

@f[
\mathbf{A}_{\text{eng}} = \sum_{i=1}^{N_{\text{atoms}}}
B_k(\mathbf{r}_i)
@f]

损失权重为 `energy_weight`（默认 1.0）除以
@f$N_{\text{atoms}}^{\text{energy_pow}}@f$，再乘以 `atom.fitness`。

### 力行（逐原子、逐笛卡尔分量）

对每个原子 @f$i@f$，力梯度贡献 @f$3@f$ 行：

@f[
\mathbf{A}_{\text{force}, i, a} = \frac{\partial F_{i,a}}{\partial \xi_k}
@f]

其中 @f$F_{i,a} = -\partial E / \partial r_{i,a}@f$。解析导数需要描述符 Jacobian 矩阵
和通过矩张量缩并树的反向传播。由 `force_weight`（默认 0.01）除以
@f$N_{\text{atoms}}^{\text{force_pow}}@f$ 加权；存在 `atomic_fitness` 时再乘以
对应原子的权重。这些是损失权重，不是设计矩阵行的乘数。

### 应力行（结构级）

通用训练器使用扩展应力张量（eV）的全部 9 个分量，而非 6 个 Voigt 分量；
非对角分量分别计入。对特征 @f$k@f$ 和笛卡尔分量 @f$a,b@f$，未加权行为：

@f[
\mathbf{A}_{\text{stress}, ab} = -\sum_i \sum_j
\frac{\partial B_k(\mathbf{r}_i)}{\partial r_{ij,a}} r_{ij,b}
@f]

其中 @f$\mathbf{r}_{ij}@f$ 是从中心原子 @f$i@f$ 指向邻居的相对位移。
损失权重为 `stress_weight`（默认 0.0001）除以
@f$N_{\text{atoms}}^{\text{stress_pow}}@f$。

## 加权方案

用户可配置的权重平衡不同数据类型的贡献：

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `energy_weight` | 1.0 | 归一化前的能量损失权重 |
| `force_weight` | 0.01 | 归一化前的力损失权重 |
| `stress_weight` | 0.0001 | 归一化前的应力损失权重 |
| `energy_pow` | 1.0 | 能量的 @f$N_{\text{atoms}}@f$ 缩放指数 |
| `force_pow` | 0.0 | 力的 @f$N_{\text{atoms}}@f$ 缩放指数 |
| `stress_pow` | 1.0 | 应力的 @f$N_{\text{atoms}}@f$ 缩放指数 |
| `weight_L2` | 1e-10 | L2 正则化强度 @f$\lambda@f$ |

此通用训练器中，`atom.fitness` 仅缩放能量观测；`atomic_fitness` 缩放对应原子的
力观测；两者均不缩放应力观测。缺失标签以及零权重观测会被跳过。

## 正则化

L2（Tikhonov / 权重衰减）正则化为 Gram 矩阵添加 @f$\lambda \mathbf{I}@f$，其作用：

- 防止对训练数据中噪声的过拟合。
- 当所有系数均受正惩罚时，可使秩不足的正规矩阵变为正定；数值条件数仍取决于
  数据尺度和惩罚强度。
- 将解偏向较小的系数范数。

正则化强度 @f$\lambda@f$ 由 `weight_L2`（默认 @f$10^{-10}@f$）控制。

## TrainerLinearPotential

`TrainerLinearPotential`（位于 `train/train_linear_potential.{h,cpp}`）编排
线性训练流程：

1. **`init()`** — 验证势是否支持线性拟合，分配 Gram 矩阵 @f$\mathbf{G}@f$ 和
   右侧向量 @f$\mathbf{b}@f$。
2. **`Assemble()`** — 遍历所有训练构型：
   - 通过 `linear_compute_descriptor_grad()` 计算逐原子描述符及其参数梯度。
   - 按观测的损失权重累加 @f$\mathbf{G}@f$ 和 @f$\mathbf{b}@f$。
   - 支持跨节点的 MPI 分布式组装。
3. **`Solve()`** — 求解带正则化的正规方程：
   - 主求解器：LDLT 分解（类 Cholesky，对对称正定 @f$\mathbf{G} + \lambda\mathbf{I}@f$ 高效）。
   - 备用方案：若 LDLT 失败则使用全选主元 LU 分解。
   - 将解向量 @f$\mathbf{w}^*@f$ 广播到所有 MPI 进程。
4. **`Train()`** — `Assemble()` + `Solve()`，保存模型后，在通过 C++ 接口配置的
   可选留出验证集上评估误差。

## 验证

通用线性训练器只执行一次组装和求解，不进行带早停的迭代优化。C++ 验证入口
在拟合结束后评估一次。Python `LinearTrainer.train()` 仅接受训练数据，留出
结构应单独评估。选择正则化时比较验证误差，最终测试集留待模型选择结束后使用。

## 与非线性训练的对比

| 方面 | 线性 | 非线性 |
|--------|--------|-----------|
| **使用场景** | 线性模型；固定径向系数的 MTP | MTP 径向拟合、NN-MTP 等非线性模型 |
| **求解** | 求解线性系统（精度受数值条件影响） | 迭代（L-BFGS、SGD） |
| **速度** | 快（单次组装 + 求解） | 较慢（需要多个 epoch） |
| **正则化** | 通过正规方程 L2 | 通过目标函数 L1/L2 |
| **导数** | 解析设计矩阵组装 | 参数梯度，包括径向、张量或神经网络导数 |
| **训练类** | `TrainerLinearPotential`（通用）；`Trainer_MTP`（MTP） | `Trainer_MTP`、`TrainNonlinear` |

## 类总结

| 类 | 模块 | 角色 |
|-------|--------|------|
| `TrainerLinearPotential` | @ref train | 通用线性最小二乘训练器，例如用于 `descriptor_linear` |
| `ErrorMonitor` | @ref train | 训练/验证误差跟踪 |

**参见：** @ref theory_mtp, @ref theory_training
