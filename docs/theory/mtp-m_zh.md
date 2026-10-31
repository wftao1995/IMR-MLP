# MTP_M：多通道与非线性能量头 {#theory_mtp_m}

[English](mtp-m.md)

@tableofcontents

MTP_M（也写作 MTP-M，代码类为 `PotMTP_M`，势类型为 `mtp_m`）在共享的
MTP 标量不变量上构建多个可训练通道，再由能量头得到位点能量。
它保留 MTP 的局域性和对称性，通过改变能量对描述符的依赖形式扩展模型容量。
本页描述本仓库实现；矩张量、径向基和缩并图的构造见 [MTP 理论](mtp_zh.md)。

## 从普通 MTP 到 MTP_M

普通 MTP 的短程位点能量为

@f[
E_i=\epsilon_{z_i}+\sum_{\alpha=1}^{P}\xi_\alpha
B_\alpha(\mathfrak n_i;\mathbf c).
@f]

这里 @f$\mathfrak n_i@f$ 是截断半径内的局域环境，@f$z_i@f$ 是中心元素，
@f$P@f$ 是标量不变量数，@f$\mathbf c@f$ 是共享的径向参数。
固定径向参数后，能量、力和维里对 @f$\epsilon,\xi@f$ 呈线性关系。

MTP_M 先把同一组不变量投影为 @f$q@f$ 个通道：

@f[
Q_{ik}=b_{k,z_i}+\sum_{\alpha=1}^{P}A_{k\alpha}
B_\alpha(\mathfrak n_i;\mathbf c),\qquad k=1,\ldots,q,
@f]

再计算

@f[
E_i=\epsilon_{z_i}+F_{\boldsymbol\theta}(\mathbf Q_i),
\qquad E=\sum_i E_i.
@f]

@f$A@f$ 是可训练投影矩阵，@f$b_{k,z_i}@f$ 是每通道、每元素的偏置，
@f$\boldsymbol\theta@f$ 是能量头参数。外部元素能量偏移 @f$\epsilon_{z_i}@f$
在能量头之后相加，与通道偏置是两组不同参数。

**通道数不等于 MTP level、矩张量秩或多体阶数。** `potential_tag` 中的整数
`q` 记录短程通道数；`li` 为 1，`q2` 和 `fs` 为 2，神经网络头由 `n_input` 决定。
通道是拟合得到的内部变量，没有预设的电荷、密度或轨道含义，也不要求彼此独立。

所有通道复用径向基、基本矩和缩并图，只改变最后的投影系数。因此增加通道不需要
完整重算多份 MTP 描述符，但会增加投影、能量头和反向传播的成本。

## 五种能量头

以下均为短程形式，元素能量偏移已单独写在 @f$F@f$ 外。
`pot:extmodel` 选择能量头；已保存的模型元数据也记录其类型。

| `extmodel` | 通道数 | 能量头 | 特点与代价 |
| --- | --- | --- | --- |
| `mtp_m_li` | 1 | @f$F=Q_1@f$ | 线性基线；能量头没有额外参数 |
| `mtp_m_q2` | 2 | 两通道的完整一次项与二次项 | 显式耦合通道；5 个头参数，外推时二次项可能快速增长 |
| `mtp_m_fs` | 2 | @f$F=Q_1+a\,\operatorname{asinh}(Q_2)@f$ | 线性主项加光滑非线性修正；1 个头参数 |
| `mtp_m_n1` | `n_input` | 单隐藏层网络，线性输出 | 可调通道数和隐藏层宽度；增加参数量和优化难度 |
| `mtp_m_n2` | `n_input` | 双隐藏层网络，线性输出 | 更深的通道混合；计算量取决于两层宽度，需独立验证收益 |

### li：与普通 MTP 的关系

令 @f$q=1@f$ 且 @f$F=Q_1@f$，有

@f[
E_i=(\epsilon_{z_i}+b_{1,z_i})+
\sum_\alpha A_{1\alpha}B_\alpha.
@f]

在相同径向基和标量基下，这与普通 MTP 具有相同的能量函数形式。
两种参数化不要求文件布局相同；@f$\epsilon_s@f$ 与 @f$b_{1,s}@f$ 还存在冗余：
只要两者之和不变，预测就不变。

### q2：通道间的二次耦合

当前实现严格使用两个通道：

@f[
F=c_1Q_1+c_2Q_2+c_{11}Q_1^2+c_{22}Q_2^2+c_{12}Q_1Q_2.
@f]

它既不是只有平方项，也不是任意通道数的全二次模型。
代入 @f$\mathbf Q=\mathbf b_s+A\mathbf B@f$，可得到
@f$B_\alpha B_\beta@f$ 形式的交叉项。但其系数受投影限制：写

@f[
H=\begin{pmatrix}c_{11}&c_{12}/2\\c_{12}/2&c_{22}\end{pmatrix},
\qquad C=A^{\mathsf T}HA,
@f]

则标量基空间中的二次系数矩阵 @f$C@f$ 的秩至多为 2。
这是用少量通道组织非线性相互作用的方式，不等价于为全部 @f$P(P+1)/2@f$
个基函数乘积分别拟合独立系数。

### fs：ArcSinh 修正

此处的 `fs` 指本实现的
@f$Q_1+a\operatorname{asinh}(Q_2)@f$，不能直接替换为传统
Finnis–Sinclair 模型的平方根嵌入项。它允许 @f$Q_2@f$ 为任意实数，且

@f[
\frac{\partial F}{\partial Q_1}=1,\qquad
\frac{\partial F}{\partial Q_2}=\frac{a}{\sqrt{1+Q_2^2}},\qquad
\frac{\partial^2F}{\partial Q_2^2}=
-\frac{aQ_2}{(1+Q_2^2)^{3/2}}.
@f]

修正在零点光滑，通道导数不会出现平方根在零附近的奇异性。这只解决零点导数；外推
稳定性仍取决于径向函数、截断和训练数据覆盖，用验证集误差和 grade 分别检查这三项。

### n1 / n2：神经网络能量头

单隐藏层头采用

@f[
F=\mathbf v^{\mathsf T}\phi(W\mathbf Q+\mathbf d).
@f]

双隐藏层头采用

@f[
F=\mathbf v^{\mathsf T}\phi\!\left(
W_2\phi(W_1\mathbf Q+\mathbf d_1)+\mathbf d_2\right).
@f]

这里按当前 MTP_M 封装直接使用通道向量；底层双隐藏层网络的输入归一化初始化为恒等变换。
两种头均无独立输出偏置，元素常数由外部
@f$\epsilon_s@f$ 等参数承担。`n1` 支持 TANH/RELU/SIGMOID/LINEAR，
`n2` 支持 TANH/RELU/SIGMOID。

非线性头可以在同一局域基组上表达更复杂的能量依赖，但不会自动扩大截断半径。
ReLU 在零点不可微，若计算声子、Hessian 或更高阶导数，应考虑激活函数及径向
截断的光滑性；程序在分支点返回的数值不能消除数学上的不可微性。

## 对称性、局域性与可辨识性

@f$B_\alpha@f$ 已满足平移、旋转及同元素原子置换不变性，投影和标量能量头
保持这些性质。位点能量求和使相互分离的子体系能量可加，力和维里均由同一总能量求导。
这些结论要求模型使用一致的元素映射和几何定义；光滑性还取决于基函数与头的具体选择。

**非线性头不等于长程物理。** 若中心元素相同且两套环境在截断半径内相同，普通短程 MTP_M
给出的位点能量也相同，无法仅凭能量头区分截断外的电荷分布。
需要 LES/SOG 等额外长程项时，必须使用包含它们的模型。

内部通道的数值通常不唯一：投影与后续头可以通过相容的缩放、混合或偏置变换
保持同一能量函数，具体自由度取决于头的形式。因此比较训练结果时应比较能量、
力与目标物性，而非要求各个 @f$Q_k@f$ 的值或全部系数一致。

## 力、维里与二阶导数

对每个位点，先把头的梯度传回标量不变量：

@f[
g_{i\alpha}=\frac{\partial E_i}{\partial B_{i\alpha}}
=\sum_{k=1}^{q}\frac{\partial F}{\partial Q_{ik}}A_{k\alpha}.
@f]

再沿普通 MTP 的缩并图和径向基反传。全体系力为

@f[
F_{j,a}=-\sum_i\sum_\alpha g_{i\alpha}
\frac{\partial B_{i\alpha}}{\partial x_{j,a}}.
@f]

通道偏置和元素能量偏移对坐标的直接导数为零；通道偏置仍会通过
@f$\partial F/\partial\mathbf Q@f$ 改变非线性模型的力。
短程维里由相同的位点邻居导数构造，符号与单位见 [MTP 力与应力](mtp_zh.md)。

令 @f$J_i=\partial\mathbf Q_i/\partial\mathbf x@f$，位点能量的坐标 Hessian 为

@f[
\nabla_{\mathbf x}^2E_i=
J_i^{\mathsf T}\left(\nabla_{\mathbf Q}^2F\right)J_i+
\sum_k\frac{\partial F}{\partial Q_{ik}}
\nabla_{\mathbf x}^2Q_{ik}.
@f]

第一项来自头本身的曲率；`li` 中该项为零，非线性头中通常必须保留。
仅把普通 MTP 的线性系数替换成当前 @f$g_{i\alpha}@f$，不足以得到完整 Hessian。
力标签训练还需要位置与参数的混合导数；见 [训练理论](training_zh.md)
和 [HVP 训练](../advanced/hvp-training_zh.md)。

## 参数布局与训练含义

设元素数为 @f$S@f$，径向参数数目为 @f$N_r@f$，短程模型的完整参数数目为

@f[
N_{\mathrm{par}}=N_r+S+qS+qP+N_{\mathrm{head}}.
@f]

| 参数 | 数目 | 实现中的位置 |
| --- | --- | --- |
| 共享径向参数 @f$\mathbf c@f$ | @f$N_r@f$ | `radial_coeffs` |
| 元素能量偏移 @f$\epsilon_s@f$ | @f$S@f$ | `species_coeffs` 的前 @f$S@f$ 项 |
| 通道元素偏置 @f$b_{k,s}@f$ | @f$qS@f$ | `species_coeffs` 的其余项，按通道排列 |
| 通道投影 @f$A_{k\alpha}@f$ | @f$qP@f$ | `moment_coeffs`，按通道排列 |
| 能量头参数 @f$\boldsymbol\theta@f$ | @f$N_{\mathrm{head}}@f$ | `model_coeffs` |

这是完整短程参数布局；冻结部分参数后，优化器看到的向量会更短。
`sr_species` 对应外部元素偏移，`sr_moment` 包含通道偏置及投影，
`sr_model` 对应能量头，`sr_radial` 对应径向参数。

固定径向参数只会固定 @f$\mathbf B@f$，不会把所有 MTP_M 头的联合训练变成线性问题。
按各头的依赖关系冻结非线性部分后，仍有可精修的条件线性块：

| 头 | `output` 精修可选择的仿射系数 |
| --- | --- |
| `li` | 元素能量偏移、通道偏置和投影系数 |
| `q2` | 元素能量偏移及 5 个二次头系数 |
| `fs` | 元素能量偏移及 ArcSinh 前的系数 @f$a@f$ |
| `n1` / `n2` | 元素能量偏移及网络最后一层的线性输出权重 |

实际选择还取决于启用的参数组。这里的“线性”是**固定其他参数后的条件线性**，
不表示可以联合线性求解投影、隐藏层和径向参数。
具体入口为 `trainnlpot` 的 `fit:linear_refit output`，见
[线性精修](../advanced/linear-refit_zh.md)；普通 MTP 的原生 VarPro 支持范围另见
[变量投影](../advanced/mtp-native-varpro_zh.md)，不能直接套用于任意 MTP_M。

`initmtpm` 用训练集初始化径向/投影参数、拟合元素能量偏移，并调整通道偏置使
训练集上的通道均值居中；通道方差和通道间相关性不受约束，不要按白化基来解读通道值。
这些统计量只从训练集估计，不用最终测试集。初始化之后通常用 `trainnlpot --do-lbfgs`
等非线性优化入口继续训练，并在独立验证集上比较模型容量、随机初始化和误差。

## 与长程模型及部署的关系

带 LES/SOG 的 MTP_M 在上述短程能量之外加入长程泛函，并使用单独的电荷或
潜变量投影系数。短程头的 @f$q@f$ 个通道与 SOG 的长程通道数是两个独立概念。
长程项的能量、响应力和晶胞导数都必须计入总预测；不能只保留短程头。
相关定义见 [LES v2](../reference/les-v2_zh.md)、[SOG](../reference/sog_zh.md)。

短程头可按支持范围使用 CPU/OpenMP 或 GPU 独立部署；原生模型组合和长程路径
需要选择兼容接口。文件后缀或转换为 GPM 都不会改变模型的数学形式或执行器支持范围。
部署选择见 [LAMMPS 接口](../../LAMMPS/README_zh.md)。

实现依据：[通道、能量头与导数](../../potentials/potential_mtp_m.h)、
[工厂与头类型](../../potentials/potential_mtp_m.cpp)、
[初始化](../../apps/initmtpm.cpp)。API 使用示例见
[势函数接口](../development/api-potential_zh.md)。
