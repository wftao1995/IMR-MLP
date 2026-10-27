# 矩张量势（MTP） {#theory_mtp}

[English](mtp.md)


@tableofcontents

实际生成模型和拟合时，请先阅读[训练工作流](../guides/workflows_zh.md)。
本页解释普通 MTP 的能量形式、对称性、基组和导数。
多通道投影及非线性能量头另见 [MTP_M 理论](mtp-m_zh.md)。

## 参考文献

1. **Wang, J. et al.** "Efficient moment tensor machine-learning interatomic potential
   for accurate description of defects in Ni-Al Alloys." *Phys. Rev. Materials* 9,
   053805 (2025). DOI: [10.1103/PhysRevMaterials.9.053805](https://doi.org/10.1103/PhysRevMaterials.9.053805)
2. Shapeev, A. "Moment Tensor Potentials: A Class of Systematically Improvable
   Interatomic Potentials." *Multiscale Model. Simul.* 14, 1153-1173 (2016).
3. Novikov, I. et al. "The MLIP package: moment tensor potentials with MPI and
   active learning." *Mach. Learn.: Sci. Technol.* 2, 025002 (2021).
4. Gubaev, K. et al. "Machine learning molecular dynamics: A concise review."
   *Phys. Status Solidi B* 255, 1700359 (2018).

## 概述

矩张量势（MTP）是一类由 Shapeev（2016）引入的、可系统性改进的原子间
势。原子的能量表示为矩张量描述符构建的标量不变量的线性组合：

@f[
E=\sum_i E_i,\qquad
E_i = \epsilon_{z_i}+\sum_{\alpha} \xi_\alpha B_\alpha(\mathfrak n_i;\mathbf c)
@f]

其中 @f$\mathfrak n_i@f$ 是原子 @f$i@f$ 的局域环境，@f$B_\alpha@f$ 是矩张量
不变量（标量基函数），@f$\mathbf c@f$ 是径向参数，@f$\xi_\alpha@f$ 是矩系数，
@f$\epsilon_{z_i}@f$ 是中心元素的能量偏移。
“线性 MTP”指固定径向参数后的线性组合；同时拟合径向参数时，整体仍是非线性问题。

旋转不变性来自对对称笛卡尔矩张量指标的完全缩并，从而得到标量基函数。
本实现的 MTP 基函数不使用 Clebsch-Gordan 系数缩并；Clebsch-Gordan 系数用于
其他描述符，例如 bispectrum 描述符。

## 局域性、对称性与模型容量

局域环境由截断半径内的相对位置和元素组成：

@f[
\mathfrak n_i=\{(\mathbf r_{ij},z_j):
0<|\mathbf r_{ij}|<r_{\mathrm{cut}}\},\qquad
\mathbf r_{ij}=\mathbf x_j-\mathbf x_i.
@f]

相对位置消除整体平移的影响；邻居求和使同元素原子的重新编号不改变结果；
张量指标完全缩并使能量对整体旋转不变。总能量是位点能量之和，因此彼此分离、
互不作用的子体系能量可加。位点能量是模型的分解，并非训练数据必须提供的唯一原子能量标签。

截断半径控制可见的空间范围，level、径向基大小和多体约束控制局域函数的容量。
增加 level 不会恢复截断外的信息；加入非线性头也不会自动产生长程相互作用。
能量、力及更高阶导数的连续性还取决于径向截断和能量头的光滑性。

两个低阶不变量有助于理解从距离到角度的构造：

@f[
M_{\mu,0}=\sum_j f_\mu(r_{ij}),\qquad
\mathbf M_{\mu,1}=\sum_j f_\mu(r_{ij})\widehat{\mathbf r}_{ij},
@f]

@f[
\mathbf M_{\mu,1}\cdot\mathbf M_{\lambda,1}
=\sum_{j,k}f_\mu(r_{ij})f_\lambda(r_{ik})
\cos\theta_{jik}.
@f]

第一个量仅依赖距离；第二个量通过点积包含邻居间夹角。双重求和还包含
@f$j=k@f$ 项，因此它具有至多三体的依赖，不应把整个表达式当作纯三体项。
更高阶缩并以类似方式组合径向与角度信息。

## 矩张量构建

### 基本矩张量

对于原子 @f$i@f$ 及其在位置
@f$\mathbf{r}_{ij} = (x_{ij}, y_{ij}, z_{ij})@f$ 处的近邻 @f$j@f$，
指数为 @f$(a,b,c)@f$ 的基本笛卡尔张量分量定义为：

@f[
M_{\mu,a,b,c}(\mathbf{r}_i) = \sum_{j} f_\mu(|\mathbf{r}_{ij}|) \cdot
\frac{x_{ij}^a \, y_{ij}^b \, z_{ij}^c}{|\mathbf{r}_{ij}|^\nu}
@f]

其中 @f$a + b + c = \nu@f$，@f$f_\mu(r)@f$ 是径向函数。用
@f$|\mathbf{r}_{ij}|^\nu@f$ 归一化后，笛卡尔因子仅依赖键的方向；
距离依赖保留在 @f$f_\mu(r)@f$ 中。
在势文件和代码中，每个基本分量作为 `alpha_index_basic` 的一行
`(mu, a, b, c)` 存储。

### 径向基函数

径向函数按模型选择的基函数展开：

@f[
f_\mu(r) = \sum_{k} c_{\mu, z_i, z_j}^{k} \, Q^{k}(r)
@f]

其中 @f$Q^{k}(r)@f$ 是选定的径向基函数，包含配置的截断函数和坐标变换；
@f$c_{\mu, z_i, z_j}^{k}@f$ 是依赖元素种类的径向系数。选择 Chebyshev 基
并不意味着加入截断或非线性坐标变换后仍然正交归一。可用类型详见
[径向基参考](../reference/radial-basis_zh.md)。

代码通过 `AnyBasis` 层级支持多种径向基类型：
- `RBasis_PolyB` — 多项式基：@f$\phi_\alpha(r) = (r_c - r)^{\alpha+2} / N_\alpha@f$
- `RBasis_ChebyshevConfigurable` — 可配置 cutoff、坐标变换及
  Chebyshev/one-minus-Chebyshev 形式

**参见：** @ref basis

### 递归笛卡尔缩并

非基本基值通过笛卡尔指标缩并生成的低阶中间量乘积与求和递归构建。
如果 @f$b_k@f$ 和 @f$b_l@f$ 是低阶中间值，则高阶中间值为：

@f[
b_j = \sum_{k,l} c_{jkl} \cdot b_k \cdot b_l
@f]

其中 @f$c_{jkl}@f$ 是缩并枚举生成的整数重数因子。最终 MTP 基函数是从
这些中间值数组中选取的标量项，因此对刚性旋转不变。

该分解生成一个存储于以下变量中的缩并树：
- `alpha_index_basic` — 基本矩张量规格（mu, a, b, c）
- `alpha_index_times` — 乘法/缩并规则 {id1, id2, multiplier, output_id}
- `alpha_moment_mapping` — 所有矩到标量不变量的映射

### 级别结构

基函数按级别组织。矩张量的级别定义为
@f$\operatorname{lev}(M_{\mu,\nu}) = w_\mu \mu + w_\nu \nu + w_d@f$，其中 @f$w_\mu, w_\nu, w_d@f$
是可配置的权重参数。在 `mtmbasis` 中，当前默认值为
@f$w_\mu = 2@f$、@f$w_\nu = 1@f$、@f$w_d = 2@f$。
对于由 @f$n@f$ 个矩张量构建的标量基函数，
@f$\operatorname{lev}(B_\alpha) = \sum_{i=1}^n \operatorname{lev}(M_{\mu_i,\nu_i})@f$。

所有级别 @f$\leq d@f$（级别阈值）的基函数构成基组。
可以施加额外的约束：
- 最大多体阶数 @f$\text{nbody}_{\max}@f$
- 最大多项式次数 @f$\text{npoly}_{\max}@f$
- 最大矩张量秩 @f$\nu_{\max} \leq 3@f$（推荐，因为 @f$\nu@f$ 秩
  张量有 @f$(\nu+1)(\nu+2)/2@f$ 个独立分量，复杂度随 @f$O(\nu^2)@f$ 增长）

这些权重通过 `imrmlp mtmbasis` 的顶层 `mu_wei`、`nv_wei` 和 `dim_wei`
设置暴露。

## 标量基生成

### 缩并指标矩阵

给定一组矩张量 @f$\{M_{\mu_1,\nu_1}, \ldots, M_{\mu_n,\nu_n}\}@f$，
缩并规则由一个 @f$(n+1) \times (n+1)@f$ 的指标矩阵唯一确定：

@f[
\boldsymbol{\alpha} = \begin{pmatrix}
0 & \mu_1 & \mu_2 & \cdots & \mu_n \\ \hline
\mu_1 & 0 & \alpha_{12} & \cdots & \alpha_{1n} \\
\mu_2 & \alpha_{12} & \ddots & \ddots & \alpha_{2n} \\
\vdots & \vdots & \ddots & \ddots & \vdots \\
\mu_n & \alpha_{1n} & \alpha_{2n} & \cdots & 0
\end{pmatrix}
@f]

右下角的 @f$n \times n@f$ 子矩阵 @f$\tilde{\boldsymbol{\alpha}}@f$ 编码了
缩并方式：@f$\tilde{\alpha}_{ij}@f$ 指定了 @f$M_i@f$ 和 @f$M_j@f$ 的
多少个指标被缩并。秩求和约束 @f$\sum_j \tilde{\alpha}_{ij} = \nu_i@f$（对所有
@f$i@f$）确保结果是标量。由于矩张量是完全对称的，
缩并结果与具体选择哪些指标无关。

**示例：** 对于 @f$\{M_{1,3}, M_{2,2}, M_{3,1}\}@f$ 且
@f$\tilde{\boldsymbol{\alpha}} = \begin{pmatrix} 0 & 2 & 1 \\ 2 & 0 & 0 \\ 1 & 0 & 0 \end{pmatrix}@f$，
标量基为 @f$B_\alpha = (M_{1,3})_{ijk} (M_{2,2})_{ij} (M_{3,1})_k@f$。

### 等价性检查

两个指标矩阵 @f$\boldsymbol{\alpha}@f$ 和 @f$\boldsymbol{\beta}@f$ 表示
等价的缩并规则，当且仅当存在一个置换矩阵 @f$\mathbf{P}@f$，
重排张量因子并保持辅助的首行/首列不变，使得
@f$\boldsymbol{\beta} = \mathbf{P} \boldsymbol{\alpha} \mathbf{P}^\mathrm{T}@f$。
当前实现通过同时置换张量因子，构造精确的规范表示进行比较。
先排除径向索引或秩列表不匹配的情况，再比较规范化的整数缩并矩阵。
因子数最多为七，因此可以枚举置换。仅有相同的特征值并不足以证明等价。

### 计算路径：树分解

标量基函数 @f$B_\alpha@f$ 通过将其分解为中间张量缩并的二叉树来计算。
根节点是标量输出，叶节点是基本矩张量，每个内部节点是两个子张量的缩并：

@f[
b_n = \sum_i d_i \cdot b_{k_i} \cdot b_{l_i}
@f]

其中 @f$d_i@f$ 是对称性简并因子，@f$(n, k_i, l_i)@f$ 是
存储在 `alpha_index_times` 中的乘法规则。

分解不是唯一的——不同的策略会产生不同数量的
存储的矩分量数 @f$n_b@f$ 和乘法规则数 @f$n_t@f$，直接
影响计算成本。对于典型的 level-28 基组，可能的分解路径数
超过 @f$10^{2195}@f$，使得穷举搜索不可能实现。

### 缩并布局搜索

`mtmbasis` 通过顶层设置 `mode` 选择搜索方法：`0` 为默认的 tempering，
`1` 为随机/局部搜索，`2` 为遗传算法（GA），`3` 为坐标下降。
这些都是启发式搜索，不保证找到全局最优解。GA 的基本流程如下
[Wang et al., 2025]：

1. **枚举**：对于每个标量函数 @f$B_\alpha@f$，枚举所有有效的
   分解树 @f$\{T_i^\alpha\}@f$。
2. **编码**：每个个体是一个基因序列 @f$\{k_1, \ldots, k_n\}@f$，其中
   @f$k_i@f$ 选择第 @f$i@f$ 个基函数的分解树。
3. **代价函数**：最小化 @f$F(R)=w_1 n_b+w_2 n_t@f$，其中 @f$n_b@f$ 是
   存储的矩分量数量，@f$n_t@f$ 是乘法规则数量。顶层设置 `w1`、`w2`
   控制两项代价，当前默认值为 `1` 和 `4`。
4. **适应度**：Sigmoid 型函数
   @f$f(R) = 1 - 1/(1 + e^{-k(F(R) - F_{\min}) + c})@f$，
   其中 @f$F_{\min}@f$ 跟踪已找到的最佳解。
5. **操作**：变异随机替换一个或多个基因；交叉从另一亲代复制选定位置的基因。
   `elite_count` 大于零时保留精英；其默认值为 `0`，不会自动启用精英保留。
6. **动态剪枝**：每个中间张量在整个计算图中必须具有唯一的分解。
   当按顺序处理基函数时，与先前固定的分解不一致的树分支被剪除，在
   维持搜索空间效率的同时强制执行全局一致性。

**初始化**默认采用维度最小化筛选，再按子张量复用和等大小分解评分。
`randominit` 可改用随机初始猜测；GA 的 `initial_population` 默认是 `legacy`，
`diverse` 为可选的多样化初始种群。

### 实现

| 类/方法 | 作用 |
|-------------|------|
| `PickTensors::gen_alpha_pool_by_shapeev_level()` | 按级别阈值生成基组 |
| `PickTensors::gen_alpha_basic_index()` | 构建基本矩张量列表 |
| `PickTensors::gen_alpha_index_times()` | 构建缩并乘法规则 |
| `alpha_index_type` 结构体 | 通过精确置换检查等价性的指标矩阵 |
| `calc_level()` | 使用可配置权重计算矩张量的级别 |

`imrmlp mtmbasis` 使用 `PickTensors::gen_alpha_pool_by_shapeev_level()` 生成
alpha 池，然后通过 `PickTensors::gen_pot_by_alphapool_GA()` 搜索缩并树布局。
该命令将候选基组模板写为 `potential.mtp_<iteration>`，并写出所选缩并图的诊断文件。

**参见：** @ref basis 、`mtmbasis/basisgen.cpp`

## 系数布局

回归系数数组具有以下布局：

```
[径向系数 | 物种系数 | 矩系数]
```

| 分段 | 大小 | 描述 |
|---------|------|-------------|
| `radial_coeffs` | `species_count^2 * radial_func_count * rb_size` | 物种对径向系数 @f$c_{\mu,z_i,z_j}^{k}@f$ |
| `species_coeffs` | `species_count` | 每个物种的恒定能量偏移 |
| `moment_coeffs` | `alpha_scalar_moments` | 标量不变量的线性系数 |

## 力和应力计算

### 通过反向传播计算力

力是能量对原子位置导数的负值：

@f[
F_{j,a}=-\frac{\partial E}{\partial x_{j,a}} =
-\sum_i \frac{\partial E}{\partial b_i} \cdot
\frac{\partial b_i}{\partial x_{j,a}}
@f]

反向传播分为三个步骤：

1. **模型梯度**：从能量模型（线性或 NN）得到 @f$\partial E / \partial B_\alpha@f$
2. **递归反向传播**：对每条规则 @f$b_j\mathrel{+}=c_{jkl}b_kb_l@f$，
   向两个输入分别累加贡献（@f$k=l@f$ 时也要累加两次）：
   @f[
   g_k\leftarrow g_k+c_{jkl}g_j b_l,\qquad
   g_l\leftarrow g_l+c_{jkl}g_j b_k,\qquad
   g_j=\frac{\partial E}{\partial b_j}
   @f]
3. **Jacobian 乘法**：对于基本张量，乘以 @f$\partial b_i / \partial x_{j,a}@f$

### 应力计算

令 @f$D_{ij,a}=\partial E_i/\partial r_{ij,a}@f$。每条位点邻居导数对中心和邻居
分别贡献 @f$+\mathbf D_{ij}@f$ 与 @f$-\mathbf D_{ij}@f$，并累加内部维里：

@f[
S_{ab}=-\sum_i\sum_j D_{ij,a}\,r_{ij,b}.
@f]

若形变按 @f$r'_a=r_a+\sum_b\eta_{ab}r_b@f$ 作用，则
@f$S_{ab}=-\partial E/\partial\eta_{ab}@f$。
内部 `Atoms::stress_` 存储该广延张量，单位为 eV；与 GPa 或 LAMMPS 压力比较时，
还需按体积、单位及相应输出的符号约定转换。
短程 MTP_M 复用同一构造；长程模型还须包括显式晶胞依赖项。

## 系数训练

### 关于系数的梯度

对于径向系数 @f$c@f$，固定线性系数或外部模型参数后，能量梯度通过矩的
依赖关系传播：

@f[
\frac{\partial E}{\partial c} =
\sum_i \frac{\partial E}{\partial b_i} \cdot
\frac{\partial b_i}{\partial c}
@f]

线性系数的直接导数需另外计入。普通 MTP 的逐原子能量为
@f$E_i=\epsilon_{z_i}+\sum_\alpha\xi_\alpha B_\alpha(\mathfrak n_i)@f$，其中
@f$\epsilon_s@f$ 是物种 @f$s@f$ 的常数能量偏移，因此

@f[
\frac{\partial E_i}{\partial\xi_\alpha}=B_\alpha(\mathfrak n_i),\qquad
\frac{\partial E_i}{\partial\epsilon_s}=\delta_{z_i,s}.
@f]

总能量的导数再对各原子求和。

对于力的梯度，需要二阶混合导数：

@f[
\frac{\partial F_{j,a}}{\partial c_k} =
-\frac{\partial^2 E}{\partial x_{j,a}\partial c_k}
@f]

**参见：** @ref theory_training

## 神经网络扩展

本仓库的多通道与可选能量头形式由 `PotMTP_M` 实现，详见
[MTP_M：多通道与非线性能量头](mtp-m_zh.md)。它先投影出通道，再计算位点能量，
应与直接把全部不变量输入网络的一般形式区分。

模板方法 `CalcSiteEFS<Model>` 允许插入外部模型
（例如，神经网络）来替代线性能量表达式：

@f[
E_i = \mathcal{N}\!\left(\{B_\alpha(\mathbf{r}_i)\}_\alpha\right)
@f]

模型必须提供一个 `propagate()` 方法，返回 @f$E@f$、@f$\partial E/\partial B@f$，
以及可选的 @f$\partial^2 E/\partial B\partial B@f$。

**参见：** @ref nnbase

## 实现类

| 类 | 作用 |
|-------|------|
| `PotMTP` | 完整的 MTP 实现，支持训练 |
| `PotMTP_M` | 包含多个隐通道，支持线性、多项式或神经网络头的 MTP-M |
| `DescriptorMTM` | 包装为 `Descriptors` 的 MTP 描述符，用于 MaxVol |

**参见：** @ref potentials
