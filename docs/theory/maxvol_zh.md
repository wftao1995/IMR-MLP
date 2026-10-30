# MaxVol 算法与主动学习技术说明 {#theory_maxvol}

[English](maxvol.md)

@tableofcontents


## 1. 文档范围

MaxVol（Maximum Volume）从候选矩阵中选择一组尽可能线性独立的行，使这些行
张成的平行多面体体积尽可能大。在机器学习原子势（MLIP）中，候选行通常是
描述符，或能量、力、应力对势参数的导数。因而 MaxVol 可以同时用于：

- 构建数值条件较好的初始训练集；
- 判断新构型是否落在已有训练域之外；
- 在主动学习或 learning-on-the-fly（LOTF）中增量扩充训练集；
- 用较少的代表性构型覆盖很大的候选池。

首次选样可先按[训练与主动学习工作流](../guides/workflows_zh.md)操作；
单个选项可查[命令参考](../reference/commands_zh.md)。
本页解释这些命令背后的算法，包括经典方阵 MaxVol 以外的实现。内容包括经典算法、
矩形广义体积、B 矩阵构造、三种选择后端、外推等级、数值稳定性、复杂度、
命令行用法、性能调优和常见问题。

最重要的使用结论是：

1. 默认 `select-mode=stream`、`stream-mode=exact` 适合大多数离线选择；
2. `full` 适合中小矩阵、参考结果和调试；
3. `online` 内存最低，但不回看已经拒绝的历史候选，因此与输入顺序有关；
4. grade 是“更换主动行后体积能放大多少”的代数指标，不是能量误差、概率或
   置信区间；
5. B 矩阵定义、权重、原子数缩放和势参数化一旦改变，旧 grade 就不可直接比较。


## 2. 符号与问题定义

设候选行矩阵为

@f[
A =
\begin{bmatrix}
\mathbf a_1 \\
\mathbf a_2 \\
\vdots \\
\mathbf a_m
\end{bmatrix}
\in \mathbb R^{m\times n},
@f]

其中：

- @f$m@f$ 是候选 B 行数；
- @f$n@f$ 是描述符维数或参与选择的势参数数目；
- @f$I=(i_1,\ldots,i_k)@f$ 是选中行索引；
- @f$C=A[I,:]\in\mathbb R^{k\times n}@f$ 是主动矩阵；
- @f$\tau\ge 1@f$ 是用户阈值，默认 `1.001`；
- @f$s@f$ 表示实际发生的行交换数。

经典 MaxVol 取 @f$k=n@f$，目标是

@f[
I_* = \arg\max_{|I|=n} |\det A[I,:]|.
@f]

穷举需要检查 @f$\binom{m}{n}@f$ 个子矩阵，实际不可行。本实现采用带主元 QR
初始化和贪心单行交换，求得满足阈值的<b>单交换局部最优解</b>；不能把它理解成
对全局最大行列式的严格保证。


## 3. 基本数学原理

### 3.1 体积与 D-optimality

对方阵 @f$C\in\mathbb R^{n\times n}@f$，其行向量张成的体积为

@f[
\operatorname{vol}(C)=|\det C|=\prod_{j=1}^{n}\sigma_j(C),
@f]

其中 @f$\sigma_j@f$ 是奇异值。体积大意味着各方向不容易塌缩，通常也意味着
所选行具有较好的线性独立性。不过，最大体积和“最小条件数”不是完全等价的
优化问题，文档和日志中不应把二者混为严格等价。

若线性化模型写为

@f[
\mathbf y \approx B\boldsymbol\theta,
@f]

且观测噪声经过权重处理后可视为独立同方差，则 Fisher 信息矩阵或正规矩阵为
@f$F=B^\mathsf T B@f$。对于方形主动矩阵，

@f[
\det(C^\mathsf T C)=\det(C)^2.
@f]

因此最大化 @f$|\det C|@f$ 等价于最大化信息矩阵行列式，这就是 MaxVol 与
D-optimal experimental design 的联系：它倾向于减小参数置信椭球的体积，而
不是专门优化某一个能量或力误差。

### 3.2 候选行的展开系数

假设 @f$C@f$ 可逆。所有候选行相对于主动基的展开系数为

@f[
R = A C^{-1}\in\mathbb R^{m\times n},
\qquad
\mathbf a_l = \sum_{j=1}^{n}R_{lj}\mathbf c_j.
@f]

若用候选行 @f$\mathbf a_l@f$ 替换 @f$C@f$ 的第 @f$j@f$ 行，得到
@f$C'@f$，由矩阵行列式引理可得

@f[
\frac{|\det C'|}{|\det C|}=|R_{lj}|.
@f]

所以 @f$|R_{lj}|@f$ 既是展开系数，也是这次交换的精确体积放大倍数。经典
MaxVol 每轮寻找

@f[
(l_*,j_*)=\arg\max_{l\notin I,\,j}|R_{lj}|.
@f]

若最大值大于 @f$\tau@f$，就执行交换；否则停止。

### 3.3 收敛条件与局部最优性

经典方阵路径的停止条件为

@f[
\max_{l\notin I,j}|(AC^{-1})_{lj}|\le\tau.
@f]

当 @f$\tau=1+\varepsilon@f$ 时，任意一次单行替换最多只能把当前体积放大
@f$1+\varepsilon@f$。本仓库的用户参数是 `threshold`=@f$\tau@f$；底层
`MaxVol::tolerance` 则是 @f$\varepsilon=\tau-1@f$。

较小阈值意味着更严格的局部最优性、更多扫描和交换；较大阈值会更早停止。
推荐令 @f$\tau\ge1@f$。实现对小于 1 的停止阈值也会按 1 处理，因此设置成
小于 1 并不能要求“更严格于最大体积”的结果。

### 3.4 外推等级

对新行 @f$\mathbf b@f$，方阵主动集上的行级 grade 定义为

@f[
\gamma(\mathbf b;C)
=\|\mathbf b C^{-1}\|_\infty
=\max_j |(\mathbf b C^{-1})_j|.
@f]

对一个含多行 B 矩阵 @f$B_q@f$ 的原子构型 @f$q@f$，构型级 grade 是

@f[
\Gamma(q;C)=\max_{\mathbf b\in\operatorname{rows}(B_q)}
\gamma(\mathbf b;C).
@f]

其工程解释为：

- @f$\Gamma<1@f$：所有行都在当前主动基内部且展开系数较小；
- @f$\Gamma\approx1@f$：接近主动集边界；
- @f$\Gamma>\tau@f$：至少存在一次可显著增大体积的行替换；
- @f$\Gamma\gg1@f$：相对于当前 B 矩阵定义，构型具有很强的外推性。

grade 没有能量单位。它不直接给出能量/力误差，也不保证 grade 较小就没有模型
偏差：若描述符、势函数形式或候选池本身遗漏了某种物理状态，MaxVol 无法发现
这种表示之外的偏差。

### 3.5 矩形主动矩阵的广义体积

底层 `MaxVol::select_rows(A, k)` 支持 @f$k\ne n@f$。本实现定义

@f[
\operatorname{vol}(C)=
\begin{cases}
\sqrt{\det(CC^\mathsf T)}, & k\le n,\\
\sqrt{\det(C^\mathsf T C)}, & k\ge n.
\end{cases}
@f]

两种表达式都等于非零奇异值之积，并在 @f$k=n@f$ 时退化成
@f$|\det C|@f$。

当 @f$k<n@f$ 且 @f$C@f$ 满行秩时，记

@f[
G=CC^\mathsf T,
\quad
\boldsymbol\alpha=\mathbf bC^\mathsf T G^{-1},
\quad
r^2=\|\mathbf b\|_2^2-
\boldsymbol\alpha(\mathbf bC^\mathsf T)^\mathsf T.
@f]

用 @f$\mathbf b@f$ 替换第 @f$i@f$ 行的广义体积增益为

@f[
g_i^2=\alpha_i^2+r^2(G^{-1})_{ii}.
@f]

残差项 @f$r^2@f$ 很重要：即使候选行在已有行空间中的投影为零，只要它带来
新的正交方向，仍会得到较高分数。

当 @f$k>n@f$ 且 @f$C@f$ 满列秩时，令

@f[
\boldsymbol\alpha=\mathbf bC^+,
\quad h_b=\|\boldsymbol\alpha\|_2^2,
\quad h_i=\bigl(CC^+\bigr)_{ii},
@f]

则替换第 @f$i@f$ 行的增益为

@f[
g_i^2=(1-h_i)(1+h_b)+\alpha_i^2.
@f]

矩形路径用 @f$\max_i g_i@f$ 作为 row score，并执行精确的一行交换直至满足
同样的阈值。`ActiveSelector` 的普通 data 初始化通常取
@f$k=\min(m,n)@f$，所以用户最常见的仍是方阵情形；矩形公式主要服务底层 API、
数据不足和增广选择。


## 4. 从原子构型构造 B 矩阵

### 4.1 数据流与行到构型的映射

本仓库中的完整数据流为：

```text
Atoms 构型池
    │
    ▼
BMatrixBuilder ──► B 行 + BRowRef 元数据
    │
    ▼
full / stream / online MaxVol ──► 主动行矩阵 C
    │                                  │
    ├──► 去重后的 selected configs     ├──► grade
    └──► score file                    └──► .sel state
```

选择的基本单位是 <b>B 行</b>，输出的基本单位是 <b>原子构型</b>。一个构型可能贡献
多行，也可能有多行同时进入主动矩阵；输出时会按构型索引去重。因此：

- 主动行数通常接近参数/描述符维数 @f$n@f$；
- 输出构型数可以明显小于主动行数；
- `selected.xyz` 的构型数不是用户指定的固定采样数。

`BRowRef` 记录行所属构型、局部/全局行号、行类型、原子编号和分量，使在线交换、
状态恢复以及“行选择到构型输出”的映射保持一致。

### 4.2 描述符构造器

`dtype=mtm`、`behler`、`bondorder` 使用
`DescriptorBuilder`。设原子 @f$i@f$ 的局域描述符为
@f$\mathbf d(\mathcal N_i)\in\mathbb R^n@f$，构型原子数为 @f$N@f$，则共同
缩放因子为

@f[
s_N=\frac{w}{N^p},
@f]

其中 `weight`=@f$w@f$，`pow-scale`=@f$p@f$。

`mode=nbh` 时每个原子贡献一行：

@f[
\mathbf b_i=s_N\mathbf d(\mathcal N_i),
\qquad m_q=N.
@f]

`mode=cfg` 时一个构型只贡献一行：

@f[
\mathbf b_q=s_N\sum_{i=1}^{N}\mathbf d(\mathcal N_i),
\qquad m_q=1.
@f]

`nbh` 更敏感于罕见局域环境，但总行数约为总原子数；`cfg` 内存和评分成本更低，
但局域异常可能在求和中被平均或抵消。

### 4.3 势函数梯度构造器

`dtype=pot_efs`（别名 `mtp_energy`）使用 `PotEFSBuilder`。设参与选择的势参数为
@f$\boldsymbol\theta@f$，可以按非零分量权重生成以下行：

@f[
\begin{aligned}
\mathbf b_E       &\propto w_E\,\frac{\partial E}{\partial\boldsymbol\theta},\\
\mathbf b_{F,ia}  &\propto w_F\,\frac{\partial F_{ia}}{\partial\boldsymbol\theta},\\
\mathbf b_{S,ab}  &\propto w_S\,\frac{\partial S_{ab}}{\partial\boldsymbol\theta},\\
\mathbf b_{\text{site},i}
                   &\propto w_{\text{site}}\,
                     \frac{\partial E_i}{\partial\boldsymbol\theta}.
\end{aligned}
@f]

每个构型的行数为

@f[
m_q=
\mathbb 1_{w_E\ne0}
+3N\mathbb 1_{w_F\ne0}
+9\mathbb 1_{w_S\ne0}
+N\mathbb 1_{w_{\text{site}}\ne0}.
@f]

默认 `w-energy=1`，其余分量权重为 0，所以默认每个构型只有一条能量梯度行。
对 MTP，默认只使用线性系数；`all-coeffs=true` 才把径向等非线性系数也纳入
B 矩阵。线性系数通常已足以刻画标量基多样性，而且显著降低 @f$n@f$ 和选择成本。

### 4.4 权重和原子数缩放

`row-scaling=uniform` 时，所有启用的行都乘 @f$w/N^p@f$，再乘相应的
`w-energy`、`w-force`、`w-stress` 或 `w-site`。

`row-scaling=mlip` 只改变 `pot_efs` 的局域量缩放：

| 行类型 | `uniform` | `mlip` |
|---|---:|---:|
| energy | @f$w/N^p@f$ | @f$w/N^p@f$ |
| stress | @f$w/N^p@f$ | @f$w/N^p@f$ |
| force | @f$w/N^p@f$ | @f$w@f$ |
| site energy | @f$w/N^p@f$ | @f$w@f$ |

权重改变了几何度量，进而会改变选择结果和 grade。只有在 builder 类型、势参数
顺序、所有权重、缩放模式和 `pow-scale` 都一致时，两个状态的 grade 才可直接
比较。


## 5. 算法设计与实现

### 5.1 模块职责

| 类/入口 | 职责 |
|---|---|
| `MaxVol` | 通用稠密矩阵选行、矩形广义体积、评分、增广和底层 checkpoint |
| `OnlineMaxVolState` | 保存紧凑主动矩阵，执行批次式在线交换和无分配重复评分 |
| `BMatrixBuilder` | 把 `Atoms` 转换为 B 行及 `BRowRef` 元数据 |
| `ActiveSelector` | 连接构型、builder、选择后端、grade、MPI 和 `.sel` 状态 |
| `maxvol_select` | 从候选池构建主动集 |
| `maxvol_sample` | 用已有主动集只读评分 |
| `maxvol_extend` | 从 state 或训练集增量扩充主动集 |

### 5.2 初始化策略

#### data 初始化

`init-mode=data` 先对 @f$A^\mathsf T@f$ 做列主元 Householder QR，QR 的主元列
正好对应 @f$A@f$ 的候选行。它比随机初始化稳定、可复现，并提供较好的初始体积。

当 `init-mode=auto` 且 builder 不是 `PotEFSBuilder` 时，默认使用 data 初始化。

#### MLIP 虚拟单位阵初始化

`init-mode=mlip` 从

@f[
C_0=\delta I_n,
@f]

开始，其中 `init-value`=@f$\delta@f$，默认 @f$10^{-6}@f$。真实候选行通过
MaxVol 交换逐步替换虚拟行。没有被数据支撑的参数方向可以继续保留为虚拟行，
不会为了填满形式上的 @f$n@f$ 个方向而强行输出无意义构型。

虚拟行的 `global_row=-1`，不会映射到输出构型；命令结束时会报告
`virtual_rows`。`init-mode=auto` 对 `pot_efs` 默认选择该策略，其他 builder 默认
选择 data。@f$\delta@f$ 同时设定虚拟方向的尺度，调整它会改变初始 grade，除非
明确理解其量纲和缩放，否则建议保留默认值。

### 5.3 经典交换与快速更新

方阵路径的概念伪代码如下：

```text
I = pivoted_qr_rows(A, n)
C = A[I, :]
factorize C and form R = A C^{-1}

repeat:
    (l, j, gain) = argmax |R[l, j]| over non-active rows
    if gain <= threshold:
        converged
    replace active row j by A[l, :]
    update C^{-1}, R and log-volume by low-rank formulas
until an iteration/swap cap is reached
```

若每次交换都重新求逆并计算 @f$AC^{-1}@f$，代价很高。实现利用
Sherman--Morrison 公式更新逆矩阵和整个系数矩阵。若新旧行差为
@f$\mathbf d=\mathbf a_l-\mathbf c_j@f$，则

@f[
C'=C+\mathbf e_j\mathbf d,
@f]

从而可用秩一公式更新 @f$C^{-1}@f$；体积日志直接更新为

@f[
\log\operatorname{vol}(C')
=\log\operatorname{vol}(C)+\log|R_{lj}|.
@f]

### 5.4 三种 ActiveSelector 后端

| 后端 | 候选历史 | 收敛性质 | 主要存储 | 适用场景 |
|---|---|---|---|---|
| `full` | 完整 B 常驻内存 | 对当前完整矩阵做局部阈值检查 | 多个 @f$m\times n@f$ 工作区 | 中小池、基准和调试 |
| `stream/exact` | 完整 B 缓存在内存或 mmap | 反复重扫完整缓存；在扫描上限内可做全池阈值检查 | 分块缓存 + 紧凑 C | 默认、大池、MPI |
| `stream/fast` | 完整分块缓存 | 有界轮数和 shortlist，可能未收敛 | 分块缓存 + shortlist + C | 更重视吞吐的预筛选 |
| `online` | 只保留当前 C；拒绝的旧行不保留 | 顺序相关，不能证明对累计历史池满足阈值 | C、逆/Gram 因子和当前批次 | LOTF、只追加数据、低内存 |

`stream/exact` 的“exact”是指反复扫描缓存的完整候选池并在结束时做全池
post-check，而不是指找到了组合优化的全局最大体积。data 初始化路径每轮接收
一个确定性的全局最佳交换；MLIP 虚拟单位阵初始化会从一次全池扫描形成的全局
shortlist 中批量交换，再进入下一轮全池扫描。它和 `full` 都可能得到不同但都
满足单交换阈值的局部解，因为初始化及交换路径可以不同。

`stream-cache-mode` 的行为如下：

- `auto`：先放内存；每进程超过 `stream-memory-limit-mb` 后，新分块转入 mmap；
- `memory`：只用内存，超过上限时报错；
- `mmap`：全部放到临时映射文件；
- `off`：使用旧流式路径，可能在每轮重新构建描述符，通常只用于兼容或回退。

临时 mmap 文件在映射后立即 unlink，进程结束或映射释放后自动回收。目录由
`stream-cache-dir`、`TMPDIR`、`/tmp` 依次确定。

### 5.5 OnlineMaxVolState

在线状态保存 @f$C@f$、行引用和因子分解，不保存已经拒绝的历史候选。每个新
构型的 B 行只在当前批次内参与交换；被换出的主动行会留在当前局部池中，允许在
同一批次的后续交换重新进入。

首次流式初始化后会释放历史 B 行缓存。后续在线更新只构建追加构型，不再自动
重建并评分整个历史池；没有新增构型时不计算描述符。如需检查历史池，可显式调用
`compute_grades()` / `grade_all()`。保留完整流式缓存时，`compute_grades()` 复用
缓存行；紧凑在线状态则在收到这项显式请求时重新构建历史行。

当当前 @f$C@f$ 行数少于列数时，先用两遍正交化残差挑选新的独立方向；方阵且
良态时保存 @f$C^{-1}@f$，宽矩阵或秩亏时保存正则化的
@f$(CC^\mathsf T+\lambda I)^{-1}@f$。这使在线模式能从数据不足的状态逐步恢复
缺失方向。

代价是：后续交换可能让很早以前被拒绝的行重新变得外推，而在线状态无法回看
该行。因此在最终提交训练集之前，若要求对完整累计池满足阈值，应再运行
`stream/exact` 或 `full`。

### 5.6 数值稳定性

实现包含以下防护：

- QR 主元初始化，避免随机选择出明显退化的起点；
- 用 SVD 构造伪逆，截断阈值约为
  @f$\epsilon_{\rm mach}\max(k,n)\sigma_{\max}@f$；
- 秩亏的宽主动矩阵使用与谱尺度相关的正则项；
- Sherman--Morrison 分母非有限或绝对值小于 @f$10^{-14}@f$ 时完整重分解；
- 底层 `MaxVol` 默认每 20 次快速更新强制完整重算一次；
- 在线状态每 16 次更新抽样检查分解漂移，最多 128 次更新后强制重分解；
- 底层 `MaxVol` 的良态方阵使用带主元的 LU；保守的逆条件数检查确保病态或秩亏
  状态仍回退到 SVD 截断；
- 用 LU 主元或奇异值的对数之和保存 `log|det|`/广义 log-volume，避免行列式上溢或下溢；
- 输入包含 NaN/Inf、索引重复或维度不匹配时拒绝选择/评分。

对于严格秩亏矩阵，日志中的有限 `log|det|` 使用了奇异值下限或正则化尺度，主要
用于诊断，不应解释成真实非零行列式。

### 5.7 收敛状态与持久化

完整矩阵和带缓存的流式选择结束后会做全池 post-check，并记录：

- `converged`：剩余最大 grade 是否满足阈值；
- `max_remaining_grade`：post-check 得到的最大构型/行 grade；
- `termination_reason`：`threshold`、`max-swaps`、`fast-pass-limit`、
  `postcheck-failed` 或 `online-approximate` 等；
- `virtual_rows`：MLIP 初始化后仍留在 C 中的虚拟行数。

在线增量更新不检查历史池，因此返回 `converged=false`、
`max_remaining_grade=inf`、`termination_reason=online-approximate`；触及交换上限时
原因是 `max-swaps`。这里的无穷大表示全池最大值未知，不是测得的外推评分。
可以显式评分整个池来检查结果，或重新执行 `full` / `stream/exact`，得到带全池
收敛检查的选择结果。在线模式首次使用流式构建时仍报告该次全池检查结果。

达到上限但未收敛的 state 仍会保存；后续 `sample`/`extend` 会发出警告。不要只看
“已写出 selected.xyz”，还应检查 `Selection status` 和最大剩余 grade。

当前 `ActiveSelector` state 格式为 v10，扩展名为 .sel 的文件保存紧凑主动矩阵、行元数据、
训练构型、已选构型、builder 元数据、缩放参数、在线进度和收敛信息。运行时分解
缓存会在加载后重建。状态适合断点续传，但不是跨不同势参数化的通用模型文件。


## 6. 算法特点与边界

### 6.1 优点

- <b>与模型线性化直接相关</b>：使用梯度 B 矩阵时，选择目标直接对应参数信息量；
- <b>无需标签即可预选</b>：描述符或模型梯度通常不需要候选的 DFT 能量/力；
- <b>外推判据统一</b>：选择和在线检测使用相同的体积增益；
- <b>可解释</b>：每次交换都有进入行、离开行和体积增益；
- <b>可增量更新</b>：在线状态适合 LOTF；
- <b>支持大池</b>：缓存流式、mmap、OpenMP 和 MPI 减少内存或构建时间压力；
- <b>确定性较强</b>：相同输入顺序和运行模式下使用确定性的 QR/全局 winner 规则。

### 6.2 局限

- 返回单行交换局部最优，而非组合意义的全局最大体积；
- 对列尺度敏感；未经物理权重或标准化的列可能主导选择；
- grade 只能衡量当前表示和当前线性化中的外推，不能检测模型形式误差；
- 非线性势参数更新后，原梯度矩阵可能失效，需要重建；
- `online` 与输入顺序相关；`fast` 可能因轮数上限而未收敛；
- 很大的 @f$n@f$ 会带来二次乃至三次线性代数成本；
- 同一构型贡献多行时，选中行数与输出构型数不是一一对应。

对 MTP 而言，若只更新线性系数，旧的线性 B 行仍有明确意义；若径向基等影响
描述符/梯度的非线性参数发生变化，应完整重建主动集，而不是只追加旧 state。


## 7. 运行效率与复杂度

### 7.1 主要维度

性能主要由以下量决定：

- @f$N_{\rm cfg}@f$：构型数；
- @f$m=\sum_q m_q@f$：总 B 行数；
- @f$n@f$：B 列数；
- @f$k@f$：主动行数，常见值为 @f$n@f$；
- @f$s@f$：交换次数；
- @f$q@f$：对完整行缓存执行的评分扫描次数；
- @f$T_{\rm build}@f$：邻居表、描述符或势梯度的构建时间。

在科学计算数据上，`pot_efs` 的梯度构建或复杂描述符计算常常比单次 MaxVol
扫描更贵，所以“描述符只构建一次”与减少 @f$m,n@f$ 同样重要。

### 7.2 底层方阵 MaxVol

对 @f$m\gg n@f$、@f$k=n@f$：

- QR 初始化和初次系数矩阵计算：约 @f$O(mn^2+n^3)@f$；
- 每次最佳元素搜索：@f$O(mn)@f$；
- 每次 Sherman--Morrison 更新：@f$O(mn+n^2)@f$；
- 每隔固定次数的完整重分解：@f$O(mn^2+n^3)@f$；
- 主工作内存：@f$O(mn+n^2)@f$。

因而在快速更新有效时，总成本近似为

@f[
O(mn^2+s\,mn),
@f]

另加周期性完整重分解。矩形路径每次广义交换需要重新形成相关投影，典型代价约
为 @f$O(mnk)@f$，因此比方阵秩一更新更重。

### 7.3 各后端的成本

设一次完整评分扫描需要把 @f$m\times n@f$ 候选矩阵乘以 @f$n\times k@f$
系数映射：

| 后端 | 近似时间 | 主要内存/磁盘 |
|---|---|---|
| `full` | @f$T_{\rm build}+O(mn^2+s\,mn)@f$，另有重分解 | RAM 中多个 @f$O(mn)@f$ 矩阵 |
| `stream/exact` | @f$T_{\rm build}+O(qmnk)@f$ | RAM/mmap 缓存约 @f$8mn@f$ 字节，加 @f$O(kn+k^2)@f$ |
| `stream/fast` | @f$T_{\rm build}+O(pmnk)@f$，@f$p@f$ 为有界 pass 数 | 同上，外加有界 shortlist |
| `online` | 每批次构建并评分；典型方阵评分 @f$O(m_qn^2)@f$ | @f$O(n^2+m_qn)@f$，不保存历史池 |

data 初始化的 exact 路径通常有 @f$q\lesssim s+1@f$；MLIP shortlist 路径可在
一次扫描中接受多次交换。默认 exact 安全上限为 @f$q=100@f$。这些式子只给
数量级。Eigen/MKL 的矩阵乘法、builder 成本、缓存带宽、mmap I/O、
OpenMP 调度和 MPI 通信都会影响实际墙钟时间。

仅一个 @f$m\times n@f$ double 矩阵就需要 @f$8mn@f$ 字节。例如
@f$m=10^6,n=1000@f$ 时约为 8 GB（约 7.45 GiB），而 `full` 还会有副本和工作区；
这类规模应优先使用 `stream` + mmap/MPI，或先从建模上减少行数和列数。

### 7.4 并行设计

- <b>OpenMP</b>：builder 可克隆时，B 行构建和分块评分按结构/分块并行；内部将
  MKL 局部线程数设为 1，避免 OpenMP 与 BLAS 过度嵌套；
- <b>MPI full 构建</b>：`add_candidates_mpi()` 按构型索引分片构建，rank 0 重组
  完整 B、选择并广播紧凑评分状态；
- <b>MPI stream</b>：构型按 `config_index % mpi_size` 分布，每个 rank 只缓存本地
  B 行；exact 模式归约确定性的全局 winner 并广播进入行，避免在 rank 0 聚合
  完整矩阵；
- <b>MPI fast</b>：每个 rank 提交有界 shortlist，再形成全局 shortlist。

MPI stream 的行缓存理想情况下约按 rank 数分摊，但所有 rank 都会复制紧凑主动
矩阵和因子。exact 模式每次交换仍需要一次全局 winner 通信，因此交换很多且每个
rank 本地工作很小时，通信会限制扩展性。

### 7.5 性能调优顺序

建议按下列顺序优化：

1. 先确认 `dtype` 和行类型确实服务于目标；不需要的 force/site/stress 权重设为 0；
2. MTP 优先保留默认线性系数，只有确有需要时才用 `all-coeffs=true`；
3. 能接受构型级信息时用 `mode=cfg`，需要局域异常检测时再用 `mode=nbh`；
4. 大池采用 `stream/exact`，内存不足用 `auto` 或 `mmap`；
5. 调整 `stream-chunk-size`，使单块足够大以摊薄调度成本，又不造成临时峰值过高；
6. 用 `--num_threads` 并行 builder，超大池再使用 MPI stream；
7. 只做预筛选时才用 `stream/fast`，并检查最终收敛状态；
8. LOTF 的严格低内存路径可用 `online`，阶段结束后用完整累计池复核。


## 8. 使用方法

### 8.1 构建

```bash
make imrmlp
```

默认可执行文件位于 `build/release/imrmlp`。查看当前版本实际支持的参数：

```bash
build/release/imrmlp maxvol_select --help
build/release/imrmlp maxvol_sample --help
build/release/imrmlp maxvol_extend --help
```

三个命令的设置文件是位置参数；未显式给出时分别使用 `select.ini`、`sample.ini`
和 `select_add.ini` 作为默认文件名。设置文件中的共享选项推荐使用 `builder:` 和
`selector:` 命名空间。

### 8.2 从构型池选择主动集

一个面向 MTP 标量描述符的 `select.ini` 示例：

```ini
builder:dtype mtm
builder:potfile potential.mtp
builder:pottype mtp
builder:mode nbh
builder:row-scaling uniform

selector:threshold 1.001
selector:weight 1.0
selector:pow-scale 0.5
selector:init-mode auto
selector:max-swaps 0
selector:select-mode stream
selector:stream-mode exact
selector:stream-cache-mode auto
selector:stream-memory-limit-mb 3072
selector:stream-chunk-size 32
selector:verbose true
```

运行：

```bash
build/release/imrmlp maxvol_select select.ini \
  --pool pool.xyz \
  --output selected.xyz \
  --score-file maxvol_scores.dat
```

命令会：

1. 写出唯一的入选构型到 `selected.xyz`；
2. 写出每个池构型的 `idx grade selected` 到 `maxvol_scores.dat`；
3. 自动把 state 保存为 `potfile + ".sel"`，本例为 `potential.mtp.sel`；
4. 打印选中构型数、log-volume、收敛状态、最大剩余 grade 和虚拟行数；
5. `verbose=true` 且使用 stream 时额外打印构建、初始化、评分时间、扫描次数、
   交换数、线程数以及 RAM/mmap 字节数。

若希望选择与势参数敏感度直接对应，可改用梯度 builder：

```ini
builder:dtype pot_efs
builder:potfile potential.mtp
builder:pottype mtp
builder:all-coeffs false
builder:row-scaling mlip
builder:w-energy 1.0
builder:w-force 0.1
builder:w-stress 0.0
builder:w-site 0.0

selector:init-mode auto
selector:init-value 1e-6
selector:select-mode stream
selector:stream-mode exact
selector:threshold 1.001
```

这里每个构型会产生 1 条能量行和 @f$3N@f$ 条力行。`w-force=0.1` 只是示例，
实际值应与训练目标、单位和噪声模型一致，不能把不同量纲的默认数值机械混合。

### 8.3 使用已有主动集评分

`maxvol_sample` 不改变主动集，只输出 grade 超过保存阈值的构型：

```bash
build/release/imrmlp maxvol_sample sample.ini \
  --potfile potential.mtp \
  --state-file potential.mtp.sel \
  --input candidates.xyz \
  --output sampled.xyz \
  --threshold-save 2.1 \
  --threshold-break 11.0
```

默认行为：

- grade @f$>2.1@f$ 的构型写入输出；
- 输出构型增加 `MV_grade` feature；
- 顺序扫描中一旦遇到 grade @f$>11.0@f$ 的构型就停止，后续构型不再处理；
- 这两个采样阈值是工作流策略，不是选择时的 `selector:threshold=1.001`。

评分前应使用与 state 一致的 potential、builder、系数集合、权重和缩放。尤其不要
用 `all-coeffs=false` 生成 state，再用 `all-coeffs=true` 的 B 行解释它。

检查或导出 state：

```bash
build/release/imrmlp maxvol_sample sample.ini \
  --state-file potential.mtp.sel --dump-state true

build/release/imrmlp maxvol_sample sample.ini \
  --state-file potential.mtp.sel \
  --extract-configs state_selected.xyz --selected-only true
```

### 8.4 增量扩充

推荐从已有 state 扩充：

```bash
build/release/imrmlp maxvol_extend select_add.ini \
  --potfile potential.mtp \
  --state-file potential.mtp.sel \
  --candidates new_pool.xyz \
  --output new_selected.xyz
```

输出只包含本轮新进入主动集的候选构型，state 会原地更新。还可以用 `--train`
加入 state 之外的额外训练构型。

没有 state 时可执行三段式流程：

```bash
build/release/imrmlp maxvol_extend select_add.ini \
  --potfile potential.mtp \
  --train train.xyz \
  --candidates new_pool.xyz \
  --output new_selected.xyz
```

该模式先从训练集建立主动集，再评估新候选，最后对合并数据重新稳定。若
`select-mode=online` 或 `stream`，实现会对训练和候选的合并序列使用相应后端。

### 8.5 MPI 大池选择

MPI build 下，离线 `maxvol_select` 在多 rank 且 `select-mode=stream` 时自动使用
分布式 stream：

```bash
mpirun -np 4 build/mpi/imrmlp-mpi maxvol_select select.ini \
  --pool huge_pool.xyz --output selected.xyz
```

共享输出和 state 只由 rank 0 写入。每个 rank 的
`stream-memory-limit-mb` 是<b>每进程</b>上限，不是所有 rank 的总和。

### 8.6 C++ 底层 API

若已经有数值矩阵，可直接使用 `MaxVol`：

```cpp
#include "maxvol/maxvol.h"

Eigen::MatrixXd A = /* m x n design matrix */;

MaxVol mv;
mv.tolerance = 1e-3;       // stop at gain <= 1.001
mv.max_swaps = 0;          // no extra swap cap
mv.verbose = false;

const int k = std::min<int>(A.rows(), A.cols());
std::vector<int> selected = mv.select_rows(A, k);
Eigen::VectorXd row_scores = mv.compute_scores(A, selected);

double log_volume = mv.log_determinant();
const auto &history = mv.swap_history();
```

只评分新矩阵时，先用同一主动集和原矩阵准备缓存：

```cpp
mv.warm_cache(selected, A);
Eigen::VectorXd candidate_scores = mv.score_rows(B_candidate, selected);
```

`score_rows()` 不会在缺少匹配缓存时偷偷重新分解；缓存未准备、列数变化、主动行
索引变化或主动行数值变化时会返回空向量。高层 `ActiveSelector::prepare_grading()`
负责完成对应准备工作。


## 9. 关键参数

### 9.1 Builder 参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `dtype` | `mtm` | `mtm`、`pot_efs`、`behler`、`bondorder` |
| `potfile` | `potential.mtp` | 势文件；MTM 和梯度 builder 需要 |
| `pottype` | `mtp` | 势类型 |
| `mode` | `nbh` | 描述符逐原子或逐构型；主要用于 descriptor builder |
| `all-coeffs` | `false` | MTP 梯度是否包含全部系数 |
| `row-scaling` | `uniform` | `uniform` 或 `mlip` |
| `w-energy` | `1.0` | 能量梯度行权重；0 表示不生成 |
| `w-force` | `0.0` | 力梯度行权重；0 表示不生成 |
| `w-stress` | `0.0` | 9 个应力梯度行权重；0 表示不生成 |
| `w-site` | `0.0` | 位点能量梯度行权重；0 表示不生成 |

### 9.2 Selector 参数

| 参数 | 默认值 | 含义与注意事项 |
|---|---:|---|
| `threshold` | `1.001` | 用户停止阈值 @f$\tau@f$ |
| `weight` | `1.0` | B 行共同权重 |
| `pow-scale` | `0.5` | 原子数缩放指数 @f$p@f$ |
| `init-mode` | `auto` | `auto`、`mlip` 或 `data` |
| `init-value` | `1e-6` | 虚拟单位阵尺度 @f$\delta@f$ |
| `max-swaps` | `0` | 额外交换上限；0 表示不设置此上限 |
| `select-mode` | `stream` | `stream`、`full` 或近似 `online` |
| `stream-mode` | `exact` | `exact` 或 `fast` |
| `stream-chunk-size` | `32` | 每个构建/缓存分块的构型数 |
| `stream-fast-passes` | `2` | fast 模式最多全局扫描轮数 |
| `stream-fast-shortlist` | `0` | 0 为自动；否则是全局 shortlist 大小 |
| `stream-cache-mode` | `auto` | `auto`、`memory`、`mmap`、`off` |
| `stream-memory-limit-mb` | `3072` | 每进程常驻行缓存上限 |
| `stream-cache-dir` | 空 | mmap 临时目录；空时用 `TMPDIR` 或 `/tmp` |
| `verbose` | `false` | 输出交换和 stream 性能统计 |

`max-swaps=0` 只表示没有用户设置的交换数上限。底层 `MaxVol` 和 exact stream
仍有默认 100 轮的安全上限；若在此之前未满足 post-check，state 会标记为
`postcheck-failed`。当前 CLI 没有单独暴露底层 `max_iterations`。


## 10. 结果解释与推荐工作流

### 10.1 初始训练集

1. 用物理合理的结构生成候选池；
2. 用 `stream/exact` 做无标签 MaxVol 选择；
3. 只对 `selected.xyz` 做昂贵的 DFT 标注；
4. 训练势并检查独立验证集；
5. 用该势重新构建梯度型主动集时，保持权重和系数集合有记录可追溯。

### 10.2 迭代主动学习

```text
初始池 ─► maxvol_select ─► DFT 标注/训练
                              │
                              ▼
新轨迹/候选 ─► maxvol_sample ─► 高 grade 构型
                                      │
                                      ▼
                              DFT + maxvol_extend
                                      │
                                      └──► 重新训练并循环
```

实践中常使用两个策略阈值：较低的 `threshold-save` 收集候选，较高的
`threshold-break` 立即终止不可信轨迹并触发高精度计算。阈值应通过具体体系的
误差--grade 相关性标定，默认的 2.1/11.0 不是对所有材料都通用的物理常数。

### 10.3 应记录的可复现实验信息

- 势文件版本和系数顺序；
- `dtype`、`mode`、各分量权重、`row-scaling`、`pow-scale`；
- 选择后端、阈值、初始化模式和随机性相关设置；
- 输入构型顺序，尤其是 `online`；
- MPI rank 数、OpenMP 线程数和 stream cache 后端；
- `converged`、`termination_reason`、最大剩余 grade、虚拟行数和交换数；
- `.sel` state 与最终选中构型。


## 11. 常见问题与排查

### 11.1 选中构型数为什么小于系数数？

MaxVol 选的是行，不是构型。`nbh`、force 或 site 模式下一个构型会贡献多行；
多个主动行可来自同一构型，最终输出会去重。MLIP 初始化还可能保留不对应真实
构型的虚拟行。

### 11.2 为什么两个模式选出的结构不同？

最大体积子集可能不唯一。`full`、`stream/exact` 使用不同初始化/交换路径；
`online` 又有顺序依赖。应优先比较是否满足阈值、验证误差和覆盖效果，而不是要求
索引逐项相同。

### 11.3 为什么 grade 全部很大？

依次检查：

1. state 与当前 builder 的列数、系数顺序是否一致；
2. potential 是否已经改变了非线性描述符/径向参数；
3. `weight`、`pow-scale`、`row-scaling` 和分量权重是否一致；
4. `init-value` 是否被改得过小；
5. state 是否未收敛或包含大量虚拟方向；
6. 候选是否真的含训练集未覆盖的新化学环境。

### 11.4 为什么 grade 返回极大值？

高层评分在构型没有生成任何 B 行、缓存未准备、维度不匹配或出现非有限分数时
使用 `1e10` 作为失败/强外推哨兵值。检查权重是否把所有行都关闭、builder 是否
成功构造以及 state 是否与当前 potential 兼容。

### 11.5 为什么 `log|det|` 很负？

体积带单位并随行/列缩放指数变化；当行数多、单位尺度小或存在虚拟
@f$10^{-6}I@f$ 行时，负的 log-volume 很正常。它只适合在相同 @f$k,n@f$、相同
builder 和相同缩放下比较，不应跨设置当作通用质量分数。

### 11.6 如何判断 fast/online 结果能否直接使用？

查看 `Selection status`。若 `termination_reason=fast-pass-limit`、
`online-approximate` 或 `max-swaps`，说明没有证明完整历史池满足阈值。可把累计池
交给 `stream/exact` 做最终复核。


## 12. 测试与验证

本仓库的相关 Catch2 测试位于：

- `unitest/ut_maxvol.cpp`：方阵/矩形选择、单交换占优、评分、缓存、增广、序列化；
- `unitest/ut_online_maxvol.cpp`：在线评分工作区、秩亏恢复、宽矩阵残差、
  保存/加载继续运行；
- `unitest/ut_active_selector.cpp`：构型映射、compact grading state、stream/MPI
  相关高层行为。

修改 MaxVol 核心或 builder 后应运行：

```bash
make test
```

性能评估至少应分别记录 `build_seconds`、`initialization_seconds`、
`score_seconds`、`score_passes`、`swaps`、常驻/映射字节数和总时间。只报告总时间
难以判断瓶颈在描述符、线性代数、I/O 还是 MPI 通信。

`score_seconds` 和 `score_passes` 包含最终的并行全池检查。评分按 B 行分块，
独立于构型数和缓存分块，因此单个大构型也能使用多个线程。每个线程的系数/Gram
评分中间矩阵共享 4 MiB 预算（每个矩阵至少容纳一行）；主动矩阵、构建器缓冲、
行缓存和输出评分另占空间。`stream-memory-limit-mb` 仅约束常驻行缓存，不能作为
进程 RSS 上限。线性 MTP 的 E/F/S 行采用批量分量 Jacobian，通用势及非线性
系数保留逐分量梯度路径。


## 13. 与 FPS、CUR 的区别

| 方法 | 选择准则 | 是否依赖当前主动集 | 典型用途 |
|---|---|---|---|
| MaxVol | 最大化子矩阵体积/单行替换增益 | 是 | D-optimal 主动学习和外推检测 |
| FPS | 最大化到已选集合的最小描述符距离 | 是 | 几何均匀覆盖、固定数量采样 |
| CUR | SVD 杠杆分数 | 可一次性从全矩阵计算 | 代表性降采样和低秩重构 |

`diverse_select --method=fps|cur` 适合“明确选固定数量结构”的数据压缩任务；
`maxvol_select` 的自然目标是让主动行达到局部体积阈值，其输出构型数不是固定值。
三者可以组合使用，例如先用 FPS/CUR 把极大池压缩，再用 MaxVol 构造与势参数
线性化相关的主动集。

### 13.1 CUR 分解

CUR 分解是 MaxVol 的互补行选择策略。MaxVol 最大化选中子矩阵的行列式，而
CUR 基于奇异值分解（SVD）得到的**统计杠杆分数（leverage scores）**选择行。

#### 算法

给定描述符矩阵 @f$\mathbf{A} \in \mathbb{R}^{m \times n}@f$：

1. 计算薄 SVD：@f$\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T@f$；
2. 取杠杆秩 @f$r = \min(k, n)@f$，其中 @f$k@f$ 是要选择的行数；
3. 计算每一行 @f$i@f$ 的杠杆分数：
   @f[
   p_i = \frac{\|\mathbf{U}(i, 1:r)\|^2}{r}
   @f]
   其中 @f$\mathbf{U}(i, 1:r)@f$ 是 @f$\mathbf{U}@f$ 第 @f$i@f$ 行的前 @f$r@f$ 个分量；
4. 用两种策略之一选择 @f$k@f$ 行：
   - **概率式**：无放回加权随机采样，行 @f$i@f$ 被采到的概率正比于 @f$p_i@f$；
   - **确定式**：按杠杆分数降序取前 @f$k@f$ 行。

#### 重构保证

杠杆分数采样下的 CUR 分解具有可证明的误差界：

@f[
\|\mathbf{A} - \mathbf{C} \mathbf{C}^+ \mathbf{A}\|_F \leq (1 + \epsilon) \|\mathbf{A} - \mathbf{A}_k\|_F
@f]

其中 @f$\mathbf{C}@f$ 是选中的行子矩阵，@f$\mathbf{A}_k@f$ 是最佳
秩-@f$k@f$ 近似。该保证确保选中行捕获描述符矩阵中的主要谱信息。

#### 实现

`cur_select_rows()`（位于 `maxvol/cur_selection.{h,cpp}`）：

```cpp
CURResult cur_select_rows(
    const Eigen::MatrixXd& A,
    int k,
    bool deterministic = false,
    int seed = 42,
    bool compute_error = false
);
```

`CURResult` 结构体包含：
- `selected_indices` — 被选中的 @f$k@f$ 行索引；
- `scores` — 每行的杠杆分数；
- `reconstruction_error` — 可选的**相对**重构误差
  @f$\|\mathbf{A} - \mathbf{C}\mathbf{C}^+\mathbf{A}\|_F / \|\mathbf{A}\|_F@f$。

#### CUR 与 MaxVol 对比

| 方面 | MaxVol | CUR |
|--------|--------|-----|
| **准则** | 最大化 @f$\det(\mathbf{C})@f$ | 最大化杠杆分数 |
| **算法** | 贪心行交换迭代 | SVD + 分数选择 |
| **确定性** | 确定式 | 确定式或概率式 |
| **误差界** | 经验性 | 可证明的重构保证 |
| **最适合** | 良态主动集 | 大池的代表性子集 |
| **复杂度** | 每次交换 @f$O(m \cdot n^2)@f$ | @f$O(\min(m,n)^2 \cdot \max(m,n))@f$（SVD） |
| **CLI 入口** | `maxvol_select`、`maxvol_sample` | `diverse_select --method cur` |


## 14. 参考资料与代码入口

理论参考：

1. S. A. Goreinov, E. E. Tyrtyshnikov, N. L. Zamarashkin,
   “A theory of pseudoskeleton approximations,” *Linear Algebra and its
   Applications* 261, 1--21 (1997),
   [DOI: 10.1016/S0024-3795(96)00301-1](https://doi.org/10.1016/S0024-3795(96)00301-1).
2. S. A. Goreinov, I. V. Oseledets, D. V. Savostyanov, E. E. Tyrtyshnikov,
   N. L. Zamarashkin, “How to Find a Good Submatrix,” in *Matrix Methods:
   Theory, Algorithms and Applications*, 247--256 (2010),
   [DOI: 10.1142/9789812836021_0015](https://doi.org/10.1142/9789812836021_0015).
3. E. V. Podryabinkin, A. V. Shapeev, “Active learning of linearly
   parametrized interatomic potentials,” *Computational Materials Science*
   140, 171--180 (2017),
   [DOI: 10.1016/j.commatsci.2017.08.031](https://doi.org/10.1016/j.commatsci.2017.08.031).

主要代码入口：

- `maxvol/maxvol.{h,cpp}`；
- `maxvol/online_maxvol.{h,cpp}`；
- `maxvol/b_matrix_builder.{h,cpp}`；
- `maxvol/active_selector.{h,cpp}`；
- `apps/maxvol_select.cpp`；
- @ref theory_training （训练理论）；
- @ref theory_mtp （MTP 理论）。

**参见：** @ref maxvol, @ref theory_training
