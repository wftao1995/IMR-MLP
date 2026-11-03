# 约束下的弛豫计算

[English](relax-controls.md)

结构里总有些部分不该动：固定的基底、只允许面内松动的表面、必须守住的键长键角，
或者外压下要达到的平衡构型。这些要求看起来都是"给弛豫加个限制"，在代码里却落在
完全不同的位置。混着用的时候，多半只会换来一条点名冲突双方的报错，至于为什么冲突，
还得回源码里找。本页按**约束作用在哪一层**来组织：从删自由度、换坐标，一直到把约束
写进目标函数；文件格式、互斥关系和能直接照抄的例子放在后面。

读这一页之前，应该已经能用[结构弛豫](../guides/relaxation_zh.md)跑通固定胞和变胞弛豫，
清楚 `gtol`、`isif` 和 `relax:driver` 各自管什么。键的类型、默认值与别名不在这里重复，
见[配置参考](../reference/configuration_zh.md)；命令行开关见
[命令参考](../reference/commands_zh.md)的 `relax` 一节。`relax:keep_symm` 的机制、
修正量判据与退出码优先级，另有[对称性约束弛豫](relax-symmetry_zh.md)专讲。

| 我要做的事 | 主要控制 | 详见 |
| --- | --- | --- |
| 固定基底、只放开面内、逐方向放开 | `--position-mask`、`--filter`、POSCAR 掩码 | 自由度掩码 |
| 只放开部分应变分量 | `--cell-mask`、`relax:strain_mask` | 自由度掩码 |
| 施加压力或应力张量 | `relax:pressure`、`relax:external_stress` | 目标函数里的约束，外部载荷一节 |
| 键长、键角、体积、二面角必须满足给定关系 | `--file-constraints` | 声明式几何约束 |
| 只沿指定振动模式弛豫 | `relax:vibffilter`、`relax:with_constraint_vib` | 坐标变换 |
| 保持初始空间群 | `relax:keep_symm` | [对称性约束弛豫](relax-symmetry_zh.md) |

## 约束作用在哪一层

约束说到底是在限定"哪些构型算合法"。但限制加在不同位置，能观察到的东西差别很大。
比起记住一串开关名，先判断一个控制项属于哪一层往往更有用：只有写进目标函数的约束
会留下残差，其余各层的效果，只能从输出结构、掩码标志和梯度上间接看出来。

| 层 | 作用对象 | 代表控制 | 有残差可查 | 失败时的表现 |
| --- | --- | --- | --- | --- |
| 变量空间，删分量 | 优化向量本身 | 掩码族、`fix_shape`、`fix_vol`、`couple`、`periodic_subspace`、`rigid_groups`、`keep_symm` | 无 | 该方向仍有力，结构却不动 |
| 变量空间，换坐标 | 优化向量的基 | LC、LF、VIB、VIB2 | 无 | 维度写错，直接报错或崩溃 |
| 梯度 | 搜索方向 | `fixcm`、`freeze_small_gd`、`symm_force` | 无 | 梯度收敛但结构不合理 |
| 目标函数 | 残差与乘子 | `--file-constraints` 配 AUL、SQP 或 NLC | 有 | 残差不降，退出码 `2` |
| 外部条件 | 平衡条件 | `relax:pressure`、`relax:external_stress` | 无（看最终应力） | 收敛到错误的应力 |

选型时按这个顺序往下走。先看这个自由度能不能直接删掉——能删就是精确的，既没有容差要调，
也不拖慢收敛。删不掉，但换一组坐标就行，用 LC、LF 或 VIB。两样都不行，才轮到把约束
写进目标函数，例如"键长必须等于 0.98 Å"这种退不掉的关系，代价是要多养一个求解器。

压力和应力不算约束。它们改的是"什么算平衡"，进不了残差，也就不会把退出码变成 `2`。
反过来，凡是带残差的约束，都要求优化器同时盯着目标与约束，求解器一旦选错会当场报错：
带上下界的约束只在 `driver sqp` 或 `nlc` 下可用。

## 自由度掩码：逐分量的取舍

掩码的做法是把分量从优化向量里拿掉，而不是给它加一条约束。所以它没有残差，没有容差，
也无从放松；想知道它有没有生效，只能去输出结构里看掩码标志，或者看被冻结方向上
还留着多少力。

### 五个入口

| 入口 | 粒度 | 数值形式 | 生效条件 |
| --- | --- | --- | --- |
| POSCAR `Selective dynamics` | 逐原子三分量 | `T`/`F` | 需 `relax:selective_dynamics true` |
| `--position-mask` | 全部原子共用的三分量 | 严格 `0`/`1` | 需原子弛豫，拒绝 `isif` 5 到 7 |
| `--cell-mask`、`relax:strain_mask`、`relax:lattice_mask` | 晶胞坐标 | 6 或 9 个有限数，大于 0 即视为可动 | 需 `relax:relax_cell true` 且 `fix_shape=false` |
| `--filter`、`relax:seldyn_file` | 逐分量，含晶胞坐标 | `n` 加恰好 n 个字面 `0`/`1` | 需 `relax:selective_dynamics`；`--filter` 自动打开 |
| `--seldyn` | 逐帧、逐原子加 3×3 晶胞 | 块格式 | 仅当 `relax:selective_dynamics=true` 时读取 |

`--cell-mask`、`relax:strain_mask` 和 `relax:lattice_mask` 写的是同一个设置。六值形式的
顺序是 `xx yy zz yz xz xy`，只有在 `relax:cell_strain` 与 `relax:symm_strain` 同时为真
（默认如此）时才合法；否则得给九个行主序分量，对应原始晶格的 3×3。值不限于 0 和 1，
任何有限数，大于 0 就算可动。走六值模式时，逐帧给出的 3×3 晶胞掩码不参与计算，
起作用的只有这里的全局掩码。

`--position-mask` 的三个标志对所有原子一视同仁，表达不了"底层全固定、顶层放开面内"。
它作用在优化坐标上，而优化坐标是笛卡尔还是分数，取决于 `relax:cartesian` 与
`relax:atomic_force_recast`，所以 `1 1 0` 未必就是实验室坐标里的 xy 面。需要逐原子区分时，
只能走 `--filter` 或结构文件里的掩码。

`--filter` 和 `relax:seldyn_file` 用的是同一种格式，命令行给的文件会覆盖设置文件里的路径。
首行那个 n 是**进入任何过滤器之前**的全维度：先是晶胞坐标，长度取决于坐标模式
（不变胞 0、`fix_shape` 1、对称应变 6、其余 9，`periodic_subspace` 则是 `rank(rank+1)/2`），
接着是 3N 个原子坐标。固定胞的四原子结构因此写成 12。
注意这个 n 与 LC、LF 文件里的 n 不是一回事，下一节会说清差别。

`--seldyn` 走另一套格式，好处是每帧可以用不同的掩码。它只有在 `relax:selective_dynamics`
为真时才会被读取，开关没打开时文件被静默跳过，连警告都没有。

### 掩码文件格式

优化坐标掩码（`--filter` 和 `relax:seldyn_file`）是纯数字流，不能写注释。固定胞的
四原子结构共 12 维：

```text
12
0 0 0  0 0 0  1 1 0  1 1 0
```

首行给出维度，后面必须正好跟这么多个字面的 `0` 或 `1`。多写一个、少写一个都会报错，
报错以 `Invalid optimization-coordinate mask: ` 开头；`0.0`、`+1` 这类写法同样不接受。

`--seldyn` 的块格式能装多帧：先写帧数，每帧依次是配置号（从 1 起）与原子数、3×3 的
晶胞掩码，再逐行写"原子号（从 0 起）加三个分量"。没有列出来的原子保持可动：

```text
1
1 4
1 1 1
1 1 1
1 1 1
0 0 0 0
1 0 0 0
```

换成话说：第 1 帧 4 个原子，晶胞三个方向都可动，0 号和 1 号原子被完全钉住。
两种格式不能互换，把 `--filter` 的文件传给 `--seldyn`，第一行就会报格式错。

## 结构性减少：固定、耦合与刚体

逐分量掩码之外，还有几种办法是拿整块结构开刀的。它们动的是坐标参数化本身，
因此不能和按分量裁剪的掩码叠加。

| 控制 | 做什么 | 代价或副作用 | 关键限制 |
| --- | --- | --- | --- |
| `relax:fix_shape` | 只保留各向同性缩放，晶胞形状不变 | 会强制关闭 `relax_cell` 并给出警告 | 与胞掩码、raw9 晶格约束、原子几何约束冲突 |
| `relax:fix_vol` | 自由度计数不变，在坐标反变换末尾把体积缩回，并从应力中扣掉静水压分量 | 每步多一次重标定 | 与 `keep_symm`、`periodic_subspace`、raw9 晶格约束冲突 |
| `relax:couple` | `1` a-b、`2` b-c、`3` c-a、`4` 全耦合；把耦合轴的对角应力取平均，让优化器看到相同分量 | 不是锁坐标，晶格矢量仍可各自变化 | `keep_symm` 会把它清零；与 raw9 晶格约束、原子几何约束、`periodic_subspace` 冲突 |
| `relax:periodic_subspace` | 周期方向用分数坐标、开放方向用笛卡尔；晶胞自由度降为 `rank(rank+1)/2` | 只适用于秩亏周期性 | 与 `cartesian`、`fix_shape`、`fix_vol`、`couple`、`keep_symm`、`symm_force`、压力与外应力、胞掩码全部冲突 |
| `relax:rigid_groups` | 每组变成质心平移加转动自由度，组内几何严格不变 | 只支持固定胞 | 要求笛卡尔或重铸坐标、无掩码、无 LF/LC；driver 仅 `lbfgs`、`cg`、`gd`、`fire` |

`relax:rigid_groups` 用分号分组，组内用逗号或空格分隔零基原子序号，例如 `0,1,2;5,6`。
组与组之间不能重叠，同一组里也不能出现重复原子。

`keep_symm` 站的是同一层：它把变量限制在初始空间群的不变子空间里，效果等同于按对称性
删自由度。也正因为如此，它跟几乎所有其它约束都互斥，掩码、文件约束、`fix_vol`、
`fix_shape`、VIB、线性变换无一例外。它自己的修正量判据与失败处理不在本页展开。

## 坐标变换：LC、LF 与 VIB

这一层不改变可行域本身，只改变优化器看到的变量。三种机制的区别，在于各自映到
哪个子空间：

- LC 把变量限制在仿射子空间 `x = b0 + Q2 v`。实现上走 SVD 求零空间，优化器在 n−k 维的
  v 上工作，行数 k 不能超过 n，且 A 必须行满秩。
- LF 做可逆映射 `x = A x + b`，梯度按 `(Aᵀ)⁻¹` 回拉。k 必须等于 n，A 必须可逆。
- VIB 与 VIB2 沿一组简正模式工作。VIB 只投影梯度，VIB2 真的换坐标，并把
  `abs(λ) < relax:ztol` 的模式置零。

| 文件 | 首行 | 后续 | 校验 |
| --- | --- | --- | --- |
| LC | `k n` | k×n 的 A（行主序），再 k 个 b | `k ≤ n`、`n` 等于当前优化维度、A 行满秩 |
| LF | `k n` | k×n 的 A，再 k 个 b | `k == n`、`n` 等于当前优化维度、A 可逆 |
| VIB | `n` | n 个模式因子、n 个初始位移、n×n 投影矩阵 | 不校验 |
| VIB2 | `n` | n 个模式因子、n 个特征值、n 个初始位移、n×n 投影矩阵 | 不校验 |

四者都是空白分隔的纯数值流，不认注释，也不要求正交归一。启用条件是开关为真、
且文件确实存在：`relax:with_linear_constraint` 读 `relax:lc_file`（默认 `LINEAR_CONSTRAINT`），
`relax:with_linear_transform` 读 `relax:lf_file`（默认 `LINEAR_TRANSFORM`），
`relax:vibffilter` 与 `relax:with_constraint_vib` 共用 `relax:vib_file`（默认 `VIB_CONSTRAINT`）。
后两者互斥，同时打开会报 `VIB filter and constraint_vib_modes cannot be used together`。

维度是最容易抄错的一处。LC、LF 文件里的 n 是**过滤器链走到那一步时**的维度，也就是
`keep_symm` 降过维、VIB 换过坐标之后的值；掩码文件里的 n 则取进入所有过滤器之前的
全维度。开了对称性或振动变换，两个 n 就不再相等。

VIB 与 VIB2 的维度根本不校验。n 与当前维度对不上时没有可读的报错，只会在后面的
矩阵运算里直接崩掉。手写这类文件之前，先确认 `relax:cartesian`、`keep_symm` 与
`relax:relax_cell` 的组合，再回头核对 n。

## 梯度投影不是约束

`relax:fixcm` 抹掉质心漂移，`relax:freeze_small_gd` 把小于 `relax:ztol` 的分量清零，
`relax:symm_force` 按空间群把力和应力对称化。三者动的都只是梯度：可行域没变，
自由度计数没变，也没有残差可报。所以它们**不是约束**，只能让优化提前"看起来收敛"，
结构并不会因此满足任何几何关系。

反过来，它们跟真正处理约束的求解器是冲突的。开启 SQP 或 NLC 时，这三个开关任意一个
都会触发 `SQP/NLC requires conservative coordinate gradients`；`keep_symm` 与它们的冲突
另有一份完整清单，在[对称性约束弛豫](relax-symmetry_zh.md)里。

一个简单的自查：梯度已经收敛而结构明显不对，就先关掉这三个开关，用同一初始结构
再算一遍，然后才去怀疑势函数或初始构型。

## 目标函数里的约束：AUL 与 SQP/NLC

声明式几何约束（下一节的主角）靠两种求解器兜底。选哪个不是风格问题：一旦有上下界，
就只剩 SQP 和 NLC 可选。

### 增广拉格朗日

AUL 既不改坐标，也不组装约束 Jacobian。它的做法是把约束折进目标函数，再在外面套一层
乘子更新，让普通优化器继续按无约束的方式跑。目标（含外载）与梯度如下：

\[
F = E + pV + \sum_i \left( \lambda_i f_i + \tfrac{1}{2} w_i f_i^2 \right)
\]

\[
\frac{\partial F}{\partial x} = \frac{\partial E}{\partial x}
  + \sum_i \left( \lambda_i + w_i f_i \right) \nabla f_i
\]

这里只有一阶项，二阶的 \( \sum_i w_i \nabla f_i \otimes \nabla f_i \) 没有组装，那是
SQP 的活。约束梯度在梯度滤波之前就并进了总梯度，所以万一还有哪个滤波器被放行，
它照样会把约束那部分一起投影掉。

| 设置 | 默认 | 作用 |
| --- | --- | --- |
| `relax:with_constraint_aul` | false | 打开 AUL；文件里有非 `symmetry` 声明且没选 sqp/nlc 时会隐式打开 |
| `relax:constraint_loops` | 5 | 外层乘子更新的轮数，必须为正 |
| `relax:gtol0` | 0.1 | 第一轮内层的梯度容差，必须不小于 `gtol` |
| `relax:aul_auto` | false | 用几何 Jacobian 估初始权重，只支持等式 |

内层容差逐轮收紧，规则是 `gtol_k = max(gtol, gtol0·0.1^k)`，最后一轮强制回到 `gtol`。
按默认值算，五轮的序列是 `0.1 → 0.01 → 1e-3 → 1e-3 → 1e-3`：第三轮起容差已经压在
`gtol` 上，再往后的外层只做对偶修正，既不再放松，也不会更严。把 `constraint_loops`
设成 0，或者 `gtol0 < gtol`，会直接报
`AUL requires positive constraint_loops and finite gtol0 >= gtol`。

乘子按 `λ += w·f` 更新，其中的 w 取**本轮开始时**的值。之后权重自己再递推一次，
`w ← weight_multply·w + weight_step`，三个常数默认为 1，所以实际效果是每轮 `w+1`。
参与更新的只有仍未满足的约束，已经满足的项在检查阶段就被标记跳过，乘子不再变动。

`relax:aul_auto` 走的是另一条路。它先取 \( \lVert \nabla f_i \rVert^2_{M^{-1}} \) 的倒数
当初始权重，再乘上梯度的量级 \( \max(1, \lVert g \rVert) \)，同时把乘子清零。此后某条
残差如果没有降到上一轮的一半，它的权重就乘 10，**不设上限**。残差一旦卡住，权重会
一路涨到数值溢出，报出 `Non-finite or nonpositive constraint multiplier/weight state`。
所以自动模式适合量纲差别大、而残差确实在往下走的情形；拿它硬顶一条收敛不了的约束
并不合适。

每过一轮外层，内层就重跑一次（`run_normal()`），而且优化器是新建的：LBFGS 和 CG
重新创建，FIRE 重置步长与温度、速度清零，历史不跨轮继承。内外层共用同一份
`maxiteration` 预算，第 k 轮内层能用的是剩下的迭代数，所以外层轮数越多，
每轮内层的步数就越少。

日志里只多出一行 `AUG Loop: k`，乘子、权重、逐条残差都不打印。步进行那行
`iter N, f = ..., gmax = ..., gmin = ...` 里，`f` 是**增广目标**（含 `λf + ½wf²`
和外载，不是能量），`gmax`/`gmin` 是梯度的最大、最小分量，带符号，不是最大绝对值。
想看逐条残差只有两个时机：运行结束时的 `Constraint j: residual=..., tolerance=...`
（v1 的 `distance` 与 `angle` 还会给出 value/target/deviation），或者自己进程序里
读约束数组。

| 现象 | 原因 | 动作 |
| --- | --- | --- |
| 外层跑完仍不满足，退出码 2 | 没有残差停滞判据，轮数用完即止 | 加 `constraint_loops`、放宽容差或换 driver |
| `V2 bounds require driver=sqp or nlc` | 文件里有 `lower`/`upper` | 换 SQP/NLC，AUL 只有等式 |
| `Automatic AUL currently supports equality regulators` | `aul_auto` 遇到带上下界的约束 | 关掉 `aul_auto` |
| `Non-finite or nonpositive constraint multiplier/weight state` | 自动权重无上限增长或残差为 NaN | 关 `aul_auto`，手动给权重 |

AUL 能提供的收敛证据只有两条：所有残差落在各自容差内，且增广梯度不超过 `gtol`。
它不算 KKT 残差，也不做对偶可行性检查，要这组证据得换下一小节的求解器。另外，
Python 的 `Relaxation` 里 `constraint_loops` 默认是 10（C++ 是 5），`gtol0` 没有提供、
固定 0.1，照抄 C++ 配置时要留意。

### SQP 与 NLC 是同一个求解器

`driver sqp` 和 `driver nlc` 是同一份实现，底层都是串行的稠密 SQP，参数完全一致。
两者的差别只留在历史命名上，选哪个结果都一样。`relax:with_constraint_sqp`、
`relax:with_constraint_nlc` 只是把 driver 改成对应名字的空壳；
`relax:with_constraint_lc` 同样只改名字，并不带任何线性约束数据。

固定胞下，纯位置且仿射的约束会被退化成盒式边界或线性约束，其余一律按非线性约束处理。
收敛判据要求 KKT 残差和每条约束的余量同时满足：`state_vars.converged` 需要求解器
正常终止、对偶可行性成立、所有约束残差落在各自容差内，且拉格朗日梯度的最大分量
不超过 `gtol`。这里的 `gtol` 比的不是"最大力分量"，无约束弛豫的直觉不能直接搬过来。

开 SQP 或 NLC 会顺手关掉 AUL（哪怕 `with_constraint_aul` 明确打开了），并且拒绝
一切非保守的梯度过滤器。

### 怎么选

| 情形 | 选择 |
| --- | --- |
| 只有等式，且希望省内存 | AUL，或 SQP |
| 有上下界、壁面、区间 | 必须 SQP 或 NLC |
| 需要 KKT 残差与对偶可行性作为收敛证据 | SQP |
| 已启用 LC、LF、VIB 或 `keep_symm` | 不能进 SQP；回到掩码或坐标层 |
| 多 MPI 进程 | 不能使用文件约束；改用掩码或坐标变换 |

### 外部载荷改的是目标应力

`relax:pressure` 是各向同性压力，单位 GPa，正值表示压缩；`relax:external_stress`
接受 3 个或 6 个分量，顺序 `xx yy zz yz xz xy`，单位同样是 GPa。两者都进不了残差，
改的是平衡条件，所以收敛后该查的是最终应力有没有到目标值，而不是某条残差。

一条硬限制：v1 的晶格类文件约束要求零压、零外应力，外载和这类约束不能同时用。

## 声明式几何约束：`--file-constraints`

### 何时用

几何量必须精确满足、或者必须落在某个区间里，需要逐帧绑定到各自的初始几何，
要用壁面和公式——这些场合才值得动用它。代价是一串运行前提：

- 只在单 MPI 进程下可用，多进程直接报 `file-constraints supports only a single MPI rank`。
- 只在 CPU 构建下可用，GPU 构建报 `file-constraints requires a CPU build`。
- 文件不超过 64 KiB、最多 32 条约束、字符串不超过 4096 字节、必须是纯 ASCII、不支持注释。
- 重复键、未知键都是硬错误，报错里会点名具体的键。

命令行的入口只有 `--file-constraints <path>`，同一路径也可以写进设置文件。

### 顶层结构与单条键

顶层只有两个键。`version` 必须是整数记号，只能取 `1` 或 `2`，`1.0`、`"1"`、`true`
都会被拒；`constraints` 是长度 1 到 32 的数组，不能为空。

单条约束能写的键就是下表这些，多写一个会报 `unknown constraint key`：

| 键 | 类型 | 适用 kind | 说明 |
| --- | --- | --- | --- |
| `kind` | 字符串 | 全部，必需 | 见下一张表 |
| `target` | 数值 | 除 `symmetry` | 目标值；省略则绑定该帧初始几何 |
| `lower`、`upper` | 数值 | v2，除 `symmetry` | 残差上下界；与 `target` 互斥 |
| `residual_tolerance` | 数值 | 除 `symmetry`，必需 | 残差容差，量纲见下一张表 |
| `axis` | `"a"`、`"b"`、`"c"` | `lattice_length` | 单轴 |
| `axes` | 两元素数组 | `lattice_angle`、`lattice_ratio` | 两个不同的轴 |
| `atoms` | 整数数组 | `distance` 两个、`angle` 三个、`dihedral` 四个、`position` 一个、`wall` 一个 | 零基下标 |
| `component` | `"x"`、`"y"`、`"z"` | `position` | 分量 |
| `normal` | 三元素数组 | `wall` | 会被归一化 |
| `expression` | 字符串 | `formula` | 表达式，见下 |
| `operations`、`sites` | 整数数组 | `symmetry` | 初始对称操作与受限原子 |

表里没有 `weight`、`tol`、`scale` 这类字段。权重和乘子归求解器管，写进文件只会被
当成未知键。

### 十一种 kind 与残差量纲

| kind | 起始版本 | 附加键 | `target` 单位 | 残差（v1 → v2） |
| --- | --- | --- | --- | --- |
| `volume` | v1 | — | Å³ | `det − V` → `V − V0` |
| `lattice_length` | v1 | `axis` | Å | `L² − L0²` → `L − L0` |
| `lattice_angle` | v1 | `axes` | 度 | `cosθ − cosθ0` → `θ − θ0` |
| `lattice_ratio` | v1 | `axes` | 长度比 | `r² − r0²` → `r − r0` |
| `distance` | v1 | `atoms`×2 | Å | `d² − d0²` → `d − d0` |
| `angle` | v1 | `atoms`×3 | 度 | `cosθ − cosθ0` → `θ − θ0` |
| `dihedral` | v2 | `atoms`×4 | 度 | — → `φ − φ0` |
| `position` | v2 | `atoms`、`component` | Å（笛卡尔绝对值） | — → `x − x0` |
| `wall` | v2 | `atoms`、`normal` | Å | — → `n̂·r`，用上下界给区间 |
| `formula` | v2 | `expression` | 按表达式 | — → `f − f0` |
| `symmetry` | v2 | `operations`、`sites` | — | 无残差，隐式打开 `keep_symm` |

最后两列才是要害：**同一个 kind 在两版里的残差量纲并不相同**。v1 的六种各走专用残差，
所以 `distance` 的容差单位是 Å²、`angle` 的容差无量纲；v2 里除 `symmetry` 外每一种
都翻译成表达式、由自动微分求导，残差就是物理量本身的差，容差单位也回到 Å、度或 Å³。
`residual_tolerance` 描述的是残差那一列，既不是物理量，也不是能量梯度。

几个细节容易踩。`angle` 的三个下标里，中间那个才是顶点，`[0,1,2]` 和 `[2,1,0]`
会被当成同一条。`dihedral` 的 `target` 会平移 360° 的整数倍到初始值附近，免得跨分支
跳变。`position` 的 `target` 是绝对笛卡尔坐标，不是位移，原子按初始帧的最近镜像绑定。
`wall` 必须同时有 `normal` 和 `lower`/`upper` 之一，只给 `target` 会被拒。

`formula` 的表达式语言只有三样东西：几何原语、五个函数、一个常量 `pi`。

| 类别 | 内容 |
| --- | --- |
| 原语 | `x(i)`、`y(i)`、`z(i)`、`distance(i,j)`、`angle(i,j,k)`、`dihedral(i,j,k,l)`、`lattice_angle(i,j)`、`lattice_ratio(i,j)`、`volume()` |
| 函数 | `sqrt`、`exp`、`log`、`sin`、`cos`、`^`（指数必须是常数） |
| 单位 | 长度 Å、角度度、体积 Å³；`sin` 与 `cos` 的参数是弧度 |

`volume()` 在 `periodic_subspace` 下没有定义，用它会报
`Periodic-subspace relaxation does not define a three-dimensional volume constraint`。

### target 省略、去重与逐帧绑定

省略 `target` 的意思是"这条约束绑定到该帧的初始几何"。绑定不发生在解析阶段，而是在
每帧开始时，所以同一份文件跑多帧时，省略 target 的约束在每帧得到的目标值都不一样，
残差总是从零开始累积；显式写了 target 的则跨帧不变。`--seq-read` 下每帧都会重新绑定。

去重比较的是几何量，不是整条声明：`kind`、轴、原子下标、表达式（或分量、法向）相同
就算同一条，`target`、`lower`、`upper` 不参与比较。于是"同一个键长给两个不同区间"
这种写法无法表达，会报 `duplicate constraint identity`；两条 `symmetry` 也会因身份
相同而冲突。

原子下标在解析期只检查非负和互异，越界要等到绑定阶段才报，报错里会指明是第几条约束。

### 布局限制

v1 的晶格类约束（`volume`、`lattice_length`、`lattice_angle`、`lattice_ratio`）落在
原始的九分量晶格上，所以要求 `relax:relax_cell true`、`relax:cell_strain false`、
`fix_shape` 关闭、`relax:cartesian false`，外加全周期的右手胞。它们还与一串功能互斥：
`fix_vol`、`couple`、`keep_symm`、`symm_force`、VIB 与 VIB2、LC 与 LF、
`selective_dynamics`、`freeze_small_gd`、`fixcm`、`dprecond`，以及零压零外应力。
在应用层，这些约束也不能和 `--position-mask`、`--filter`、`--cell-mask`、`--seldyn`
或结构文件里的冻结标志同时出现。

原子类约束（`distance`、`angle`、`dihedral`、`formula`、`position`、`wall`）宽松得多：
可以和掩码共存，但不能与 `keep_symm`、`symm_force`、VIB、`fixcm`、`freeze_small_gd`
同用，也不支持 `fix_shape`，变胞下同样不支持 `fix_vol` 和 `couple`。变胞时还要求
`dprecond=false`，或者保持 `atomic_force_recast=true`，否则报
`Atomic geometry constraints with variable cell require dprecond=false or atomic_force_recast=true`。

文件里一旦出现 `lower` 或 `upper`，就必须显式设置 `driver sqp` 或 `nlc`。v2 的每条约束
都会翻译成表达式，边界检查在准备阶段就会触发，报错文本是
`V2 bounds require driver=sqp or nlc`。

### 完整示例

取一个三原子示意分子，固定胞，`relax:cartesian true`，三个原子的坐标依次是
`(0.00, 0.00, 0.00)`、`(0.98, 0.00, 0.00)`、`(0.49, 0.85, 0.00)`：0–1 号距离 0.98 Å，
以 1 号为顶点的键角约 60.0°。后面的文件都基于这个结构。

v1 的等式约束，第二条故意省掉目标值：

```json
{
  "version": 1,
  "constraints": [
    {"kind": "distance", "atoms": [0, 1], "target": 0.98, "residual_tolerance": 1e-6},
    {"kind": "angle", "atoms": [0, 1, 2], "residual_tolerance": 1e-4}
  ]
}
```

v2 把键角放宽成区间，同时显式点名 SQP：

```json
{
  "version": 2,
  "constraints": [
    {"kind": "distance", "atoms": [0, 1], "target": 0.98, "residual_tolerance": 1e-6},
    {"kind": "angle", "atoms": [0, 1, 2], "lower": 58, "upper": 62, "residual_tolerance": 0.2}
  ]
}
```

```ini
relax:driver        sqp
relax:gtol          1e-4
relax:maxiteration  600
```

两份文件只有两处实质差异：第二份的键角是 58° 到 62° 的区间，容差 0.2 的单位是度；
第一份的容差 `1e-4` 无量纲，因为它约束的是 `cosθ − cosθ0`。

## 约束文件格式速查

五种格式分散在各自的小节里，因为它们都和维度语义绑在一起。想知道"这个文件的第一行
该写什么"，先看这张表。

| 格式 | 由谁产生 | 对应开关或键 | 维度 | 校验强度 |
| --- | --- | --- | --- | --- |
| JSON 约束集 | 手写 | `--file-constraints` | 每条约束自描述，无全局维度 | 最严：版本、键、单位、下标、身份全部检查 |
| 优化坐标掩码 | 手写或脚本 | `--filter`、`relax:seldyn_file` | 全优化维度，晶胞坐标在前 | 严：维度与取值逐项核对 |
| `--seldyn` 块 | 手写或脚本 | `--seldyn` | 逐帧、逐原子 | 较严：帧号、原子数、重复号都会报 |
| LC 与 LF | 外部工具或脚本 | `lc_file`、`lf_file` | 过滤器链中当时的维度 | 中：检查维度与秩或可逆性 |
| VIB 与 VIB2 | 声子分析工具 | `vib_file` | 同上 | 无：维度不符时在矩阵运算处崩溃 |

## 互斥关系与运行前提

下面按"主导控制"列出源码里会显式拒绝的组合。没列出来的组合不代表验证过，
只代表没有专门的检查。

| 主导控制 | 不能与以下同用 |
| --- | --- |
| `relax:keep_symm` | 选择性动力学；任何文件约束；`fix_vol`、`fix_shape`；VIB 与 VIB2；LC 与 LF；`fixcm`；`freeze_small_gd`；`cartesian` |
| `--position-mask` | `--filter`、`--seldyn`、`--cell-mask`；`keep_symm`；VIB 与 VIB2；LC 与 LF；活动的 `seldyn_file`；`isif` 5 到 7 |
| `--cell-mask`、`strain_mask` | 不变胞或 `fix_shape`；`keep_symm`；VIB 与 VIB2；LC 与 LF；活动的掩码文件 |
| `--filter`、活动的 `seldyn_file` | `--position-mask`；`--seldyn`；胞掩码；`keep_symm`；VIB 与 VIB2；LC 与 LF；结构文件里的冻结原子 |
| VIB 与 VIB2 | 彼此 |
| v1 晶格约束 | 掩码族；`fix_vol`；`couple`；`keep_symm`；`symm_force`；VIB 与 VIB2；LC 与 LF；`fixcm`；`freeze_small_gd`；`dprecond`；非 raw9 坐标；非全周期胞；非零压力或外应力 |
| 原子类文件约束 | `keep_symm`；`symm_force`；VIB；`fixcm`；`freeze_small_gd`；`fix_shape`；变胞下的 `fix_vol` 与 `couple`；变胞加 `dprecond` 且未重铸 |
| `periodic_subspace` | `cartesian`；`fix_shape`；`fix_vol`；`couple`；`keep_symm`；`symm_force`；压力与外应力；胞掩码 |
| `rigid_groups` | 变胞；任何掩码；LC 与 LF；driver 不是 `lbfgs`、`cg`、`gd`、`fire` |

还有四条运行前提，写错时最容易被误判成"约束没生效"：

1. 单 MPI 进程与 CPU 构建只属于 `--file-constraints`。`driver sqp` 与 `nlc` 本身
   没有这两条限制。
2. SQP 与 NLC 只接受保守坐标梯度，`fixcm`、`freeze_small_gd`、VIB 梯度滤波、
   `symm_force` 都会被拒。
3. v1 晶格约束要 raw9 布局：`relax_cell=true`、`cell_strain=false`、`fix_shape=false`、
   `cartesian=false`、全周期右手胞、零压零外应力。
4. 原子类约束配变胞时，`dprecond` 必须关闭，或保持力重铸开启。

## 端到端示例

### 示例一：同一 slab 的三种掩码写法

一个正交胞、四原子、固定晶胞的示意 slab（不对应具体材料），要求底层 0、1 号原子
整体固定，顶层 2、3 号只放开面内。

```sh
imrmlp relax --input slab.vasp --output relaxed.cfg --potfile model.mtp \
  --pottype mtp --position-mask "1 1 0"
```

这条命令放开的是所有原子的 xy、固定所有原子的 z，本该全固定的底层也跟着放开了。
粒度不够，换成逐分量掩码文件 `slab.mask`（固定胞，12 = 3×4）：

```text
12
0 0 0  0 0 0  1 1 0  1 1 0
```

```sh
imrmlp relax --input slab.vasp --output relaxed.cfg --potfile model.mtp \
  --pottype mtp --filter slab.mask
```

第三种写法把掩码挪进结构文件：在 POSCAR 的 `Selective dynamics` 行后给每个原子一组
`T`/`F`，再在设置文件里写 `relax:selective_dynamics true`。

三者的差别在粒度和 n 的含义上。`--position-mask` 只有三个标志，所有原子共用；
`--filter` 的十二个标志逐分量对应，且 n 是进入过滤器之前的全维度；POSCAR 掩码跟着
结构走，换结构时不必改命令行。收敛判据三者一致：只看允许自由度上的最大梯度分量，
被冻结的方向上力可以不为零。

### 示例二：同一几何约束的 v1 与 v2

仍旧用上一节的三原子分子。v1 文件只写等式，第一条显式给目标值，第二条绑定初始几何：

```sh
imrmlp relax --input mol.xyz --output relaxed.cfg --potfile model.mtp \
  --pottype mtp --file-constraints eq.json
```

没有指定 driver，程序会自动走 AUL：外层最多五轮，容差从 `gtol0` 起逐轮收紧。
第二条约束的初始残差为零，因为它的目标就是初始键角。

换成 v2 的区间文件，就得显式点名求解器：

```sh
imrmlp relax --input mol.xyz --output relaxed.cfg --potfile model.mtp \
  --pottype mtp --file-constraints bounds.json --settings sqp.ini
```

| 对比项 | v1（`eq.json`） | v2（`bounds.json`） |
| --- | --- | --- |
| 距离残差 | `d² − d0²`，容差单位 Å² | `d − d0`，容差单位 Å |
| 键角约束 | 等式，可省略 target | 58° 到 62° 的区间 |
| 可用 kind | 6 种 | 11 种 |
| 求解器 | 未指定时自动 AUL | 必须 `sqp` 或 `nlc` |
| 收敛判据 | 各约束残差达容差，且外层梯度达 `gtol` | KKT 残差与对偶可行性，加各约束残差 |

v1 和 v2 不能混在一个文件里：`lower`、`upper`、`position` 及其后新增的 kind、
`expression` 这些字段，在 `version: 1` 下都会报
`extended constraint fields require version 2`。

## 收敛判据与排错

有约束之后，"梯度达标"不再是充分条件。要分别看三件事：允许自由度上的梯度是否小于
`gtol`，每条约束的残差是否小于它自己的 `residual_tolerance`，以及退出码是 `0` 还是
`2`。冻结方向上力不为零是正常的；SQP 与 NLC 下 `gtol` 比的是拉格朗日梯度，
含义与无约束时不同。

| 报错或症状 | 原因 | 处理 |
| --- | --- | --- |
| `file-constraints supports only a single MPI rank` | 多进程运行 | 降到单进程 |
| `file-constraints requires a CPU build` | GPU 构建 | 换 CPU 构建 |
| `file-constraints is incompatible with position-mask, filter, cell-mask and seldyn` | v1 晶格约束与掩码族同时使用 | 二者择一；或改用 v2 |
| `V2 bounds require driver=sqp or nlc` | 用了上下界但没换求解器 | 显式设 `driver sqp` |
| `Automatic AUL currently supports equality regulators` | 开了 `aul_auto` 又有区间约束 | 关掉 `aul_auto`，或改用 SQP |
| `SQP/NLC requires conservative coordinate gradients` | 同时开了 `fixcm`、`freeze_small_gd`、VIB 滤波或 `symm_force` | 关掉这些开关 |
| `AUL requires positive constraint_loops and finite gtol0 >= gtol` | 外层次数或内层容差非法 | `constraint_loops` 为正，`gtol0 ≥ gtol` |
| `Cell mask requires 6 components for the final cell coordinate mode` | 6 值与 9 值配错了坐标模式，报错里的数字随模式变化 | 按 `cell_strain` 与 `symm_strain` 的实际取值给分量数 |
| `Lattice constraints require raw9 cell coordinates` | 开了应变坐标或 `fix_shape` | 关 `cell_strain`，用原始晶格 |
| 残差不降，退出码 `2` | 约束互相矛盾、初始结构离可行域过远、或约束被掩码改写了坐标 | 先看是哪一条残差，再检查同层是否有其他控制 |

通用的输入错误，比如元素映射、近邻、模型适用范围，不在本页范围内，见
[结构弛豫](../guides/relaxation_zh.md)的常见错误一节。

## 其余控制：优化器、预条件、多帧

约束之外的那些设置没有变，这里只列和本页内容有交互的部分。

`relax:driver` 默认 `lbfgs`，另有 `lbfgspp`、`fire`、`cg`、`gd`、`mdmin`、`quickmin`，
以及专供约束问题的 `sqp` 与 `nlc`。换优化器之后，该比的是最终能量和梯度，
迭代数说明不了什么。

预条件在约束下有两处牵连：`relax:dprecond` 与 v1 晶格约束、原子类约束配变胞的
组合冲突（见上一节）；`relax:precond_file` 和 `hessian_precond_file` 的维度同样是
全优化维度。`relax:thin_cell_scaling` 对含真空的胞有用，`relax:sparse_precond`
只用于固定胞的 `gd` 或 `fire`。

多帧输入用 `--seq-read` 逐帧读取，每帧都会重新绑定一次文件约束；轨迹由
`relax:write_traj`、`relax:trajfile`、`relax:traj_append` 和写出间隔 `relax:nblock`
控制。

## 相关

- [结构弛豫](../guides/relaxation_zh.md)：常规流程、收敛判据与梯度复核
- [对称性约束弛豫](relax-symmetry_zh.md)：`relax:keep_symm` 的机制与限制
- [配置参考](../reference/configuration_zh.md)：全部 `relax:` 键、默认值与约束文件路径
- [命令参考](../reference/commands_zh.md)：`relax` 一节的命令行开关
