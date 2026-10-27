# MTP 基组生成、筛选与复核 {#basis_generation_selection}

[English](basis-generation-selection.md)

本页串联 `mtmbasis → trainmtp/trainnlpot → fastfit → calc_efs/calc_err`，适用于普通线性头 MTP。先完成[基本训练流程](../guides/workflows_zh.md)，准备元素映射一致的 `train.xyz`、`valid.xyz` 和独立的 `test.xyz`。数据约定见结构文件格式，模型布局见MTP 文件格式。

无需外部数据即可运行的完整脚本见基组生成与筛选示例。其他头和长程模型的初始化、训练、重载见模型家族流程。

## 1. 先分清三种“基组优化”

| 任务 | 工具 | 改变什么 | 依据 |
|---|---|---|---|
| 定义候选标量不变量 | `mtmbasis` 枚举、采样 | 函数空间、候选规模 | 矩张量阶数、level、缩并条件 |
| 优化缩并计算结构 | `mtmbasis` 搜索 | 中间矩与乘积规则、共享子表达式 | 结构计数，不读取训练 E/F/S |
| 按拟合质量筛选标量 | `fastfit` | 保留的标量集合及重拟合的线性系数 | 训练／验证数据、相关性、可选实测成本 |
| 判断重排前后是否等价 | `mtpbasis` | 只读判等，或迁移已有参数 | 精确符号表达式及兼容性检查 |

`mtmbasis` 的张量数量少，不等于验证误差低；`fastfit` 的标量数量少，也不等于运行更快。不要把 `mtpbasis`（比较／映射）与 `mtmbasis`（生成）混淆。筛选基函数与 `maxvol_select` 等筛选**数据构型**是不同操作。

普通 MTP 在固定径向参数后，对物种偏置和标量 moment 系数是线性的。`fastfit` 固定输入模型的径向参数，反复拟合线性部分；它不执行径向非线性训练。LES/SOG、MTP-M 和独立 NN 头不在这个普通 MTP 筛选流程内。

## 2. 用 mtmbasis 定义候选池

在独立工作目录中创建 `mtmbasis.ini`。以下是便于跑通流程的小基组，不是生产精度推荐值：

```ini
max_lev 8
max_dim 3
max_mu 2
max_nv 2
max_poly 4
mu_wei 2
nv_wei 1
dim_wei 2
mode 3
maxiter 20
w1 1
w2 4
```

这些是顶层设置，不放在 `[pot]` 或 `[fit]` 中。对含 d 个矩张量的乘积，当前生成器使用

\[
\mathrm{level}=w_\mu\sum_{i=1}^{d}\mu_i+w_\nu\sum_{i=1}^{d}\nu_i+w_d d.
\]

| 设置 | 默认值 | 含义 |
|---|---|---|
| `max_lev` | 16 | 组合的 level 上限 |
| `max_dim` | 4 | 每个缩并不变量的矩张量因子数上限，支持 1–7；不是空间维数 |
| `max_mu` | 4 | 自动生成候选 `(mu,nu)` 时的 mu 上限 |
| `max_nv` | 10 | 单个矩张量角向阶数 nu 的上限 |
| `max_poly` | 10000 | 组合中 nu 总和的上限 |
| `mu_wei`, `nv_wei`, `dim_wei` | 2, 1, 2 | level 中三个权重 |

当前实现还会将 `max_dim` 限制为 `min(max_dim, max_lev / 2)`（整数除法）；调整权重后也要考虑这个额外限制。

mu 是径向通道标签，不是径向展开项数 `radial_basis_size`；nu 是张量阶数。d 个邻居求和因子的乘积可包含至多 d+1 体的贡献，但也包含重复邻居等低体阶项，不能把 `max_dim` 当作严格的物理体阶。不同文献／生成器的 level 定义未必相同，比较时应同时记录权重和全部截断条件。

先保存候选池，再优化其实现：

```sh
imrmlp mtmbasis mtmbasis.ini --generate-only true
imrmlp mtmbasis mtmbasis.ini --alphapool alphapool.dat --potfile basis.mtp
```

当前 `mtmbasis` 的布尔命令行选项需要显式值，例如 `--generate-only true`、`--sample true`、`--advanced-sample true`。`--generate-only` 单独出现会报缺少参数。

需要限制候选矩张量时，用 `--moments moments.txt` 替代自动矩张量列表；文件第一个数是数量，其后恰好是这些 `(mu,nu)` 对，例如：

```text
3
0 0
1 0
0 2
```

level、因子数和总阶数限制仍然适用。负值、重复对和多余／缺失数据会被拒绝。已有 `--alphapool` 直接提供标量候选池，应保留它对应的生成配置。

## 3. 缩并搜索、采样与输出

缩并搜索的代价为 `w1 * moments_count + w2 * rules_count`，默认权重为 1 和 4。它统计中间 moment 和乘积累加规则，没有使用目标机器上的实测时间。

| `mode` | 搜索方式 | 预算与特点 |
|---|---|---|
| `0` | 模拟退火 | `T_init`、`maxiter`；默认模式 |
| `1` | 随机起点加局部搜索 | `maxiter` |
| `2` | 遗传搜索 | `popsize`、`childsize`、`maxiter` 及变异／交换参数 |
| `3` | 坐标下降 | 逐分支搜索，直到一轮没有改进；当前不以 `maxiter` 限制轮数 |

输出前缀由 `--potfile` 控制，上例得到 `basis.mtp_<编号>`；默认前缀是 `potential.mtp`。保存的是初始候选和改进候选，不保证每个搜索迭代都输出文件。结合日志中的 moments、rules、score 选择模板，记录具体文件，不要在后续脚本里依赖不明确的通配符。

`mid_tid_mapping_<编号>.log`、`scalar_id_tensor_id_mapping_<编号>.log`、`scalar_pool_<编号>.dat` 和 `conns_exp.log` 用于追溯表达式与缩并路径。启用自动加回路径标量等选项时，还应复核最终标量空间；需要判断两个输出是否等价，使用 `mtpbasis`，不要仅比较文件长度或 moment 数量。

候选池很大时可以先采样：

```sh
imrmlp mtmbasis mtmbasis.ini --alphapool alphapool.dat --sample true --sample-rate 0.5 --generate-only true
```

`--sample` 写出 `sampled_alphas.dat`，但**本身不会退出**；上例再用 `--generate-only true` 保存采样后的 `alphapool.dat` 并退出，因此会覆盖当前目录的原池。应在副本目录中运行。按因子数采样则用 `--advanced-sample true` 和顶层 `sample_rates`，其前六个位置对应 d=2…7，未给出的比例为 0，单因子标量保留；这一分支保存 `sampled_alphas.dat` 后退出。比例应在 `[0,1]`。

枚举和抽样会改变候选函数空间；缩并搜索主要改变计算路径。当前池抽样使用 `random_device`，没有公开 seed 选项；复现应保存抽样后的池，不能只保存采样率。

## 4. 把基组模板变成可筛选的模型

生成文件是 `MTP / version = 1.1.0` 模板，默认 `species_count=1`、`RBChebyshev`、`min_dist=2`、`max_dist=5`、`radial_basis_size=8`。它包含拓扑，但没有训练后的径向与线性系数。加载成功不表示它已经可用于材料预测。

在**未训练模板**上检查并设置物种数、元素顺序、距离范围和径向展开项数。`radial_funcs_count` 由基组的 mu 索引决定，不能任意修改；训练后的模型也不能靠改表头扩展物种或径向维度。多元素、pair-split 和可配置径向基见MTP 文件格式和径向基参考。

将选中的模板复制为 `template.mtp`，准备 `index.map`。下面的最小训练配置使用当前目录的数据：

```ini
trainset train.xyz
validset valid.xyz
trainset-format extxyz
validset-format extxyz
potfile template.mtp
index index.map
fit:energy_weight 1
fit:force_weight 0.1
fit:stress_weight 0.001
fit:weight_L2 1e-8
fit:random_init false
fit:force_randomradial false
fit:force_randomlinear false
```

保存为 `train.ini`，运行：

```sh
imrmlp trainmtp train.ini --linear-fit
cp _Trained_Potential_.mtp full.mtp
```

这个步骤验证生成到训练的连接。生产流程通常还要优化径向参数，例如按训练指南执行 `trainmtp --do-lbfgs` 或 `trainnlpot --do-varpro`，再将选定模型保存为 `full.mtp`。`fastfit` 会固定这些径向参数。不要使用 `train_linear` 代替普通 MTP 的专用拟合器。

## 5. 用 fastfit 建立基准，再筛选

创建 `fastfit.ini`，这里同样使用**顶层键**，而不是 `fit:` 权重或 `pot:` 模型键：

```ini
potfile full.mtp
trainset train.xyz
validset valid.xyz
format extxyz
index index.map
weight-energy 1
weight-force 0.1
weight-stress 0.001
weight-l2 1e-8
energy-power 1
force-power 0
stress-power 1
seed 42
```

`format` 同时用于训练和验证；默认是 `mlpcfg`，extxyz 必须显式指定。模型必须具有与数据一致的物种顺序和已确定的径向布局。训练集拟合参数，验证集比较方案；最终测试集不放进这份配置。

| 模式 | 用途 | 主要输出／前置条件 |
|---|---|---|
| `0` | 标量正规矩阵的特征值诊断 | `EIGENSYS`、`EIGVEC`、`EIGVAL`；不是直接对原始设计矩阵做 SVD |
| `1` | 全基组加权线性重拟合 | `__Trained__.mtp`；`--cmp` 仅比较求解器，不走通常导出分支 |
| `2` | 应用现有选集并重新拟合、压缩 | 读 `selected_ones.dat`，写 `__REDUCED__.mtp` |
| `3` | 逐项试删的敏感性诊断 | 逐个标量的损失报告，不是连续贪心剔除，也不导出最终缩减势 |
| `4` | 基于 CUR 分数的筛选 | 读模式 0 的 `EIGVEC`、`EIGVAL`；写 `selected_ones.dat`、`removed.dat`，随后用模式 2 导出 |
| `5` | 固定标量数量的 SA／贪心交换 | 读上述两个互补且非空的选集，写 `__TEMPERED__.mtp` |
| `6` | NSGA-II 多目标子集搜索 | 至少两个标量，写 `__NSGA2__.mtp` 及对应选集 |
| `7` | 阈值／聚类／VIF 去相关 | 写选集与 `correlation_*`；`--corr-save-reduced` 额外拟合并写 `__DECORR__.mtp` |

最短的“建立基准 → 筛选 → 导出”路径：

```sh
imrmlp fastfit fastfit.ini --mode 1 --num_threads 1
imrmlp fastfit fastfit.ini --mode 0 --num_threads 1
imrmlp fastfit fastfit.ini --mode 4 --discard-rate 0.25 --num_threads 1
imrmlp fastfit fastfit.ini --mode 2 --num_threads 1
```

模式 4 必须先有模式 0 生成的 `EIGVEC` 和 `EIGVAL`，不能在空目录中直接接在模式 1 后运行。它默认按分数确定性选择；`--cur` 才开启随机采样。`--discard-rate` 是丢弃比例。`--addback` 可保留计算路径中的标量，因此实际保留数量可能不同于简单比例。

可在原选集上执行 `--mode 5 --search-steps 100 --method greedy`，或独立尝试 `--mode 6 --population 16 --generations 10`。模式 5 默认 legacy 分数使用训练损失加可用的验证损失。模式 6 的 legacy 双目标在有验证集时为训练损失／验证损失，没有验证集时为训练损失／标量数；有验证集时不能把它解释为显式“误差与速度”优化。搜索预算只是示例，不保证收敛。

需要在验证误差上限内追求速度，使用模式 5/6 的 `selection-objective error-cost`，见[按 E/F/S 成本选基](fastfit-cost-selection_zh.md)。它要求非空验证集，以估算运算量搜索，再对少量候选实测。不要用标量数量代替实际 CPU/GPU/LAMMPS 时间。

## 6. 选集文件、矩阵缓存与可复现性

`selected_ones.dat` 第一项是数量，后面是从 0 开始的**标量序号**。例如保留 0、2、5：

```text
3 0 2 5
```

该编号是原模型 `alpha_moment_mapping` 中的位置，既不是其 moment ID，也不是加上 `species_count` 后的线性参数下标。物种偏置始终隐式保留。模式 5 的 `removed.dat` 必须是同一原模型的补集，无重复、越界、遗漏或交集。基组压缩后会重新编号，不能把旧选集直接应用到压缩模型；应归档输入模型、选集和输出模型。

默认 `fastfit` 从模型与数据组装矩阵，不要求提前生成 `MATATA_*`。`--use-matrices` 会读取由 `trainmtp --do-matrix-ata` 导出的未加权 E/F/S 块，并重新组装当前问题进行一致性验证；它不是绕过数据读取的缓存捷径。修改数据、径向系数、类型映射、归一化或基组后要重新生成。`--method svd` 等方法在这里求解的是正规方程，不能把它与对原始设计矩阵直接做 QR/SVD 的数值性质等同。

模式 4 加载的特征系统文件只检查维度及数值有限性，不会证明它们来自当前模型和数据。修改模型、数据、权重或归一化后，应重新运行模式 0；不要误用同尺寸的旧 `EIGVEC`、`EIGVAL`。

保存输入文件与二进制版本、全部权重和原子数幂、seed、线程数／MPI rank 数、选集和性能测量环境。随机种子相同不保证跨 MPI 划分或机器负载的逐位一致；性能复核见fastfit profile。每个实验使用独立目录，避免固定名称输出互相覆盖。

## 7. 独立复核与部署

先对验证集比较完整基准与缩减模型；完成方案选择后，再对独立测试集运行一次最终评估：

```sh
imrmlp calc_efs --potfile __REDUCED__.mtp --input test.xyz --output test-predicted.xyz --index index.map
imrmlp calc_err --input test.xyz --input-predict test-predicted.xyz --index index.map --output-summary test-errors.txt
```

保留 E、F、virial 各自误差，检查目标应用中的极端构型、弛豫／MD 稳定性，并在实际部署后端比较吞吐量。缩减后可再做非线性训练，但这会成为新的模型候选，需重新验证。仅重排计算图、要求保持预测的场景使用 `mtpbasis --source ... --target ... --report ...`；发生参数迁移后仍需比较 E/F/virial。

完整命令参数见[命令参考](../reference/commands_zh.md)。实现核对入口为 `apps/genmtm.cpp`、`mtmbasis/mtm.cpp`、`apps/fastfit.cpp`；回归覆盖包括 `unitest/ut_mtmbasis.cpp`、`unitest/ut_fastfit_regression.cpp` 和 `unitest/ut_fastfit_correlation.cpp`。
