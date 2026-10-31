# MTP 基组的生成、优化与选择 {#guide_basis_selection}

[English](basis-selection.md)

基组决定模型的函数形式上限：`mtmbasis` 生成标量候选池与缩并结构，`trainmtp` 把模板
训练成模型，`fastfit` 按拟合质量删减标量，`calc_efs`/`calc_err` 复核结果。适用于普通
线性头 MTP。

不涉及：LES/SOG、MTP-M 及独立神经网络头（`fastfit` 只重拟合普通线性部分）、
数据量或标签质量问题（见[训练、评估与部署](workflows_zh.md)）。

## 前置输入

| 文件 | 内容 | 来源 |
| --- | --- | --- |
| `index.map` | 与训练数据一致的元素顺序 | 模型或其训练记录 |
| `train.xyz`、`valid.xyz` | 训练集与验证集 | [训练流程](workflows_zh.md)的划分结果 |
| `test.xyz` | 独立最终测试集，不参与候选比较 | 同上 |
| `template.mtp` | 待训练的基组候选 | `mtmbasis` 输出 |

参数在训练集上拟合，候选在验证集上比较，最终测试集只用于复核。

## 判断问题是否出在基组

| 观察到的现象 | 该动哪里 |
| --- | --- |
| 增加数据后验证误差不再下降，且偏差集中在特定键角或配位环境 | 候选池可能缺少相应不变量：提高 `max_lev` 或 `max_nv` 后重新生成 |
| 模型拟合精度够用，但计算成本过高 | 用 `fastfit` 删减标量；径向函数保持不变 |
| 需要证明两套基组给出相同预测，或换布局后复用已有参数 | 用 `mtpbasis` 判等，不要比较文件长度或 moment 数量 |
| 数据不足、标签不一致，或径向函数尚未优化 | 不是基组问题，回到[训练流程](workflows_zh.md) |
| LES/SOG、MTP-M 或独立神经网络头 | 不在本页范围内 |

改基组通常会使已训练的模型作废，因此先排除数据与径向的问题。基组变小不会自动带来更低的
验证误差或更快的计算，两项的判据见 [MTP 基组生成、筛选与复核](../advanced/basis-generation-selection_zh.md) §1。

## 生成候选池

在独立工作目录中创建 `mtmbasis.ini`。下列键是顶层设置，不放在 `[pot]` 或 `[fit]` 下；
取值小是为了跑得快，不是生产精度推荐值：

```ini
# ===== Topology =====
max_lev             8
max_dim             3
max_mu              2
max_nv              2
max_poly            4
mu_wei              2
nv_wei              1
dim_wei             2

# ===== Search =====
mode                3
maxiter             20
w1                  1
w2                  4
```

生成与缩并搜索分两次运行，代价高的候选池可以只建一次、反复搜索：

```sh
imrmlp mtmbasis mtmbasis.ini --generate-only true
imrmlp mtmbasis mtmbasis.ini --alphapool alphapool.dat --potfile basis.mtp
```

`mtmbasis` 的布尔选项需要显式取值：单独写 `--generate-only` 会报缺少参数，必须写成
`--generate-only true`。搜索为每个被接受的候选写一个文件，命名为 `<前缀>_<编号>`；
`--potfile basis.mtp` 得到 `basis.mtp_0`，默认前缀则得到 `potential.mtp_<编号>`。
`conns_exp.log` 与各 `*_mapping_*.log` 记录每个候选用到哪些表达式和缩并路径。

搜索最小化 `w1 * moments_count + w2 * rules_count`，即中间 moment 与乘积累加规则的
结构计数。它不读取能量、力或应力，因此无法按精度给候选排序。

候选池很大时通常先采样：

```sh
imrmlp mtmbasis mtmbasis.ini --alphapool alphapool.dat --sample true --sample-rate 0.5 --generate-only true
```

`--sample` 写出 `sampled_alphas.dat` 后继续运行；再加 `--generate-only true` 才会把
采样后的池保存为 `alphapool.dat` 并退出，因此会覆盖当前目录中的原池，采样应在副本目录
里做。池采样使用 `random_device`，没有 seed 选项，复现要保留采样后的文件，而不是只记录
采样率。

## 训练模板

生成的文件是拓扑，不是势：只有一个元素、`RBChebyshev` 径向函数、2–5 Å 区间和 8 个径向
基函数，没有任何训练后的系数。要在未训练的文件上设置元素数与顺序、距离范围和径向基大小；
`radial_funcs_count` 由基组的 mu 索引决定，不能任意修改。训练后的模型也不能靠改表头
增加元素或径向维度。已有搜索好的拓扑时不必重跑这一步，可从 MTM-new 取用，见
[模板选择](workflows_zh.md)。

把选中的候选复制为 `template.mtp`，用后面筛选时相同的训练数据拟合：

```ini
# ===== Input/Output =====
trainset            train.xyz
validset            valid.xyz
trainset-format     extxyz
validset-format     extxyz
index               index.map

# ===== Potential =====
potfile             template.mtp

# ===== Loss Weights =====
fit:energy_weight   1
fit:force_weight    0.1
fit:stress_weight   0.001
fit:weight_L2       1e-8
```

```sh
imrmlp trainmtp train.ini --linear-fit
cp _Trained_Potential_.mtp full.mtp
```

在意精度时再加上径向优化，例如 `--do-lbfgs` 或 `trainnlpot --do-varpro`。这一步定下径向
函数：`fastfit` 固定它们，只重拟合线性部分，因此标量筛选能做的只是在径向给出的上限之内
取舍。优化器或 `--linear-fit` 要显式指定，单独运行 `trainmtp train.ini` 不执行任何非线性
优化。

## 基准与标量删减

`fastfit` 的设置文件同样使用顶层键：

```ini
# ===== Input/Output =====
trainset            train.xyz
validset            valid.xyz
format              extxyz
index               index.map

# ===== Potential =====
potfile             full.mtp

# ===== Loss Weights =====
weight-energy       1
weight-force        0.1
weight-stress       0.001
weight-l2           1e-8

# ===== Normalization =====
energy-power        1
force-power         0
stress-power        1

# ===== Reproducible Search =====
seed                42
```

`format` 同时作用于训练集与验证集，默认是 `mlpcfg`，数据为 `extxyz` 时必须显式指定。
模型的元素顺序要与数据一致。最终测试集不写进这份文件。

最短的“基准 → 筛选 → 导出”序列：

```sh
imrmlp fastfit fastfit.ini --mode 1 --num_threads 1
imrmlp fastfit fastfit.ini --mode 0 --num_threads 1
imrmlp fastfit fastfit.ini --mode 4 --discard-rate 0.25 --num_threads 1
imrmlp fastfit fastfit.ini --mode 2 --num_threads 1
```

| 步骤 | 产出 |
| --- | --- |
| `--mode 1` | `__Trained__.mtp`：全基组加权重拟合，作为误差基准 |
| `--mode 0` | 标量正规矩阵的 `EIGENSYS`、`EIGVEC`、`EIGVAL` |
| `--mode 4` | 按 CUR 重要度得分得到的 `selected_ones.dat` 与 `removed.dat` |
| `--mode 2` | 依该选集重拟合、压缩后的 `__REDUCED__.mtp` |

模式 4 需要模式 0 在同一目录生成的 EIGENSYS，不能只接在模式 1 之后运行。`--discard-rate`
是丢弃比例，`--addback` 会保留计算路径中用到的标量，所以最终数量可能与该比例不同。
`fastfit` 的布尔选项是不带值的开关，与 `mtmbasis` 不同。

细化选集而不一次删到位时，可用 `--mode 5 --search-steps ... --method greedy`，或另外
运行 `--mode 6 --population ... --generations ...`。两者都支持设置文件中的
`selection-objective error-cost`，用于在验证误差上限内追求速度，见
[按 E/F/S 成本选基](../advanced/fastfit-cost-selection_zh.md)。模式 7 按阈值、聚类或
VIF 去除相关标量。

## 复核缩减后的模型

先在验证集上比较完整模型与缩减模型，再对独立测试集评估选定模型：

```sh
imrmlp calc_efs --potfile __REDUCED__.mtp --input test.xyz --output test-predicted.xyz
imrmlp calc_err --input test.xyz --input-predict test-predicted.xyz --output-summary test-errors.txt
```

分列读取单原子能量、力与 virial 误差，并检查偏差最大的构型。速度由压缩后的模型、体系规模
和后端共同决定，标量数只是其中一项；压缩前后在同一批构型上实测耗时。缩减后再做非线性训练
会得到新的候选，需要重新走一遍验证。

压缩会重新编号标量，旧的 `selected_ones.dat` 不能用于压缩后的模型；模型、选集与输出应
一起归档。若只是重排计算图并要求预测不变，使用
`mtpbasis --source ... --target ... --report ...`，并在参数迁移后比较 E/F/virial。

## 常见错误

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| `--generate-only` 报缺少参数 | `mtmbasis` 的布尔选项需要显式取值 | 写成 `--generate-only true` |
| 原有候选池被覆盖 | `--sample` 配合 `--generate-only true` 会重写 `alphapool.dat` | 在副本目录中采样 |
| 训练报错或模型行为异常 | 修改了 `radial_funcs_count`，或给已训练模型增加元素 | 在未训练模板上改表头，模型文件不可后加元素 |
| 筛选结果与基准不可比 | 改动模型、数据、权重或归一化后沿用旧 `EIGVEC`、`EIGVAL` | 重跑 `--mode 0` |
| 压缩模型无法按选集加载 | 压缩重新编号了标量 | 选集与压缩模型配套保存 |
| 混淆了工具 | `mtpbasis` 比较基组、迁移参数；`mtmbasis` 生成基组 | 按用途选用 |

## 相关

- [MTP 基组生成、筛选与复核](../advanced/basis-generation-selection_zh.md)：生成器全部设置、四种搜索模式、fastfit 模式 0–7
- [按 E/F/S 成本选基](../advanced/fastfit-cost-selection_zh.md)：模式 5、6 的 `error-cost` 目标
- [MTP 完整示例](../../examples/mtp-workflow/README_zh.md)：生成两套模板、分别训练并二选一
- [命令参考](../reference/commands_zh.md)：`mtmbasis`、`fastfit`、`mtpbasis` 的全部选项
