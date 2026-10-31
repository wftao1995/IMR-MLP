# 势训练后的线性精修 {#doc_docs_mtp_linear_refit}

[English](linear-refit.md)

线性精修用于在固定非线性参数的前提下改善已训练模型。`trainnlpot` 对选中的系数
求解加权最小二乘问题，核验完整目标后保存结果。此功能默认关闭。它降低的是加权
训练目标；选择精修后的模型前，还应单独检查验证集误差。

## 精修已保存模型

将下列内容保存为 `refit.ini`，替换绝对路径并提前创建输出目录。数据单位及元素
顺序必须与原训练一致；示例权重仅作演示。

```text
# ===== Input/Output =====
index               /absolute/path/index.map
trainset            /absolute/path/train.xyz
trainset-format     extxyz

# ===== Potential =====
pot:pottype         mtp
pot:potfile         /absolute/path/input.mtp
pot:typemap         /absolute/path/index.map

# ===== Runtime =====
fit:device          cpu

# ===== Initialization =====
fit:random_init     false
fit:force_random    false

# ===== Training Schedule =====
fit:training_schedule single

# ===== Loss Function =====
fit:energy_weight   1
fit:force_weight    0.05
fit:stress_weight   0.001
fit:energy_pow      1
fit:force_pow       0
fit:stress_pow      1
fit:penalty_weight  1e-8

# ===== Linear Refit =====
fit:linear_refit    sr
fit:linear_refit_report /absolute/path/output/refit.json

# ===== Output =====
fit:save_to         /absolute/path/output/fitted.mtp
```

```sh
imrmlp trainnlpot --settings refit.ini --linear-refit-only \
  --output /absolute/path/output/fitted.mtp
```

输入模型必须已含初始化系数。只精修模式要求 `training_schedule single`，禁止随机
初始化及输入扰动。`--linear-refit-only` 不能与优化器选项同时使用。

## 选择要精修的系数

“仿射参数块”指其他参数固定后，预测值对该组系数呈线性关系。可用参数块取决于
模型及启用的 request 标志；不支持或为空的选择会报错。

| 模型 | `fit:linear_refit` | 修改内容 |
| --- | --- | --- |
| 普通 `mtp` | `sr` | 元素能量偏置和短程矩系数 |
| `mtp_les`，Coulomb LES v2 | `sr` | 短程系数；电荷和 Born 响应保持不变 |
| `mtp_les`，SOG v3/v4 | `sr` 或 `sr_sog` | 仅短程，或短程加全部通道的 Gaussian 权重 |
| `mtp_m`（`li`、`q2`、`fs`、`n1`、`n2`） | `head` / `output` | 受支持的仿射输出系数；隐藏特征保持不变 |
| 带 SOG 的 `mtp_m` | `output,sog` | 输出系数和 Gaussian 权重，须启用相应 request 标志 |
| `mtpnn1` | `head` / `output` | 网络末层权重和元素偏置 |
| `descriptor_linear` | `head` / `linear` | 线性描述符系数 |
| `hybrid_scale` | 如 `pot1/sr` | 指定子势的明确仿射参数块 |

`all` 选择模型声明的所有参数块；逗号分隔列表选择明确的块名。`head` 选择
`output`，若模型的输出头名为 `linear` 则选择后者。这些选项不会使任意非线性
参数变成线性参数。

对 SOG，径向参数、潜变量头和带宽保持不变。Gaussian 权重拟合包含潜变量响应力
和 virial、有符号权重、零模/自能约定及已有权重正则。应保留原正则化设置：精修
保留选中参数实际生效的 L2，拒绝非零的选中参数 L1，不自动启用自适应 ridge。

## 非线性训练结束后精修

在训练配置中加入 `fit:linear_refit sr`（或其他受支持的选择），并保留常规
`--do-lbfgs` 等选项，即会在所选非线性阶段后精修。CPU CG 配合
`training_schedule single` 时，在最后接受点执行一次终点精修；选中系数也必须
在 CG 可训练掩码中。GPU 精修及多阶段 CG 精修尚不支持。

终点精修不会改变 CG 终止原因，也不会将耗尽迭代预算的 CG 标为收敛。若希望在
**每次试探**中求解线性问题，同时优化径向参数，请使用[变量投影](mtp-native-varpro_zh.md)。

MPI 使用相同配置，例如：

```sh
LD_LIBRARY_PATH=$PWD/build/mpi:$LD_LIBRARY_PATH \
  mpirun -np 4 build/mpi/imrmlp-mpi trainnlpot --settings refit.ini \
  --linear-refit-only --output /absolute/path/output/fitted.mtp
```

先以 `BUILD_MODE=mpi` 构建，再从仓库根目录运行。每个 rank 都须能访问同一组
输入。并行结果允许浮点容差范围内的差异。

## 标签、核验与输出

精修复用能量/力/virial 权重、原子数幂次、fitness 和参考力归一化。支持 CPU HVP
及完整 Hessian 标签，采用与非线性训练一致的中心力差分。仅当模型支持总电荷且
选中系数不影响总电荷时，才可保留总电荷项；该项在精修中是常数。SOG 本身不支持
物理总电荷标签。

拟合使用内部完整九分量 virial。`calc_err` 与 VarPro 的应力分量 RMSE 为完整张量
平方误差除以有标签构型数的九倍后开方；GPa 指标先按各构型体积除 virial。
力矢量 RMSE 和力分量 RMSE 是不同指标。

求解前，输入模型保存为 `<fit:save_to>.pre-refit`。接受结果要求冻结参数不变、
完整目标不增、矩阵目标与完整预测一致，且模型保存/重载核验成功。失败会恢复系数
并返回错误。成功后写入 `fit:save_to` 和 CLI `--output` 目标；原生模型若需要附属
文件，则保留其文件布局。CG 还保留 `.cg.snapshot` 检查点。检查点保存系数，
不保存优化器历史。

| 设置 | 默认值 | 含义 |
| --- | --- | --- |
| `fit:linear_refit_solver` | `qr_svd` | 受支持的求解器 |
| `fit:linear_refit_rtol` | `1e-12` | 数值秩相对阈值 |
| `fit:linear_refit_atol` | `0` | 数值秩绝对阈值 |
| `fit:linear_refit_tolerance` | `1e-10` | 目标核验容差，按 `max(1, abs(objective_before))` 缩放 |
| `fit:linear_refit_report` | `linear-refit.json` | 目标、秩、参数增量、正则化及耗时报告 |

固定成分数据不能独立确定所有元素偏置。精修保留无法确定的组合，在其余缩放坐标
中求最小增量解。秩亏时有效解不唯一，系数会随求解器与阈值变化；比较预测值和目标是
唯一稳定的判据。独立验证/测试集只用于报告，不参与求解。
