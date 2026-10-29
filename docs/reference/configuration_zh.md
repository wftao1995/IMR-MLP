# 配置系统 {#guide_config}

[English](configuration.md)

MTP 势类型使用 `mtp`，训练命令使用 `trainmtp`。

@tableofcontents


## 概述

IMR-MLP 使用基于 `ConnectSettings` 的 ini 风格配置系统。
设置文件将参数组织在**带命名空间的前缀**下，这些前缀将值
路由到相应的子系统。

## 文件格式

```text
# 注释以 # 或 ! 开头
key value
section:key value
```

键与值用空白分隔。等号会成为值的一部分，通常会导致解析失败。此规则针对设置文件，
不适用于使用独立语法的 `.mtp` 势文件。相对路径从进程当前工作目录解析。

### 方括号 Section 语法

除了逐行为每个键添加前缀，还可以使用 `[section]` 标头
为后续所有键设置命名空间前缀：

```text
[pot]
pottype    mtp
potfile    potential.mtp

[relax]
driver         lbfgs
maxiteration   500
```

规则：
- `[section]` 标头为后续的 key-value 行设置命名空间前缀。
- section **仅作用于连续行** — 空行会结束当前 section，后续键恢复为顶层命名空间。
- 纯注释行（`# ...` 或 `! ...`）**不会**结束 section。
- 显式 `section:key` 应写在方括号 section 之外（先插入空行），否则会再次添加当前 section 前缀。
- 重复键保留**第一次**出现的值，不要依靠后面的同名键覆盖。

### 值类型

```text
# 标量
cutoff 5.0             # double
max_iter 500           # int
verbose yes            # bool (yes/no, true/false, y/n, t/f, 1/0)

# 向量用空格分隔，前提是该参数支持向量
relax:strain_mask 1 1 1 0 0 0

# 用行尾反斜杠延续一个设置值
fit:trainable_groups \
    sr_radial sr_linear \
    les_species les_moment
```

### 命名空间前缀

每个子系统前缀用于路由参数。无前缀的值将进入顶层设置容器。

| 前缀 | 子系统 |
|--------|-----------|
| *(无)* | 顶层 / 全局设置 |
| `relax:` | 结构弛豫 |
| `fit:` | 势训练（`trainmtp`、`trainnlpot`、`train_linear`） |
| `al:` | 弛豫中的外推监控和候选收集 |
| `neb:` | Nudged Elastic Band (NEB) |
| `dimer:` | Dimer 鞍点搜索（`calc_dimer`） |
| `pot:` | 势 |
| `builder:` | MaxVol、FPS/CUR 与拥挤度的描述符/B-matrix 构造 |
| `selector:` | MaxVol 活动集选择（`maxvol_*`） |
| `behler:` | Behler/ACSF 描述符设置 |
| `bondorder:` | Bond-order 描述符设置 |
| `shellfile:` | 外部 shell 提供者（`pottype shellfile`），见 @ref reference_potential_types |
| `descriptor_linear:` | 线性描述符势（`pottype descriptor_linear`） |
| `potN:` | `hybrid`/`hybrid_scale` 的第 N 个子势（N = 1, 2, ...） |

只有 `npbc_extent` 由 GPIO 层自身注册。常用键（`input`、`output`、
`input-format`、`output-format`、`index`）由各命令注册为顶层设置，
因此同样没有 `io:` 前缀。

当前实现中，`mtmbasis` 是一个例外：其生成器设置连接为顶层键，
而不是通过 `mtmbasis:` 前缀读取。

MaxVol、FPS/CUR 与拥挤度命令从 `builder:` 和 `selector:` 块读取构造器与选择器选项；
构造器内部的势和描述符子设置仍使用 `pot:`、`behler:`、`bondorder:` 等前缀。

---

## 弛豫 (`relax:`)

`relax` 的主要配置。

### 基本控制

| 键 | 类型 | 默认值 | 描述 |
|-----|------|---------|-------------|
| `relax:driver` | str | `lbfgs` | 优化算法：`lbfgs`、`lbfgspp`、`fire`、`cg`、`gd`、`mdmin`、`quickmin`、`sqp`、`nlc` |
| `relax:maxiteration` | int | `1000` | 最大优化迭代次数 |
| `relax:constraint_loops` | int | `5` | 应用约束时的最大外迭代次数 |
| `relax:stepmax` | float | `0.1` | 每次变量更新的最大允许步长 |
| `relax:pressure` | float | 0.0 | 外部静水压 (GPa) |
| `relax:external_stress` | str | — | 外部应力张量，3 或 6 个分量 (GPa) |
| `relax:stiffness` | str | — | 应变模式的刚度，6 个分量 (GPa) |

### 几何控制

| 键 | 类型 | 默认值 | 描述 |
|-----|------|---------|-------------|
| `relax:relax_atom` | bool | yes | 是否弛豫原子位置（可掩码） |
| `relax:relax_cell` | bool | no | 是否弛豫晶胞参数（可掩码） |
| `relax:fix_vol` | bool | no | 固定体积（与约束冲突） |
| `relax:fix_shape` | bool | no | 弛豫体积时固定晶胞形状 |
| `relax:cell_strain` | bool | yes | 使用基于应变的晶胞弛豫（更快，更稳定） |
| `relax:symm_strain` | bool | yes | 使用对称应变张量以防止刚性旋转 |
| `relax:atomic_force_recast` | bool | yes | 为应变一致性重铸力 |
| `relax:selective_dynamics` | bool | no | 启用选择性动力学过滤 |
| `relax:cartesian` | bool | no | 使用笛卡尔坐标 (yes) 或分数坐标 (no) |
| `relax:car_mode` | bool | no | cartesian 的别名 |
| `relax:couple` | int | — | 晶格矢量耦合模式：`0` 不耦合，`1` a-b，`2` b-c，`3` c-a，`4` 全耦合 |
| `relax:strain_mask` | str | — | 晶格/应变分量掩码（应变模式 6 个值，晶格模式 9 个值；1=弛豫 0=固定） |
| `relax:lattice_mask` | str | — | strain_mask 的别名 |

### 收敛容差

| 键 | 类型 | 默认值 | 描述 |
|-----|------|---------|-------------|
| `relax:gtol` | float | `1e-3` | 梯度容差 — 最大梯度低于此值时停止 |
| `relax:ftol` | float | `0` | 能量容差 — 能量变化低于此值时停止 |
| `relax:xtol` | float | `0` | 步长容差 — 位移变化低于此值时停止 |
| `relax:ztol` | float | `1e-6` | 冻结小梯度的零阈值 |
| `relax:eps` | float | `1e-6` | 防止除零的机器 epsilon（浮点精度上限） |

### 约束与过滤器

| 键 | 类型 | 默认值 | 描述 |
|-----|------|---------|-------------|
| `relax:with_constraint_aul` | bool | no | 增广拉格朗日方法 |
| `relax:with_constraint_sqp` | bool | no | 序列二次规划 |
| `relax:with_constraint_nlc` | bool | no | 非线性约束 |
| `relax:with_constraint_lc` | bool | no | 线性约束过滤器 |
| `relax:with_constraint_vib` | bool | no | VIB2 过滤器（变换坐标） |
| `relax:vibffilter` | bool | no | VIB 过滤器（无坐标变换） |
| `relax:with_linear_constraint` | bool | no | 线性约束 (LC) 过滤器 |
| `relax:with_linear_transform` | bool | no | 线性变换 (LF) 过滤器 |
| `relax:fixcm` | bool | no | 固定质心（仅调整梯度） |
| `relax:freeze_small_gd` | bool | no | 将低于阈值的梯度置零 |
| `relax:symm_force` | bool | no | 通过空间群对称化力/应力 |

### 约束文件

| 键 | 类型 | 描述 |
|-----|------|-------------|
| `relax:lc_file` | path | 线性约束定义文件 |
| `relax:lf_file` | path | 线性变换定义文件 |
| `relax:vib_file` | path | 振动简正模式文件 |
| `relax:seldyn_file` | path | 选择性动力学配置文件 |
| `relax:precond_file` | path | 预处理配置文件 |

### 轨迹输出

| 键 | 类型 | 默认值 | 描述 |
|-----|------|---------|-------------|
| `relax:write_traj` | bool | no | 写入优化轨迹 |
| `relax:traj_append` | bool | no | 追加到现有轨迹文件 |
| `relax:trajfile` | path | `relax-trajectory.xyz` | 输出轨迹文件名 |
| `relax:nblock` | int | `10` | 每 N 次迭代写入一次快照 |

### 初始条件

| 键 | 类型 | 默认值 | 描述 |
|-----|------|---------|-------------|
| `relax:apply_disp` | bool | no | 应用随机初始位移 |
| `relax:disp` | float | `0.01` | 位移幅度 |
| `relax:disp_gaussian` | bool | no | 高斯分布位移 |

### 驱动特定

所有驱动特定参数使用 `relax:` 前缀（表中缩写前缀为简写形式）：

| 键 | 默认值 | 描述 |
|-----|---------|-------------|
| `relax:bfgs_m` | `15` | L-BFGS 记忆参数 |
| `relax:gd:step_max_init_cell` | `0.1` | GD：晶胞初始最大步长 |
| `relax:gd:step_upscale` | `1.2` | GD：步长放大因子 |
| `relax:gd:step_downscale` | `0.2` | GD：步长缩小因子 |
| `relax:fire:ts_max` | `10.0` | FIRE：最大时间步长 |
| `relax:fire:ts_min` | `0.001` | FIRE：最小时间步长 |
| `relax:fire:ts_init` | `2.0` | FIRE：初始时间步长 |
| `relax:fire:alpha_start` | `0.25` | FIRE：初始阻尼 |
| `relax:fire:alpha_reduce` | `0.99` | FIRE：阻尼减小因子 |
| `relax:fire:ts_grow` | `1.1` | FIRE：时间步长增长因子 |
| `relax:fire:ts_reduce` | `0.5` | FIRE：时间步长减小因子 |
| `relax:fire:ndelay` | `20` | FIRE：延迟步数 |
| `relax:fire:do_backward` | `true` | FIRE：启用步长回溯 |
| `relax:fire:initial_delay` | `true` | FIRE：启用初始延迟 |
| `relax:fire:downhill_check` | `false` | FIRE：检查下坡条件 |
| `relax:lbfgspp:epsilon` | 见 LBFGSpp | LBFGSpp：梯度容差 |
| `relax:lbfgspp:past` | 见 LBFGSpp | LBFGSpp：收敛检查的历史步数 |
| `relax:lbfgspp:delta` | 见 LBFGSpp | LBFGSpp：收敛 Delta |
| `relax:lbfgspp:max_iterations` | 见 LBFGSpp | LBFGSpp：最大迭代次数 |
| `relax:lbfgspp:epsilon_rel` | 见 LBFGSpp | LBFGSpp：相对梯度容差 |
| `relax:lbfgspp:linesearch` | 见 LBFGSpp | LBFGSpp：线搜索算法 |
| `relax:lbfgspp:max_linesearch` | 见 LBFGSpp | LBFGSpp：最大线搜索次数 |
| `relax:lbfgspp:min_step` | 见 LBFGSpp | LBFGSpp：最小线搜索步长 |
| `relax:lbfgspp:max_step` | 见 LBFGSpp | LBFGSpp：最大线搜索步长 |
| `relax:lbfgspp:ftol` | 见 LBFGSpp | LBFGSpp：Armijo 参数 |
| `relax:lbfgspp:wolfe` | 见 LBFGSpp | LBFGSpp：Wolfe 参数 |
| `relax:lbfgspp:initial_step` | 见 LBFGSpp | LBFGSpp：初始线搜索步长 |
| `relax:lbfgspp:damping` | 见 LBFGSpp | LBFGSpp：曲率阻尼 |
| `relax:lbfgspp:first_initial_step` | 见 LBFGSpp | LBFGSpp：首次初始步长 |
| `relax:lbfgspp:first_step` | 见 LBFGSpp | LBFGSpp：首步长 |

弛豫驱动额外接受的键：

| 键 | 默认值 | 描述 |
|-----|---------|-------------|
| `relax:periodic_subspace` | 否 | 使用周期方向分数坐标与非周期方向笛卡尔坐标；变胞仅改变周期 Gram 矩阵 |
| `relax:dprecond` | `0` | 对角预条件强度（大于 9 的级别默认隐藏） |
| `relax:force_switchoff_scaling` | — | 在切换区平滑缩放力 |
| `relax:keep_symm` | 否 | 保持初始空间群对称性 |
| `relax:cell_factor` | — | 晶胞坐标步长缩放因子 |
| `relax:hessian_precond_file` | — | 用作预条件的 Hessian 文件 |
| `relax:sparse_precond` | 否 | 使用稀疏预条件 |
| `relax:sparse_shift` | — | 稀疏预条件对角项偏移 |
| `relax:precond_floor` | — | 预条件特征值下界 |
| `relax:fc_update_interval` | — | 力常数预条件更新间隔 |
| `relax:aul_auto` | — | 自动自适应步长控制 |
| `relax:thin_cell_scaling` | — | 薄/扁平晶胞的步长缩放 |
| `relax:symmetry_operations` | — | 约束弛豫的显式对称操作 |
| `relax:symmetry_sites` | — | 对称性位点映射 |
| `relax:rigid_groups` | — | 保持刚性的原子组 |
| `relax:gtol0` | — | 分级收敛的初始梯度容差 |
| `relax:first_step` | — | 首步步长覆盖 |
| `relax:index` | `index.map` | 弛豫局部物种映射（顶层 `index` 同样生效） |

---

## 训练 (`fit:`)

`trainmtp`、`trainnlpot` 和 `train_linear` 的参数。所有键使用 `fit:` 前缀。

这些表覆盖多种训练器；只有所选训练器实际提供的键才会生效。GPU、阶段日程、
VarPro 和条件线性精修控制属于 `trainnlpot`。可用 `imrmlp <command> --help`
查看该命令连接的设置。以下默认值不是完整训练方案；应根据数据选择标签权重和收敛容差。

@note GPU 键（`fit:device`、`fit:gpu_id` 及以下 `fit:gpu_*` 键）仅适用于
`trainnlpot`；`trainmtp` 和 `train_linear` 在 CPU 上运行。

### 数据集与 I/O

| 键 | 类型 | 描述 |
|-----|------|-------------|
| `fit:save_to` | path | 已接受迭代的系数检查点基础路径（`trainnlpot` 追加 `.iter-NNNNNN.mtp`） |
| `fit:save_every` | int | 每 N 个已接受的优化器迭代保存一次系数检查点 |
| `fit:lfit_eng_only` | bool | 线性拟合仅使用能量 |
| `fit:lfit_fastalgo` | bool | 使用快速线性拟合算法 |
| `fit:device` | `cpu`（默认）、`cuda`、`hip` 或 `dcu` | 训练计算后端；`dcu` 是 `hip` 的别名。GPU 支持兼容的普通 MTP、MTP-M 与 SOG 模型，限制见 [GPU 指南](../development/cuda_zh.md) |
| `fit:gpu_id` | int | GPU 设备编号；`-1` 按节点内 MPI rank 自动映射 |

### GPU 训练

| 键 | 类型 | 默认值 | 描述 |
|-----|------|---------|-------------|
| `fit:gpu_batch_atoms` | int | 0 | 每个保持结构完整性的 GPU 分块的最大原子数；`0` 自动选择 |
| `fit:gpu_batch_structures` | int | 0 | 每个 GPU 分块最大结构数；`0` 自动选择 |
| `fit:gpu_memory_fraction` | float | 0.75 | 每个分块可使用的当前空闲显存比例 |
| `fit:gpu_device_loss` | bool | true | 在 GPU 上构造 E/F/S 残差、指标和损失伴随量 |
| `fit:gpu_deterministic_reduction` | bool | false | 使用有序、可复现的 GPU 归约进行诊断 |
| `fit:gpu_radial_parameter_tile` | int | 128 | 每个径向参数部分梯度分块包含的参数数目 |
| `fit:gpu_cache_batches` | bool | true | 在优化器重复回调间缓存固定全量数据的主机端打包批次和标签 |

缓存可避免在重复全量回调中重建 CSR 拓扑和标签，代价是与打包数据量
成比例的主机内存。随机或部分 SGD 选择不会使用全量缓存。多 GPU 采用
一进程一卡，梯度继续通过现有 MPI 路径归约。

### 损失权重

能量、力和应力的权重如下。HVP 标签及其训练权重见
[HVP 训练指南](../advanced/hvp-training_zh.md)。

| 键 | 类型 | 默认值 | 描述 |
|-----|------|---------|-------------|
| `fit:energy_weight` | float | 1.0 | 能量损失权重 |
| `fit:force_weight` | float | 0.01 | 力损失权重 |
| `fit:stress_weight` | float | 0.0001 | 应力损失权重 |
| `fit:energy_pow` | float | 1.0 | 能量权重幂缩放 |
| `fit:force_pow` | float | 0.0 | 力权重幂缩放 |
| `fit:stress_pow` | float | 1.0 | 应力权重幂缩放 |
| `fit:penalty_weight` | float | 1e-6 | 正交性惩罚权重 |
| `fit:force_wgt_d` | float | 0.1 | 力权重归一化分母 d |
| `fit:force_wgt_f` | float | 0 | 力权重归一化因子 f |
| `fit:force_wgt_p` | float | 1.0 | 力权重归一化幂 p |

### 正则化

| 键 | 类型 | 默认值 | 描述 |
|-----|------|---------|-------------|
| `fit:weight_L1` | float | 0 | L1 正则化权重 |
| `fit:weight_L2` | float | 1e-10 | L2 正则化权重 |
| `fit:weight_rad_w0` | float | 1e-8 | 径向光滑性：0 阶导数 |
| `fit:weight_rad_w1` | float | 0 | 径向光滑性：1 阶导数 |
| `fit:weight_rad_w2` | float | 0 | 径向光滑性：2 阶导数 |
| `fit:radial_reg_type` | int | 0 | 径向正则化类型 |
| `fit:quad_count` | int | 100 | 径向积分的求积点数 |
| `fit:safe_fit` | bool | true | 安全（较慢）线性拟合模式 |

### 初始化

| 键 | 类型 | 默认值 | 描述 |
|-----|------|---------|-------------|
| `fit:random_init` | bool | no | 随机初始化系数 |
| `fit:rand_low` | float | -0.1 | 随机初始化：下限 |
| `fit:rand_up` | float | 0.1 | 随机初始化：上限 |
| `fit:force_random` | bool | no | 强制通用非线性训练器 (`trainnlpot`) 随机初始化 |
| `fit:force_randomlinear` | bool | no | 强制随机初始化线性系数 |
| `fit:force_randomradial` | bool | no | 强制随机初始化径向系数 |
| `fit:skip_preinit` | bool | no | 跳过预初始化步骤 |
| `fit:pre_init_iter` | int | 75 | 预初始化迭代次数 |
| `fit:apply_disp` | bool | no | 对训练数据施加位移 |
| `fit:disp_sigma` | float | 1e-6 | 位移标准差 |

### SGD 设置

| 键 | 类型 | 默认值 | 描述 |
|-----|------|---------|-------------|
| `fit:learning_rate` | float | 0.0001 | 初始学习率 |
| `fit:momentum` | float | 0.1 | 动量系数 |
| `fit:lr_scheduler` | str | none | 学习率调度：`none`、`exponential`、`step` 或 `cosine` |
| `fit:lr_decay_rate` | float | 0.95 | `step` 的乘性因子；`exponential` 中 `exp(-rate * t)` 的 rate，`t` 为小批次更新数 |
| `fit:lr_decay_steps` | int | 10 | 仅用于 `step`，每个衰减区间的小批次更新数 |
| `fit:lr_final` | float | 0.0 | 最终学习率下限 |
| `fit:lr_warmup_steps` | int | 0 | 可选的线性预热步数，作用于所选调度器 |
| `fit:lr_warmup_init` | float | 0.0 | 可选预热的初始学习率 |
| `fit:batch_size` | int | 100 | SGD 小批量大小 |
| `fit:epoch_limit` | int | 100 | 最大训练 epoch 数 |
| `fit:clip_gradient` | float | 0.0 | 梯度裁剪阈值（0 = 禁用） |

### BFGS 特定

| 键 | 类型 | 默认值 | 描述 |
|-----|------|---------|-------------|
| `fit:bfgs_m` | int | 5 | L-BFGS 记忆（保存的步数） |
| `fit:bfgs_conv_tol` | float | 1e-3 | BFGS 收敛容差（函数变化） |
| `fit:bfgs_grad_tol` | float | 1e-3 | BFGS 梯度范数容差 |
| `fit:bfgs_step_tol` | float | 1e-3 | BFGS 步长容差 |
| `fit:bfgs_maxiteration` | int | 1000 | BFGS 最大迭代次数 |
| `fit:bfgs_iteration_offset` | int | 0 | 系数重启后给已接受迭代日志和检查点编号增加偏移；ALGLIB 历史会重置 |
| `fit:linear_fit_bfgs` | bool | no | ALGLIB BFGS 在各阶段第 25、100、200 步及训练结束时线性拟合；ALGLIB 按已接受迭代计数，拟合后重启并保留累计步数 |
| `fit:lfit_eng_only` | bool | no | 线性拟合仅使用能量 |
| `fit:lfit_fastalgo` | bool | yes | 使用快速线性拟合算法 |
| `fit:bfgs_rescale` | bool | no | BFGS 后重新缩放变量 (`trainmtp`) |
| `fit:set_scale` | bool | no | 从外部文件设置变量缩放 |
| `fit:precondition` | bool | no | 使用对角 Hessian 预处理 |
| `fit:fix_linear_coeffs` | bool | no | 固定线性 MTP 系数 (`trainmtp`) |
| `fit:fix_species_coeffs` | bool | no | 固定物种系数 (`trainmtp`) |
| `fit:fix_radial_coeffs` | bool | no | 固定径向系数 (`trainmtp`) |
| `fit:do_orthogonalization` | bool | yes | 训练时对 MTP 基组正交化 (`trainmtp`) |

### LBFGS++ 嵌套键

`trainmtp` 还会在 `[fit]` 下读取 LBFGS++ 控制键。使用 bracket 语法时写作
`lbfgspp:epsilon`、`lbfgspp:maxit` 等；直接前缀语法则写完整键，例如
`fit:lbfgspp:epsilon`。

| 键 | 描述 |
|-----|------|
| `fit:lbfgspp:epsilon` | 绝对梯度容差 |
| `fit:lbfgspp:epsgmax` | 最大梯度分量容差 |
| `fit:lbfgspp:epsrel` | 相对梯度容差 |
| `fit:lbfgspp:m` | L-BFGS 记忆 |
| `fit:lbfgspp:damping` | 阻尼因子 |
| `fit:lbfgspp:init_step` | 初始 line-search 步长 |
| `fit:lbfgspp:first_step` | 第一次初始步长 |
| `fit:lbfgspp:past` | 用于收敛判断的历史迭代数 |
| `fit:lbfgspp:delta` | 函数变化收敛阈值 |
| `fit:lbfgspp:maxit` | 最大 LBFGS++ 迭代数 |
| `fit:lbfgspp:linesearch` | line-search 算法选择 |
| `fit:lbfgspp:max_linesearch` | 最大 line-search 尝试次数 |
| `fit:lbfgspp:min_step` | 最小 line-search 步长 |
| `fit:lbfgspp:max_step` | 最大 line-search 步长 |
| `fit:lbfgspp:ftol` | Armijo 条件参数 |
| `fit:lbfgspp:wolfe` | Wolfe 条件参数 |
| `fit:lbfgspp:f_dec_coeff` | 函数下降系数 |
| `fit:lbfgspp:s_curv_coeff` | 曲率系数 |
| `fit:lbfgspp:cautious_factor` | 谨慎更新（cautious update）因子 |
| `fit:lbfgspp:max_move` | 坐标空间最大移动量 |

训练器额外读取的 `fit:` 键：

| 键 | 默认值 | 描述 |
|-----|---------|-------------|
| `fit:do_ladder` | 否 | 启用径向拟合阶梯调度 |
| `fit:ladder_cut` | — | 阶梯截断级别 |
| `fit:ladder_last` | — | 最后一个阶梯阶段 |
| `fit:popsize` | — | 差分进化种群规模 |
| `fit:df_orthogonal_constraints` | — | 差分进化的正交约束 |
| `fit:linear_precondition` | — | 线性拟合预条件模式 |
| `fit:rescale_mode` | — | 系数缩放模式 |
| `fit:hvp_weight` | `0` | Hessian-向量标签权重 |
| `fit:hvp_pow` | `0` | HVP 原子数幂次 |
| `fit:hvp_fd_step` | `1e-4` | HVP 有限差分步长 |
| `fit:hessian_weight` | `0` | 稠密 Hessian 标签权重 |
| `fit:hessian_pow` | `0` | Hessian 原子数幂次 |
| `fit:hessian_fd_step` | `1e-4` | Hessian 有限差分步长 |
| `fit:objective-convention` | `legacy` | `legacy` 或共享 mean-moments 目标 |
| `fit:l2-reference` | `native` | `native`、`mean` 或 `sum` |
| `fit:objective-report` | — | 可选的 TSV 目标报告 |
| `fit:validation_interval` | `1` | 验证求值间隔迭代数 |
| `fit:charge_weight` | `0` | 电荷标签权重 |
| `fit:charge_pow` | `1` | 电荷原子数幂次 |
| `fit:charge_target_attrib` | `total_charge` | 电荷目标所在属性 |
| `fit:charge_default_target` | `0` | 缺省电荷目标 |
| `fit:training_schedule` | `single` | `single`、`mtp_les_from_scratch`、`mtp_les_resume`、`mtp_sog_from_scratch`、`mtp_sog_resume`；仅 `trainnlpot` |
| `fit:varpro_report` | `varpro.jsonl` | VarPro 报告路径 |
| `fit:varpro_radial_scale_file` | — | VarPro 径向缩放文件 |
| `fit:varpro_linear_refinement_steps` | `2` | VarPro 内线性精化步数 |
| `fit:varpro_linear_gradient_tolerance` | `1e-7` | VarPro 内线性梯度容差 |
| `fit:validation_metric` | `force` | 验证结果排序指标 |
| `fit:refit_train_valid` | 否 | 在 train+valid 合并集上重拟合 |
| `fit:regularization_search` | 否 | 启用正则化乘子搜索 |
| `fit:regularization_multipliers` | — | 候选正则化乘子 |
| `fit:validation_max_energy` | `-1` | 验证能量上限 |
| `fit:validation_max_force` | `-1` | 验证力上限 |
| `fit:validation_max_stress` | `-1` | 验证应力上限 |
| `fit:linear_refit` | `off` | 关闭或选择可线性拟合的参数块：`sr`、`sr_sog`、`head`、`all` 或模型声明的块名；见[线性精修](../advanced/linear-refit_zh.md) |
| `fit:linear_refit_solver` | `qr_svd` | 线性精修求解器 |
| `fit:linear_refit_report` | `linear-refit.json` | 线性精修报告路径 |
| `fit:linear_refit_rtol` | `1e-12` | 数值秩的相对阈值 |
| `fit:linear_refit_atol` | `0` | 数值秩的绝对阈值 |
| `fit:linear_refit_tolerance` | `1e-10` | 精修目标核验的相对容差 |
| `fit:trainable_groups` | — | 暴露给优化器的命名参数组 |
| `fit:sr_stage_iterations` | `100` | 短程阶段迭代数 |
| `fit:les_stage_iterations` | `50` | LES 阶段迭代数 |
| `fit:joint_stage_iterations` | `0` | 联合阶段迭代数 |
| `fit:charge_warmup_factor` | `10` | 电荷预热倍率 |
| `fit:les_zero_init_range` | `1e-3` | LES 零初始化范围 |
| `fit:sog_zero_init_range` | `0.1` | SOG 零初始化范围 |
| `fit:les_zero_init_seed` | `5489` | LES 初始化种子 |
| `fit:curvature_precondition` | — | 曲率预条件模式 |
| `fit:gn_probes` | — | Gauss-Newton 探针数 |
| `fit:gn_update` | — | Gauss-Newton 更新间隔 |
| `fit:gn_seed` | — | Gauss-Newton 探针种子 |
| `fit:gn_floor` | — | Gauss-Newton 特征值下界 |
| `fit:parameter_scaling` | `none` | 逐参数缩放模式 |
| `fit:scale_floor` | `1e-3` | 参数缩放下界 |
| `fit:scale_ceiling` | `0.1` | 参数缩放上界 |
| `fit:scaled_step_max` | `0` | 缩放坐标中的最大步长 |

---

## 主动学习 (`al:`)

`relax` 通过 `ALPotential` 和 `ALModifier`，根据已保存的 MaxVol 主动集监控外推程度。
等级超过 `threshold` 的构型进入候选池；超过 `threshold_break` 时先收集构型，
再停止该结构的弛豫。输入结构处理结束后，对候选池采样并导出。
参考标注与模型重训练是独立步骤。

| 键 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `al:mode` | str | `none` | `maxvol` 启用监控，`none` 关闭 |
| `al:state_file` | path | MaxVol 必需 | 已保存的主动集（`.sel`，v10） |
| `al:index_file` | path | — | 评分器可选的物种索引表 |
| `al:threshold` | float | 2.01 | 超过此等级时收集候选 |
| `al:threshold_break` | float | 100.0 | 超过此等级时由调用方停止弛豫 |
| `al:interval` | int | 1 | 每 N 次求值检查一次，必须为正数 |
| `al:export_file` | path | `al_candidates.xyz` | 候选输出文件 |
| `al:sparsify_to` | int | 1000 | 导出候选池的最大大小；非正数清空候选池 |
| `al:sparsify_method` | str | `fps` | `fps` 或 `random` |
| `al:source` | str | `external` | `external` 外部活动集或 `embedded` GPM 内嵌会话 |
| `al:committee_prefix` | str | — | committee 成员模型文件前缀 |

在普通弛豫配置中加入以下 section，再执行 `imrmlp relax relax.ini`：

```text
[al]
mode            maxvol
state_file      active.sel
threshold       2.01
threshold_break 100
export_file     candidates.xyz
```

---

## NEB (`neb:`)

`calc_neb` 的 Nudged Elastic Band (NEB) 参数。

| 键 | 类型 | 默认值 | 描述 |
|-----|------|---------|-------------|
| `neb:spring` | float | `5.0` | 镜像间的弹簧常数 |
| `neb:climb` | bool | no | 启用爬坡镜像 NEB (CI-NEB) |
| `neb:climb_mode` | string | `highest` | 选择最高能量镜像（标准 CI-NEB）或全部 `all_local` 局部极大值 |
| `neb:climb_delay` | int | `0` | 延迟 N 步启动 CI-NEB；`stable-v1` 按接受步计数，`legacy` 按力评估计数 |
| `neb:vcneb` | bool | no | 启用变胞 NEB |
| `neb:variablespring` | bool | no | 可变弹簧常数 |
| `neb:iref` | int | `-1` | 参考晶胞：-1=最终，-2=平均 |
| `neb:selective_dynamics` | bool | no | 选择性动力学 |
| `neb:cmask` | str | — | 晶胞参数掩码（9 个分量） |
| `neb:symm_strain` | bool | yes | 对称应变张量 |
| `neb:recast_atomic_force` | bool | yes | 为 VC-NEB 配合 symm_strain 重铸力 |
| `neb:pressure` | float | 0.0 | VC-NEB 的外部压力 |
| `neb:write_traj` | bool | no | 写入优化轨迹 |
| `neb:trjfile` | path | — | 轨迹文件名 |
| `neb:trjformat` | str | `mlpcfg` | 轨迹文件格式 |
| `neb:nblocks` | int | `5` | 轨迹输出间隔；`stable-v1` 按接受步计数，`legacy` 按力评估计数 |
| `neb:cellweight` | float | 100.0 | VC-NEB 切线、弹簧与广义力统一使用的晶胞坐标度量权重 |
| `neb:min_images` | int | 5 | 最小总镜像数（包含端点） |
| `neb:max_images` | int | 20 | 最大总镜像数（包含端点） |
| `neb:solver_policy` | str | `legacy` | 数值策略：`legacy` 或 `stable-v1` |
| `neb:max_cell_strain` | float | — | 单步最大晶胞应变 |
| `neb:max_force_evaluations` | int | — | 力求值次数上限 |
| `neb:diagnostics` | path | 空 | 可选的接受状态诊断 CSV 文件名 |
| `neb:vally` / `neb:valley` | bool | 否 | 启用 valley 路径修正 |
| `neb:pin` | bool | 否 | 固定爬坡和 valley 镜像 |
| `neb:ciival` | 6 floats | `100 100 100 50 50 50` | 晶胞坐标缩放正权重，顺序为 `xx yy zz yz xz xy` |
| `neb:apply_scal` | bool | — | 应用标量路径修正 |
| `neb:omp_mode` | str | — | 力求值的 OpenMP 调度模式 |
| `neb:check_mic` | bool | 是 | 检查最小镜像约定 |
| `neb:restore_F_cholesky` | bool | — | 在 Cholesky 坐标系恢复力 |
| `neb:periodic_subspace` | bool | 否 | 在部分周期边界变胞 NEB 中使用周期/非周期混合坐标 |
| `neb:keep_symm` | bool | 否 | 保持路径初始对称性 |
| `neb:symprec` | float | `1e-5` | 对称性容差 |
| `neb:symmetry_operations` | — | — | 显式对称操作 |
| `neb:symmetry_diagnostics` | path | 空 | 可选的对称性报告文件名 |
| `neb:perturb_amplitude` | float | — | 路径扰动幅度 |
| `neb:perturb_seed` | int | — | 路径扰动种子 |
| `neb:symmetry_release` | path | 空 | 运行结束后对称性释放检查的新输出目录；空值禁用 |
| `neb:symmetry_release_trials` | int | — | 对称释放尝试次数 |
| `neb:symmetry_release_seed` | int | — | 对称释放种子 |
| `neb:symmetry_release_steps` | int | — | 对称释放优化步数 |
| `neb:symmetry_release_evaluations` | int | — | 对称释放力求值次数 |
| `neb:symmetry_release_amplitude` | float | — | 对称释放位移幅度 |
| `neb:symmetry_release_strain` | float | — | 对称释放晶胞应变 |
| `neb:symmetry_release_energy_tol` | float | — | 对称释放能量容差 |
| `neb:traj` | path | — | `neb:trjfile` 的别名 |
| `neb:traj-format` | str | — | `neb:trjformat` 的别名 |
| `neb:springmax` | float | — | 最大弹簧常数 |
| `neb:springmin` | float | — | 最小弹簧常数 |
| `neb:variableimage` | bool | 否 | 优化过程中插入/删除镜像 |
| `neb:img_insert_thresh` | float | — | 变镜像插入阈值 |
| `neb:img_remove_thresh` | float | — | 变镜像删除阈值 |
| `neb:img_adjust_interval` | int | — | 镜像调整间隔 |
| `neb:ttype` | int | 1 | 切线类型：1 改进型，2 带保护的原始型，3 EB |
| `neb:stype` | int | 1 | 弹簧投影类型，通常与 `ttype` 相同 |
| `neb:mdist_r` | float | — | 路径过滤使用的最小距离半径 |
| `neb:mdist_factor` | float | — | 最小距离惩罚因子 |
| `neb:mdist_power` | float | — | 最小距离惩罚幂次 |
| `neb:apply_mindist` | bool | — | 应用最小距离惩罚 |
| `neb:init_method` | str | — | 路径初始化方法 |
| `neb:idpp_fmax` | float | — | IDPP 力收敛目标 |
| `neb:idpp_max_steps` | int | — | IDPP 迭代上限 |
| `neb:neb_ftol` | float | — | NEB 力容差 |
| `neb:gtol` | float | — | `neb:neb_ftol` 的别名 |
| `neb:check_saddle` | bool | 否 | 收敛后校验鞍点 |
| `neb:debug` | int | 0 | 调试输出级别 |

顶层 `cellweight` 和 `--cellweight`/`-cwt` 已弃用，请使用 `neb:cellweight`。两者同时提供时顶层值优先，并输出警告。


---

## 势 (`pot:`)

`PotMTP` 及其他势的 MTP 特定设置，在设置文件的 `[pot]` 部分配置。

| 键 | 类型 | 默认值 | 描述 |
|-----|------|---------|-------------|
| `pot:potfile` | path | — | 要加载的势文件 |
| `pot:pottype` | str | `mtp` | 势类型标识符。全部 21 种类型与文件格式见 @ref reference_potential_types "势类型总览"：`mtp`、`mtpnn1`、`mtp_m`、`hmtp`、`shellfile`、`eamalloy`、`eamfs`、`eampoly`、`openkim`、`acelmp`、`mindist`、`lj612`、`zbl`、`mtp_les`、`descriptor_linear`、`pair_table`、`threebody_table`、`sw`、`tersoff`、`hybrid`、`hybrid_scale` |
| `pot:radial_split` | bool | no | 每对物种使用独立的径向基 |
| `pot:typemap` | str | — | 势的显式元素顺序 |
| `pot:kim_model` | str | — | `pottype openkim` 使用的 OpenKIM 模型名 |
| `pot:potlevel` | int | — | `hmtp` 实际使用的子势数量 |
| `pot:radial_local_cache` | bool | — | 缓存局域径向值 |
| `pot:full_symmetry_reuse` | bool | — | 在多次求值间复用完整对称展开 |
| `pot:cutoff` | float | — | `mindist` 的截断半径 |
| `pot:factor` | float | — | `mindist` 的最小距离惩罚因子 |
| `pot:power` | float | — | `mindist` 的最小距离惩罚幂次 |
| `pot:les_kernel` | str | — | LES 核函数族 |
| `pot:les_recurrence` | — | — | LES 递推模式 |
| `pot:les_cache` | bool | — | 缓存 LES 中间量 |
| `pot:sog_terms` | int | — | SOG 项数 |
| `pot:sog_channels` | int | — | SOG 通道数 |
| `pot:sog_log_bandwidth_min` | float | — | 对数带宽下界 |
| `pot:sog_log_bandwidth_max` | float | — | 对数带宽上界 |
| `pot:lr_sigma` | float | — | 长程平滑宽度 |
| `pot:les_dl` | float | — | LES 有限差分步长 |
| `pot:les_taper_fraction` | float | — | LES 渐隐比例 |
| `pot:les_cutoff_mode` | str | — | LES 截断模式 |
| `pot:les_init_range` | float | — | LES 初始化范围 |
| `pot:les_balance_initial_charges` | bool | — | 平衡初始电荷 |
| `pot:les_hessian_fd_nudge` | float | — | Hessian 有限差分微扰 |
| `pot:remove_self_interaction` | bool | — | 移除长程自相互作用 |
| `pot:les_response_parity` | str | — | 响应宇称约定 |

@note `level`、`species_count`、`min_dist`、`max_dist`、`radial_basis_size` 和
`radial_basis_type` 是 `.mtp` 文件格式的标签，不是设置键——在 `.ini` 中设置它们
不会生效；这些值从势文件本身读取。

---

## 矩张量基组生成

`imrmlp mtmbasis` 的设置。这些键在设置文件顶层读取。

### 池生成

生成器枚举满足以下加权级别的矩张量乘积：

```text
level = mu_wei * sum(mu_i) + nv_wei * sum(nu_i) + dim_wei * dim
```

并保留 `level <= max_lev` 的项。`max_dim` 还会在内部被限制为
`min(max_dim, max_lev / 2)`。

| 键 | 类型 | 默认值 | 描述 |
|-----|------|---------|-------------|
| `max_lev` | int | 16 | 生成标量不变量的最大加权级别 |
| `max_dim` | int | 4 | 内部限制前，单个标量不变量中矩张量数量的最大值 |
| `max_mu` | int | 4 | 未提供 `--moments` 时的最大径向指标 `mu` |
| `max_nv` | int | 10 | 未提供 `--moments` 时的最大张量秩 `nu` |
| `max_poly` | int | 10000 | 单个标量不变量中张量秩之和的最大值 |
| `seed` | int | — | GA 缩并搜索的随机种子 |
| `elite_count` | int | — | 每代保留的精英个体数 |
| `initial_population` | int | — | GA 初始种群规模 |
| `mu_wei` | float | 2 | 级别公式中 `mu` 的权重 |
| `nv_wei` | float | 1 | 级别公式中 `nu` 的权重 |
| `dim_wei` | float | 2 | 级别公式中标量维度的权重 |
| `kselect` | int | 2 | 构建缩并候选时保留的低维分解组数量 |
| `add_new_scalar_to_pot` | bool | yes | 将新发现的标量中间量加入生成的矩列表 |
| `sample_rates` | str | — | `--advanced-sample` 使用的空格分隔采样率；依次对应维度 2、3、... |

提供 `--moments`/`-mts` 时，矩张量对文件的第一个值是数量，
后续列出相同数量的 `mu nu` 对。未提供时，会考虑所有
`0 <= mu <= max_mu` 且 `0 <= nu <= max_nv` 的组合。

生成器以确定的规范顺序枚举 1–7 维缩并图，只有在同一顶点置换下
`mu` 标签和缩并矩阵都完全一致时才去重。矩张量对和 alpha pool 输入会进行
严格校验。由于不再使用旧版手写枚举，重新生成的 tensor/moment ID 可能改变；
应将 pool 与 restart/select 诊断文件成套重新生成。

### 缩并搜索

`mtmbasis` 先构建所有有效缩并树，然后最小化
`w1 * moments + w2 * contraction_rules`。加权代价相同时优先选择缩并规则更少的
候选，再优先选择矩更少的候选。模式 3 会重复进行坐标扫描，直到没有分支能改善
这一排序。真实拓扑生成过程中学到的逐连接精确计数可用于提前排除不会改善的候选；
任何可能成为最优解的候选仍会完整生成并通过拓扑校验。
精确计数始终启用。

| 键 | 类型 | 默认值 | 描述 |
|-----|------|---------|-------------|
| `mode` | int | 0 | 搜索模式：0 为模拟退火式温度调度（tempering），1 为随机搜索加局部改进，2 为遗传算法，3 为收敛到局部最优的坐标下降 |
| `T_init` | float | 200 | `mode = 0` 的初始温度 |
| `w1` | float | 1.0 | 矩数量的代价权重 |
| `w2` | float | 4.0 | 缩并规则数量的代价权重 |
| `uacc` | float | 0.5 | `mode = 0` 中代价不变移动的接受概率 |
| `maxiter` | int | 1000000 | 搜索迭代次数 |
| `restart` | bool | no | 可用时从已写出的选择文件重新开始 |
| `randominit` / `rndinit` | bool | no | 使用随机初始树选择 |
| `chop_trees_first` | bool | yes | 初始构建期间剪除不一致的树 |
| `chop_trees` | bool | yes | 后续搜索/评估期间剪除不一致的树 |
| `reassign` | bool | no | 剪枝改变可用选择时重新分配选中的分支 |
| `filter_init` | bool | no | 初始处理阶段应用重用过滤 |
| `filter_second` | bool | no | 第二处理阶段应用重用过滤 |

### 遗传算法控制

这些键由 `mode = 2` 使用。

| 键 | 类型 | 默认值 | 描述 |
|-----|------|---------|-------------|
| `popsize` | int | 10 | 保留的种群成员数量 |
| `childsize` | int | 30 | 每次迭代生成的子候选数量 |
| `kfit` | float | 0.002 | 适应度 sigmoid 斜率 |
| `cfit` | float | 0.01 | 适应度 sigmoid 偏移 |
| `frac_mutation_min` | float | 0.2 | 变异比例下限 |
| `frac_mutation_max` | float | 1.0 | 变异比例上限 |
| `max_mutation_sites` | int | 4 | 单次最多变异的基因位置数量 |
| `min_ex_frac` | float | 0.1 | 交叉交换比例下限 |
| `max_ex_frac` | float | 0.5 | 交叉交换比例上限 |
| `rstart` | int | 0 | 受限搜索包含的第一个基因索引 |
| `rstop` | int | -1 | 受限搜索包含的最后一个基因索引；-1 表示到末尾 |

典型最小文件：

```text
max_lev 16
max_dim 4
max_mu 4
max_nv 10

mu_wei 2
nv_wei 1
dim_wei 2

mode 2
popsize 10
childsize 30
maxiter 1000000
```

---

## I/O

通用 I/O 设置。这些是顶层键（没有 `io:` 前缀）：`npbc_extent` 由
`GPIO::InitSettings` 注册，而 `input`、`output`、`input-format`、
`output-format`、`index` 由各命令注册。

| 键 | 类型 | 默认值 | 描述 |
|-----|------|---------|-------------|
| `input` | path | — | 输入文件（格式自动检测） |
| `output` | path | — | 输出文件 |
| `input-format` | str | — | 强制输入格式 |
| `output-format` | str | — | 强制输出格式 |
| `npbc_extent` | float | 5.0 | 非 PBC 结构的晶胞填充 |
| `index` | path | — | 种类索引文件 |

---

## 混合势

提供两种组合方式：

- `pottype = hybrid` 直接累加各子势的能量、力和应力。
- `pottype = hybrid_scale` 根据局域特征和平滑归一权重混合 MTP 或
  EAMPoly 的原子局域能。

| 键 | 类型 | 默认值 | 描述 |
|-----|------|---------|-------------|
| `npot` | int | — | 要组合的子势数量 |
| `spcount` | int | 自动推断 | 物种数量；必须与所有子势一致 |
| `potN:pottype` | str | — | 第 `N` 个子势的类型 |
| `potN:potfile` | path | — | 第 `N` 个子势的参数文件 |
| `feature` | str | `smooth_coordination` | `smooth_coordination` 或 `min_distance` |
| `transition` | str | `smoothstep` | 五次平滑过渡函数 |
| `feature_r0` | float | 0 | 平滑配位特征的内半径 |
| `feature_r1` | float | 组合势截断半径 | 平滑配位特征的外半径 |
| `boundaryN` | 2 floats | — | 从第 `N` 个势向后过渡的递增特征区间 |

三个及以上子势的有序过渡区可以重叠；顺序切棍（stick-breaking）权重始终非负
且总和为 1。使用 `trainnlpot` 时，`hybrid_scale` 将子势参数拼接为
`pot1/all` 等参数组，过渡边界和特征半径保持固定。保存训练结果会生成
一个清单及相邻的 `.potN` 子势文件，可通过 `pot:potfile` 重新加载清单。

`hybrid` 使用 MTP 共享物种参数布局，并通过
`radial`、`species`、`moments` 参数组接入 `trainnlpot`；固定的 ZBL 子势
不会增加可训练参数。

```text
pot:pottype hybrid_scale
pot:npot 2
pot:feature smooth_coordination
pot:feature_r0 0.0
pot:feature_r1 5.0
pot:boundary1 0.3 0.8
pot:pot1:pottype mtp
pot:pot1:potfile short_range.mtp
pot:pot2:pottype eampoly
pot:pot2:potfile background.eampoly
```

---

## 程序化 API

大多数使用 `ConnectSettings` 的类遵循以下模式：

```cpp
class MyClass : public ConnectSettings {
    void InitSettings() {
        Connect(param, "section:key", "Description of this parameter");
    }

    void InitBySettings(GPSettings &settings) {
        InitSettings();
        ApplySettings(settings);
    }
};

// 使用：
GPSettings settings;
settings["section:key"] = "value";
myobj.InitBySettings(settings);
```

从文件加载：

```cpp
GPSettings settings;
settings.Load("config.ini");
myobj.InitBySettings(settings);
```

## 示例文件

### 完整弛豫配置

```text
# relax.ini
relax:driver lbfgs
relax:maxiteration 500
relax:gtol 1e-4
relax:ftol 1e-6
relax:relax_cell yes
relax:cell_strain yes
relax:symm_strain yes
relax:strain_mask 1 1 1 1 1 1
relax:pressure 0.0
relax:write_traj yes
relax:trajfile relax.traj
relax:nblock 10
relax:bfgs_m 5

pot:potfile potential.mtp

input POSCAR
output relaxed.vasp
index index.map
```

### 完整训练配置

```text
# train.ini
trainset train.cfg
validset valid.cfg
trainset-format mlpcfg
validset-format mlpcfg
potfile init.mtp
index index.map

[fit]
energy_weight 1.0
force_weight 1.0
stress_weight 0.1
weight_L2 1e-6
weight_rad_w0 1e-6
bfgs_maxiteration 1000
bfgs_m 5
save_to checkpoint.mtp
save_every 50
```

### NEB 配置

```text
# neb.ini
neb:spring 5.0
neb:climb yes
neb:climb_delay 50
neb:vcneb no
neb:symm_strain yes
neb:write_traj yes
neb:trjfile neb_path.cfg
neb:nblocks 10

relax:driver lbfgs
relax:gtol 1e-3
relax:maxiteration 200

pot:potfile potential.mtp
```

## 检查点兼容性

MaxVol v2 检查点不再受支持；v3 检查点仍受支持。

## 参见

- @ref guide_commands — CLI 命令参考
- @ref guide_workflows — 端到端工作流示例
- @ref api_core — C++ API 示例
