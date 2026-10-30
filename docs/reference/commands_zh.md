# 命令参考 {#guide_commands}

[English](commands.md)

@tableofcontents


输入文件与配置语法见[输入文件与配置约定](../guides/conventions_zh.md)，
单点计算见[单点计算](../guides/single-point_zh.md)。
[弛豫](../guides/relaxation_zh.md)、[NEB](../guides/neb_zh.md)、
[晶体性质](../guides/properties_zh.md)和[选样](../guides/active-learning_zh.md)
提供完整流程；本页用于查命令和选项。

## CLI 结构

```
imrmlp <command> [options] <input> <output>
```

命令同时支持**短标志**（`-x val`）、**长标志**（`--option=val`）
和**位置参数**（输入/输出文件）。零参数布尔标志写为 `--flag` 或
`--flag=true`；显式关闭写为 `--flag=false`。部分布尔选项要求提供值
（`--option true`）。省略选项会保留表中的默认值，默认值也可能是 true。
不要给零参数标志单独追加 `false`，否则它会被解析为位置参数。

运行 `imrmlp --help` 查看所有命令列表；`imrmlp <command> --help` 查看
各命令的帮助信息。

```bash
imrmlp efs -pot potential.mtp POSCAR out.cfg       # 短别名
imrmlp relax relax.ini                              # 设置文件
```

## 通用选项

以下选项适用于大多数命令：

| 选项 | 描述 |
|--------|-------------|
| `--help` | 显示命令特定帮助信息 |
| `--version` | 显示版本信息（`imrmlp --version`） |
| `--settings` | 从设置文件加载选项（无 `-i` 别名；许多命令也接受把该文件作为位置参数）。`phonopyfc` 是例外：它使用 `--settings-file` / `-i`。 |
| `--verbose` | 启用详细日志输出（仅少数命令支持） |
| `--index` / `-id` | 种类索引文件（通常为 `index.map`） |

### 压缩输入/输出文件

结构文件（POSCAR、dump/lammpstrj、extxyz、cfg 等）、`.ini` 设置文件和势函数文件支持 gzip（`.gz`）压缩：

- **读取**：通过文件 magic bytes 自动识别 gzip，扩展名作为兜底。`POSCAR.gz`、`dump.gz` 和没有压缩扩展名的 gzip 文件都可使用常规 `--input` 选项。
- **写入**：当文件名以 `.gz` 结尾时自动压缩（例如 `--output out.xyz.gz`）。

gzip 默认启用，可用 `make USE_GZIP=0` 关闭。禁用后仍会识别 gzip 并报错，
不会按明文处理。若手动在 CXXFLAGS 中添加 `-DUSE_GZIP`，还需在链接参数中添加 `-lz`。

Zstandard 支持及其构建开关已移除。`.zst` 路径和识别为 Zstandard 的输入会被拒绝。
旧 `.zst` 文件需先在外部解压，或转换成 gzip 后再使用。

---

## 核心 EFS 与预测

EFS（能量、力、应力）命令是所有基于势函数工作的基础。
它们使用训练好的势函数对输入结构进行求值——这是验证、分析或生产 MD 运行的第一步。
每个轨迹分析、弹性常数计算或弛豫流程在底层都以 EFS 评估开始。

**典型流程**: 训练势函数 → 用 `calc_efs` + `calc_err` 验证 → 用于生产（弛豫、MD、分析）。

### `calc_efs` — 能量、力、应力 *（别名：`efs`）*

计算一个或多个结构的能量、力和应力张量。这是最基础的命令：
给定一个势文件和一组输入结构，运行势函数并将计算出的 EFS 属性附加到输出中。

@note **何时使用 `--energy-only`**：当只需要总能量时（例如能量-体积曲线、凸包分析）
使用 `-eno`。跳过力和应力计算对于大体系可显著减少运行时间。

```bash
imrmlp efs -pot potential.mtp --input=input.cfg --output=output.cfg
imrmlp calc_efs --potfile=potential.mtp --input=POSCAR --output=out.cfg
imrmlp calc_efs --potfile=potential.mtp --input=POSCAR --output=out.cfg --device=cuda --gpu-id=0
imrmlp efs --settings=calc_efs.ini
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--input` | `-in` | `POSCAR` | 输入结构文件（位置参数，必需） |
| `--output` | `-out` | `output-efs.cfg` | 附加 EFS 后的输出文件（位置参数，必需） |
| `--potfile` | `-pot` | `potential.mtp` | 势参数文件 |
| `--pottype` | `-pt` | `mtp` | 势类型标识符 |
| `--device` | | `cpu` | 计算后端：`cpu`、`cuda` 或 `hip`（`dcu` 是 `hip` 的别名） |
| `--gpu-id` | | `-1` | GPU 设备编号；`-1` 使用当前 GPU 设备 |
| `--gpu-batch-atoms` | | `0` | 每个 GPU 分块的最大原子数；`0` 按显存自动选择 |
| `--gpu-batch-structures` | | `0` | 每个 GPU 分块的最大结构数；`0` 自动选择 |
| `--gpu-memory-fraction` | | `0.75` | 每个分块可使用的当前空闲显存比例 |
| `--gpu-reuse-topology` | | `false` | 为拓扑完全兼容的连续单结构轨迹复用拓扑 |
| `--input-format` | `-if` | auto | 输入格式 |
| `--output-format` | `-of` | auto | 输出格式 |
| `--index` | `-id` | `index.map` | 原子类型到索引的映射文件 |
| `--no-site-energy` | `-ns` | `false` | 输出中排除位点能量（布尔标志） |
| `--energy-only` | `-eno` | `false` | 仅输出能量（布尔标志） |
| `--show-progress` | `-sp` | `false` | 显示已完成构型数量（布尔标志） |
| `--settings` | | `input.ini` | 配置文件 |

@note GPU 求值需要对应的 `BUILD_MODE=cuda` / `mpi-cuda` 或
`BUILD_MODE=hip` / `mpi-hip` 可执行文件。支持普通 MTP、MTP-M
（`li`、`q2`、`fs`、`n1`、`n2`），包括兼容的 SOG v3/v4 模型。
此 GPU 路径不支持 Coulomb LES v2。模型和径向基限制见
[GPU 指南](../development/cuda_zh.md)。输入按完整结构分组，在显存预算内分块；
单个结构超过预算会报错。启用 `--gpu-reuse-topology` 时，兼容的连续帧逐帧处理，
`--gpu-batch-structures` 应设为 0 或 1。`--gpu-memory-fraction` 必须在 `(0,1]` 内。
`--energy-only` 会跳过 GPU 力与 virial 求值。

### `testpot` — 测试势

通过力一致性检验（有限差分检验与解析梯度对比）验证势的正确性。

```bash
imrmlp testpot --settings=test_settings.ini
imrmlp testpot --settings=settings.ini --input=POSCAR --test-which=0 --nugedvalue=0.001
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--settings` | | `settings.ini` | 设置文件（位置参数，必需） |
| `--input` | | `POSCAR` | 要测试的结构文件 |
| `--format` | | `POSCAR` | 结构文件格式 |
| `--index` | | `index.map` | 索引文件 |
| `--test-which` | | `-1` | 待测试原子索引，从 0 开始；任意负值测试全部原子 |
| `--nugedvalue` | | `0.001` | 有限差分步长（Å） |
| `--test-stress` | | `0` | 设为 `1` 时同时通过晶胞有限差分测试 virial；必须提供值 |
| `--stress-nudge` | | `1e-6` | 应力测试的有限差分位移（Å） |

---

## 弛豫与优化

几何优化命令在势能面上寻找局部极小值（弛豫）或一阶鞍点（过渡态）。
这是势函数训练后最常见的生产任务。

**如何选择鞍点方法**：参见本节末尾的对比表。

### `relax` — 结构弛豫

用已有势函数优化几何结构，支持多种优化引擎
（FIRE、L-BFGS、CG、GD）、变胞弛豫、选择性动力学（逐原子和逐应变分量固定/释放）
以及修改器（如用于即时主动学习的 `ALModifier`）。

```bash
imrmlp relax relax.ini
```

@note **为何推荐设置文件？** 弛豫涉及大量相互依赖的参数（优化器选择、收敛容差、
晶胞弛豫标志、约束掩码、压力等）。将它们全部放入结构化的 `.ini` 文件可以避免
CLI 混乱，并使配置可复现。参见 @ref guide_config 了解所有支持的参数。

虽然设置文件是主要接口，以下 CLI 标志也可以直接覆盖设置文件中的值：
`--potfile`/`-pot`、`--pottype`/`-pt`、`--input`/`-in`、
`--output`/`-out`、`--input-format`/`-if`、`--output-format`/`-of`、
`--index`/`-id`、`--gtol`、`--pressure`/`-p`、`--isif`、`--selective_dynamics`/`-sd`、
`--cell-mask`、`--strain_mask`、`--seldyn`、
`--stepmax`、`--dprecond`、`--write-force`、`--write-stress`、`--write-ateng`、
`--create-input`、`--settings`、`--tags`/`--vals`（在命令行直接添加 `relax:` 设置）、
`--seq-read`/`--seq-write`（逐帧流式读写）、`--position-mask`（全局 0/1 原子标志）、
`--seldyn`、`--filter`（优化器坐标掩码文件）、`--file-constraints`（JSON 约束集）
以及 `--discard-all`（输出时丢弃额外属性）。

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--potfile` | `-pot` | — | 势文件 |
| `--pottype` | `-pt` | `mtp` | 势类型 |
| `--input` | `-in` | `POSCAR` | 输入结构 |
| `--output` | `-out` | `output-relaxed.cfg` | 输出结构 |
| `--input-format` | `-if` | 自动检测 | 输入格式 |
| `--output-format` | `-of` | 根据文件名推断 | 输出格式 |
| `--index` | `-id` | `index.map` | 物种索引映射 |
| `--settings` | | — | 设置文件 |
| `--gtol` | | — | 力收敛容差 |
| `--pressure` | `-p` | `0` | 外部压力（GPa） |
| `--isif` | | — | 晶胞/变胞弛豫模式 |
| `--selective_dynamics` | `-sd` | `false` | 启用选择性动力学 |
| `--seldyn` | | — | 逐原子/逐晶胞约束掩码 |
| `--cell-mask` | | — | 晶胞坐标掩码 |
| `--strain_mask` | | — | 应变坐标掩码 |
| `--stepmax` | | `0.1` | 单次更新最大步长 |
| `--dprecond` | | `0` | 启用对角预条件（布尔值） |
| `--write-force` | | `true` | 输出时写力 |
| `--write-stress` | | `true` | 输出时写应力 |
| `--write-ateng` | | `true` | 输出时写逐原子能量 |
| `--create-input` | | `false` | 生成示例输入文件后退出 |
| `--tags` | | — | 附加 `relax:` 设置键 |
| `--vals` | | — | 与 `--tags` 配对的值 |
| `--seq-read` | | `false` | 逐帧流式读取输入结构 |
| `--seq-write` | | `false` | 逐帧流式写出结构 |
| `--position-mask` | | — | 全局 0/1 原子标志 |
| `--filter` | | — | 优化器坐标掩码文件 |
| `--file-constraints` | | — | JSON 约束集 |
| `--discard-all` | | `false` | 输出时丢弃额外属性 |

### `modefollowing` — 模式跟随

沿特定振动模式（Hessian 矩阵的特征向量）从极小点向上搜索鞍点。
最适合当你知道哪个模式指向目标过渡态时——例如从先前的声子计算中识别出的
软声子模式。

```bash
imrmlp modefollowing --settings=modefollowing.ini --potfile=potential.mtp --input=structure.cfg
imrmlp modefollowing --settings=modefollowing.ini --potfile=potential.mtp --input=structure.cfg --imode=1 --amplitude=0.1
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--settings` | | `modefollowing.ini` | 设置文件（必填） |
| `--potfile` | `-pot` | `potential.mtp` | 势文件（MTP 类型） |
| `--input` | `-in` | `POSCAR` | 输入结构 |
| `--input2` | `-in2` | `POSCAR_LAM` | 可选第二结构，用于双结构模式/重叠比较 |
| `--input-format` | `-if` | `POSCAR` | 输入结构格式 |
| `--index` | `-id` | `index.map` | 索引文件 |
| `--max-loop` | | `100` | 每个模式跟随阶段的最大迭代次数，必须非负 |
| `--next-tol` | `-ntol` | `-0.1` | 下一步容差 |
| `--ifac` | | `0` | 被跟随模式的力因子 |
| `--ofac` | | `1` | 非跟随正交模式的弛豫因子 |
| `--sfac` | | `1` | 非跟随模式的弛豫因子 |
| `--cycle` | | `false` | 生成模式序列时使用循环（布尔） |
| `--updateref` | | `false` | 晶胞变化较大时更新参考晶格（布尔） |
| `--modecar-out` | `-mcout` | `MODECAR` | MODECAR 输出文件（模式 9） |
| `--imode` | `-i` | `4` | 待跟踪的模式索引 |
| `--mode` | `-m` | `1` | 作业类型 |
| `--amplitude` | `-a` | `0.1` | 位移幅度 |
| `--trial-step` | | `0.1` | 尝试步长 |
| `--symprec` | `-tol` | `1e-3` | 对称容差 |
| `--frames` | | `16` | 轨迹帧数 |
| `--pressure` | `-p` | `0` | 外部压力（GPa） |
| `--hessian` | | | 从文件加载 Hessian 矩阵 |
| `--nocell` | | `false` | 固定晶胞（布尔标志） |
| `--linetraj` | | `false` | 创建线性轨迹（布尔标志） |
| `--car-rep` | | `false` | 使用笛卡尔坐标（布尔标志） |
| `--request-efs` | `-efs` | `false` | 请求 EFS 输出（布尔标志） |
| `--eigzero-tol` | | `0.008` | 零特征值容差 |

@note 许多其他参数（ifac、ofac、sfac、cycle、updateref、next-tol 等）
可用——运行 `imrmlp modefollowing --help` 查看完整列表。

变胞分数坐标模式使用 `C-I`（`xx yy zz yz xz xy`）与分数原子坐标，
其中 `H = H_ref chol(C)`；Hessian、模式位移和弛豫使用同一约定。
`--car-rep` 使用独立的原始晶胞与笛卡尔原子坐标。
比较两个结构时，第二个结构先整体旋转到相同的 Cholesky 晶胞约定，再进行位移或模式投影。

### `calc_neb` — Nudged Elastic Band (NEB) *（别名：`neb`）*

在两个端点结构之间搜索最小能量路径。

```bash
imrmlp neb neb_config.ini
imrmlp calc_neb neb_config.ini
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--settings` | | | 设置文件（位置参数，必需） |
| `--create-input` | | `false` | 创建 NEB 计算示例输入文件（布尔） |
| `--input-format` | `-if` | auto | 输入结构格式 |
| `--nimgs` | | `0` | 镜像数（含两个端点）；通常由设置文件给出 |
| `--index` | `-id` | `index.map` | 原子类型到物种的索引文件 |
| `--ninterp` | | `100` | MEP 曲线插值点数 |
| `--cellweight` | `-cwt` | `100` | 已弃用；请使用 `neb:cellweight`，顶层值优先。 |
| `--mepout` | | `mep.xyz` | MEP 轨迹输出文件 |
| `--mepformat` | | `extxyz` | MEP 轨迹格式 |
| `--summary` | | `neb.dat` | MEP 摘要输出 |
| `--neb:path_pbc` | | | 从 POSCAR 导入完整路径时的显式 PBC 位 |
| `--neb:reference_cell` | | | 变胞路径的参考晶胞；与重启不兼容 |
| `--neb:profile` | | | MPI+OpenMP 计时 JSON 文件名 |
| `--driver` | | `lbfgs` | 优化器：`lbfgs`、`fire`、`cg` 或 `sgd` |
| `--gtol` | `-gtol` | `lbfgs:epsgmax` | 活动 NEB 力容差；覆盖 `neb:gtol` |
| `--maxiteration` | `-maxit` | `lbfgs:max_iterations` | 单次优化的迭代上限 |
| `--max_total_iter` | | `1000000` | 跨镜像调整重启的总迭代预算 |
| `--initial_step` | | `lbfgs:initial_step` | 初始步长 |
| `--max_step` | | `lbfgs:max_step` | 最大步长 |
| `--max_move` | | `lbfgs:max_step_move` | 单次迭代最大累计位移 |
| `--first_step` | | `lbfgs:first_step` | L-BFGS 历史重置后的首步长 |
| `--epsilon` | | `lbfgs:epsilon` | 绝对力范数精度 |
| `--epsilon_rel` | | `lbfgs:epsilon_rel` | 相对力范数精度 |
| `--past` | | `lbfgs:past` | L-BFGS 历史步数 |
| `--delta` | | `lbfgs:delta` | 加到容差上的收敛偏移 |
| `--damping` | | | L-BFGS 混合参数 |
| `--s_curv_coeff` | | | 曲率条件系数 |
| `--lbfgs_m` | | `15` | L-BFGS 历史长度 |
| `--next_step` | `-r` | `false` | 从 `nebimage_XX_final.vasp` 重启（布尔） |

@note NEB 参数（弹簧常数、climbing image 选项、对称释放、变镜像阈值等）在设置文件的
`neb:` 前缀键中配置。上表 CLI 标志是常见情形的显式覆盖；完整键列表参见 @ref guide_config "配置说明"。

### 鞍点方法选择指南

| 命令 | 方法 | 需要输入 | 最适合 |
|---------|--------|----------|----------|
| `calc_neb` | NEB + Climbing Image | 两个端点（初态+末态） | 未知鞍点位置；需要完整 MEP |
| `modefollowing` | 模式跟随 | 极小点+目标模式方向 | 跟随特定的软声子模式 |

@note **典型工作流**: 运行 `calc_neb` 绘制能垒图并定位近似鞍点。
当反应坐标由声子分析明确指定时使用 `modefollowing`。

---

## 训练与拟合

训练命令将势函数参数拟合到参考数据（通常为 DFT）。
它们是势函数开发工作流的核心：在收集和筛选参考结构后，
根据模型类型和优化需求选择合适的训练命令。

@note **选择合适的训练命令** — 参见本节末尾的对比表。

### `trainmtp` — 训练 MTP 势

矩张量势的主要训练命令。将径向基系数和线性种类/矩系数同时拟合到
参考 DFT 数据（能量、力、应力）。


```bash
imrmlp trainmtp training.ini --do-lbfgs
imrmlp trainmtp --trainset=train.cfg --validset=valid.cfg --potfile=init.mtp --do-lbfgs
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--settings` | | | 设置文件（位置参数） |
| `--trainset` | `-train` | `train.cfg` | 训练数据集文件 |
| `--validset` | `-valid` | | 验证数据集文件 |
| `--trainset-format` | `-if` | | 训练数据集格式（空 = 自动检测） |
| `--validset-format` | `-vf` | | 验证数据集格式（空 = 自动检测） |
| `--potfile` | `-pot` | `potential.mtp` | 势文件 |
| `--index` | `-id` | `index.map` | 索引文件 |
| `--linear-fit` | `-lf` | `false` | 仅线性拟合，固定径向（布尔标志） |
| `--do-rescale` | `-rs` | `false` | 在线性拟合中搜索径向缩放；须同时指定 `--linear-fit` |
| `--balance-at` | `-bw` | `false` | 负载均衡（布尔标志） |
| `--do-sgd` | `-sgd` | `false` | 使用 SGD 优化（布尔标志） |
| `--do-lbfgs` | `-lbf` | `false` | 使用 L-BFGS 优化（布尔标志） |
| `--do-cg` | `-cg` | `false` | 使用共轭梯度（布尔标志） |
| `--do-df` | `-df` | `false` | 使用差分进化（布尔标志） |
| `--do-random` | `-random` | `false` | 随机径向搜索（布尔标志） |
| `--species-only` | | `false` | 仅通过通用 L-BFGS 训练器拟合物种系数（布尔标志） |
| `--do-matrix-ata` | `-ata` | `false` | 输出 A^T A 矩阵（布尔标志） |
| `--do-matrix-a` | `-a` | `false` | 输出完整 A 矩阵（布尔标志） |
| `--calc-grad` | `-g` | `false` | 计算诊断梯度（布尔标志） |
| `--check-grad` | | `false` | 验证解析梯度（布尔标志） |
| `--check-grad-all` | | `false` | 完整梯度检查（布尔标志） |
| `--test-curv` | | `false` | 测试曲率计算（布尔标志） |
| `--test-curv-which` | | `0` | 曲率分量索引 |
| `--test-curv-nudge` | | `0.001` | 曲率测试步长 |
| `--debug` | | `false` | 调试输出（布尔标志） |

非线性拟合需显式选择 `--do-lbfgs` 等优化器；MTP 线性系数拟合使用
`--linear-fit`。`fit:lfit_eng_only true` 控制线性拟合使用的标签，不会自行选择
优化器。`train_linear` 是另一条命令，有自己的描述符与设置要求。`trainmtp`
最终输出为当前工作目录下的 `_Trained_Potential_.mtp`，`fit:save_to` 指定训练检查点。

`--do-matrix-a` 对每个有应力标签的构型输出六行，顺序为
`xx, xy, xz, yy, yz, zz`。非对角行的基函数系数和目标值同时乘 `sqrt(2)`，
使六行等权平方误差等于完整对称张量的平方误差；对角行保持原值。
二进制记录仍为 `int 构型编号`，后接 `double` 类型的系数和目标值；旧导出未缩放剪切行。
这里仅包含张量分量的几何权重，不包含训练配置中的权重及按原子数归一化。

训练超参数从 `[fit]` 部分读取；势文件由扁平 `potfile` 键给出（默认
`potential.mtp`，`trainmtp` 不读取 `[pot]` 部分——那是 `trainnlpot` 的行为）。

### `trainnlpot` — 训练非线性势

使用显式选择的优化器训练通用非线性势（例如 NN-MTP、HMTP）。
势类型在设置文件的 `[pot]` 部分或 `pot:` 命名空间中指定，不是 CLI 参数。

```bash
imrmlp trainnlpot --settings=training.ini --do-lbfgs
imrmlp trainnlpot --settings=training.ini --trainset=train.cfg --validset=valid.cfg --do-sgd -o output_potential
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--settings` | | | 设置文件（位置参数，必需） |
| `--trainset` | `-train` | `train.cfg` | 训练数据集文件 |
| `--validset` | `-inv` | | 验证数据集文件 |
| `--trainset-format` | `-if` | | 训练数据格式（空 = 自动检测） |
| `--validset-format` | `-vf` | | 验证数据集格式（空 = 自动检测） |
| `--testset` | | | 仅用于最终拟合后评估的留出数据集 |
| `--testset-format` | | | 留出数据集格式（空 = 自动检测） |
| `--fixed-core` | | | 已有固定核心；对残差标签拟合普通 MTP 并输出 Hybrid manifest |
| `--fixed-core-type` | | `zbl` | 固定核心类型：`zbl` 或 `mtp` |
| `--output` | `-o` | `_Trained_Potential_` | 输出势文件前缀 |
| `--index` | `-id` | `index.map` | 索引文件 |
| `--debug` | | `false` | 调试输出（布尔值） |
| `--do-lbfgs` | `-lbf` | `false` | 使用 L-BFGS 优化器（布尔标志） |
| `--do-sgd` | `-sgd` | `false` | 使用 SGD 优化器（布尔标志） |
| `--do-varpro` | | `false` | 对径向参数做 profile、每次试探后线性重拟合（普通 CPU MTP） |
| `--linear-refit-only` | | `false` | 仅对已初始化模型的线性系数重拟合；需要 `fit:linear_refit` |
| `--do-cg` | `-cg` | `false` | 使用共轭梯度（布尔标志） |
| `--calc-grad` | `-g` | `false` | 计算诊断梯度（布尔标志） |
| `--check-grad` | | `false` | 数值验证梯度（布尔标志） |
| `--check-grad-all` | | `false` | 完整梯度检查（布尔标志） |
| `--test-curv` | | `false` | 测试曲率计算（布尔标志） |
| `--test-curv-which` | | `0` | 曲率分量索引 |
| `--test-curv-nudge` | | `0.001` | 曲率测试步长 |
| `--balance-atomic-fitness` | `-bw` | `false` | 补偿不完整力数据（布尔标志） |

@note 非线性训练器超参数从 `[fit]` 部分读取。

双分量 HMTP 设置为 `pot:pottype hmtp`、`pot:potfile source` 和
`pot:potlevel 2`。加载器读取 `source_sub1` 与 `source_sub2`；两者的元素数量及
顺序必须一致。径向范围可以不同，组合近邻截断取二者最大值，各分量梯度仍按
自己的近邻范围计算。`trainnlpot` 优化组合系数向量，支持纯能量及能量/力/应力
梯度。`--output fitted` 写出 `fitted_sub1` 与 `fitted_sub2`，重载时两份都需要。

使用有限的 `fit:bfgs_maxiteration`（SGD 使用 `fit:epoch_limit`），并在独立参考集
上评估保存模型。未选择优化器时不会执行常规非线性优化。


### `fastfit` — 快速 MTP 拟合

MTP 势能的快速线性拟合与基函数选择。支持多种模式：
模式 0 计算特征值谱，模式 1 执行简单加权最小二乘拟合，
模式 2 应用预计算的基函数选择，模式 3 暴力逐个剔除搜索，
模式 4 使用基于 CUR 的重要性评分选择基函数，模式 5 通过
模拟退火或贪心交换精化选择，模式 6 使用 NSGA-II 多目标
遗传优化寻找最佳基函数子集，模式 7 进行相关性分析和去相关。

```bash
imrmlp fastfit training.ini
imrmlp fastfit --settings=settings.ini --potfile=init.mtp --trainset=train.cfg --validset=valid.cfg
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--settings` | | `settings.ini` | 设置文件（位置参数，必需） |
| `--potfile` | `-pot` | `potential.mtp` | 势文件 |
| `--trainset` | `-ts` | `train.cfg` | 训练集 |
| `--validset` | `-val` | `valid.cfg` | 验证集 |
| `--format` | `-fmt` | `mlpcfg` | 数据格式 |
| `--index` | `-id` | `index.map` | 索引文件 |
| `--weight-energy` | `-we` | `1.0` | 能量权重 |
| `--weight-force` | `-wf` | `0.1` | 力权重 |
| `--weight-stress` | `-ws` | `0.001` | 应力权重 |
| `--weight-l2` | `-ww` | `1e-8` | L2 正则化权重 |
| `--mode` | `-m` | `0` | 拟合模式 (0–7) |
| `--method` | | `llt` | 求解方法；模式 5 中用 `greedy` 启用贪心交换 |
| `--sparse-method` | `-dm` | `0` | 模式 4：0=选高分，1=丢弃低分 |
| `--discard-rate` | `-drate` | `0.1` | 模式 4：标量丢弃比例 |
| `--epslow` | `-epslow` | `1e-6` | 模式 4：特征值下界 |
| `--cur` | `-cur` | `false` | 模式 4：启用 CUR 随机采样（布尔） |
| `--cur-nsearch` | `-ntry` | `10` | 模式 4：最大 CUR 搜索迭代次数 |
| `--addback` | `-addback` | `false` | 模式 4：加回路径上的标量（布尔） |
| `--corr-strategy` | `-cs` | `0` | 模式 7：去相关策略，0=threshold，1=clustering，2=greedy VIF |
| `--corr-pearson` | `-cp` | `0.95` | 模式 7：Pearson 阈值 |
| `--corr-vif` | `-cv` | `10.0` | 模式 7：VIF 阈值 |
| `--corr-max-samples` | `-cms` | `50000` | 模式 7：相关性分析最多采样原子数 |
| `--corr-max-features` | `-cmf` | `0` | 模式 7：最多保留特征数，0 表示不限制 |
| `--corr-addback` | `-caddback` | `false` | 模式 7：去相关后加回 DAG 路径标量（布尔） |
| `--corr-save-reduced` | `-csave` | `false` | 模式 7：选择后保存降维势（布尔） |
| `--cmp` | `-cmp` | `false` | 模式 1：比较求解器方法（布尔） |
| `--objective-convention` | | `legacy` | `legacy` 或共享 mean-moments EFS/L2 目标 |
| `--l2-reference` | | `native` | L2 输入参考：`native`、`mean` 或 `sum`；后者要求 mean-moments |
| `--objective-report` | | | 可选的 TSV 目标契约报告，含有效 L2 项 |
| `--selection-objective` | | `legacy` | `legacy` 或 `error-cost`（模式 5/6） |
| `--max-valid-loss` | | | error-cost 选择所需的平均加权验证损失上限 |
| `--cost-report` | | `fastfit-cost.tsv` | 成本选择 TSV 报告 |
| `--cost-benchmark-topk` | | `8` | 最终实测候选数（不含基线） |
| `--cost-benchmark-configs` | | `16` | 代表性验证构型数 |
| `--cost-archive-size` | | `128` | 跨代候选档案上限 |
| `--cost-archive-rule` | | `spread` | `spread`（旧默认）或有界 `pareto` 档案 |
| `--cost-export-dir` | | | 导出已验证模型、Pareto 前沿、目标与计时轮次 |
| `--cost-benchmark-rounds` | | `5` | 重复测量轮数（至少 3） |
| `--cost-benchmark-seconds` | | `0.01` | 每个候选/路径/轮次的最短测量秒数 |
| `--profile` | | `false` | 报告各阶段独占耗时与峰值 RSS（布尔） |
| `--profile-output` | | | 写出 profile TSV，同时启用 profiling |
| `--use-matrices` | | `false` | 读取导出的未加权 E/F/S 矩阵块（布尔） |
| `--energy-power` | | `1` | 能量平方残差的原子数幂次 |
| `--force-power` | | `0` | 力平方残差的原子数幂次 |
| `--stress-power` | | `1` | virial 平方残差的原子数幂次 |
| `--force-norm-d` | | | 力归一化分母偏移 |
| `--force-norm-f` | | | 力归一化尺度（0 关闭） |
| `--force-norm-p` | | | 力归一化幂次 |
| `--seed` | | | 可复现随机种子 |
| `--search-steps` | | `-1` | SA/贪心步数（-1 选择自动预算） |
| `--population` | | `0` | NSGA-II 种群规模（0 = 自动） |
| `--generations` | | `0` | NSGA-II 代数（0 = 自动） |
| `--mutation-rate` | | `0.05` | NSGA-II 单标量翻转概率；-1 使用 1/nscalar |
| `--population-init` | | `half` | NSGA-II 初始化：`half`、`stratified` 或 `dependency` |
| `--group-mutation-rate` | | `0` | 标量变异后丢弃一个依赖组的概率 |
| `--group-mutation-kind` | | `basic` | 依赖组族：`basic`、`radial` 或 `mixed` |
| `--sa-temperature` | | | 平均每构型损失的初始温度 |
| `--sa-cooling` | | | (0, 1] 内的 SA 降温因子 |
| `--matE` | `-me` | | 能量设计矩阵块 |
| `--matF` | `-mf` | | 力设计矩阵块 |
| `--matS` | `-ms` | | 应力设计矩阵块 |
| `--vecE` | `-ve` | | 能量目标向量 |
| `--vecF` | `-vf` | | 力目标向量 |
| `--vecS` | `-vs` | | 应力目标向量 |

选集文件采用“数量 + 未偏移的标量矩 ID”格式，物种系数始终隐式保留；模式
2、4、5、6、7 使用同一格式。MPI 运行时，候选拟合与误差评估由所有 rank
共同执行，输出文件只由 rank 0 写入。

### `train_linear` — 训练线性描述符势

使用最小二乘回归训练线性描述符势。一次性解析求解正规方程
\f$(\mathbf{A}^T\mathbf{W}\mathbf{A} + \lambda\mathbf{I})\mathbf{w} = \mathbf{A}^T\mathbf{W}\mathbf{y}\f$，
无需迭代梯度下降。支持跨节点的 MPI 分布式组装。

```bash
imrmlp train_linear --settings=training.ini
imrmlp train_linear --trainset=train.cfg --validset=valid.cfg -o fitted_potential
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--settings` | | | 设置文件（位置参数，必需） |
| `--trainset` | `-in` | `train.cfg` | 训练数据集文件 |
| `--validset` | `-inv` | | 验证数据集文件（默认：无） |
| `--trainset-format` | `-if` | `mlpcfg` | 训练数据格式：mlpcfg、extxyz、vaspabn |
| `--output` | `-o` | `_Trained_Potential_` | 输出势文件前缀 |
| `--index` | `-id` | `index.map` | 原子类型索引文件 |
| `--debug` | | `false` | 启用调试输出（布尔） |

@note 设置文件使用 `[fit]` 部分配置 TrainerLinearPotential 参数
（energy_weight、force_weight、stress_weight、weight_L2 等），
`[pot]` 部分配置势函数。参见 @ref theory_linear 了解数学背景。

### `nnmtp` — 创建 NN-MTP 模板

从 MTP 模板创建神经网络 MTP 势模板。这是在使用 `nnmtp` 势类型的
`trainnlpot` 之前的准备步骤。

```bash
imrmlp nnmtp --settings=nnmtp.ini
imrmlp nnmtp -pot init.mtp -n 10 -n2 5 -o nnmtp_template.mtp
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--settings` | | | 设置文件（位置参数） |
| `--potfile` | `-pot` | `potential.mtp` | 输入 MTP 势模板 |
| `--data` | `-d` | `train.cfg` | 用于自动检测的训练数据 |
| `--input-format` | `-f` | auto | 训练数据格式 |
| `--nneuron` | `-n` | `10` | 第一隐藏层神经元数 |
| `--nneuron2` | `-n2` | `5` | 第二隐藏层神经元数 |
| `--nnlay` | `-l` | `1` | 隐藏层数：1 或 2 |
| `--index` | `-id` | `index.map` | 索引文件 |
| `--initmtp` | `-m` | `true` | 用 MTP 系数初始化（布尔） |
| `--extent` | `-e` | `1.0` | 归一化标准差乘数，必须为正有限数 |
| `--output` | `-o` | `potential_specific` | 输出 NN-MTP 模板 |
| `--normalize` | `-norm` | `true` | 归一化输入特征（布尔） |
| `--cvnv` | `-cvnv` | `false` | 将输出前缀对应的 `.nn` 文件转换为文本 |

数据文件必须包含非空结构。缺少能量标签的结构仍参与归一化，但在拟合元素能量偏移时
会被跳过并给出警告；如果全部缺少能量标签，则警告并将偏移初始化为零。
每个使用中的隐藏层，其神经元数必须为正数。

### `initmtpm` — 初始化 MTP_M

使用随机系数初始化 PotMTP_M（多阶 MTP）势文件。
在没有预训练势时用作起点。MTP 等级和物种数量在设置文件的 `[pot]` 部分配置。

```bash
imrmlp initmtpm --settings=settings.ini
imrmlp initmtpm -pot initial.mtp --dataset=train.cfg
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--settings` | | `settings.ini` | 设置文件（位置参数，必需） |
| `--potfile` | `-pot` | `potential.mtp` | 输出势文件 |
| `--dataset` | `-d` | `train.cfg` | 用于基组检测的参考数据 |
| `--format` | `-f` | | 数据格式（空 = 自动检测） |
| `--index` | `-id` | `index.map` | 索引文件 |
| `--randomize` | `-r` | `true` | 随机化系数（布尔值） |
| `--rand-scale` | | `1.0` | 随机缩放因子 |
| `--rand-lb` | | `-0.1` | 随机下界 |
| `--rand-ub` | | `0.1` | 随机上界 |

`--randomize=false` 保留径向、矩和模型系数，仍根据数据集初始化物种能量偏移及属性偏置。
缺少能量标签的结构参与基组统计，但跳过能量偏移拟合并给出警告；全部结构都没有能量标签时，
能量偏移初始化为零。数据集缺失、为空或读取失败时，命令报错。

### 训练命令选择指南

| 命令 | 模型类型 | 方法 | 适用场景 |
| --- | --- | --- | --- |
| `trainmtp --linear-fit` | 普通 MTP | 加权线性最小二乘 | 固定径向基，只拟合元素偏置与线性矩系数 |
| `trainmtp --do-lbfgs` | 普通 MTP | L-BFGS；另有 SGD / CG / DE | 同时拟合径向与线性参数 |
| `trainnlpot` | MTP、MTP-M、NN-MTP、HMTP、LES/SOG 等 | 内置 L-BFGS / SGD / CG；普通 CPU MTP 还支持 VarPro | 选择与模型及损失兼容的通用非线性训练路径 |
| `train_linear` | `descriptor_linear` | 线性最小二乘 | 固定描述符上的线性系数拟合 |
| `fastfit` | 普通 MTP | 线性拟合与 SA / NSGA-II 等基组选择 | 比较误差、基组规模和计算成本 |

普通 MTP 先选择兼容模板，再用 `trainmtp --linear-fit` 或 `trainmtp --do-lbfgs`。
`initmtpm` 面向独立的 MTP-M 模型。组合不同训练器前先阅读[训练流程](../guides/workflows_zh.md)。

---

## 主动学习（MaxVol / FPS / CUR）

主动学习命令从外部准备的候选结构池中筛选信息量高的结构，减少昂贵的参考计算。

- **MaxVol 选择**：`maxvol_select`、`maxvol_sample` 和 `maxvol_extend`
  用于选择独立描述符行、评估外推程度和扩充主动集。
- **多样性采样**：`diverse_select` 使用 FPS 或 CUR 筛选具有代表性的候选子集。
- **弛豫监控**：`relax` 通过 `[al]` 进行外推评分、收集候选，并在超过停止阈值时中止。
  标注与重训练分别执行，设置见[主动学习配置](configuration_zh.md)。

### `maxvol_select` — MAXVOL 选择

使用 MaxVol 从结构池中构建最大独立训练集。

```bash
imrmlp maxvol_select --potfile=potential.mtp --pool=pool.cfg -o selected.cfg
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--pool` |  | `pool.xyz` | 输入结构池 |
| `--potfile` |  | `potential.mtp` | 势文件用于 B-矩阵 |
| `--pottype` |  | `mtp` | 势类型：mtp、eampoly、lj612... |
| `--dtype` |  | `mtm` | 描述符类型：mtm、pot_efs、behler、bondorder |
| `--index` | `-id` | `index.map` | 索引文件 |
| `--input-format` | `-if` | auto | 输入池文件格式 |
| `--output-format` | `-of` | auto | 输出格式 |
| `--output` | `-o` | `selected.xyz` | 输出选中的结构 |
| `--score-file` |  | — | 输出每个结构的 MaxVol 评分 |
| `--threshold` |  | `1.001` | 选择阈值 |
| `--max-swaps` |  | `0` | 交换限制（0 = 无限制） |
| `--weight` |  | `1.0` | B-矩阵行权重 |
| `--pow-scale` |  | `0.5` | 原子计数缩放指数 |
| `--init-mode` |  | `auto` | 活动集初始化：`auto`、MLIP 兼容虚拟单位阵 `mlip`，或旧的真实行初始化 `data` |
| `--init-value` |  | `1e-6` | `mlip` 模式的虚拟单位行尺度 |
| `--verbose` | `-v` | `false` | 详细输出（布尔值） |
| `--all-coeffs` |  | `false` | 使用 ALL 系数（不仅是线性系数，布尔值） |
| `--row-scaling` |  | `uniform` | 行缩放：`uniform`，或仅对能量/应力做原子数缩放的 `mlip` |
| `--select-mode` |  | `stream` | 选择后端：`stream`、`full` 或与输入顺序相关的近似 `online` |
| `--stream-mode` |  | `exact` | `exact` 扫描缓存的完整候选池直至收敛；`fast` 使用有界的全局 shortlist |
| `--stream-chunk-size` |  | `32` | 每个并行构建/缓存分块包含的结构数 |
| `--stream-fast-passes` |  | `2` | `fast` 流式模式的最大全局扫描轮数 |
| `--stream-fast-shortlist` |  | `0` | `fast` 模式的全局 shortlist 大小；0 表示自动选择 |
| `--stream-cache-mode` |  | `auto` | 行缓存后端：`auto`、`memory`、`mmap` 或旧实现 `off` |
| `--stream-memory-limit-mb` |  | `3072` | 每进程常驻行缓存上限；`auto` 会把后续分块转为 mmap |
| `--stream-cache-dir` |  | — | 临时 mmap 目录；默认依次使用 `TMPDIR` 和 `/tmp` |
| `--mode` |  | `nbh` | 行模式：`nbh`（逐原子）或 `cfg`（逐结构） |
| `--w-energy` |  | `1.0` | 能量 B-矩阵权重 |
| `--w-site` |  | `0.0` | 位点能量权重 |
| `--w-force` |  | `0.0` | 力 B-矩阵权重 |
| `--w-stress` |  | `0.0` | 应力 B-矩阵权重 |
| `--settings` |  | `select.ini` | 设置文件（必需） |

MaxVol 状态文件（`.sel`）使用 v10 格式，模型元数据解析独立于该状态格式。

配置文件中，共享的 builder/selector 选项推荐使用命名空间 key：

```ini
builder:dtype mtm
builder:potfile potential.mtp
builder:pottype mtp
builder:mode nbh
builder:all-coeffs false
builder:row-scaling uniform
builder:w-energy 1.0
builder:w-site 0.0
builder:w-force 0.0
builder:w-stress 0.0

selector:threshold 1.001
selector:weight 1.0
selector:pow-scale 0.5
selector:init-mode auto
selector:init-value 1e-6
selector:max-swaps 0
selector:select-mode stream
selector:stream-mode exact
selector:stream-cache-mode auto
selector:stream-memory-limit-mb 3072
selector:verbose false
```

`online` 模式只保留当前 active matrix，不保存此前被拒绝的历史候选。后续交换
可能使旧候选的评分重新超过阈值，因此结果与候选到达顺序有关，是对累计候选池
的近似选择。如果最终 active set 必须对所有输入行满足阈值，应使用 `full`，或
使用 `stream-mode exact` 的流式模式。缓存式精确流只构建一次描述符，之后每次
接受交换仅重新扫描缓存行。`auto` 会在每进程常驻缓存达到上限后，把新增分块
透明地放入临时 mmap。MPI 运行时按 rank 划分结构，同时保持确定性的全局行顺序。

`init-mode auto` 对 `pot_efs` 和 `mtp_energy` 默认启用 MLIP 兼容初始化：
活动矩阵从 `init-value * I` 开始，只把数据能够支撑的虚拟方向替换为真实 B 行，
从而避免为形式上存在但在当前数据中不活跃的系数方向强行选择构型。其他描述符
仍使用 `data` 初始化。选择结束会报告是否收敛、剩余最大 grade 和虚拟行数；
达到交换/扫描上限的 state 仍会保存，但标记为未收敛，采样和扩展时会警告。

### `maxvol_sample` — MAXVOL 采样

针对现有的 MaxVol 主动集对结构进行评分（外推检测）。

```bash
imrmlp maxvol_sample --potfile=potential.mtp --input=pool.cfg --state-file=state.bin -o candidates.cfg
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--settings` | | `sample.ini` | 设置文件（位置参数，必需） |
| `--potfile` | | `potential.mtp` | 势文件 |
| `--pottype` | | `mtp` | 势类型 |
| `--input` | | `candidates.xyz` | 输入构型文件 |
| `--output` | `-o` | `sampled.xyz` | 输出评分后的结构 |
| `--state-file` | | | Active Set state 文件（默认：`potfile` + `.sel`） |
| `--dtype` | | `mtm` | 描述符类型 |
| `--index` | `-id` | `index.map` | 索引映射文件 |
| `--input-format` | `-if` | | 输入格式 |
| `--output-format` | `-of` | `extxyz` | 输出格式 |
| `--mode` | | `nbh` | 行模式：`nbh`（逐原子）或 `cfg`（逐结构） |
| `--threshold` | | `1.001` | 共享 selector 控制台提供的 MaxVol 选择阈值 |
| `--threshold-save` | | `2.1` | 超出此阈值 → 保存到输出 |
| `--threshold-break` | | `11.0` | 超出此阈值 → 中止评分 |
| `--weight` | | `1.0` | B-矩阵行权重 |
| `--pow-scale` | | `0.5` | 原子计数缩放指数 |
| `--w-energy` | | `1.0` | 能量 B-矩阵权重 |
| `--w-site` | | `0.0` | 位点能量权重 |
| `--w-force` | | `0.0` | 力 B-矩阵权重 |
| `--w-stress` | | `0.0` | 应力 B-矩阵权重 |
| `--add-grade` | | `true` | 在输出中添加 `MV_grade` 特征（布尔值） |
| `--dump-state` | | `false` | 打印 state 文件摘要，跳过评分（布尔值） |
| `--extract-configs` | | | 从 state 文件导出构型到指定路径 |
| `--selected-only` | | `false` | 仅导出选中的构型（与 --extract-configs 配合，布尔值） |
| `--verbose` | `-v` | `false` | 详细输出（布尔值） |
| `--all-coeffs` | | `false` | 使用 ALL 系数（不仅是线性系数，布尔值） |
| `--row-scaling` | | `uniform` | 行缩放：`uniform` 或 `mlip` |
| `--init-mode` | | `auto` | 活动集初始化：`auto`、`mlip` 或 `data` |
| `--init-value` | | | MLIP 虚拟单位行尺度 |
| `--max-swaps` | | `0` | 交换上限（0 = 无限制） |
| `--select-mode` | | `stream` | 选择后端：`stream`、`full` 或 `online` |
| `--stream-mode` | | `exact` | 流式后端：`exact` 或 `fast` |
| `--stream-chunk-size` | | | 流式构型分块大小 |
| `--stream-fast-passes` | | | fast 模式最大扫描轮数 |
| `--stream-fast-shortlist` | | `0` | fast 模式全局 shortlist 大小（0 = 自动） |
| `--stream-cache-mode` | | `auto` | 流式缓存：`auto`、`memory`、`mmap` 或 `off` |
| `--stream-memory-limit-mb` | | `3072` | 每进程常驻流式缓存上限（MiB） |
| `--stream-cache-dir` | | | 流式 mmap 临时文件目录 |

`maxvol_sample` 使用与 `maxvol_select` 相同的 `builder:*` 和
`selector:*` 配置块；`threshold-save` 和 `threshold-break` 仍是命令级顶层 key。

### `maxvol_extend` — MAXVOL 扩充

基于已有 Active Set 增量扩充信息量高的候选结构。

两种模式：

**模式 1 — 从 state 文件加载（推荐迭代主动学习）：**

```bash
imrmlp maxvol_extend --potfile=potential.mtp \
  --state-file=state.bin --candidates=pool.cfg -o new_selected.cfg
```

从 `maxvol_select` 生成的 `.sel` state 文件加载已有 Active Set，对候选结构进行选择，
输出新入选的结构。可选的 `--train` 参数提供超出 state 中已有训练集的额外训练构型。

**模式 2 — 三段式选择（原始方式，无 state 文件）：**

```bash
imrmlp maxvol_extend --potfile=potential.mtp \
  --train=train.cfg --candidates=pool.cfg -o new_selected.cfg
```

从 `--train` 构建初始 Active Set，评估候选结构，合并且重新稳定。
适用于一次性训练集构建。

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--potfile` |  | `potential.mtp` | 势文件用于 B-矩阵 |
| `--pottype` |  | `mtp` | 势类型：mtp、eampoly、lj612... |
| `--state-file` |  | — | 主动集 state 文件 (`.sel`) — 跳过 Stage 1，加载已有 Active Set |
| `--train` |  | — | 训练集文件 |
| `--candidates` |  | `candidates.xyz` | 候选结构文件 |
| `--output` | `-o` | `new_selected.xyz` | 输出新入选的候选结构 |
| `--dtype` |  | `mtm` | 描述符类型：mtm、pot_efs、behler、bondorder |
| `--index` | `-id` | `index.map` | 索引文件 |
| `--input-format` | `-if` | auto | 输入文件格式 |
| `--output-format` | `-of` | `extxyz` | 输出格式 |
| `--mode` |  | `nbh` | 行模式：`nbh`（逐原子）或 `cfg`（逐结构） |
| `--threshold` |  | `1.001` | 选择阈值 |
| `--weight` |  | `1.0` | B-矩阵行权重 |
| `--pow-scale` |  | `0.5` | 原子计数缩放指数 |
| `--w-energy` |  | `1.0` | 能量 B-矩阵权重 |
| `--w-site` |  | `0.0` | 位点能量权重 |
| `--w-force` |  | `0.0` | 力 B-矩阵权重 |
| `--w-stress` |  | `0.0` | 应力 B-矩阵权重 |
| `--max-swaps` |  | `0` | 交换限制（0 = 无限制） |
| `--verbose` | `-v` | `false` | 详细输出（布尔值） |
| `--all-coeffs` |  | `false` | 使用 ALL 系数（不仅是线性系数，布尔值） |
| `--row-scaling` |  | `uniform` | 行缩放：`uniform` 或 `mlip` |
| `--init-mode` |  | `auto` | 活动集初始化：`auto`、`mlip` 或 `data` |
| `--init-value` |  | `1e-6` | MLIP 虚拟单位行尺度 |
| `--select-mode` |  | `stream` | 选择后端：`stream`、`full` 或 `online` |
| `--stream-mode` |  | `exact` | 流式后端：`exact` 或 `fast` |
| `--stream-chunk-size` |  | `32` | 流式构型分块大小 |
| `--stream-fast-passes` |  | `2` | fast 模式最大扫描轮数 |
| `--stream-fast-shortlist` |  | `0` | fast 模式全局 shortlist 大小（0 = 自动） |
| `--stream-cache-mode` |  | `auto` | 流式缓存：`auto`、`memory`、`mmap` 或 `off` |
| `--stream-memory-limit-mb` |  | `3072` | 每进程常驻流式缓存上限（MiB） |
| `--stream-cache-dir` |  | — | 流式 mmap 临时文件目录 |
| `--settings` |  | `select_add.ini` | 设置文件（旧版默认名：`select_add.ini`，必需） |

`maxvol_extend` 使用与 `maxvol_select` 相同的 `builder:*` 和
`selector:*` 配置块；训练集、候选集和 state 路径仍是命令级顶层 key。

### `diverse_select` — 多样性代表性结构选择 *（别名：`fps_select`）*

在描述符空间中使用最远点采样 (FPS) 或 CUR (SVD leverage-score) 分解选择多样化/代表性子集。别名: `fps_select`。

```bash
# FPS (默认): 几何多样性选择
imrmlp diverse_select --n-select=50 --pool=pool.xyz --output=selected.xyz

# CUR: 统计 leverage-score 采样
imrmlp diverse_select --method=cur --n-select=50 --pool=pool.xyz --output=selected.xyz

# 使用余弦距离度量的 FPS
imrmlp diverse_select --method=fps --metric=cosine --n-select=50 --pool=pool.xyz
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--pool` | | `pool.xyz` | 输入结构池文件 |
| `--potfile` | | `potential.mtp` | 势文件（用于基于描述符的 B-矩阵） |
| `--pottype` | | `mtp` | 势类型 |
| `--dtype` | | `mtm` | 描述符类型：`mtm`、`pot_efs`、`behler`、`bondorder` |
| `--index` | `-id` | `index.map` | 索引文件 |
| `--format` | `-f` | | 输入格式 |
| `--output` | `-o` | `selected.xyz` | 输出选中结构文件 |
| `--n-select` | `-n` | `0` | 要选择的结构数量上限（CUR 必需） |
| `--method` | | `fps` | 选择方法：`fps`（默认）或 `cur` |
| `--metric` | | `euclidean` | FPS 距离度量：`euclidean`（默认）或 `cosine` |
| `--threshold-distance` | | `0.0` | 最小距离阈值（提前停止选择） |
| `--first-idx` | | `-1` | 第一个选中结构的索引（-1 = 随机） |
| `--cur-deterministic` | | `false` | 使用确定性 CUR（布尔值） |
| `--seed` | | `42` | 可复现性的随机种子 |
| `--compute-error` | | `false` | 计算 CUR 重构误差（布尔值） |
| `--score-file` | | | 输出每个结构的评分 |
| `--mode` | | `cfg` | 行模式：`cfg`（逐结构）或 `nbh`（逐原子） |
| `--all-coeffs` | | `false` | 使用 ALL 系数（不仅是线性系数，布尔值） |
| `--verbose` | `-v` | `false` | 详细输出（布尔值） |
| `--row-scaling` | | `uniform` | 行缩放：`uniform` 或 `mlip` |
| `--w-energy` | | `1.0` | `pot_efs` 的能量行权重 |
| `--w-site` | | `0.0` | `pot_efs` 的位点能行权重 |
| `--w-force` | | `0.0` | `pot_efs` 的力行权重 |
| `--w-stress` | | `0.0` | `pot_efs` 的应力行权重 |
| `--settings` | | `diverse_select.ini` | 设置文件（必需） |

配置文件中，`diverse_select` 的描述符/B-矩阵构建使用同样的 `builder:*`
配置块。`method`、`n-select`、`metric`、`seed` 等 FPS/CUR 控制项仍为顶层 key。

FPS 要求 `--n-select` 或 `--threshold-distance` 至少一个为正数。
同时指定时，达到数量上限或距离阈值任一条件即停止选择。

### `calc_crowding` — 描述符空间拥挤度权重

计算有界内存的逐原子描述符拥挤度权重，并写入 `atomic_fitness`
（可选写入构型 `fitness`）。命令对输入做三次流式遍历，因此无需把全部
描述符保存在内存中即可处理大数据集。

```bash
imrmlp calc_crowding --potfile=potential.mtp --input=pool.xyz --output=pool-crowded.xyz
imrmlp calc_crowding --potfile=potential.mtp --input=pool.xyz --kernel=gaussian --apply-to-energy
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--input` | `-in` | `pool.xyz` | 输入结构 |
| `--output` | `-out` | `pool-crowded.xyz` | 带 `atomic_fitness` 的输出结构 |
| `--input-format` | `-if` | auto | 输入格式 |
| `--output-format` | `-of` | auto | 输出格式（`extxyz`、`mlpcfg` 或 `mlipbin`） |
| `--settings` | | `crowding.ini` | 设置文件（位置参数） |
| `--index` | `-id` | `index.map` | 物种索引映射 |
| `--kernel` | | `cosine` | 核函数：`cosine` 或 `gaussian` |
| `--sigma` | | auto | 高斯带宽（≤0 = 自动） |
| `--standardize` | | `false` | 核函数前标准化描述符分量（布尔） |
| `--alpha` | | | 反密度指数 |
| `--clip-min` | | | 权重下界裁剪 |
| `--clip-max` | | | 权重上界裁剪 |
| `--sample` | | | 每元素参考行数（-1 = 精确小数据模式） |
| `--seed` | | | 确定性采样种子 |
| `--batch-rows` | | | 密度查询批大小 |
| `--bucket` | | `true` | 按化学元素分桶 |
| `--apply-to-energy` | | `false` | 以平均原子权重写入构型 fitness |
| `--potfile` | | `potential.mtp` | `dtype=pot_efs` 使用的势文件 |
| `--pottype` | | `mtp` | 势类型 |
| `--dtype` | | `mtm` | 描述符类型：`mtm`、`mtp`、`mtp_scalar`、`behler`、`bondorder` |
| `--mode` | | `nbh` | 行模式；本命令强制为 `nbh` |
| `--all-coeffs` | | `false` | 使用全部势系数 |
| `--row-scaling` | | `uniform` | 行缩放：`uniform` 或 `mlip` |
| `--w-energy` | | `1.0` | `pot_efs` 能量行权重 |
| `--w-site` | | `0.0` | `pot_efs` 位点能行权重 |
| `--w-force` | | `0.0` | `pot_efs` 力行权重 |
| `--w-stress` | | `0.0` | `pot_efs` 应力行权重 |

描述符构造设置从 `builder:` 块读取。

### 主动学习策略指南

| 策略 | 命令 | 原理 | 输出 | 最适合 |
|----------|-----------|-----------|--------|----------|
| **MaxVol 选择** | `maxvol_select` | 最大化描述符子矩阵的行列式 | 最大独立子集 | 构建初始良态训练集 |
| **MaxVol 采样** | `maxvol_sample` | 相对现有集的外推评分 | 逐结构评分 (>1 = 外推) | 筛选候选池中信息量高的结构 |
| **MaxVol 扩充** | `maxvol_extend` | 采样 + 添加高评分结构 | 扩充后的训练集 | 增量增长已有训练集 |
| **多样性 FPS** | `diverse_select --method fps` | 最大化最小成对距离 | 均匀多样化子集 | 确保构型空间的几何覆盖 |
| **多样性 CUR** | `diverse_select --method cur` | SVD 杠杆分数采样 | 统计代表性子集 | 带重构保证的大池降采样 |

@note **典型主动学习循环**：在外部准备候选结构 → `maxvol_select`（选择初始参考集）→
执行参考计算 → `trainmtp` 或 `train_linear`（拟合单个势）→ `maxvol_sample`（候选评分）→
`maxvol_extend`（扩充主动集）→ 为选中结构补充参考标签并重新拟合。

---

## 结构操作

结构操作命令在不计算能量或力的情况下变换原子构型。它们是预处理步骤——
在将结构送入 EFS 评估或弛豫之前，制备超胞、标准化晶胞、重排轴或使用 `findint` 插入间隙原子。

@note 这些命令不需要势函数，纯几何操作。

### `mk_super` — 创建超胞

使用三个对角整数或九个行主序整数构造超胞。

```bash
imrmlp mk_super --input=POSCAR 2 2 2 --output=super.vasp
imrmlp mk_super --input=POSCAR 2 0 0 0 2 0 0 0 2 --output=super.vasp   # 3x3 矩阵（行主序）
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--input` | `-in` | `POSCAR` | 输入结构（位置参数） |
| `--output` | `-out` | `POSCAR_REV` | 输出结构（位置参数） |
| `--supercell` | `-s` | | 3 个对角整数或 9 个行主序整数（位置参数） |
| `--input-format` | `-if` | auto | 输入格式 |
| `--output-format` | `-of` | auto | 输出格式 |
| `--index` | `-index` | `index.map` | 物种索引映射 |
| `--check` | `-c` | `true` | 左手晶胞时翻转第一行晶格矢量 |
| `--overwrite` | `-ov` | `false` | 覆盖已有输出文件 |

### `mk_lower` — 将晶胞转换为下三角形式

将晶胞矩阵转换为下三角形式，便于比较或对称性分析前标准化周期晶胞。

```bash
imrmlp mk_lower --input=POSCAR --output=lower.vasp
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--input` | `-in` | `POSCAR` | 输入结构（位置参数） |
| `--output` | `-out` | `POSCAR_REV` | 输出结构（位置参数） |
| `--ax` | `-ax` | | 与 x 轴对齐的晶格矢量（`a`、`b` 或 `c`） |
| `--order` | `-order` | | 轴顺序：`xyz`、`xzy`、`yxz`、`yzx`、`zxy`、`zyx`（或 `abc` 形式） |
| `--input-format` | `-if` | auto | 输入格式 |
| `--output-format` | `-of` | auto | 输出格式 |
| `--index` | `-index` | `index.map` | 物种索引映射 |
| `--overwrite` | `-ov` | `false` | 覆盖已有输出文件 |

### `mk_orth` — 寻找（近似）正交晶胞

保持晶格矢量 **a** 不变。搜索角度接近 90° 的等价晶胞。

```bash
imrmlp mk_orth --input=structure.cfg --output=orth.cfg
imrmlp mk_orth --input=POSCAR --axis-choice 1 1 0 --axis-to x --max-expand 8
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--input` | `-in` | `POSCAR` | 输入结构（位置参数） |
| `--output` | `-out` | `POSCAR_ORTH` | 输出结构（位置参数） |
| `--tol` | `-tol` | `1e-2` | 正交性判据的最大绝对余弦 |
| `--max-expand` | `-mx` | `20` | 系数/体积膨胀预算 |
| `--axis-choice` | | | 三个整数晶格方向系数（约化为本原方向）；要求全周期 |
| `--axis-to` | | `x` | 将指定方向对齐到 `x`、`y` 或 `z` |
| `--index` | `-id` | `index.map` | 物种索引映射 |
| `--one` | `-one` | `false` | 找到第一个结果后停止 |
| `--search-only` | `-s` | `false` | 只打印超胞矩阵，不写出结构 |
| `--input-format` | `-if` | auto | 输入格式 |
| `--output-format` | `-of` | auto | 输出格式 |
| `--overwrite` | `-ov` | `false` | 覆盖已有输出文件 |

### `mk_orth_search` — 搜索正交晶胞

与 `mk_orth` 类似，但搜索所有晶格矢量方向。

```bash
imrmlp mk_orth_search --input=structure.cfg --output=orth.cfg --tol=1e-4
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--input` | `-in` | `POSCAR` | 输入结构（位置参数） |
| `--output` | `-out` | `POSCAR_ORTH` | 输出结构（位置参数） |
| `--tol` | `-tol` | `1e-2` | 正交性判据的最大绝对余弦 |
| `--max-expand` | `-mx` | `20` | 系数/体积膨胀预算 |
| `--index` | `-id` | `index.map` | 物种索引映射 |
| `--one` | `-one` | `false` | 找到第一个结果后停止 |
| `--input-format` | `-if` | auto | 输入格式 |
| `--output-format` | `-of` | auto | 输出格式 |
| `--overwrite` | `-ov` | `false` | 覆盖已有输出文件 |

### `trans_reduce` — 减少平移多重性

减少结构沿晶格方向的平移多重性。

```bash
imrmlp trans_reduce --input=POSCAR --output=reduced.vasp --tol=1e-4
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--input` | `-in` | `POSCAR` | 输入结构（位置参数） |
| `--output` | `-out` | `POSCAR_TRED` | 输出结构（位置参数） |
| `--tol` | `-tol` | `1e-3` | 对称性容差 |
| `--optimize-search` | | `false` | 对 3D/1D 搜索启用精确候选剪枝/缓存 |
| `--index` | `-id` | `index.map` | 物种索引映射 |
| `--input-format` | `-if` | auto | 输入格式 |
| `--output-format` | `-of` | auto | 输出格式 |
| `--overwrite` | `-ov` | `false` | 覆盖已有输出文件 |

### `primi_reduce` — 约化到初基晶胞

寻找所有角度在 60° 到 120° 之间的初基晶胞。

```bash
imrmlp primi_reduce --input=POSCAR --output=primitive.vasp
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--input` | `-in` | `POSCAR` | 输入结构（位置参数） |
| `--output` | `-out` | `POSCAR_PRIM` | 输出结构（位置参数） |
| `--tol` | `-tol` | `1e-5` | 对称性容差 |
| `--two-stage` | `-t` | `true` | 先执行 `trans_reduce` 再做完整初基搜索 |
| `--correct-cell` | `-c` | `false` | 尽量使结果晶胞正交 |
| `--optimize-search` | | `false` | 对 3D/1D 搜索启用精确候选剪枝/缓存 |
| `--index` | `-id` | `index.map` | 物种索引映射 |
| `--input-format` | `-if` | auto | 输入格式 |
| `--output-format` | `-of` | auto | 输出格式 |
| `--overwrite` | `-ov` | `false` | 覆盖已有输出文件 |

### `swap_axis` — 重排晶格轴

```bash
imrmlp swap_axis --order=132 --input=POSCAR --output=swapped.vasp
# mode: 3 位数字排列。132 = xyz → xzy
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--order` | `-order` | `xyz` | 轴排列（位置参数）：`xyz`、`xzy`、`yxz`、`yzx`、`zxy`、`zyx`（或 `abc` 形式） |
| `--input` | `-in` | `POSCAR` | 输入结构（位置参数） |
| `--output` | `-out` | `POSCAR_REV` | 输出结构（位置参数） |
| `--tol` | `-tol` | `1e-6` | 轴交换容差 |
| `--keephand` | `-k` | `true` | 保持右手性 |
| `--tolower` | `-l` | `false` | 结果转换为下三角形式 |
| `--index` | `-index` | `index.map` | 物种索引映射 |
| `--input-format` | `-if` | auto | 输入格式 |
| `--output-format` | `-of` | auto | 输出格式 |
| `--overwrite` | `-ov` | `false` | 覆盖已有输出文件 |

### `sameq` — 检查结构等价性

检查两个结构是否等价（相同几何结构）。

```bash
imrmlp sameq --input1=structure1.cfg --input2=structure2.cfg --tol=0.1
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--input1` | `-in1` | | 第一个结构（位置参数） |
| `--input2` | `-in2` | | 第二个结构（位置参数） |
| `--tol` | `-tol` | `1e-4` | 晶胞与位置容差（Å） |
| `--input-format1` | `-if1` | auto | 第一个结构的格式 |
| `--input-format2` | `-if2` | auto | 第二个结构的格式 |
| `--index` | `-id` | `index.map` | 物种索引映射 |
| `--scale` | `-s` | `false` | 比较前将两个晶胞缩放到相同体积 |
| `--allow-reflections` | | `true` | 将镜像视为等价 |

### `find_mindist` — 最小原子间距

输出每个结构的最小原子间距及全局最小值。默认打印到标准输出；
给出 `--output` 时，每行一个距离，最后一行是全局最小值。

```bash
imrmlp find_mindist --input=traj.cfg
imrmlp find_mindist --input=traj.cfg --output=mindist.dat
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--input` | `-in` | `POSCAR` | 输入结构（位置参数） |
| `--output` | `-out` | | 可选的距离输出文件 |
| `--input-format` | `-if` | auto | 输入格式 |
| `--output-format` | `-of` | auto | 输出格式 |
| `--index` | `-index` | `index.map` | 物种索引映射 |
| `--overwrite` | `-ov` | `false` | 覆盖已有输出文件 |

## 分析

结构分析命令计算衍生物理量——从基本的结构描述符（RDF、键角）到高级量
（力常数、弹性常数）。部分命令需要势函数进行 EFS 评估；
其他命令仅基于几何结构。

`sameq` 与批量结构去重共用比较内核，检查晶格、物种及原子的一一对应，
支持平移、原子重排和晶格对称操作。仅全周期结构约化到原胞；部分周期和非周期
结构保留晶胞及真空尺寸。默认允许镜像等价，`--allow-reflections=false` 可区分镜像。
快速去重在化学组成组内比较缓存描述符，不再按量化描述符分桶。

### `calc_prop` — 结构性质

计算结构性质：RDF、键角、二面角、SRO、HA 指数、Voronoi、CNA、MSD、VACF。


| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--settings` | | | 设置文件（位置参数，必需） |
| `--input` | `-in` | `input.cfg` | 输入轨迹文件 |
| `--input-format` | `-if` | auto | 轨迹文件格式 |
| `--output-xyz` | `-o` | | 带逐原子属性的 EXTXYZ 输出轨迹 |
| `--output-sro` | `-os` | `output_sro.txt` | SRO 分析输出文件 |
| `--output-cna` | | `output_cna.txt` | 所选各帧、各原子的 CNA 计数 |
| `--output-neighbor` | `-on` | `output_neighbor.txt` | 近邻信息输出文件 |
| `--index` | `-id` | `index.map` | 索引文件 |
| `--cutoff` | `-c` | `5.0` | 邻域截止半径 |
| `--cutoff-inner` | `-ci` | `-1.0` | 壳层排除内截止半径 |
| `--pair-cutoff` | `-pc` | `false` | 类型相关成对截止半径（布尔） |
| `--rdf-kd` | `-rkd` | `false` | 使用 KD-tree 计算 RDF（布尔） |
| `--rdf-no-pbc` | `-rnp` | `false` | KD-tree RDF 禁用 PBC（布尔） |
| `--rdf-rho-all` | | `false` | 从全帧平均计算密度（布尔） |
| `--ref-rlimit` | `-rl` | `15.0` | RDF 最大半径 |
| `--framestart` | `-s` | `0` | 起始帧索引 |
| `--frameend` | `-e` | | 结束帧索引 |
| `--skip-every` | `-skip` | `1` | 分析跳帧间隔 |
| `--num-types` | `-nt` | `-1` | 原子类型数量（-1 = 自动检测） |
| `--rdf` | `-r` | `false` | 计算 RDF / 对关联函数（布尔值） |
| `--bondangle` | `-b` | `false` | 计算键角分布（布尔值） |
| `--dihedral` | `-d` | `false` | 计算二面角分布（布尔值） |
| `--sro` | `-sro` | `false` | 计算短程序参数（布尔值） |
| `--ha-index` | `-ha` | `false` | 计算 HA 指数（布尔） |
| `--voro-index` | `-voro` | `false` | 计算 Voronoi 指数（布尔） |
| `--voro-neighbor` | `-vb` | `false` | 使用 Voronoi 近邻（布尔） |
| `--voro-num-print` | `-vn` | `10` | 输出 Voronoi 指数数量 |
| `--bop` | `-bop` | `false` | 计算键序参数（布尔值） |
| `--c-cna` | `-cna` | `false` | 计算公共近邻分析（布尔） |
| `--c-wcsro` | `-wcsro` | `false` | 计算 Warren-Cowley SRO（布尔） |
| `--avgpos` | `-caps` | `false` | 计算平均原子位置（布尔值） |
| `--msd` | `-msd` | `false` | 计算均方位移（布尔值） |
| `--vacf` | `-vacf` | `false` | 计算速度自相关函数（布尔值） |
| `--output-all-indices` | | `false` | 输出全部 HA 和 Voronoi 指数（布尔） |
| `--avgpos-remove-drift` | | `false` | 计算平均位置时移除质心漂移（布尔） |
| `--avgpos-fractional` | | `false` | 对展开后的分数坐标取平均而非笛卡尔坐标（布尔） |


`--c-cna` 将 `frame atom central_type neighbor_type cna_index species_index count`
各列写入 `--output-cna`，帧和原子索引从零开始；物种合并行使用
`neighbor_type=species_index=-1`。`--output-xyz` 同时包含整数属性 `cna_total`
和 `cna_<index>`。`cna:specie_info=true` 增加物种分辨的计数行和属性列。
小周期晶胞中若多个邻居镜像共用原子编号，需扩胞后计算 CNA。

`--c-wcsro` 将所选帧汇总后的完整有向 2×2 矩阵写入 `--output-sro`；与
`--sro` 同用时追加独立的 WCSRO 区段。要求恰好两种物种且两者均有邻居。
`wcsro:r_inn`、`wcsro:r_out` 定义严格壳层 `r_inn < r < r_out`；省略或设置为
`-1` 时分别采用 `--cutoff-inner`、`--cutoff`。`wcsro:sro_type` 支持 `WC`、`PM`。
WC 定义为 `1 - P(j|i)/c_j`，配位数不等时两个非对角元可以不同。
CNA、SRO 和带属性轨迹的输出路径必须彼此独立，符号链接和硬链接也不能指向同一文件。

`bondangle:sep`、`bondangle:low` 和 `bondangle:high` 均使用弧度，输出角度使用度。
默认间隔为 1°，网格包含 0° 和 180°；观测值计入最近的网格点，末尾不足一个间隔时计入最后一格。

BOP 默认使用 `--cutoff`，可用 `bop:cutoff` 覆盖；`bop:nnn` 选择固定数量的
最近邻，`bop:minnn` 和 `bop:maxnn` 限定半径筛选后的邻居数量。开启
`--voro-neighbor` 后，BOP 直接使用 Voronoi 邻域。
`bop:average` 对球谐分量平均时包含中心原子；`bop:corase` 对邻居的 Q/W
标量取平均，可独立于 `bop:average` 使用。无邻居原子的 Q/W 置零并给出警告。
SRO 从全构型排序后的邻居距离确定壳层，结果不受原子排列顺序影响。
每条键只属于一个壳层：与该壳层最小距离之差小于 `sro:width` 的距离归入
同一壳层，后续距离开始新壳层。
`--avgpos` 按各帧晶胞沿周期晶格方向，选择相邻帧之间最近的周期像连续展开坐标后
取平均。这要求相邻帧在每个周期方向上的位移小于半个晶胞，输出位置可能在晶胞外。

### `calc_descriptors` — 计算描述符

计算结构的 ML 描述符（ACSF、bond order、MTM）。

```bash
imrmlp calc_descriptors --settings=settings.ini -pot potential.mtp -in POSCAR -out descriptors.txt
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--settings` | | `input.inp` | 设置文件（位置参数，可选） |
| `--input` | `-in` | `POSCAR` | 输入结构文件（位置参数，必需） |
| `--output` | `-out` | `output-desc.xyz` | 输出描述符文件 |
| `--input-format` | `-if` | | 输入格式（空 = 自动检测） |
| `--output-format` | `-of` | `extxyz` | 输出格式 |
| `--index` | `-id` | `index.map` | 索引文件 |
| `--potfile` | `-pot` | `potential.mtp` | 势文件 |
| `--dtype` | | `mtp` | 描述符类型：`mtp`、`mtm`、`biso3`、`biso4`、`behler`、`bondorder` |
| `--biso3-lmax` | | `4` | biso3：最大角动量 L |
| `--biso3-rmin` | | `0.0` | biso3：最小径向距离 |
| `--biso3-rmax` | | `5.0` | biso3：截断半径 |
| `--biso3-rsize` | | `8` | biso3：径向基大小 |
| `--biso3-real-harmonics` | | `false` | biso3：选择实谐函数后端 |
| `--cutoff` | `-c` | `5.0` | 描述符计算的截断半径 |
| `--fuse-species-info` | `-fs` | `false` | 融合种类信息（布尔值） |
| `--discard-local-components` | `-d` | `false` | 不输出逐原子局域描述符分量（布尔） |
| `--local-name` | | `desc` | 局域描述符数组名 |
| `--global-name` | | `desc` | 全局描述符数组名 |
| `--local-energy-components` | `-le` | `false` | 输出局域能量分量（布尔值） |

SO3 / SO4 使用示例：

```sh
imrmlp calc_descriptors --dtype biso3 --input input.xyz --input-format extxyz \
  --biso3-lmax 3 --biso3-rsize 2 --biso3-rmax 4 --biso3-real-harmonics true
imrmlp calc_descriptors --dtype biso4 --settings so4.ini --input input.xyz
```

`so4.ini` 使用以下 `biso4:` 设置。物种索引从 0 开始，需覆盖权重数组；局部输出始终是逐中心描述符。

| 设置 | 默认值 | 说明 |
| --- | --- | --- |
| `biso4:twojmax` | `9` | 最大二倍角动量阶数 |
| `biso4:cutoff` | `5` | 近邻截断 |
| `biso4:radius` | `4/3 cutoff` | S3 球半径 |
| `biso4:weights` | `10 entries, 1` | 物种密度权重（默认 10 项，每项 1） |
| `biso4:selfweights` | `10 entries, 0.6` | 中心物种权重（默认 10 项，每项 0.6） |
| `biso4:normalize_U` | `false` | 归一化 Wigner U 矩阵 |
| `biso4:quaternion_convention` | `legacy` | legacy 或 q1/q2/q3/q4（legacy 默认 q4） |
| `biso4:radial_angle_map` | `linear` | linear 或 atan 径向嵌入 |
| `biso4:average` | `outer` | outer 或 inner-density |
| `biso4:contraction_backend` | `reference` | reference 或精确稀疏 snap 收缩 |
| `biso4:wigner_backend` | `direct` | direct 或实验性 spline（twojmax <= 12） |

`inner-density` 先平均密度系数，再构造旋转不变量；`outer` 为局部描述符的平均。SO4 的 `snap` 只是收缩后端名称，无需安装 LAMMPS。spline 后端尚属实验性，不支持三阶力常数。

### `calc_eos` — 状态方程 *（别名：`ev`）*

从能量-体积或压力-体积计算拟合状态方程。自动生成应变结构、评估势函数并拟合 EOS 参数。
使用 `--curve ev|pv` 选择模式。

```bash
# 能量-体积曲线（默认；省略 --settings 时使用 input.ini）
imrmlp ev --input=POSCAR --settings=eos.ini
imrmlp calc_eos --input=POSCAR --upper=0.05 --lower=-0.05

# 压力-体积曲线（默认同样使用 input.ini）
imrmlp calc_eos --curve=pv --input=POSCAR --press-ub=10 --press-lb=-10
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--settings` | | `input.ini` | 可选的位置参数设置文件；提供 `[pot]` 和 `[relax]` 设置 |
| `--curve` | `-c` | `ev` | 曲线类型：`ev`（能量-体积）或 `pv`（压力-体积） |
| `--input` | `-in` | `POSCAR` | 输入平衡结构 |
| `--output` | `-o` | | EOS 曲线数据输出文件 |
| `--input-format` | | auto | 输入格式 |
| `--index` | `-id` | `index.map` | 索引文件 |
| `--skip-prelax` | | `false` | 跳过预弛豫（布尔） |
| `--traj-format` | | `mlpcfg` | 输出轨迹格式 |
| `--npos` | | `5` | 正向应变点数（> 参考点） |
| `--nneg` | | `5` | 负向应变点数（< 参考点） |
| `--upper` | `-ub` | `0.05` | [EV] 体积膨胀上限分数（如 0.05 = +5%） |
| `--lower` | `-lb` | `-0.05` | [EV] 体积压缩下限分数（如 -0.05 = -5%） |
| `--ninterp` | `-ninterp` | `100` | [EV] EOS 曲线插值点数，至少为 2 |
| `--out-veph` | | `veph.dat` | [EV] 体积-能量-压力-焓插值数据输出文件 |
| `--fit-only` | `-fit` | `false` | [EV] 仅拟合 EOS，跳过计算（布尔） |
| `--lossless` | | `false` | [EV] 启用 lossless 结构输出模式（布尔） |
| `--output-eos` | | `eos.dat` | [EV] EOS 拟合参数输出文件（Birch-Murnaghan） |
| `--press-ub` | `-pub` | `10` | [PV] 压力上限（GPa） |
| `--press-lb` | `-plb` | `-10` | [PV] 压力下限（GPa） |

@note 从命令行角度看，设置文件是可选的，默认使用 `input.ini`。
`[pot]` 部分配置势函数，`[relax]` 控制预弛豫；EOS 拟合参数通过命令行选项配置。

EV 拟合要求存在有限正体积处的稳定平衡点，且体模量为有限正数。
不满足时，在写出拟合参数和插值数据前报错；具有有效极小值的二次拟合仍受支持。

### `calc_elastic` — 弹性常数 *（别名：`elastic`、`mlpcalccellhess`）*

使用解析 Hessian 或数值应变微分计算的弹性常数。通过 `--method` 选择模式。
解析模式需要势具有完整的解析 Hessian 支持（如 MTP）；数值模式适用于任何支持 EFS 的势函数。

```bash
# 解析模式（默认自动检测）
imrmlp elastic --potfile=potential.mtp --input=relaxed.vasp

# 数值模式
imrmlp elastic --potfile=potential.mtp --input=relaxed.vasp --method=numerical --strain-magnitude=0.005
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--potfile` | `-pot` | `potential.mtp` | 势文件（位置参数，必需） |
| `--input` | `-in` | `POSCAR` | 输入结构（位置参数，必需） |
| `--method` | | | 计算方法：`auto`（默认自动检测）、`analytical` 或 `numerical` |
| `--input-format` | `-if` | auto | 输入格式 |
| `--index` | `-id` | `index.map` | 索引文件 |
| `--settings` | | | 设置文件 |
| `--elastic-output` | `-eo` | `elastic.out` | 弹性常数输出文件 |
| `--eigensys-output` | | `eigensys.out` | 解析本征系统输出文件 |
| `--symprec` | `-symprec` | `1e-4` | 对称容差 |
| `--eigzero-tol` | `-etol` | `1e-4` | 零特征值容差 |
| `--use-gemo-mean` | `-g` | `false` | 使用几何平均（布尔） |
| `--pressure` | `-p` | `0` | 外部压力（GPa） |
| `--debug` | | `false` | [数值模式] 调试输出（布尔） |
| `--strain-magnitude` | `-emag` | `0.01` | [数值模式] 应变步长 |
| `--num-strain-points` | `-n` | `5` | [数值模式] 应变点数（应为奇数） |
| `--mode` | | `0` | [数值模式] 0=应力-应变，1=能量-应变 |
| `--pottype` | `-pt` | `mtp` | [数值模式] 势类型 |
| `--relax-correction` | `-cr` | `true` | [解析模式] 应用离子弛豫修正（布尔） |
| `--correction-l1` | `-c1` | `false` | [解析模式] 一阶修正 1（布尔） |
| `--correction-l2` | `-c2` | `false` | [解析模式] 一阶修正 2（布尔） |
| `--compute-eigsys` | | `false` | [解析模式] 计算完整超 Hessian 特征系统（布尔） |

主刚度矩阵及 Voigt/Reuss/Hill 性质对应弛豫离子的材料张量 (C)。当
`--pressure` 非零时，还会输出用于机械稳定性本征值判断的 Born/Wallace
切线张量。
数值方法的离子弛豫参数（例如 `relax:gtol` 和 `relax:maxiteration`）应写入
`--settings` 指定的设置文件。

**方法对比：**

| 方面 | 解析 (`analytical`) | 数值 (`numerical`) |
|--------|---------------|-------------------|
| **原理** | 解析 Hessian | 数值应变有限差分 |
| **势函数要求** | 必须支持 `getEFSHessian` | 任何支持 EFS 的势函数 |
| **精度** | 精确（到数值精度） | 取决于应变步长 |
| **速度** | 单次评估 | 多次 EFS 评估（6+ 应变模式） |
| **最适合** | MTP 等支持 Hessian 的势函数 | EAM、LJ 或无 Hessian 的势函数 |

### `calc_fc` — 力常数导出 *（别名：`fc`、`mlpcalcfc`）*

从势函数计算原子 Hessian 并导出力常数；色散、DOS、热力学与振动动画使用 Phonopy。

```bash
imrmlp calc_fc --potfile potential.mtp --input POSCAR --supercell 3 3 3 --reduced=false
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--potfile` | `-pot` | `potential.mtp` | 势文件（必需） |
| `--input` | `-in` | `POSCAR` | 输入结构（必需） |
| `--potential-type` | `-pt` | `mtp` | 势类型 |
| `--input-format` | `-if` | auto | 输入格式 |
| `--index` | `-id` | `index.map` | 索引文件 |
| `--settings` | | `fc.ini` | 设置文件（可选） |
| `--supercell` | `-dim` | | 超胞尺寸（可变长度，例如 `3 3 3` 或 9 个矩阵分量） |
| `--symprec` | `-tol` | `1e-5` | 对称容差 |
| `--write-force-constant` | `-fc` | `true` | 写入力常数文件（布尔） |
| `--reduced` | `-rd` | `true` | 使用对称约化力常数（布尔） |
| `--hessian-method` | | `analytic` | 原子 Hessian 后端：`analytic`、`auto`、`force_fd`、`energy_fd` 或 `hybrid` |
| `--hessian-nudge-pos` | | `0.001` | 笛卡尔有限差分位移（Angstrom） |
| `--hessian-cutoff` | | `0` | 有限差分近邻截断；非正值使用势函数截断 |
| `--check-symm` | | `false` | 报告张量声学求和规则残差 |
| `--lowest-modes` | | `0` | 用解析 HVP 求最低的 N 个内部 Gamma 模；0 表示导出力常数 |
| `--lowest-mode-tol` | | `1e-8` | 最低模相对残差容差 |
| `--lowest-mode-max-matvecs` | | `0` | HVP 次数上限；非正值使用 max(400,20*N) |

`FORCE_CONSTANTS` 的块单位为 eV/Å²，原子顺序对应 `supercell_created.xsf`；
`sup_debug_efs.xyz` 记录同一超胞的 E/F/S。完整矩阵使用 `--reduced=false`。

默认 `analytic` 在势函数不支持解析 Hessian 时明确报错；`auto` 回退到力的有限差分。
`energy_fd` 和 `hybrid` 的求值次数随笛卡尔自由度平方增长，适合小体系验证。
`--check-symm` 仅报告残差，不修改力常数。

`--lowest-modes=N` 保留为矩阵无关的 HVP 诊断，投影掉三维质量加权平移，
仅输出 N 个内部模到 `EIGENVALS`、`EIGENVECTORS` 和 `EIGENSYS`，不写力常数。
该模式要求解析 HVP，不能与 `--check-symm` 组合。
质量读取 `index.map` 的第三列，缺省使用元素表。

Phonopy 接入见 [Python 适配器](python_zh.md#phonopy-integration)。导入解析力常数时，
先在 Phonopy 中生成超胞，将其原样写成 `calc_fc` 输入，省略 `--supercell` 并使用
`--reduced=false`；只有当晶胞及两轴原子顺序都与 `phonon.supercell` 一致时，
才可赋值给 `phonon.force_constants`，不能按原子数相同直接混用。

### `ti_fit_fc` — 有限温度正定有效力常数

`ti_fit_fc` 用于 MD/ensemble 拟合与 LAMMPS 简谐参考文件导出。必须显式选择 `--workflow md|ensemble|md-ensemble|export`：

```bash
imrmlp ti_fit_fc --workflow md-ensemble \
  --traj md.extxyz --traj-format extxyz \
  --potfile potential.mtp --index index.map \
  --temperature 600 --output-prefix BETA_600K
```

未给 `--input` 时，`md` 工作流会对固定晶胞 NVT 轨迹做连续 PBC unwrap、
去质量加权质心漂移并构造平均结构。NPT 轨迹会被拒绝；应先确定平均晶胞，
再在该晶胞补跑 NVT。默认用当前势函数重新计算所有轨迹力；只有显式指定
`--force-source trajectory` 才使用轨迹保存的力。

周期镜像在去除共同平移后选择，支持质心漂移跨越半个晶胞；原子间的相对晶体
振动仍需小于半个周期。

拟合和采样均在去除三个平移模的质量加权内部空间完成。默认先做模态刚度
更新，再对软模和近简并小块做 `sqrt` root 拟合；自洽迭代用当前 FC 生成
antithetic 高斯 ensemble、计算真实 MTP 力，并在 root 空间阻尼更新。默认
`--initial-floor-thz=1` 同时控制初始稳定化和每一轮自洽 ensemble 的物理采样
刚度，`--pd-floor-thz=1e-4` 仅是拟合 FC 的数值正定下限，二者不可混用。
收敛次数仅在 modal warmup 结束、进入耦合阶段后累计。两个拟合阶段的目标均为
按权重平均的力残差平方加上偏离当前 FC 的 `--ridge` 惩罚；重复样本不会改变
正则化强度。
超过位移或最短距离限制的 antithetic 样本会按 `+q/-q` 整对拒绝并重新生成，
不会因单个高斯尾部样本立即终止。`--max-sampling-attempts=0` 默认允许尝试
目标样本数的 20 倍；预算耗尽后程序会报告两类拒绝数、实际最大位移和最短
距离并失败，不会自动修改 sampling floor 或使用不足数量的样本拟合。

主要输出为 `<prefix>_FORCE_CONSTANTS`、平均参考结构、迭代 CSV、诊断报告、
独立验证 ensemble 和 `<prefix>_LAMMPS_REFERENCE`。后者可直接供新的
LAMMPS `pair_style harmonic/reference` 使用。运行
`imrmlp ti_fit_fc --help` 查看收敛、soft-block 与安全位移选项。

报告保留 MD 拟合阶段的训练/留出误差，并记录最终拟合参考态对应的
`final_md_force_rmse`。训练误差对应中心化的拟合目标；留出误差、
`final_md_force_rmse` 和 `harmonic_validation_force_rmse` 直接计算简谐力残差，
不再减去验证集自身的平均力偏差。以上误差均使用质量加权内部力坐标，单位为
eV/Angstrom/sqrt(amu)。`lammps_*` 诊断和 `# lammps_mode` 表始终对应实际写入
`<prefix>_LAMMPS_REFERENCE` 的 FC；TI 验收和解析谐波自由能应使用这些值。
迭代 CSV 和报告还记录采样尝试数、拒绝数、接受率及几何极值；持续偏低的
接受率表明参考态或 sampling floor 不适合，而不是应继续放宽安全限制。

LAMMPS 文件可按原子对的 3x3 FC 块稀疏导出。截断后会重新构造 onsite 块
以满足 acoustic sum rule；若最低内部频率低于目标，可在保留的连通图上加入
最小各向同性稳定项。输入参考结构已携带有限能量时，可把已有局域 FC 转换为
稀疏参考态而不调用势函数；若缺少能量，`ti_fit_fc` 会自动调用所选势函数：

```bash
imrmlp ti_fit_fc --workflow export \
  --input REFERENCE.xyz --input-format extxyz \
  --initial-fc FORCE_CONSTANTS --index index.map \
  --lammps-fc-source initial --lammps-fc-block-threshold 0.01 \
  --lammps-fc-min-frequency-thz 0.5 \
  --write-validation-ensemble false --output-prefix TI_SPARSE
```

也可用 `--lammps-fc-cutoff` 做距离截断。报告中的保留块比例、Hessian 相对变化、
稳定化前后最低频率和 ASR 误差必须一起检查。ensemble 拟合或全局谱投影后的
`fitted` FC 可能含有较大的长程块，此时直接截断会显著改变参考态；若目标是
保留原始局域模型，应使用 `--lammps-fc-source initial`。


| 选项 | 别名 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--workflow` |  | `—` | 必选工作流：md、ensemble、md-ensemble、export |
| `--traj` |  | `—` | MD 轨迹 |
| `--traj-format` |  | `—` | 轨迹格式 |
| `--input` | `-in` | `—` | 可选的显式参考结构 |
| `--input-format` | `-if` | `—` | 参考结构格式 |
| `--initial-fc` |  | `—` | 可选的初始 FORCE_CONSTANTS |
| `--potfile` | `-pot` | `potential.mtp` | 用于计算力的势文件 |
| `--potential-type` | `-pt` | `mtp` | 势类型 |
| `--index` | `-id` | `index.map` | 类型映射文件 |
| `--force-source` |  | `potential` | 力来源：potential 或 trajectory |
| `--root-representation` |  | `sqrt` | 矩阵参数化：normal、sqrt、root4 |
| `--initial-stabilization` |  | `spectral-floor` | 初始稳定化策略 |
| `--lammps-fc-source` |  | `fitted` | 稀疏导出的力常数来源：fitted 或 initial |
| `--output-prefix` |  | `TI_FC` | 输出文件前缀 |
| `--start` |  | `1` | 首帧（从 1 开始） |
| `--end` |  | `2147483647` | 末帧（包含） |
| `--stride` |  | `1` | 轨迹取帧间隔 |
| `--temperature` | `-T` | `300.0` | 采样温度（K） |
| `--ensemble-size` |  | `128` | 每轮自洽采样数 |
| `--max-sampling-attempts` |  | `0` | 候选采样预算；0 为目标数量的 20 倍 |
| `--seed` |  | `5489` | 随机种子 |
| `--initial-floor-thz` |  | `1.0` | 初始及迭代采样的物理频率下限（THz） |
| `--pd-floor-thz` |  | `1e-4` | 拟合矩阵的数值正定频率下限（THz） |
| `--ridge` |  | `1e-8` | 向当前参考态的正则化强度 |
| `--mixing` |  | `0.25` | 矩阵参数空间中的更新比例 |
| `--warmup-iterations` |  | `5` | 仅模态更新的预热轮数 |
| `--min-iterations` |  | `3` | 最少自洽轮数 |
| `--max-iterations` |  | `20` | 最多自洽轮数 |
| `--consecutive-converged` |  | `2` | 要求连续收敛的轮数 |
| `--relative-tolerance` |  | `1e-3` | 动力学矩阵相对变化容差 |
| `--frequency-tolerance-thz` |  | `0.02` | 最大频率变化容差（THz） |
| `--soft-block-frequency-thz` |  | `2.0` | 软模耦合块频率截断（THz） |
| `--soft-block-max-modes` |  | `32` | 软模耦合块最大模数 |
| `--degeneracy-gap-thz` |  | `0.05` | 近简并频率间隔（THz） |
| `--degeneracy-block-max-modes` |  | `8` | 近简并块最大模数 |
| `--max-displacement` |  | `1.0` | 采样最大位移（Angstrom） |
| `--min-distance-ratio` |  | `0.65` | 采样最短距离与参考最短距离之比 |
| `--symprec` | `-tol` | `1e-5` | 空间群识别容差 |
| `--hessian-method` |  | `analytic` | Hessian 后端 |
| `--hessian-nudge-pos` |  | `0.001` | 有限差分位移 |
| `--hessian-cutoff` |  | `0.0` | 有限差分截断 |
| `--lammps-fc-cutoff` |  | `0.0` | LAMMPS 力常数距离截断（Angstrom）；0 关闭 |
| `--lammps-fc-block-threshold` |  | `0.0` | 非对角力常数块范数阈值 |
| `--lammps-fc-min-frequency-thz` |  | `-1.0` | 稀疏参考态最低频率；负值使用 pd-floor-thz |
| `--lammps-fc-stabilize` |  | `true` | 必要时添加稀疏各向同性稳定项 |
| `--write-validation-ensemble` |  | `true` | 写出使用真实势函数计算力的验证 ensemble |

返回码：0 表示成功，2 表示自洽迭代未收敛（仍写出结果和诊断）。

### `phonopyfc` — Phonopy FORCE_SETS *（别名：`mlpphdisp`）*

从 Phonopy 位移输入生成 FORCE_SETS 文件。

```bash
imrmlp mlpphdisp -pot potential.mtp -d phonopy_disp.yaml
imrmlp phonopyfc -pot potential.mtp --disp=phonopy_disp.yaml
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--potfile` | `-pot` | `potential.mtp` | 势文件（位置参数） |
| `--disp` | `-d` | `phonopy_disp.yaml` | Phonopy 位移 YAML（位置参数） |
| `--index-file` | `-id` | `index.map` | 物种索引映射 |
| `--settings-file` | `-i` | `settings.ini` | 设置文件；势配置从 `pot:` 读取 |

### `calc_ferr` — 自由能误差估计

估计热力学积分的自由能误差。两组数据必须帧数相同、原子数相同（非零）且都带能量标签。

```bash
imrmlp calc_ferr --input=dft.cfg --prediction=mlp.cfg --temperature=300
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--input` | `-in` | `dft.cfg` | 参考数据集 |
| `--prediction` | | `mlp.cfg` | 预测数据集 |
| `--temperature` | `-t` | `300` | 温度（K），必须为正 |
| `--input-format` | `-if` | auto | 两组数据的格式 |
| `--index` | `-id` | `index.map` | 物种索引映射 |

### `calc_trjavg` — 轨迹平均

从轨迹数据计算平均结构。

```bash
imrmlp calc_trjavg trajectory.cfg average.cfg
imrmlp calc_trjavg trajectory.cfg average.cfg --start=100 --end=500 --wrap
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--input-file` | | | 输入轨迹（位置参数） |
| `--output-file` | | `POSCAR_IDEAL` | 输出平均结构（位置参数） |
| `--start` | `-st` | `1` | 起始帧（从 1 开始） |
| `--end` | `-ed` | 末帧 | 结束帧（从 1 开始，含该帧） |
| `--input-format` | `-if` | auto | 输入格式 |
| `--output-format` | `-of` | auto | 输出格式 |
| `--index` | `-id` | `index.map` | 物种索引映射 |
| `--loadrate` | `-lr` | `0` | 每 N 个结构读入一次；`1` 启用逐帧顺序模式 |
| `--wrap` | `-w` | `true` | 将平均结构折回晶胞 |
| `--average-coordinates` | | `auto` | `auto`、`cartesian` 或 `fractional` |
| `--drift` | | `auto` | `auto`、`remove` 或 `keep` 质心漂移 |
| `--append` | `-a` | `false` | 追加到已有输出文件 |
| `--overwrite` | `-ov` | `false` | 覆盖已有输出文件 |

轨迹平均与 `calc_prop --avgpos` 共用流式内核。`calc_trjavg` 固定胞时去除质量加权
质心漂移，变胞时平均展开后的分数坐标；`--avgpos` 平均展开后的笛卡尔坐标，不去漂移。
质量读取 `index.map` 第三列，缺省使用元素质量；输出平均几何，不沿用原帧 E/F/S 标签。

可用 `calc_trjavg --average-coordinates=auto|cartesian|fractional` 和
`--drift=auto|remove|keep` 显式选择平均策略，去漂移要求固定晶胞。
`calc_prop --avgpos-fractional` 选择分数坐标平均，`--avgpos-remove-drift`
去除质心漂移；两项默认均关闭。仅当每帧都有完整位点能量时输出其平均值。

### `findint` — 寻找间隙位点

使用 Voronoi 细分寻找晶体结构中的间隙位点。

```bash
imrmlp findint --input=POSCAR --output=sites.cfg --insert=Li --min-dist=1.0
imrmlp findint --input=POSCAR --insert=H --top=5 --sites-only
imrmlp findint POSCAR --near-elem=O --near-radius=2.0 --local
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--input` | | `POSCAR` | 输入结构（位置参数） |
| `--output` | `-o` | `POSCAR_INT` | 输出结构 |
| `--index-file` | `-id` | | 物种映射；原子选择请用 `--near-idx` |
| `--input-format` | `-if` | auto | 输入格式 |
| `--output-format` | `-of` | auto | 输出格式 |
| `--insert` | | | 在最佳位点插入的元素（空 = 仅列出位点） |
| `--top` | `-n` | `-1` | 按空腔尺寸保留前 N 个位点（-1 = 全部） |
| `--min-dist` | `-md` | `0.4` | 与宿主原子的最小距离（Å） |
| `--cluster-tol` | `-ct` | `0.8` | 合并重复位点的簇容差（Å） |
| `--vertices` | `-v` | `true` | 使用 Voronoi 顶点 |
| `--edges` | | `false` | 使用 Voronoi 边中点 |
| `--faces` | | `false` | 使用 Voronoi 面心 |
| `--volprec` | `-vp` | `0.5` | 最小 Voronoi 晶胞体积 |
| `--sites-only` | | `false` | 仅输出间隙位点，不含宿主原子 |
| `--no-fast-cluster` | | `false` | 使用旧版 O(N²) 候选聚类 |
| `--verbose` | `-vb` | `0` | 冗余输出级别 |
| `--near-elem` | `-ne` | | 在指定元素附近搜索，逗号分隔（如 `Fe,Cr`） |
| `--near-pos` | `-np` | | 在分数坐标 `x,y,z` 附近搜索 |
| `--near-pos-car` | `-npc` | | 在笛卡尔坐标 `x,y,z` 附近搜索 |
| `--near-idx` | `-ni` | | 在 1-based 原子索引附近搜索，逗号分隔 |
| `--near-radius` | `-nr` | `3.0` | 近邻搜索半径（Å） |
| `--local` | `-lcl` | `false` | 使用局域 Voronoi 构造 |

---

## I/O 与转换

格式转换和数据集管理命令。`convstr` 是通用格式转换器——几乎所有结构数据
都经过它。使用 `split_set` 将数据集划分为训练/验证/测试子集，去重命令用于清理冗余结构。

@note **想要去除重复？** 使用 `remove_duplicate_atoms` (`rmdups`) 去除
每个结构*内部*的重复原子（例如超胞操作后的重叠原子）。
使用 `remove_duplicate_structures` (`unique`) 从数据集中去除重复结构
（例如对称等价构型）。

### `convstr` — 转换结构

在所有支持的文件格式之间进行转换。支持全面的过滤和选择。

```bash
imrmlp convstr input.cfg output.extxyz
imrmlp convstr POSCAR output.lammps
imrmlp convstr trajectory.lammpstrj output.cfg
```

NetCDF（`.nc`）不在支持范围内。显式给出 `--input-format nc` 时读不到任何结构，
报 `no structures were read`；不给格式时 `.nc` 无法被识别。两种情况都不会写出或
覆盖目标文件。需要处理 NetCDF 数据时，先用其他工具导出为 extxyz、cfg 或 POSCAR。

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|--------|------|
| `--inputfiles` | `-in` | — | 输入结构文件（位置参数，必需，多文件） |
| `--outputfile` | `-out` | — | 输出文件（位置参数，必需） |
| `--input-format` | `-if` | 自动检测 | 输入格式 |
| `--detect-format` | | `false` | 省略 `--input-format` 时根据内容检测输入格式（布尔） |
| `--input-unit-system` | | — | 显式指定 EXTXYZ 源单位：`canonical`、`metal`、`atomic` 或 `real` |
| `--output-format` | `-of` | 自动检测 | 输出格式 |
| `--index` | `-id` | `index.map` | 索引文件 |
| `--wrap` | `-w` | `false` | 坐标环绕到周期性边界内（布尔值） |
| `--resetid` | `-rid` | `true` | 原子类型 ID 重置为 0,1,2,... |
| `--start` | `-st` | `0` | 起始镜像索引 |
| `--end` | `-ed` | `INT_MAX` | 结束镜像索引 |
| `--outrate` | `-or` | `0`（全部） | 每 N 帧写入一次 |
| `--shuffle` | | `false` | 随机打乱镜像顺序（布尔值） |
| `--sort-by` | | — | 逗号分隔的排序键：`energy`、`fenergy`、`fitness`、`volume`、`size`、`id` 或全局属性 |
| `--sort-order` | | `asc` | 排序方向，单个值或逗号分隔 |
| `--reverse` | `-r` | `false` | 反转镜像顺序（布尔值） |
| `--last` | `-last` | `false` | 仅输出最后一帧（布尔值） |
| `--drop` | `-d` | `false` | 跳过未收敛的 OUTCAR 帧（布尔值） |
| `--overwrite` | `-ov` | `false` | 覆盖现有文件（布尔值） |
| `--append` | `-a` | `false` | 追加到现有文件（布尔值） |
| `--select` | `-s` | — | 选择特定镜像索引（可变参数） |
| `--loadrate` | `-lr` | `0` | 每 N 个结构读取镜像到内存 |
| `--ostart` | `-ost` | `0` | 输出起始镜像索引 |
| `--oend` | `-oed` | `INT_MAX` | 输出结束镜像索引 |
| `--bstart` | `-b` | `true` | 从零开始 ID（布尔值） |
| `--outsuffix` | `-suff` | — | 附加到输出文件名的后缀 |
| `--separator` | `-sep` | — | 文件名与 ID 之间的分隔符 |
| `--magnetic` | `-mag` | `false` | 读取磁矩（布尔值） |
| `--magnetic-non-colinear` | `-mncl` | `false` | 假设非共线磁性（布尔值） |
| `--discard` | | — | 丢弃特定属性（可变参数） |
| `--rextent` | | `5.0` | 非 PBC 结构的晶胞扩展 |
| `--force-one` | | `false` | 全部写入单个文件（布尔值） |
| `--force-tilt` | | `false` | 即使正交也强制倾斜晶胞格式（布尔值） |
| `--write-symbol` | | `true` | 输出中写入原子符号（布尔值） |
| `--pack-atoms` | | `true` | POSCAR 输出中合并原子类型 |
| `--apply-disp` | `-ad` | `false` | 对镜像施加随机位移（布尔值） |
| `--rand-type` | `-rt` | `0` | 随机位移类型 |
| `--disp-mag` | `-dm` | `0.05` | 位移幅度 |
| `--label` | `-lb` | | 添加自定义逐结构标签（可变参数） |
| `--value` | `-lv` | | 自定义逐结构标签值（可变参数） |
| `--label-expr` | | | 标量元数据赋值，例如 `energy_per_atom=energy/size`（可变参数） |

### `split_set` — 分割数据集

将结构数据集按比例分割，或等分为 N 个子集。

```bash
imrmlp split_set dataset.cfg split_ --ratios=8,1,1 --shuffle yes
# 生成约 80% / 10% / 10% 的子集
imrmlp split_set --input=dataset.cfg --output-prefix=split_ --nsplit=2 --shuffle yes
# 生成 split_0.xyz, split_1.xyz（输出一律为 extxyz 格式）
```

| 选项 | 别名 | 描述 |
|--------|-------|-------------|
| `--nsplit` | `-n` | 等分为 N 个子集（可选，默认 10） |
| `--ratios` | | 逗号分隔的非负相对权重；至少两项且总和为正；决定子集数量 |
| `--shuffle` | `-s` | 分割前随机打乱 |
| `--input` | | 输入结构集（必需） |
| `--output-prefix` | | 输出文件前缀（必需） |

比例模式以最大余数法取整子集大小（余数相同时优先前面的子集），然后连续分段；
零大小子集会生成空输出文件。等分模式按轮转分配，空子集直接跳过。
若同时显式提供 `--nsplit` 和 `--ratios`，前者必须等于比例项数。
没有随机种子选项，复现时应保存实际划分结果。

### `remove_duplicate_atoms` *（别名：`rmdups`）*

基于距离标准移除重复原子。

```bash
imrmlp rmdups --input=overlap.cfg --output=cleaned.cfg --rtol=0.5
imrmlp rmdups --input=overlap.cfg --output=cleaned.cfg --rtol=0.5 --mode=merge
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--input` | `-in` | `POSCAR` | 输入结构（位置参数） |
| `--output` | `-o` | `POSCAR_REMOVED_DUPS` | 输出结构（位置参数） |
| `--rtol` | `-rt` | `0.5` | 重复原子距离容差（Å），必须为正 |
| `--mode` | `-m` | `delete` | `delete` 或 `merge` |
| `--input-format` | `-if` | `POSCAR` | 输入格式 |
| `--output-format` | `-of` | `POSCAR` | 输出格式 |
| `--index` | `-id` | `index.map` | 物种索引映射 |

`--mode=delete` 保留最接近重复组平均位置的代表原子并删除其余原子；
`--mode=merge` 会先将代表原子移动到平均位置，再删除其余原子。

### `remove_duplicate_structures` *（别名：`unique`）*

基于对称性等价移除重复结构。

```bash
imrmlp unique --input=dataset.cfg --output=unique_set.cfg --sym-prec=1e-3
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--input` | `-in` | | 输入数据集（位置参数） |
| `--output` | `-out` | `output-dedup.xyz` | 输出数据集（位置参数） |
| `--sym-prec` | `-sprec` | `1e-4` | 对称性容差（晶胞与空间群） |
| `--vol-prec` | `-vprec` | `1e-3` | 体积比较容差 |
| `--pos-tol` | `-ptol` | `0.1` | 原子位置匹配容差（Å） |
| `--preserve` | `-p` | `true` | 保留原晶胞而非标准化晶胞 |
| `--allow-reflections` | | `true` | 将镜像视为等价 |
| `--fastalgo` | | `true` | 使用缓存描述符快速比较 |
| `--desc-factor` | | `1.0` | `--fastalgo` 的描述符容差倍数 |
| `--index` | `-id` | `index.map` | 物种索引映射 |
| `--input-format` | `-if` | auto | 输入格式 |
| `--output-format` | `-of` | `xyz` | 输出格式 |
| `--overwrite` | `-ov` | `false` | 覆盖已有输出文件 |

数据集旁会写出节点索引文件 `output-nodup.id`。

---

## 势工具

MTP 势文件维护和转换命令。处理格式转换（转为 LAMMPS 样条表）、物种重排序、
系数重映射和基组检查——全部不改变训练好的模型的预测结果。

### `mtp2spline` — 将 MTP 转换为样条

将 MTP 类型势转换为样条表示。输出径向函数
的样条系数，用于 LAMMPS 的 `pair_style mlip spline`。

```bash
imrmlp mtp2spline potential.mtp spline_output
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--input` | | `potential.mtp` | 输入势（位置参数） |
| `--output` | | `splinedpot.mtp` | 输出样条势（位置参数） |
| `--delta-r` | `-dr` | `0.01` | 样条间隔 |
| `--r-min` | `-min` | `-1` | 低于该半径保持常数；负值使用每对元素径向基的最小半径 |

`--r-min R` 将 `R` 以下的径向表值设为常数。默认值 `-1` 表示使用每对元素
各自径向基的最小半径，也支持分元素对径向基；各元素对截断半径之外的表值为零。

### `pot2table` — 导出表格化势

在径向/角度网格上采样支持的势函数，导出 pair 和/或 three-body 相互作用表。
也可以写出 LAMMPS table 格式文件。

```bash
imrmlp pot2table --potfile=my_pot.sw --pottype=sw --pair --pair-output=pair.table
imrmlp pot2table --potfile=my_pot.sw --pottype=sw --threebody --threebody-output=tb.table
imrmlp pot2table --potfile=my_pot.tersoff --pottype=tersoff --pair --lammps
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--potfile` | `-pot` | | 势文件路径（必需） |
| `--pottype` | `-t` | | 势类型，如 `lj612`、`sw`、`tersoff`、`eampoly` |
| `--index` | `-id` | `index.map` | 索引文件 |
| `--pair-output` | `-po` | `pair_table.out` | PairTable 输出文件 |
| `--threebody-output` | `-to` | | ThreebodyTable 输出文件 |
| `--lammps-prefix` | `-lp` | | LAMMPS table 文件前缀 |
| `--nr` | `-nr` | `1000` | 径向网格点数 |
| `--ntheta` | `-nt` | `181` | 角度网格点数 |
| `--rmin` | `-rmin` | `0.01` | 最小半径 |
| `--rmax` | `-rmax` | 势截断半径 | 最大半径覆盖值 |
| `--pair` | | `false` | 导出 pair table（布尔） |
| `--threebody` | | `false` | 导出 three-body table（布尔） |
| `--lammps` | `-lm` | `false` | 同时写出 LAMMPS table 格式（布尔） |
| `--settings` | | | 势加载设置文件 |

Pair 表包含恰好 `N` 个均匀网格点，包括两个端点。导出 pair 表时 `--nr`
至少为 3；导出 three-body 表时 `--nr` 和 `--ntheta` 至少为 2。
半径必须为有限数且满足 `0 < rmin < rmax`。LAMMPS pair 表包含径向力 `-dE/dr`。

三体表格式只有一个径向维度，采样时两条键等长。对于不等长的键，
`ThreebodyTable` 使用
`V(rij, rik, theta) = (T(rij, theta) + T(rik, theta)) / 2`，并从该能量计算力。
这一近似无法表示具有两个独立径向维度的一般三体势。

### `adjustmtp` — 重排 MTP 势

重排 MTP 势的物种顺序以匹配不同的排序约定。
当训练时使用的 index.map 与生产中使用的不同时很有用——
此命令在不重新训练的情况下置换系数布局。

```bash
imrmlp adjustmtp --input=original.mtp --output=reordered.mtp --neworder 0 2 1
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--input` | `-in` | `potential.mtp` | 输入势（位置参数） |
| `--output` | `-out` | `output-adj.mtp` | 输出势（位置参数） |
| `--neworder` | | | 新物种顺序（位置参数，可多值）；顺序变长时追加随机初始化物种 |
| `--elements` | | | 输出物种顺序对应的元素符号；`--neworder` 追加物种时必需 |

### `mtpbasis` — 比较 MTP 基组并映射参数

使用精确整数符号运算判断两个经典线性 MTP 基组是否等价，允许 moment
编号、标量顺序及缩并路径不同。`auto` 先寻找逐项标量排列，再通过精确有理
消元判断两组标量是否张成相同空间。生产时生成的 `mid_tid_mapping` 文件是
可选的快速路径提示，其结果仍会由符号表达式复核。

```bash
# 只读判等，同时输出 JSON 报告
imrmlp mtpbasis --source=pot1.mtp --target=pot2.mtp \
  --equivalence=auto --report=basis-report.json

# 复核生产映射并迁移经典线性 MTP 的全部参数
imrmlp mtpbasis --source=pot1.mtp --target=pot2.mtp \
  --source-map=map1.txt --target-map=map2.txt \
  --source-index=index1.map --target-index=index2.map \
  --output=pot2-remapped.mtp
```

| 选项 | 别名 | 描述 |
|--------|-------|-------------|
| `--source` | `-s` | 参数来源 MTP 基组或势文件（必需） |
| `--target` | `-t` | 提供输出拓扑的 MTP 基组或势文件（必需） |
| `--source-map`, `--target-map` | `-sm`, `-tm` | 可选且必须成对提供的 `moment_id tensor_id` 文件 |
| `--source-index`, `--target-index` | `-si`, `-ti` | 可选且必须成对提供的物种顺序文件 |
| `--equivalence` | `-eq` | `auto`（默认）、`item` 或 `span` |
| `--output` | `-o` | 用目标拓扑写出映射参数；省略时仅判等 |
| `--report` | | 将秩、诊断、排列/变换和兼容性写为 JSON |
| `--overwrite` | `-ov` | 允许替换已有输出或报告 |

只有拓扑、径向布局以及已提供的物种映射全部兼容时才允许写出，而且输出
路径不能是任一输入。工具复制径向与物种参数，并用精确标量变换迁移全部
moment 参数。MTP_M、NN 和 LES 附加参数不会被静默丢弃，也不在本命令的
映射范围内。退出码 0 表示等价，1 表示输入合法但不等价，2 表示输入、映射
或写出错误。



### `mtmbasis` — 生成矩张量基组

从基组到普通 MTP 拟合、验证和后续计算的可运行流程见
[MTP 连续练习](../../examples/mtp-workflow/README_zh.md)。

生成矩张量基组：标量不变量、中间矩张量以及缩并规则。该命令读取设置文件中的
顶层键（如 `max_lev`、`max_dim`、`max_mu`、`max_nv` 和 GA 搜索控制参数），
而不是常规的 `[pot]`/`pot:` 势加载配置段。

```bash
imrmlp mtmbasis --settings=mtmbasis.ini
imrmlp mtmbasis --settings=mtmbasis.ini --generate-only
imrmlp mtmbasis --settings=mtmbasis.ini --alphapool=alphapool.dat
```

普通生成流程会先构建 alpha 池，然后用配置的搜索模式寻找高效的缩并树布局，
并将候选 MTP 基组模板写为 `potential.mtp_<iteration>`。命令还会写出
`conns_exp.log`，用于记录缩并计数诊断信息。生成的 MTP 文件包含
`alpha_index_basic`、`alpha_index_times` 和 `alpha_moment_mapping` 等基组元数据，
但没有训练后的系数；先补全所选模板的物种和径向设置，再用 `trainmtp` 拟合。
`initmtpm` 面向 MTP-M，不是普通 MTP 的通用初始化工具。

仅处理池的模式适合把耗时的基组枚举拆分出来：

- `--generate-only` 写出去重后的 alpha 池到 `alphapool.dat`，随后退出。
- `--sample` 按 `--sample-rate` 对生成或读取的池随机采样，并写出 `sampled_alphas.dat`。
- `--advanced-sample` 使用顶层 `sample_rates` 设置，保留所有单矩标量项，
  对更高张量维度分别采样，写出 `sampled_alphas.dat` 后退出。

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--settings` | | `settings.ini` | 设置文件。此命令没有 `-i` 别名。 |
| `--alphapool` | `-pool` | | 读取已有 alpha 池，而不是根据限制重新生成。 |
| `--moments` | `-mts` | | 限定可用 `(mu, nu)` 矩张量对的文件：第一个值是数量，之后是相同数量的 `mu nu` 对。 |
| `--generate-only` | `-g` | false | 生成并保存 `alphapool.dat`，在 GA 搜索前退出。 |
| `--sample` | `-s` | false | 对当前池采样并保存 `sampled_alphas.dat`。 |
| `--sample-rate` | `-rate` | 1.0 | `--sample` 的随机采样比例；超出 `[0,1]` 的值会被重置为 1。 |
| `--advanced-sample` | `-as` | false | 由 `sample_rates` 控制的按维度采样。 |
| `--potfile` | `-pot` | `potential.mtp` | 候选基组的输出前缀，每个被接受的候选写成 `<前缀>_<编号>`。 |
| `--convert` | | false | 旧格式转换路径；实现中标记为“not working”。 |

---

## 实用命令

### `attach_hessian` — 附加 Hessian 或 HVP 标签 *（别名：`hessian2xyz`）*

将 DFPT（`vasprun.xml`）或有限差分（`FORCE_CONSTANTS`）Hessian 附加到
extxyz 结构上。指定 `--hvp-probes` 时同时写入随机 Hessian-向量标签，
可用于 HVP 监督训练。

```bash
imrmlp attach_hessian --input POSCAR --hessian vasprun.xml --output hessian.xyz --source dfpt
imrmlp hessian2xyz --input POSCAR --hessian FORCE_CONSTANTS --output hvp.xyz --hvp-probes 8
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--input` | `-in` | `POSCAR` | 输入结构（位置参数） |
| `--output` | `-out` | `hessian.xyz` | 输出 extxyz 数据集（位置参数） |
| `--hessian` | | 输入文件 | DFPT `vasprun.xml` 或 FD `FORCE_CONSTANTS` |
| `--source` | `-s` | `auto` | Hessian 来源：`auto`、`dfpt` 或 `fd` |
| `--vasprun-key` | | `auto` | `vasprun.xml` 中的 `varray` 名称 |
| `--frame` | `-f` | `-1` | 结构帧；-1 使用最后一帧 |
| `--scale` | | `1.0` | 写出前对 Hessian 乘以该因子 |
| `--symmetrize` | | `false` | 用 `0.5*(H+H^T)` 替换 `H` |
| `--hvp-probes` | | `0` | 随机 Hessian-向量标签对数 |
| `--hvp-seed` | | `42` | HVP 方向随机种子 |
| `--hvp-only` | | `false` | 仅保存 HVP 标签，省略稠密 Hessian |
| `--input-format` | `-if` | auto | 输入格式 |
| `--output-format` | `-of` | `extxyz` | 输出格式 |

`--hvp-only` 要求 `--hvp-probes > 0`；HVP 探针要求至少两个原子。

### `calc_err` — 误差计算 *（别名：`err`）*

比较势函数预测与参考数据。计算能量、力和应力的 RMSE 和 MAE。

```bash
imrmlp err -pot potential.mtp -in reference.cfg
imrmlp calc_err -pot potential.mtp -in reference.cfg -inprd predicted.cfg -out summary.txt
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--potfile` | `-pot` | | 用于生成预测的势，也可在 `pot:potfile` 中设置；提供 `--input-predict` 时无需此项 |
| `--input` | `-in` | `valid.cfg` | 参考数据文件（必需） |
| `--input-format` | `-if` | auto | 输入格式 |
| `--input-predict` | `-inprd` | | 用于比较的预测文件 |
| `--index` | `-id` | `index.map` | 索引文件 |
| `--pottype` | `-pt` | `mtp` | 势类型 |
| `--output-energy-diff` | `-oe` | | 输出能量差异文件 |
| `--output-energy-at-diff` | `-oeat` | | 输出逐原子能量差异文件 |
| `--output-force-diff` | `-ofce` | | 输出力差异文件 |
| `--output-stress-diff` | `-ofst` | | 输出应力差异文件 |
| `--output-virial-diff` | `-ofvr` | | 输出维里差异文件 |
| `--output-summary` | `-out` | | 输出误差摘要文件（默认：stdout） |
| `--show-progress` | `-sp` | `false` | 显示已完成构型数量（布尔值） |
| `--mpi-mode` | | `replicated` | MPI 求值模式：`replicated` 或 `frames` |
| `--percentile` | `-q` | `0` | 报告分位比例（如 `0.95`；0 关闭） |
| `--settings` | | | 可选设置文件（位置参数） |

仅当配对帧都含有某项物理量时，该项才计入对应误差。参考集与预测集的帧数、
各帧原子数和元素顺序必须一致。`--input-predict` 直接比较已保存的预测值，
不计算势函数；两份文件共用 `--input-format`。能量单位为 eV 和 eV/atom，
力为 eV/Å，应力为 GPa，virial 为 eV。力矢量 RMSE 与笛卡尔分量 RMSE
归一化不同，跨计算比较时应选择同一种报告指标。

### `smooth_traj` — 平滑轨迹

对轨迹应用基于 FFT 的平滑变换并写出平滑后的轨迹。

```bash
imrmlp smooth_traj --input=trajectory.cfg --output=smoothed.cfg --timestep=0.1
```

| 选项 | 别名 | 默认值 | 描述 |
|------|------|--------|------|
| `--input` | `-i` | | 输入轨迹文件 |
| `--output` | `-o` | | 输出文件 |
| `--index` | `-id` | `index.map` | index 文件 |
| `--input-format` | `-if` | 自动检测 | 输入文件格式 |
| `--output-format` | `-of` | 自动检测 | 输出文件格式 |
| `--overwrite` | `-ov` | `false` | 覆盖已有输出文件 |
| `--timestep` | `-ts` | `0.1` | 轨迹时间步长，单位 fs |
| `--freq-cutoff` | `-fq` | `1` | 平滑截止频率，单位 fs⁻¹ |
| `--wtype` | `-w` | `ComputeSmoothFFT::SUPERGAUSSIAN` | FFT 窗口类型 |
| `--output-rate` | `-or` | `1` | 每隔多少帧输出一次 |
| `--interpolation-factor` | | `1` | 正整数 Fourier 插值因子 |
| `--scale-cell` | `-sc` | `false` | 平滑前将所有帧缩放到平均晶胞 |
| `--settings` | | `settings.ini` | 设置文件 |

平滑操作至少需要两帧。`--freq-cutoff` 和两个配置项 `freq-cutoff`、`freq_cutoff`
统一使用 fs⁻¹，例如 `0.005` 对应周期 200 fs。显式命令行值优先于这两个配置项，
未指定命令行值时，`freq-cutoff` 优先。

---

## 驻点与鞍点搜索

### `nroptimize` — Newton–Raphson 驻点精修

使用 PotMTP 解析 Hessian 精修驻点。固定晶胞时使用 `--no-cell`；
退出码 2 表示迭代预算耗尽。[用法与验证](../advanced/saddle-workflows_zh.md)。

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--input` | `-in` | `POSCAR` | 输入结构（位置参数） |
| `--output` | `-o` | `CONTCAR` | 输出结构（位置参数） |
| `--potfile` | `-pot` | `potential.mtp` | 势文件（位置参数） |
| `--index` | `-id` | `index.map` | 物种索引映射 |
| `--no-cell` | `-nocell` | `false` | 固定晶胞 |
| `--pressure` | `-p` | `0` | 外部静水压（GPa） |
| `--external-stress` | `-str` | | 外部应力张量（GPa） |
| `--gtol` | `-g` | `0.01` | 梯度容差（固定晶胞原子坐标为 eV/Å；变胞为混合单位） |
| `--max-iteration` | `-maxit` | `100` | 迭代上限 |
| `--alpha` | `-a` | `0.1` | 初始步长尺度 |
| `--alpha-reduce` | `-ar` | `0.9` | 回溯时的步长缩减因子 |
| `--max-backtracks` | `-mbt` | `10` | 最大回溯步数 |
| `--fixed-atoms` | `-fix` | | 固定原子索引，0-based（可多值） |
| `--eigzero-tol` | `-eigtol` | | 零特征值容差 |

### `calc_dimer` — Dimer 鞍点搜索 *（别名：`dimer`）*

Dimer 鞍点搜索。显式 MODECAR 输入按严格格式解析；省略 MODECAR 时
初始方向使用 `dimer:random_seed`。其余控制项位于 `dimer:` 设置命名空间。
[参数、输出与限制](../advanced/saddle-workflows_zh.md)。

```bash
imrmlp dimer dimer.ini
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--settings` | | | Dimer 设置文件（位置参数，必需） |
| `--structure` | `-in` | | 初始猜测结构 |
| `--input-format` | `-if` | auto | 输入结构格式 |
| `--index` | `-id` | `index.map` | 物种索引映射 |
| `--output` | `-out` | `dimer_final.vasp` | 收敛鞍点结构 |

模式跟随同样支持 `--mode=9 --modecar-out=MODECAR`；模式 2 接受
`--max-loop`，搜索失败时返回退出码 2。

---

## GPM 模型打包

`model_convert` 写出自包含的 GPM 文件；`model_info` 与 `model_validate`
在不读取数值块的情况下检查该文件。参见
[GPM 格式、保存模式和主动学习恢复](gpm-v1_zh.md)。

### `model_convert` — 写出 GPM 模型包

```bash
imrmlp model_convert --input old.mtp --type mtp --output model.gpm
imrmlp model_convert --settings with-al.ini --output resume.gpm --profile resume
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--input` | `-in` | | 旧格式模型或 GPM 输入 |
| `--output` | `-out` | | 输出 GPM 文件（必需） |
| `--settings` | | | 已有势配置及可选的 `al:` 配置 |
| `--type` | | `auto` | 旧格式势类型；GPM 会推断类型 |
| `--index` | | | 显式有序元素映射 |
| `--profile` | | `preserve` | `preserve`、`inference` 或 `resume` |

### `model_info` — 查看 GPM 元数据

```bash
imrmlp model_info --input model.gpm
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--input` | `-in` | | GPM 模型（必需） |

### `model_validate` — 校验 GPM 模型

```bash
imrmlp model_validate --input resume.gpm
```

| 选项 | 别名 | 默认值 | 描述 |
|--------|-------|---------|-------------|
| `--input` | `-in` | | GPM 模型（必需） |

---

## 参见

- @ref guide_workflows — 端到端工作流示例
- @ref guide_config — 配置文件格式
- @ref guide_install — 构建与安装
- @ref reference_potential_types — 势类型总览与文件格式
- @ref apps — 所有命令实现
- @ref cheatsheet — 命令速查表
