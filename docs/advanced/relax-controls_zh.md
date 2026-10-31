# 指南之外的弛豫控制

[English](relax-controls.md)

[结构弛豫](../guides/relaxation_zh.md)一页覆盖日常计算需要的部分：原子与晶胞自由度、
`isif`、选择性动力学、对称性和优化器选择。本页把其余功能集中起来，让指南保持简短。
这里只列控制项及其用途；默认值、单位与完整键表见[配置参考](../reference/configuration_zh.md)，
对应的命令行开关见[命令参考](../reference/commands_zh.md) 的 `relax` 一节。
`imrmlp relax --create-input` 会写出一份带注释的示例配置。

## 部分放开的原子与晶胞

非对称晶胞、表面和固定基底，靠逐个掩码自由度来处理。

| 目标 | 控制项 |
| --- | --- |
| 只放开部分应变分量 | `relax:strain_mask` 或 `relax:lattice_mask`：对称应变 6 个值，原始晶格 9 个行主序值 |
| 同样的掩码，走命令行 | `--cell-mask` |
| 所有原子放开相同的三个方向 | `--position-mask`，三个 0/1 标志 |
| 为每个原子和晶胞坐标单独给标志 | `--filter`，或在设置文件中用 `relax:seldyn_file` |
| 输入格式没有 POSCAR 标志时的选择性动力学 | `--seldyn` |
| 变胞时保持晶格矢量耦合 | `relax:couple`：`1` a-b，`2` b-c，`3` c-a，`4` 全耦合 |
| 只变体积、形状固定 | `relax:fix_shape`；`relax:fix_vol` 则固定体积、放开形状 |
| 含真空的表面 | `relax:periodic_subspace`，周期方向按分数坐标变胞，开放方向保持笛卡尔 |
| 固定晶胞下的刚性片段 | `relax:rigid_groups`，以分号分隔若干组从 0 开始的原子序号 |

这些掩码互相校验，不会悄悄叠加：应变掩码要求开启变胞且不能同时用 `fix_shape`；
`--filter` 拒绝与 `--position-mask`、`--cell-mask`、`keep_symm` 及各类过滤修饰同时使用；
`--position-mask` 不接受 `keep_symm` 与 `isif` 5 到 7。报错信息会点明冲突的一对。

## 外加载荷

`relax:pressure` 为各向同性压力，正值为压缩。`relax:external_stress` 给出完整应力张量，
按 `xx yy zz yz xz xy` 顺序取 3 个或 6 个分量，单位 GPa。`relax:stiffness` 提供应变刚度
用于预条件，`relax:cell_factor` 按它缩放晶胞步长。

## 坐标与预条件

变胞默认使用对称应变（`relax:symm_strain`），去掉整体旋转的零空间；默认也使用
`relax:cell_strain`，更新的是应变而不是原始晶格矢量。`relax:atomic_force_recast`
让原子力与应变度规保持一致。`relax:cartesian true` 切换到笛卡尔坐标，适用于固定晶胞。

默认优化器不需要额外缩放。薄膜、层状晶胞或某些方向特别硬时可用：

- `relax:dprecond` —— 分数对称应变坐标的自动对角缩放。
- `relax:thin_cell_scaling` —— 按周期测度做晶胞预条件，避免真空层主导步长。
- `relax:precond_file`、`relax:hessian_precond_file` —— 从文件读入对角或完整 Hessian 度规。
- `relax:sparse_precond` —— 近邻图度规，适用于 `gd` 或 `fire` 的大体系固定晶胞计算。

## 约束

有三类约束，彼此不能随意混用：

- 坐标过滤（LC、LF、VIB），由 `relax:with_linear_constraint`、
  `relax:with_linear_transform`、`relax:with_constraint_vib` 及各自的 `*_file` 文件启用，
  作用在优化变量或梯度上。
- 与优化过程同时求解的约束：`relax:with_constraint_aul`，或 `relax:driver sqp` 与 `nlc`。
- `--file-constraints` 给出的几何等式集合，JSON 第二版还支持边界、壁面、公式、二面角
  和指定的初始对称操作。这类约束要求 `driver sqp` 或 `nlc`，只在单 MPI 进程运行，
  未给出目标值的项绑定到各帧的初始几何。

`relax:fixcm` 通过调整梯度去掉质心漂移；`relax:freeze_small_gd` 把低于 `relax:ztol`
的分量置零，使其退出活动集。

## 优化器与停止条件

`relax:driver` 可取 `lbfgs`（默认）、`lbfgspp`、`fire`、`cg`、`gd`、`mdmin`、`quickmin`，
以及配合上述约束族的 `sqp` 与 `nlc`。LBFGSpp 的线搜索与收敛参数在 `relax:lbfgspp:` 下。

收敛由 `relax:gtol` 判定。`relax:ftol`、`relax:xtol` 用于停止不再有进展的计算，
`relax:gtol0` 放宽增广拉格朗日前几轮的内层容差。这些都不能替代对最终结构的梯度复核，
做法见指南。

## 多帧读写

`--seq-read` 一次读入一帧，大数据集不必整体驻留内存，后续帧失败时已写出的结果仍然保留。
轨迹由 `relax:write_traj`、`relax:trajfile`、`relax:traj_append` 和写出间隔
`relax:nblock` 控制。`relax:apply_disp` 配合 `relax:disp` 从随机位移的副本出发，
可快速判断极小值是否依赖初始猜测。

## 相关

- [约束与对称性](relax-symmetry_zh.md)：`relax:keep_symm` 做什么、排除什么
- [配置参考](../reference/configuration_zh.md)：全部 `relax:` 键及默认值
- [结构弛豫](../guides/relaxation_zh.md)：常规流程
