# 结构弛豫 {#guide_relaxation}

[English](relaxation.md)

弛豫用已有势函数优化原子位置或晶胞，在给定约束下寻找局部极小值；落到哪个极小值由初始
结构决定。

不涉及：固定几何下的能量与误差（见[单点计算](single-point_zh.md)）、
初态与终态之间的过渡路径（见 [NEB](neb_zh.md)）。

## 前置输入

| 文件 | 内容 | 来源 |
| --- | --- | --- |
| `model.mtp` | 已训练的普通 MTP 模型 | 训练输出（如 `fitted.mtp`）或模型提供方 |
| `POSCAR` | 初始结构，含元素符号 | 自己的建模或参考计算 |
| `index.map` | 模型元素顺序 | 模型或其训练记录 |

每个任务使用独立目录，配置文件与日志放在其中；相对路径按运行命令时的当前目录解析，
见[输入文件与配置约定](conventions_zh.md)。下面示例针对普通 MTP 和有效晶胞；
模型类型或边界条件不同时，应先核对其支持范围。

## 命令与配置

### 自由度选择

| 目标 | 原子位置 | 晶胞 | 建议配置 | `isif` |
| --- | --- | --- | --- | --- |
| 固定晶胞、固定体积下的缺陷或原子优化 | 放开 | 固定 | 固定晶胞弛豫（默认） | `2` |
| 体相在零压或指定压力下的平衡晶胞 | 放开 | 放开 | 变胞弛豫 + 可选 `relax:pressure` | `3` |
| 只优化晶格、保留分数坐标 | 固定 | 放开 | 关闭原子坐标更新后启用变胞 | `6` |
| 表面固定底层或受约束的局部优化 | 部分放开 | 按物理问题决定 | 选择性动力学或约束；先确认哪些分量被固定 | — |

`isif` 一列对应 VASP 的习惯写法，可用 `--isif` 开关或顶层 `isif` 键给出。其余取值
也与 VASP 一致：`4` 在固定体积下放开原子与晶胞，`5` 固定原子位置并固定体积，`7` 只做
等比体积缩放，`8` 在原子弛豫的同时改变体积。`isif` 在设置文件之后生效，会覆盖
`relax_atom`、`relax_cell`、`fix_vol`、`fix_shape`；两者同时出现时会打印警告，
因此同一份配置里只应保留一种写法。

### 固定晶胞弛豫

把以下内容保存为 `relax.ini`。输入为 POSCAR，输出为保留 E/F/应力的 EXTXYZ；
默认固定晶胞、优化原子，使用 L-BFGS，无需额外写出这些开关。

```text
# ===== Input/Output =====
input               POSCAR
output              relaxed.xyz
index               index.map

# ===== Potential =====
pot:potfile         model.mtp

# ===== Relaxation =====
relax:cartesian     true
relax:stepmax       0.05
relax:gtol          0.001
relax:maxiteration  500

# ===== Trajectory =====
relax:write_traj    true
```

```sh
imrmlp relax relax.ini > relax.log 2>&1
echo $?
tail -n 20 relax.log
```

`echo $?` 要紧接上一条命令运行。固定胞笛卡尔坐标下，`gtol` 检查活动梯度的最大
绝对分量；无约束时对应每个力分量低于 `0.001 eV/Å`。`maxiteration` 是迭代预算，
`stepmax` 控制优化器更新幅度，两者不是收敛精度。这些数值是示例，生产计算应做
容差检查。

### 变胞弛豫

复制上述配置，将 `relax:cartesian true` **替换**为以下两行：

```text
relax:relax_cell true
relax:cartesian false
```

这会同时优化原子和晶胞，默认目标为零压；指定外压时再加 `relax:pressure`，
单位 GPa，正值为压缩。变胞使用包含原子和晶胞的广义坐标，`gtol` 不能直接解释为
统一的 eV/Å 力阈值；除日志收敛外，还要检查最终应力/目标压力、体积和晶格参数。
表面或含真空的结构不宜直接照搬全自由度体相变胞设置，应按周期方向选择晶胞自由度。

### 约束、对称性与优化器

POSCAR 的 `Selective dynamics` 标志需配合 `relax:selective_dynamics true` 使用。
分量的含义与优化坐标有关，不能不加检查地把任意 T/F 掩码理解为实验室笛卡尔方向。
`relax:keep_symm true` 会限制在初始空间群允许的子空间中，可能排除真实的破对称畸变，
并与其他约束存在兼容性限制；规则见[约束与对称性](../advanced/relax-symmetry_zh.md)
及[配置参考](../reference/configuration_zh.md)。

`lbfgs` 可作为普通任务的起点，`fire`、`lbfgspp`、`cg` 是其他可选驱动；变更优化器后
应比较最终能量和梯度。

本页的开关覆盖日常情形；晶胞分量掩码、晶格矢量耦合、外应力张量、约束求解器与
预条件等，集中放在[弛豫的进阶控制](../advanced/relax-controls_zh.md)。

## 输出与判据

| 检查项 | 含义 |
| --- | --- |
| 退出码 `0` | 所有输入结构达到弛豫收敛条件 |
| 退出码 `2` | 至少一个结构正常停止但未收敛；输出仍可能包含最后一个接受的结构 |
| 退出码 `1` | 输入、求值或写出失败，查看 `relax.log` 中的具体错误 |
| `relaxed.xyz` | 查看最终结构、元素、能量、力；未收敛时同样会写出，是否收敛看退出码 |
| `relax-trajectory.xyz` | 检查是否发生原子重叠、异常跳跃或不合理变形 |

收敛只按力/梯度判据：`gtol` 比较允许自由度上最大绝对梯度分量，能量变化、步长和迭代次数
都不参与判定。有约束时，冻结方向仍可能有力。需要后续 POSCAR 文件时转换最终结果：

```sh
imrmlp convstr relaxed.xyz relaxed.vasp
```

复核做法：用同一模型对最终结构再算一次，把最大绝对力分量与 `gtol` 对照：

```sh
imrmlp calc_efs relaxed.xyz checked.xyz --potfile model.mtp
```

对上面无约束的固定胞设置，`0.0008 eV/Å` 满足 `gtol=0.001`，`0.0046 eV/Å` 不满足；
比较的是单个分量，不是每个原子的三维力模长。若两者明显不符，通常是模型或帧取错了，
也可能是约束仍在生效。

## 常见错误

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| 力异常大或线搜索失败 | 元素映射、单位、原子距离或模型适用范围不匹配 | 先核对上述几项，再调整 `stepmax` |
| 迭代耗尽仍不收敛 | 预算不足或初始结构/优化器不合适 | 查看轨迹是否仍在改善，再决定增加预算、换初始结构或换优化器 |
| 结构停在高对称位置 | 对称性或约束排除了目标畸变 | 检查 `relax:keep_symm` 与选择性动力学设置 |
| 多结构输入结果不一致 | 各帧停止状态不同 | 逐帧核对输出与停止状态，不能只看最后一帧 |

## 相关

- [晶体性质](properties_zh.md)：平衡结构确定后的声子、弹性与状态方程
- [NEB](neb_zh.md)：两个端点均弛豫后再计算路径
- [弛豫的进阶控制](../advanced/relax-controls_zh.md)：掩码、耦合、外应力与约束求解器
- [约束与对称性](../advanced/relax-symmetry_zh.md)：`relax:keep_symm` 的作用与限制
- [输入文件与配置约定](conventions_zh.md)、[命令参考](../reference/commands_zh.md)
