# 结构弛豫 {#guide_relaxation}

[English](relaxation.md)

弛豫用已有势函数优化原子位置或晶胞，在给定约束下寻找局部极小值。
它不训练模型，也不保证得到全局最低能结构。

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

| 目标 | 原子位置 | 晶胞 | 建议配置 |
| --- | --- | --- | --- |
| 固定晶胞、固定体积下的缺陷或原子优化 | 放开 | 固定 | 固定晶胞弛豫（默认） |
| 体相在零压或指定压力下的平衡晶胞 | 放开 | 放开 | 变胞弛豫 + 可选 `relax:pressure` |
| 只优化晶格、保留分数坐标 | 固定 | 放开 | 关闭原子坐标更新后启用变胞 |
| 表面固定底层或受约束的局部优化 | 部分放开 | 按物理问题决定 | 选择性动力学或约束；先确认哪些分量被固定 |

### 固定晶胞弛豫

把以下内容保存为 `relax.ini`。输入为 POSCAR，输出为保留 E/F/应力的 EXTXYZ；
默认固定晶胞、优化原子，使用 L-BFGS，无需额外写出这些开关。

```text
input POSCAR
output relaxed.xyz
index index.map
pot:potfile model.mtp
relax:cartesian true
relax:gtol 0.001
relax:maxiteration 500
relax:stepmax 0.05
relax:write_traj true
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
应比较最终能量和梯度。不要同时照抄 `isif` 和一组自由度标志：顶层 `isif` 会覆盖
`relax_atom`、`relax_cell` 等相关设置。

## 输出与判据

| 检查项 | 含义 |
| --- | --- |
| 退出码 `0` | 所有输入结构达到弛豫收敛条件 |
| 退出码 `2` | 至少一个结构正常停止但未收敛；输出仍可能包含最后一个接受的结构 |
| 退出码 `1` | 输入、求值或写出失败，查看 `relax.log` 中的具体错误 |
| `relaxed.xyz` | 查看最终结构、元素、能量、力；有文件不等于收敛 |
| `relax-trajectory.xyz` | 检查是否发生原子重叠、异常跳跃或不合理变形 |

能量变化很小、步长很小或达到最大迭代数，都不能替代力/梯度收敛。有约束时，只对
允许的自由度判断收敛，冻结方向仍可能有力。需要后续 POSCAR 文件时转换最终结果：

```sh
imrmlp convstr relaxed.xyz relaxed.vasp --index index.map
```

### 复核最终受力

用相同模型对最终几何再做单点计算：

```sh
imrmlp calc_efs relaxed.xyz checked.xyz --potfile model.mtp --index index.map
```

查看 `checked.xyz` 的 `forces` 列。对无约束固定胞设置，最大绝对力分量为
`0.0008 eV/Å` 时满足 `gtol=0.001`，为 `0.0046 eV/Å` 时不满足。这里比较力分量，
不是每个原子的三维力模长。若日志称收敛而复核值明显不符，应先核对模型、输入是否为
最终帧，以及实际生效的约束和设置。需要更紧精度时，在副本中把输入改为
`relaxed.xyz`，替换 `gtol` 并更换输出文件名，再比较最终能量、几何和受力是否稳定。

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
- [输入文件与配置约定](conventions_zh.md)、[命令参考](../reference/commands_zh.md)
