# 结构弛豫：从输入到收敛 {#guide_relaxation}

[English](relaxation.md)

弛豫用已有势函数优化原子位置或晶胞，寻找给定约束下的局部极小值。
它不会训练模型，也不保证得到全局最低能结构。先完成[单点计算](getting-started_zh.md)，
再用本页流程；需要初态与终态之间的过渡路径时，使用 [NEB](neb_zh.md)。

**还没有输入？** 先运行[连续计算练习第 1–3 步](../../examples/basic-calculations/README_zh.md)，
脚本会提供结构、势和两份弛豫配置，并把结果交给物性计算。本页正文用于自己的普通 MTP。
从基组和训练开始的完整版本见 [MTP 连续练习](../../examples/mtp-workflow/README_zh.md)。

## 1. 先决定哪些自由度可以改变

| 目标 | 原子位置 | 晶胞 | 建议起点 |
| --- | --- | --- | --- |
| 固定体积、固定晶胞下的缺陷或原子优化 | 放开 | 固定 | 使用下面的固定晶胞示例 |
| 求体相零压或指定压力下的平衡晶胞 | 放开 | 放开 | 单点与固定胞跑通后，再用变胞设置 |
| 只优化晶格、保留分数坐标 | 固定 | 放开 | 明确原子随晶胞仿射运动的约定 |
| 表面固定底层、受约束的局部优化 | 部分放开 | 按物理问题决定 | 使用选择性动力学或约束；先检查哪些分量被固定 |

准备适用的 `model.mtp`、`POSCAR` 和 `index.map`，在一个新的任务目录运行。
模型可来自训练得到的 `fitted.mtp`，结构来自自己的建模或参考计算；按实际文件名修改配置。
下例针对普通 MTP 和有效晶胞；模型类型或边界条件不同，应先核对其支持范围。

## 2. 固定晶胞，先优化原子

把以下内容保存为 `relax.ini`。输入为 POSCAR，输出为保留 E/F/应力的 EXTXYZ。
默认固定晶胞、优化原子，使用 L-BFGS；无需额外写出这些开关。

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

`echo $?` 要紧接上一条命令运行。这里使用固定胞笛卡尔坐标，`gtol` 检查活动梯度的
最大绝对分量；无约束时对应每个力分量低于 `0.001 eV/Å`。
`maxiteration` 是迭代预算，`stepmax` 控制优化器更新幅度，两者不是收敛精度。
这些数值是示例，生产计算应做容差检查。

## 3. 判断是否成功

| 检查项 | 怎样判断 |
| --- | --- |
| 退出码 `0` | 所有输入结构达到弛豫收敛条件 |
| 退出码 `2` | 至少一个结构正常停止但未收敛；输出仍可能包含最后一个接受的结构 |
| 退出码 `1` | 输入、求值或写出失败，查看 `relax.log` 中的具体错误 |
| `relaxed.xyz` | 查看最终结构、元素、能量、力；有文件不等于收敛 |
| `relax-trajectory.xyz` | 检查是否发生原子重叠、异常跳跃或不合理变形 |

能量变化很小、步长很小或达到最大迭代数，都不能替代力/梯度收敛。
有约束时，只对允许的自由度判断收敛，冻结方向仍可能有力。
需要后续 POSCAR 文件时转换最终结果：

```sh
imrmlp convstr relaxed.xyz relaxed.vasp --index index.map
```

重新做一次 `calc_efs` 可独立核对最终构型的力，尤其是在更换模型、坐标或约束之后。

### 小练习：复核最终受力

用相同模型对最终几何再做单点计算：

```sh
imrmlp calc_efs relaxed.xyz checked.xyz --potfile model.mtp --index index.map
```

查看 `checked.xyz` 的 `forces` 列。对本页的**无约束固定胞**设置，若所有原子的
最大绝对力分量为 `0.0008 eV/Å`，则满足 `gtol=0.001`；若为 `0.0046 eV/Å`，
则还不满足。这里比较力分量，不是每个原子的三维力模长。若日志称收敛而复核值
明显不符，应先核对模型、输入是否为最终帧，以及实际生效的约束和设置。
需要更紧精度时，在副本中把输入改为 `relaxed.xyz`，
替换 `gtol` 并更换输出文件名，再比较最终能量、几何和受力是否稳定。

## 4. 需要平衡晶胞时再启用变胞

复制上述配置，将 `relax:cartesian true` **替换**为以下两行：

```text
relax:relax_cell true
relax:cartesian false
```

这会同时优化原子和晶胞，默认目标为零压；指定外压时再加 `relax:pressure`，单位 GPa，正值为压缩。
变胞使用包含原子和晶胞的广义坐标；`gtol` 不能直接解释为统一的 eV/Å 力阈值。
除日志收敛外，还要检查最终应力/目标压力、体积和晶格参数。
表面或含真空的结构不宜直接照搬全自由度体相变胞设置，应按周期方向选择晶胞自由度。

## 5. 约束、对称性和优化器

POSCAR 的 `Selective dynamics` 标志需配合 `relax:selective_dynamics true` 使用。
分量的含义与优化坐标有关，不能不加检查地把任意 T/F 掩码理解为实验室笛卡尔方向。
`relax:keep_symm true` 会限制在初始空间群允许的子空间中，可能排除真实的破对称畸变；
它与其他约束存在兼容性限制。具体规则见[约束与对称性](../advanced/relax-symmetry_zh.md)
及[配置参考](../reference/configuration_zh.md)。

`lbfgs` 可作为普通任务的起点；`fire`、`lbfgspp`、`cg` 等是其他可选驱动。
变更优化器后应比较最终能量和梯度。不要同时照抄 `isif` 和一组自由度标志：
顶层 `isif` 会覆盖 `relax_atom`、`relax_cell` 等相关设置。

## 常见卡点与下一步

- 力异常大或线搜索失败：先检查元素映射、单位、原子距离和模型适用范围，再调整步长。
- 迭代耗尽：查看轨迹是否仍在改善，再决定增加预算、换初始结构或调整优化器。
- 结构停在高对称位置：检查对称性/约束是否排除了想研究的畸变。
- 多结构输入：输出与各帧停止状态都要核对，不能只看最后一帧。

得到可靠的平衡结构后，可继续做[晶体性质](properties_zh.md)；两个端点均弛豫后，
再做 [NEB](neb_zh.md)。全部 CLI 选项见[命令参考](../reference/commands_zh.md)。
