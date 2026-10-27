# 对称性约束弛豫与近邻安全性

[English](relax-symmetry.md)

设置 `relax:keep_symm true` 可在弛豫中保持初始结构的空间群约束。
先将下方设置保存为 `relax.ini`，再运行：

```sh
imrmlp relax --input POSCAR --output relaxed.cfg --potfile potential.mtp \
  --pottype mtp --index index.map --settings relax.ini
```

```ini
relax:keep_symm true
relax:relax_cell true
relax:relax_atom true
relax:driver lbfgs
relax:gtol 1e-5
relax:ftol 0
relax:xtol 0
relax:stepmax 0.01
relax:maxiteration 500
```

## 严格对称性

初始化阶段识别对称操作与保元素原子映射，把晶胞度规投影到其不变子空间，
并把结果晶胞对齐到输入取向。原子坐标被修正到周期性仿射约束上。原子顺序、
元素种类和原子数保持不变。`symmetry_atom_correction` 与
`symmetry_cell_correction` 分别记录以 Å 为单位的最大原子位移和晶格矢量修正量。
修正量超过 `symprec` 即失败。被冻结的分量不允许要求超过 `1e-10 Å` 的修正。

坐标遵循固定参考 `x = x0 + J q`；约化梯度为 `J.transpose() * g`。
每个轨道上的小型 Jacobian 保留了耦合的位点模式。两种
`atomic_force_recast` 设置都受支持。参考晶胞在一次运行中保持不变。
周期性 wrap 不改变优化变量。即使允许的自由度为 0，仍需要一次有效的初始求值。

`keep_symm` 与以下功能互斥：选择性动力学、固定体积或形状、附加结构约束、
笛卡尔模式、振动变换或滤波、线性变换或约束、质心过滤以及小梯度过滤。
晶胞弛豫要求对称应变、Cholesky 复原及其精确应变导数。
`keep_symm` 直接投影原始梯度，不会额外施加 `symm_force`。
更改坐标模式或结构约束后需重新初始化。

## 优化器状态

Python 结果的 `converged` 表示是否收敛，`n_iter` 是被接受的优化步数，
`stop_reason` 说明正常终止或失败原因；每次运行会重新计算这些状态。

只有满足 `gtol` 的有限最终梯度才算收敛；增广约束也必须满足各自的容差。
能量、步长和迭代上限都可能让未收敛的运行停止。LBFGSpp 额外的
`epsilon` 与 `epsilon_rel` 判据默认为零。

内部会保留最后一个被接受的点。非法试探点会恢复到该点并向上传递错误，
而不会自动重启优化器。正常结束时恢复被接受的结构及其已求值的能量与梯度，
不再重新求值。Python 返回该结构、最终能量以及本次运行的初始能量。

线搜索遵守位移上限；搜索失败会报错，不会把被拒绝的试探结构当作结果。

## 输出与退出状态

- `0`：所有输入都收敛，包括合法的零自由度输入。
- `2`：至少一个输入未收敛停止，且没有失败项。
- `1`：某个输入或计算失败。优先级为 `1 > 2 > 0`。

正常输出包含未收敛运行的最后一个被接受结构，日志会将其标注为未收敛。
失败的输入会被报告并跳过，其余独立输入继续运行。
内存分配失败会停止整批任务，但保留此前已完成的输出。
不会自动生成 JSON 状态文件或逐输入 checkpoint 文件。

## 输入几何与近邻

检查结构坐标和截断半径是否有限、原子是否异常重叠，并确保所用模型支持输入的
周期边界。部分周期结构允许非周期方向的晶格矢量为零；但空间群保对称弛豫需要
满足其自身的周期性和晶胞条件。斜交晶胞会增加近邻数和内存消耗。
近邻构建报错时，应先修正输入几何或截断设置再重新运行。
