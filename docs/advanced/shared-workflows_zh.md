# 结构 IO 与非线性训练

[English](shared-workflows.md)

## 结构 IO

`imrmlp.iread()` 逐帧读取轨迹，`imrmlp.write_iter()` 在一个输出会话中写入可迭代对象。
支持的格式和读写方向限制见 `imrmlp.io`。独立会话分别维护帧计数和编解码设置。
消费到损坏帧或截断的压缩流时会报告错误。

`imrmlp.detect_format()` 可检查文件内容。显式格式优先，文件名格式推断只检查基本文件名。
`open_reader()` 和 `open_writer()` 提供可配置、可复用的 Python 会话对象。

单位转换必须显式指定物理量与源单位，接口包括 `unit_conversion_factor()`、
`convert_units()`、`source_unit_system()` 和 `canonicalize_atoms()`。
原生单位为 Angstrom、eV、ps 和 amu；`Atoms.stress` 保存能量维度的 virial。
源单位覆盖适用于 EXTXYZ。

XSF 标量网格通过独立的 `read_xsf_fields()`、`write_xsf_fields()` 以及
`ScalarField`、`DataGridBlock` 读写，保留网格布局与跨距矢量。
转换时的派生标签见[标量表达式（英文）](convstr-scalar-labels.md)，
选择性 HA 分类见[HA 类别（英文）](ha-selected-indices.md)。

## 训练

`trainnlpot` 与 Python `NonlinearTrainer` 共用 CPU 非线性训练实现。
`hessian_weight`、`hessian_pow`、`hessian_fd_step` 支持完整 Cartesian Hessian 标签，
可以与 E/F/S、HVP 观测共同训练。训练使用结构副本，位置转换不修改输入标签。

普通 MTP 可以针对固定普通 MTP 或 ZBL 核心势的残差训练；复合势输出保留两个分量。
训练集、验证选择和最终测试集评估分别处理。命名参数组和支持的分阶段调度控制拟合坐标。

CPU CG 可在最后接受点执行一次终端线性重拟合，保留径向参数并校验目标函数。
校验失败时恢复接受点参数。该步骤不改变 CG 轨迹。
配置方式和支持的参数块见[线性精修](linear-refit_zh.md)。

## 弛豫与路径

V2 弛豫支持位置与晶胞掩码、版本化物理约束文件、调用方提供的非线性坐标及相应预条件器。
部分周期与无晶胞结构使用对应的坐标模式；需要完整晶胞的控制在输出前拒绝奇异几何。

NEB 提供求解预算、稳定试探步、可选的对称性诊断、可复现扰动和解除对称性后的检查。
详见[NEB 并行计算与路径对称性](neb-parallel-symmetry_zh.md)。新增控制按需启用，
有限迭代预算结束与实际收敛分别报告。
