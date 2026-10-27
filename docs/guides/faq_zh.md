# 常见问题 {#guide_faq}

[English](faq.md) · [命令参考](../reference/commands_zh.md) · [配置说明](../reference/configuration_zh.md)

这里回答第一次运行前最常见的问题，偏“该用哪个”“如何定义”。
按症状排错请见[故障排查](troubleshooting_zh.md)；第一次计算先看[任务引导](getting-started_zh.md)。

## 训练势函数用哪个命令？

| 目标 | 命令 | 说明 |
|---|---|---|
| 矩张量势（MTP） | `trainmtp` | 非线性优化需显式加 `--do-lbfgs`，只拟合线性系数用 `--linear-fit`；默认两者都不做。 |
| NN-MTP / HMTP 等非线性势 | `trainnlpot` | 必须显式选择优化器，势类型来自 `[pot]` 段。 |
| 线性描述符势 | `train_linear` | 最小二乘，可带岭正则。 |
| 快速线性拟合与基组选择 | `fastfit` | 模式 0 计算特征值谱，模式 1 加权最小二乘，其余模式用于基组/NSGA-II 选择。 |

这些命令使用相同的空白分隔设置语法，但支持的键和势函数定义不同，不能直接
互换训练配置。`trainmtp` 的最终模型是工作目录下的
`_Trained_Potential_.mtp`，训练检查点由 `fit:save_to` 控制。

## 怎么查命令和选项？

```sh
imrmlp --help                # 列出全部已注册命令
imrmlp trainmtp --help       # 查看单个命令的选项
```

可用命令以当前程序的帮助输出为准；`efs`、`err`、`elastic`、`neb`、`dimer`、`fps_select` 等
别名指向规范命令，行为完全一致。[命令参考](../reference/commands_zh.md)
给出完整选项表，[速查表](../reference/cheatsheet_zh.md)按任务归类。

## 为什么我的配置项没生效？

设置文件用空白分隔键和值：`key value`。等号会被当成值的一部分，
因此 `cutoff = 5.0` 并不会设置 `cutoff`。`.mtp` 模型文件的语法是另一套，
它确实使用等号。`[section]` 规则、重复键与相对路径解析见
[配置说明](../reference/configuration_zh.md)。

## 元素顺序如何确定？

元素顺序必须显式给出，不会从文件名或结构出现顺序推断：

- 顶层 `index` 指向 `index.map`，每行一个元素；所选势函数需要映射时，
  也应设置 `pot:typemap`。
- CFG 中的 species 是该映射的索引，不是原子序数。
- LAMMPS 类型从 1 开始；数字模型 species 映射从 0 开始，新输入推荐在 `pair_coeff` 中写元素名。
- Python/ASE 需要按模型顺序显式传入 `elements` 列表。

## 使用什么单位？

能量 eV、长度 Å、力 eV/Å。外应力与压强设置的单位为 GPa；原生的广延应力
单位为 eV，ASE 计算器返回按体积归一化的 eV/Å³。温度与时间步沿用调用方
驱动器的单位（例如 LAMMPS 的 `units metal`）。

## 选样该用哪个命令？

| 场景 | 命令 |
|---|---|
| 构建第一批最大无关训练集 | `maxvol_select` |
| 用已有主动集给新结构打分 | `maxvol_sample` |
| 增量扩充已有主动集 | `maxvol_extend` |
| 按 FPS 或 CUR 杠杆分数选多样性子集 | `diverse_select`（别名 `fps_select`） |

很大的生产级候选池建议使用 CLI；Python 侧的 `MaxVol`、`ActiveSelector`、
`select_from_pool`、`cur_select` 更适合内存中的矩阵和自定义循环。

## 应该选择哪种势类型？

一般从普通 `mtp` 开始；多通道和非线性头见 [MTP_M](../theory/mtp-m_zh.md)。
LES/SOG 属于 `mtp_les` 家族，
需要把短程势与 `zbl` 等子势组合时用 `hybrid`/`hybrid_scale`。
[势类型总览](../reference/potential-types_zh.md)列出全部 21 种已注册类型、
各自的输入产物与构建开关。

## 支持 MPI 和 GPU 吗？

支持，但属于独立的构建模式：`make BUILD_MODE=mpi imrmlp` 产出
`imrmlp-mpi` 与 `libimrmlp-mpi.so`。CUDA 使用 `BUILD_MODE=cuda` 和
`CUDA_HOME`/`CUDA_ARCH`；ROCm HIP 使用 `BUILD_MODE=hip` 和 `HIP_ROOT`/`HIP_ARCH`。
组合模式为 `mpi-cuda`、`mpi-hip`。提供 `--device` 的评估命令可选择 `cpu`、
`cuda` 或 `hip`（`dcu` 是 `hip` 的别名）。详见[安装说明](installation_zh.md)
与 [GPU 支持](../development/cuda_zh.md)。GPU 构建不改变命令行接口；
不支持的组合会在构造势函数时报错。

## CLI 和 Python 可以混用吗？

可以。二者绑定同一个库：用 `trainmtp` 训练的模型在 Python 中直接加载，
`convstr` 写出的结构可用 `imrmlp.read` 读取。混用前应核对两侧的势类型、物种映射与 IO 格式。
例如 `Potential(...)` 默认 `pottype="auto"`：GPM 读取包内类型，旧格式回退为 `mtp`；
其他旧格式模型需显式指定类型，不按扩展名猜测。
详见 [Python API](../reference/python_zh.md)。

## 如何让结果可复现？

固定同一构建、固定线程数（`OMP_NUM_THREADS`、`MKL_NUM_THREADS`），
并设置所用功能暴露的种子，例如 `dimer:random_seed`。
Dimer 的 `random_seed=0` 在同一构建下可复现，`-1` 表示非确定性初始化。
每次运行使用独立目录，避免 `_Trained_Potential_.mtp` 与日志被覆盖。
