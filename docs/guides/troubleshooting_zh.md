# 故障排查 {#guide_troubleshooting}

[English](troubleshooting.md)

## 找不到模型、POSCAR 或示例目录

先确认当前目录（`pwd`）和上一阶段是否成功。`model.mtp`、`POSCAR` 等是自备输入的
占位名，不会由任意命令自动下载；没有自己的输入时走
[连续计算练习](../../examples/basic-calculations/README_zh.md)。物性目录的 `relaxed.vasp`
来自其中晶胞弛豫结果的显式复制，NEB 端点来自两次端点弛豫后的导出。
训练入门默认输出到 `build/quickstart/`；如果使用了其他 `--work`（包括旧示例的
`build/first-run/`），后续页面也要换成实际路径。

脚本提示输出目录非空时，复用已成功的结果，或指定一个新的 `--work`；不需要删除原结果。
`convstr` 从多帧文件导出 POSCAR 时可能添加编号后缀；仅需一帧时使用
`--force-one --select 0`，如 [LAMMPS 入门核对](../../LAMMPS/README_zh.md)所示。

## 配置未生效，或文件名以 `=` 开头

设置文件使用 `key value`，不写 `key = value`。注释用 `#` 或 `!`；空行结束
`[section]`，显式 `fit:key` 应放在 section 外。重复键保留第一次出现的值。
相对路径从进程当前工作目录解析。`.mtp` 势文件是独立语法，仍使用等号。

## 训练结束了，却没有预期的优化迭代

非线性 L-BFGS 需显式执行 `imrmlp trainmtp train.ini --do-lbfgs`；仅拟合线性
系数用 `--linear-fit`。`--check-grad` 和矩阵导出属于诊断操作。检查日志中的
接受步数和停止原因。最终势文件是 `_Trained_Potential_.mtp`，`fit:save_to`
指定训练器检查点。每次训练使用独立目录，避免覆盖之前的模型。

ALGLIB 的 `M>N` 错误表示 L-BFGS 历史长度超过自由参数数目；调小 `fit:bfgs_m`，
微型入门模型使用 2。

## 找不到动态库、符号错误或误用了旧程序

使用明确的可执行文件路径，并检查加载的库：

```sh
build/release/imrmlp --version
ldd build/release/imrmlp
export LD_LIBRARY_PATH="$PWD/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

程序和动态库应来自相同构建模式。安装后把前缀下的 `bin` 加入 `PATH`，`lib`
加入 `LD_LIBRARY_PATH`。项目静态可执行文件仍依赖外部系统库。修改头文件或
影响 ABI 的编译选项后，重新编译使用这些接口的模块。

## 缺少依赖或编译内存不足

安装开发头文件和运行库。`MKLROOT` 指向可选的 oneMKL 安装目录，Make 只在该环境
提供 `include/mkl.h` 时启用该后端；`USE_MKL=0` 可完全不用 oneMKL，改用 Eigen
自带内核编译。`USE_KIM=0` 可关闭可选 KIM 接口。
编译器及标准库需支持本项目的 C++23 特性，详见[安装指南](installation_zh.md)。
编译进程因内存不足被杀死时，降低并行编译任务数。

## 物种不匹配或能量、力异常

核对模型物种顺序和 `index.map`（每行一个元素符号）。CFG 物种是索引，不是
原子序数。LAMMPS 原子类型从 1 开始，MTP `pair_style` 映射参数从 0 开始。
Python/ASE 显式指定按模型顺序排列的 `elements`。不要从文件名猜测物种顺序。
坐标和晶胞单位为 Å，能量为 eV，力为 eV/Å；原生广延应力为 eV，ASE 返回
除以体积后的应力，单位 eV/Å³。

## LAMMPS 找不到 pair style，或接口编译失败

按照 [LAMMPS 安装说明](../../LAMMPS/README_zh.md) 安装接口后重新编译 LAMMPS。
`make imrmlp` 不会构建 LAMMPS。复制的接口文件应来自同一源码版本。目前接口
通过公开的 `neighbor->add_request` API 申请完整邻居表；如果错误涉及直接写入
`NeighRequest::half/full`，通常仍在编译旧的接口副本。

核对 `units metal`、物种映射及对应 style 的 `newton`、atom style 要求。先用
`run 0` 与 `calc_efs` 比较能量和力，再开始 MD。修改 CPU/OpenMP/spline/transport
路径后运行[跨 style 回归脚本](../../LAMMPS/USER-MLIP/transport/tests/run_pair_parity.sh)。
Spline、LES/SOG 模型需选择各自对应的 style。

## Python 导入或 ASE 计算失败

构建、安装、运行应使用同一个 Python 解释器。用
`python3 -c "import imrmlp; print(imrmlp.__file__)"` 检查导入路径，并检查扩展的
动态库依赖。注意当前目录可能遮蔽已安装的软件包。ASE 计算器需另外安装 ASE。
`Potential(filename)` 默认 `auto`：GPM 推断包内类型，旧格式按 `mtp` 加载；
其他旧格式类型需显式传入。
详见 [Python 指南](../reference/python_zh.md)。

## 弛豫有输出文件，或 NEB 有路径，但没有收敛

先检查退出码和日志。`relax` 返回 2 表示有未收敛结构；NEB 返回 2 表示正常停止但
未收敛，路径摘要里还有 `# converged` 和 `# status`。保留最后接受的结构是为了
检查和续算，不代表容差已经达到。先检查映射、原子距离、自由度和端点，再决定增加
预算或调整初始化。逐步流程见[弛豫](relaxation_zh.md)和 [NEB](neb_zh.md)。

## NEB 插值重叠、原子穿越或得到意外机制

核对端点逐行的原子身份、晶胞与周期位移分支。程序不自动重排等价原子。
查看初始化图像，再考虑 IDPP 或显式提供中间图像；改变优化器不能修复错误的原子对应。

## 回归测试需要仓库里没有的数据

历史科学回归可能需要外部势和数据集，先核对测试的输入要求；缺少数据本身
不表示算法失败。[入门示例](../../examples/quickstart/README_zh.md)只使用提供的
合成输入，适合验证新安装。

报告问题时提供命令、最小配置、模型与物种信息、程序版本、构建模式、编译器和
依赖版本及相关日志。只分享有权分发的最小复现输入。
