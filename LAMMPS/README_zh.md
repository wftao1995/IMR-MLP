# LAMMPS 接口

[English](README.md)

**普通短程 MTP / MTP-M 模型做 MD 或结构优化，先用 CPU 接口 `mlip/opt`。**
它支持 MPI，不需要链接 `libimrmlp`，也不需要先转换成 spline 或 GPM。
先按下面的最小流程跑通，再根据 CPU 线程、GPU 或特殊观测量的需求选择其他接口。
LES/SOG 等含长程贡献的模型，按后面的专用入口选择。

本项目名称为 IMR-MLP，命令行为 `imrmlp`；LAMMPS 中沿用 `mlip/...` 风格名。

## 先跑通最常用的短程模型

### 1. 安装 CPU 接口

准备一份干净的 LAMMPS 源码、Python 3、CMake、C++ 编译器、MPI 和 Eigen 头文件。
在 IMR-MLP 仓库根目录运行，替换 LAMMPS 路径和 Eigen 路径：

```sh
python3 LAMMPS/USER-MLIP/install.py /path/to/lammps/src
cmake -S /path/to/lammps/cmake -B /path/to/lammps/build-imrmlp \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_MPI=ON \
  -DCMAKE_CXX_FLAGS="-I/usr/include/eigen3"
cmake --build /path/to/lammps/build-imrmlp -j8
```

这条路线直接把接口源码装入 LAMMPS；无需先编译 IMR-MLP 主程序或训练库。
没有 MPI 时可改用 `-DBUILD_MPI=OFF` 构建串行版。

### 2. 用自己的模型做一次单点计算

准备适用于目标元素和模拟条件的已训练模型，以及包含原子质量的 LAMMPS 数据文件。
下面假设数据文件有两种原子类型：类型 1 为 Al，类型 2 为 Ni；模型含相应元素元数据。
将输入保存为 `in.mtp`：

```text
units metal
atom_style atomic
boundary p p p
newton on
read_data system.data

pair_style mlip/opt
pair_coeff * * model.mtp Al Ni

thermo 1
thermo_style custom step atoms pe press
run 0
```

```sh
/path/to/lammps/build-imrmlp/lmp -in in.mtp
```

`model.mtp` 和 `system.data` 是占位文件名，需替换为自己的文件。
单元素 Al 模型只写 `pair_coeff * * model.mtp Al`。
若模型没有元素元数据，见下文“元素映射与旧模型”。先确认能量、力和压力与
IMR-MLP 对同一构型的计算一致，再添加所需的积分器、温控或压控命令进行 MD。

### 3. 没有自己的数据时，用入门模型核对 CLI 与 LAMMPS

先完成 [CPU CLI 安装](../docs/guides/installation_zh.md)及
[训练入门](../examples/quickstart/README_zh.md)，再从 IMR-MLP 仓库根目录运行。
已生成 `build/quickstart/` 时不用重跑训练；自定义了 `--work` 时替换复制路径。

```sh
mkdir -p build/lammps-first-run
cp build/quickstart/fitted.mtp build/quickstart/valid-input.cfg \
  build/quickstart/index.map build/lammps-first-run/
cd build/lammps-first-run
imrmlp convstr valid-input.cfg single.vasp --index index.map --force-one --select 0
imrmlp convstr single.vasp system.data --index index.map --output-format lmpdat
imrmlp adjustmtp fitted.mtp named.mtp --elements Al
imrmlp calc_efs single.vasp cli.xyz --potfile named.mtp --pottype mtp \
  --index index.map --input-format POSCAR --output-format extxyz
```

请使用一个新的 `build/lammps-first-run` 目录。`--select 0` 取四帧中的第一帧，
`--force-one` 让单个 POSCAR 精确写到 `single.vasp`，避免多帧导出时的编号后缀。
`adjustmtp` 为已知单元素教学模型补写名称映射。保存以下内容为该目录的 `in.demo`：

```text
units metal
atom_style atomic
boundary p p p
newton on
read_data system.data
mass 1 26.9815385
pair_style mlip/opt
pair_coeff * * named.mtp Al
thermo 1
thermo_style custom step atoms pe press
run 0
write_dump all custom forces.dump id type x y z fx fy fz modify sort id
```

导出的 data 默认没有质量，因此显式设置 `mass`。仍在该目录运行已安装接口的 LAMMPS：

```sh
/path/to/lammps/build-imrmlp/lmp -in in.demo
```

`log.lammps` 应显示 3 个原子、`PotEng` 约 `0.46195372 eV`，与 `cli.xyz` 的 `energy`
一致；`forces.dump` 按 id 排序后，各原子的 `fx fy fz` 应与 CLI 的 `forces` 三列一致，
允许文本输出精度造成的舍入差异。LAMMPS `metal` 的 `press` 是 bar，不能直接与
GPa 数值比较。这个合成模型仅用于接口核对，不能拿来做真实铝的 MD。
完成后用 `cd ../..` 回到 IMR-MLP 仓库根目录。

## 短程模型：各接口怎么选

下表主要面向普通短程 MTP 和短程 MTP-M。**先选模型兼容的接口，再比较运行速度；
换接口不会提高模型本身的物理精度。**

| 接口 | 适用范围与优势 | 代价与限制 | 建议何时选 |
| --- | --- | --- | --- |
| **`mlip/opt`** | CPU/MPI；解析径向基；支持普通 MTP、短程 MTP-M 的 li/q2/fs/n1/n2 头及相应 GPM；无需 `libimrmlp` | 不计算 LES/SOG 长程项，也不提供 transport 热流 compute | **普通 MD、结构优化的默认起点** |
| `mlip/opt/omp` | 与 `opt` 相同的模型范围；MPI + OpenMP，可让每个 rank 使用多个 CPU 线程 | 需启用 OPENMP 包并设置线程数；每线程工作区增加内存，线程越多不一定越快 | CPU 节点上比较纯 MPI 与 MPI×线程的实测效率 |
| `mlip/gpudirect` | CUDA 或 HIP/DCU；独立短程推理，不链接 `libimrmlp`；支持短程 MTP/MTP-M | 需 GPU 工具链；不支持 LES/SOG 或 transport compute；MTP-M 使用 FP64；小体系的传输和启动开销可能抵消收益 | 有 GPU、希望加速短程 MD，且可用实际模型和体系做基准比较 |
| `mlip/spline` | CPU/MPI；读取预先转换的 `RBSpline` 普通 MTP，以插值代替解析径向基求值 | 多一步转换；有网格插值误差，需检查能量、力和 MD 稳定性；不能当作通用 MTP-M 接口 | 径向基求值成本明显、愿意验证插值精度时再尝试 |
| `mlip/native` | 调用 IMR-MLP 的原生模型实现；适合需要保留原生神经网络、混合势或更完整模型组合的部署；支持的模型可用 CPU/CUDA/HIP | 必须链接匹配的 MPI `libimrmlp` 并管理其运行依赖；后端支持取决于具体模型；部署步骤较多 | `opt` / `gpudirect` 不支持的模型，或需要与原生模型工作流统一时 |
| `mlip` | 原始 CPU 接口，可继续运行已有输入 | 不应套用 `opt` / native 的扩展模型支持范围 | 维护既有计算；新任务从 `opt` 开始 |

GPU 与 OpenMP 都没有固定加速比。用自己的模型、原子数和硬件测量，比较时保持
MPI rank 数、线程数和输出频率可追溯。普通能量、力、压力计算不需要 transport 接口。
完整模型范围见 [CPU 风格说明（英文）](USER-MLIP/opt/README.md)、
[standalone 说明（英文）](USER-MLIP/standalone/README.md)和
[native 说明（英文）](USER-MLIP/native/README.md)。

### 使用 OpenMP

复制安装时加 `--with-omp`，CMake 配置加 `-DPKG_OPENMP=ON`，然后重新构建。
输入中在 `read_data` 前添加 `package omp 4`，并把 pair style 改为：

```text
pair_style mlip/opt/omp
pair_coeff * * model.mtp Al Ni
```

这里每个 MPI rank 使用 4 个线程，需同时规划 rank 数与分配到的 CPU 核数。
已用安装脚本安装的源码需要先卸载再安装，见下面的安装维护说明。

### 使用 spline

仅对普通 MTP 尝试转换，再用同一构型比较原模型与转换模型：

```sh
imrmlp mtp2spline model.mtp model.rbspline.mtp
```

```text
pair_style mlip/spline
pair_coeff * * model.rbspline.mtp Al Ni
```

网格间距等转换选项见 [mtp2spline 命令](../docs/reference/commands_zh.md)。
转换文件不会自动适用于其他模型家族。

## 安装方式与 GPU 部署

**接口名决定怎样计算，安装方式决定怎样把它放进 LAMMPS。**
例如 standalone 和复制安装都可提供 `mlip/opt`；standalone 不是一种新的势函数。
每份 LAMMPS 源码/构建选择一条路线，避免重复安装相同风格名。

| 安装方式 | 适合的使用场景 | 依赖与取舍 |
| --- | --- | --- |
| 本页的 CPU 复制安装 | 已有 LAMMPS 源码，希望一次安装 CPU 短程、LES/SOG、transport 和简谐参考接口 | 修改 LAMMPS `src`，需 Eigen 头文件和重编译；这些 CPU 风格无需 `libimrmlp` |
| [standalone 标准 `lmp` 构建（英文）](USER-MLIP/standalone/README.md) | 只部署短程 MTP/MTP-M；CPU/OpenMP 或 CUDA/HIP；可导出独立源码包给计算节点 | 无需训练库、MKL 或系统 Eigen；使用随包 Eigen 头文件；仅含支持的短程风格；标准 `lmp` 入口不修改上游源码 |
| [native 标准 `lmp` 构建](USER-MLIP/native/lammps/README_zh.md) / [native 插件（英文）](USER-MLIP/native/README.md) | 部署原生模型组合、长程模型或其支持的 GPU 路径 | 链接匹配的 MPI `libimrmlp`；插件还要求目标 LAMMPS 启用 PLUGIN，且版本、MPI 和编译器 ABI 匹配 |

短程 GPU 部署优先按 standalone 的标准 `lmp` 流程构建。下面以 NVIDIA CUDA 为例；
需要 CMake 3.22+，并将 `80` 换成实际设备架构：

```sh
cmake -S LAMMPS/USER-MLIP/standalone/lammps -B build/lmp-standalone-cuda \
  -DLAMMPS_SOURCE_DIR=/path/to/clean/lammps \
  -DMTP_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/lmp-standalone-cuda -j8
```

在前面的单点输入中替换 pair 声明，使用新构建的 `lmp` 运行：

```text
pair_style mlip/gpudirect device cuda precision fp64 gpu_id auto
pair_coeff * * model.mtp Al Ni
```

HIP/DCU 对应 `-DMTP_BACKEND=HIP`、`CMAKE_HIP_ARCHITECTURES` 和 `device hip`。
CPU 部署使用 `-DMTP_BACKEND=CPU`，OpenMP 另加 `-DMTP_OPENMP=ON`。
GPU 构建保留 CPU 风格，便于在同一可执行文件中比较。具体模型、径向基、显存控制、
设备验证及插件部署方法见 [standalone 说明（英文）](USER-MLIP/standalone/README.md)。

## 元素映射与旧模型

`pair_coeff` 中每个 LAMMPS 原子类型对应一个模型元素，按类型 1、2、…… 的顺序书写，
与原子在数据文件中出现的顺序无关。例如类型 1 为 Ni、类型 2 为 Al：

```text
pair_style mlip/opt
pair_coeff * * model.mtp Ni Al
```

元素名须与模型中声明的元素表匹配。同一条命令不能混用名称与数字，也不接受 `NULL`。
三种原子类型就提供三个条目，例如 `Al Ni Co`；多个类型也可映射到同一模型元素。
上述语法同样适用于本页其他 `mlip/...` pair 风格。

旧模型若没有元素元数据，先从训练设置确认 species 顺序，再用从 0 开始的数字映射：

```text
pair_coeff * * legacy.mtp 0 1      # 类型 1 -> species 0，类型 2 -> species 1
```

对于已确认 species 顺序为 `Al Ni` 的普通 MTP，可显式补写元数据：

```sh
imrmlp adjustmtp legacy.mtp named.mtp --elements Al Ni
```

核对预测后再改用元素名。重新保存本身不能恢复未知的元素身份。
原始声明 `pair_style mlip/opt model.mtp 0 1` 加 `pair_coeff * *` 仍受支持，
模型只声明一次。`fix mlip/maxvol` 有独立的数字映射语法，见其专用说明。

## 长程及原生模型

模型含长程贡献或需要原生模型组合时，按下表选择。普通短程 MTP 无需配置电荷或隐变量 compute。

| 模型与需求 | 接口与优势 | 限制与选择依据 |
| --- | --- | --- |
| 线性 MTP 短程头 + Coulomb LES v2 / SOG v3、v4，在 CPU 上运行 | [`mlip/opt-les`（英文）](USER-MLIP/opt-les/README.md)：专用 CPU/MPI 实现，无需 `libimrmlp` | 要求三维全周期；不支持非线性 MTP-M 头；restart 后需重新声明模型和 compute |
| 长程 MTP-M、原生神经网络、MTP 混合势，或需要长程 GPU | [`mlip/native`（英文）](USER-MLIP/native/README.md)：复用原生模型及其 CPU/GPU 路径 | 链接 MPI `libimrmlp`；具体保存格式和模型/设备组合必须受支持；没有 CPU 回退 |

例如，已通过 CPU 安装脚本安装的 SOG 接口：

```text
pair_style mlip/opt-les
pair_coeff * * sog.mtp Ni Al
compute latent all mlip/sog/latent
```

Coulomb LES 使用 `compute q all mlip/les/charge`；SOG 的隐变量无量纲，不能解释为物理电荷。
这些模型已包含其长程贡献，不应把短程接口当作等价替代。

native 插件加载 GPM 的示例（直接编入 `lmp` 时省略 `plugin load`）：

```text
plugin load /path/to/mlip_native_plugin.so
# 在前述 units、建盒和 read_data 命令之后定义 pair：
pair_style mlip/native device cpu precision fp64 pottype auto
pair_coeff * * model.gpm Al Ni
run 0
```

GPM 可自动推断势类型；旧格式需显式 `pottype`，可能还需 `settings` 和配套资源。
电荷/隐变量 compute、GPU 设备分配和总电荷约束按 [native 说明（英文）](USER-MLIP/native/README.md)设置。
GPM 是模型文件格式，转为 GPM 不会扩大接口的模型支持范围；见 [GPM 模型说明](../docs/reference/gpm-v1_zh.md)。

## 热流、主动学习与热力学积分

这些功能按任务添加，不是普通 MD 的必选步骤。

| 任务 | 入口与优势 | 使用范围与限制 |
| --- | --- | --- |
| 热导率、全体系热流或静态界面能流 | [`mlip/transport`、`/omp`、`/spline`（英文）](USER-MLIP/transport/README.md)，配合专用 heat compute | CPU/MPI；仅线性短程 MTP / 相应 spline；不支持 MTP-M、LES/SOG 或 GPU；总热流 compute 拒绝 pair hybrid 及额外 bonded/kspace 能量项 |
| 在 MD 中识别陌生构型并收集候选 | [`fix mlip/maxvol`](USER-MLIP/maxvol/README_zh.md)，可附加在已有 pair 上 | 独立 MPI `libimrmlp` 插件；需预先生成匹配的主动集；只打分和保存候选，不改变 pair 力，也不生成参考标签 |
| 简谐参考与热力学积分 | [`harmonic/reference` 工作流](../docs/advanced/harmonic-ti_zh.md) | CPU 复制安装已包含；需准备参考结构、力常数及 TI 设置 |

热流最小声明（用它替换普通 `opt` pair）：

```text
pair_style mlip/transport
pair_coeff * * model.mtp Al Ni
compute J all mlip/heat/flux
```

该 compute 输出全体系广延热流，需按分析方法处理体积归一化。
加载 GPM 不会自动启用主动学习；MaxVol 的模型范围、主动集生成和候选输出见其专用说明。
GPU pair 配合 CPU MaxVol 打分器需要单独验证。

## 更换模型、续算与安装维护

更换已加载模型前，先删除依赖它的观测量 compute，再清空 pair 并重新声明：

```text
pair_style none
pair_style mlip/opt
pair_coeff * * other.mtp Al Ni
```

需要的 compute 随后重新创建。`opt` / `opt/omp` / `gpudirect` 的短程 MTP-M/GPM
restart 仍要求原模型路径可读且内容不变；native 也需保留引用的模型和设置文件。
跨目录续算建议使用绝对模型路径。`opt-les` 续算须重新声明模型和 compute。
各风格的具体 restart 限制见对应说明。

CPU 安装脚本不覆盖已有文件；更新接口时先用哈希校验卸载，再安装同一源码版本的文件：

```sh
python3 LAMMPS/USER-MLIP/install.py --uninstall /path/to/lammps/src
```

安装或卸载后都要重新配置并重建 LAMMPS。同一份源码上不要混用本安装脚本和
transport 专用安装脚本。MaxVol 和 native 插件不包含在 CPU 复制安装中。

## 接口验证与开发资料

首次部署先比较同一构型的能量、力及六个维里分量，再验证目标运行条件下的短程 MD。
开发或更新接口时，可使用以下回归入口；合成夹具的通过不代表材料势的适用性验证。

- CPU/LES/SOG：[构建与 MPI 回归说明（英文）](USER-MLIP/opt-les/README.md)。
- 跨风格一致性：[opt/transport 对比脚本](USER-MLIP/transport/tests/run_pair_parity.sh)；
  [全 CPU 构建脚本](USER-MLIP/opt-les/tests/build_regression.py)的 `--all-cpu` 可构建完整测试运行器。
- 元素映射、新旧语法、MPI 一致性：[元数据回归脚本](USER-MLIP/opt/tests/run_metadata_regression.py)。
- 短程 CPU/OpenMP/GPU 与 restart：[standalone 测试说明（英文）](USER-MLIP/standalone/README.md)。
- 原生模型与 GPU：[native 回归说明（英文）](USER-MLIP/native/README.md)。
- 热流与 spline：[transport 回归说明（英文）](USER-MLIP/transport/README.md)；
  简谐参考：[hyb-ti 回归说明（英文）](USER-MLIP/hyb-ti/README.md)。

## 模型元数据参考

模型通过单行 JSON 标签记录元素顺序，例如：

```text
potential_tag = {"schema":"gpml.mtp.metadata","version":1,"elements":["Al","Ni"],"type":"MTP","model":"linear"}
```

schema 名称有意保持共享，没有为 IMR-MLP 改名。MTP-M 使用 `type: "MTP_M"`，
在 `model` 中给出头名称、整数 `q`、`model_coeffs_count`，神经头另用
`nn.activation` / `nn.hidden_layers`。LES/SOG 使用 `type: "MTP_LES"` 及
`long_range.kernel` / `long_range.channels`；长程 MTP-M 仍为 `type: "MTP_M"`，
另带独立的 `long_range` 字段。
既有的空标签、文本标签与 `imrmlp_meta_v1:` 标签仍可读取。保存时写入规范
schema，旧文本保留在 `legacy_tag` 中，扩展字段也会保留。旧的 IMR q
名称/定义与头扩展保留在 `imrmlp_q` 与 `imrmlp_external_model` 中。
旧版 GPU 读取器使用的 `# mtp_*` 页脚字段仍然保留，并与 JSON 做一致性检查。
版本非法、键/元素重复、维度不一致和元素顺序冲突都会失败。
元素扩展必须显式给出每个输出元素：
`imrmlp adjustmtp ... --neworder ... --elements ...`。

## 许可

源自 LAMMPS 的适配器保留其原始的 Sandia/LAMMPS 声明，并按
[GPLv2](COPYING.GPL-2.0) 分发，不会被重新许可为本仓库的 MIT 许可。
新增的集成改动使用 IMR-MLP contributors 版权，头文件会标明适用许可。
独立维护的 `runtime/mtp` 核心与安装脚本复制的项目辅助头文件保留其 MIT 声明。
合并后的 LAMMPS 可执行文件在再分发时必须与 GPL 及其他依赖许可保持一致，
详见[第三方许可说明](../THIRD_PARTY_NOTICES.md)。
