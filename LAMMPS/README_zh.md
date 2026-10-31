# LAMMPS 接口

[English](README.md)

IMR-MLP 通过一组 `mlip/...` pair 风格接入 LAMMPS。除本页介绍的 CPU 复制安装外，
另有 standalone 与 native 两条安装路线，见第 3 节。

对普通短程 MTP 与 MTP-M 模型做分子动力学或结构优化，应优先使用 CPU 接口
`mlip/opt`：它支持 MPI，不依赖 `libimrmlp`，也不需要预先将模型转换为 spline 或 GPM。
建议先完成第 1 节的最小流程，再根据 CPU 线程、GPU 或特殊观测量的需求改用其他接口；
LES/SOG 等含长程贡献的模型见第 5 节。

## 1. 快速开始：短程模型的最小流程

本节给出安装接口与执行单点计算的最小步骤，适用于短程 MTP 与 MTP-M 模型。

### 1.1 安装 CPU 接口

依赖项包括干净的 LAMMPS 源码树、Python 3、CMake、C++ 编译器、MPI 与 Eigen 头文件。
在 IMR-MLP 仓库根目录执行以下命令，并替换其中的 LAMMPS 路径与 Eigen 路径：

```sh
python3 LAMMPS/USER-MLIP/install.py /path/to/lammps/src
cmake -S /path/to/lammps/cmake -B /path/to/lammps/build-imrmlp \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_MPI=ON \
  -DCMAKE_CXX_FLAGS="-I/usr/include/eigen3"
cmake --build /path/to/lammps/build-imrmlp -j8
```

该路线将接口源码直接装入 LAMMPS 源码树，不需要预先编译 IMR-MLP 主程序或训练库。
没有 MPI 时可用 `-DBUILD_MPI=OFF` 构建串行版本。

### 1.2 用自有模型做单点计算

需要两类输入：适用于目标元素与模拟条件的已训练模型，以及包含原子质量的 LAMMPS
数据文件。下例假定数据文件含两种原子类型，类型 1 为 Al、类型 2 为 Ni，且模型带有
相应的元素元数据。将输入保存为 `in.mtp`：

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

`model.mtp` 与 `system.data` 是占位文件名，使用时替换为实际文件；单元素 Al 模型
只需写成 `pair_coeff * * model.mtp Al`。模型缺少元素元数据时，参见第 4 节。
在添加积分器、温控或压控命令进行 MD 之前，应先确认同一构型的能量、力与压力
同 IMR-MLP 的计算一致。

## 2. 短程接口选择

本节面向普通短程 MTP 与短程 MTP-M 模型。应先取与模型兼容的接口，再比较运行速度；
更换接口不会提高模型本身的物理精度。

| 接口 | 适用范围与优势 | 代价与限制 | 选用条件 |
| --- | --- | --- | --- |
| **`mlip/opt`** | CPU/MPI；解析径向基；支持普通 MTP、短程 MTP-M 的 li/q2/fs/n1/n2 头及相应 GPM；不依赖 `libimrmlp` | 不计算 LES/SOG 长程项，也不提供 transport 热流 compute | **普通 MD 与结构优化的默认起点** |
| `mlip/opt/omp` | 模型范围与 `opt` 相同；MPI + OpenMP，每个 rank 可使用多个 CPU 线程 | 需启用 OPENMP 包并设置线程数；每线程工作区增加内存占用，线程越多不一定越快 | 单个节点上还想再压榨 CPU 时 |
| `mlip/gpudirect` | CUDA 或 HIP/DCU；独立完成短程推理，不链接 `libimrmlp`；支持短程 MTP 与 MTP-M | 需 GPU 工具链；不支持 LES/SOG 与 transport compute；MTP-M 使用 FP64；体系较小时传输与启动开销可能抵消收益 | 有 GPU 可用，想让短程 MD 跑得更快时 |
| `mlip/spline` | CPU/MPI；读取预先转换的 `RBSpline` 普通 MTP，以插值代替解析径向基求值 | 增加一步转换；存在网格插值误差，转换后需核对能量与力；多数情况下不比 `opt` 快；不能作为通用 MTP-M 接口 | 径向基求值占主要耗时、且能接受插值误差时 |
| `mlip/native` | 调用 IMR-MLP 的原生模型实现；适合需要保留原生神经网络、混合势或更完整模型组合的部署；受支持的模型可用 CPU/CUDA/HIP | 必须链接版本匹配的 MPI `libimrmlp` 并管理其运行依赖；后端支持范围取决于具体模型；部署步骤较多；同一模型上通常比 `opt` / `gpudirect` 慢 | `opt` / `gpudirect` 不支持该模型，或需与原生模型工作流统一时 |
| `mlip` | 原始 CPU 接口，可继续运行已有输入 | 不应套用 `opt` / native 的扩展模型支持范围 | 维护既有计算；新任务从 `opt` 开始 |

GPU 与 OpenMP 的加速比取决于模型、体系规模与硬件，换接口或加线程都不保证变快。
普通能量、力与压力计算不需要 transport 接口。

对标准短程 MTP，CPU 上慢约 1.5–2 倍，GPU 上差距更大。native 主要用来跑 `opt`
读不了的模型，即长程 MTP-M、原生神经网络与混合势。

完整模型范围见 [CPU 风格说明（英文）](USER-MLIP/opt/README.md)、
[standalone 说明（英文）](USER-MLIP/standalone/README.md)与
[native 说明（英文）](USER-MLIP/native/README.md)。

### 2.1 OpenMP

复制安装时加 `--with-omp`，CMake 配置加 `-DPKG_OPENMP=ON -DBUILD_OMP=ON`，然后重新构建。
两个开关都要给：只装 `PKG_OPENMP` 而编译期没有 OpenMP 支持时，LAMMPS 只打一条警告，
实际仍按单线程运行。

在输入文件中把 `package omp N` 放在 `read_data` 与 `pair_style` 之前，并将 pair 声明改为：

```text
pair_style mlip/opt/omp
pair_coeff * * model.mtp Al Ni
```

`N` 是**每个 MPI rank** 的线程数，既不是总核数也不是 rank 数；下文示例统一写成 `4`，那只是
占位值，应按作业实际分到的核数改写。总并行度是 rank 数乘 `N`，不应超过节点的物理核数。
每线程各持一份工作区，线程增多会增加内存占用，也不保证更快。`N = 0` 表示沿用
`OMP_NUM_THREADS` 的设置。

同样的设置也可以从命令行给出，且顺序要求自动满足：

```sh
lmp -pk omp 4 -in in.mtp    # 等价于输入文件里的 package omp 4
lmp -sf omp -in in.mtp      # 为 mlip/opt 自动选中 mlip/opt/omp
```

两种方式只选一种。用 `-sf omp` 时线程数取自 `OMP_NUM_THREADS`，该变量未设置就只有单线程；
没有 `/omp` 变体的其他样式会自动退回原实现。

`package omp` 也不能挪到 `pair_style` 之后：LAMMPS 在模拟盒子定义之后即拒绝该命令，而
`/omp` 样式在构造时就要读取线程数，缺这条是报错，不是静默退回单线程。

若源码已由安装脚本安装，需先卸载再重新安装，见第 7 节的安装维护说明。

### 2.2 spline

`mlip/spline` 以查表插值代替径向基的解析求值。只有当径向基计算确实占据主要耗时、
且查表的额外内存访问不足以抵消省下的浮点运算时，它才更快；多数体系下 `mlip/spline`
与 `mlip/opt` 速度相当甚至更慢，不宜作为加速手段。转换的意义主要在于得到 spline
形式的模型文件，供需要该形式的流程使用。

转换仅适用于普通 MTP，命令如下：

```sh
imrmlp mtp2spline model.mtp model.rbspline.mtp
```

```text
pair_style mlip/spline
pair_coeff * * model.rbspline.mtp Al Ni
```

网格间距等转换选项见 [mtp2spline 命令](../docs/reference/commands_zh.md)。
转换后的文件不会自动适用于其他模型族。

## 3. 安装路线与 GPU 部署

接口名决定计算方式，安装路线决定接口如何进入 LAMMPS。例如 standalone 路线与复制安装
都可提供 `mlip/opt`，standalone 并不是另一种势函数。每份 LAMMPS 源码或构建只应选择
一条路线，以避免重复安装同名风格。

| 安装路线 | 适用场景 | 依赖与取舍 |
| --- | --- | --- |
| 本页的 CPU 复制安装 | 已有 LAMMPS 源码，希望一次装入 CPU 短程、LES/SOG、transport 与简谐参考接口 | 修改 LAMMPS `src`，需 Eigen 头文件并重新编译；这些 CPU 风格不依赖 `libimrmlp` |
| [standalone 标准 `lmp` 构建（英文）](USER-MLIP/standalone/README.md) | 只部署短程 MTP/MTP-M；CPU/OpenMP 或 CUDA/HIP；可导出独立源码包供计算节点使用 | 不需要训练库、MKL 或系统 Eigen，使用随包的 Eigen 头文件；只含受支持的短程风格；标准 `lmp` 入口不修改上游源码；标准短程模型上通常快于 native |
| [native 标准 `lmp` 构建](USER-MLIP/native/lammps/README_zh.md) / [native 插件（英文）](USER-MLIP/native/README.md) | 部署原生模型组合、长程模型或其所支持的 GPU 路径 | 链接版本匹配的 MPI `libimrmlp`；插件还要求目标 LAMMPS 启用 PLUGIN，且版本、MPI 与编译器 ABI 匹配；标准短程模型上通常慢于 standalone |

短程 GPU 部署优先按 standalone 的标准 `lmp` 流程构建。下例以 NVIDIA CUDA 为例，
需要 CMake 3.22 及以上版本，并将 `80` 替换为实际设备架构：

```sh
cmake -S LAMMPS/USER-MLIP/standalone/lammps -B build/lmp-standalone-cuda \
  -DLAMMPS_SOURCE_DIR=/path/to/clean/lammps \
  -DMTP_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/lmp-standalone-cuda -j8
```

这里的架构号就是设备的 compute capability，去掉小数点书写：A100 为 `80`，H100 为
`90`，RTX 4090 为 `89`。各型号的对应值见
[NVIDIA CUDA GPU 计算能力对照表](https://developer.nvidia.com/cuda-gpus)。

在前面的单点输入中替换 pair 声明，再用新构建的 `lmp` 运行：

```text
pair_style mlip/gpudirect device cuda precision fp64 gpu_id auto
pair_coeff * * model.mtp Al Ni
```

HIP/DCU 对应 `-DMTP_BACKEND=HIP`、`CMAKE_HIP_ARCHITECTURES` 与 `device hip`；
CPU 部署使用 `-DMTP_BACKEND=CPU`，OpenMP 另加 `-DMTP_OPENMP=ON`。
GPU 构建保留 CPU 风格，便于在同一可执行文件中比较。模型与径向基的限制、显存控制、
设备检查及插件部署方法见 [standalone 说明（英文）](USER-MLIP/standalone/README.md)。

## 4. 元素映射与旧模型

`pair_coeff` 中每个 LAMMPS 原子类型对应一个模型元素，按类型 1、2、…… 的顺序书写，
与原子在数据文件中出现的顺序无关。例如类型 1 为 Ni、类型 2 为 Al：

```text
pair_style mlip/opt
pair_coeff * * model.mtp Ni Al
```

元素名须与模型中声明的元素表匹配。同一条命令不能混用名称与数字，也不接受 `NULL`。
三种原子类型需提供三个条目，例如 `Al Ni Co`；多个原子类型也可映射到同一模型元素。
上述语法同样适用于本页其他 `mlip/...` pair 风格。

旧模型若没有元素元数据，应先从训练设置确认 species 顺序，再使用从 0 开始的数字映射：

```text
pair_coeff * * legacy.mtp 0 1      # 类型 1 -> species 0，类型 2 -> species 1
```

对已确认 species 顺序为 `Al Ni` 的普通 MTP，可显式补写元数据：

```sh
imrmlp adjustmtp legacy.mtp named.mtp --elements Al Ni
```

改用元素名之前需核对预测结果；仅重新保存不能恢复未知的元素身份。
原有声明 `pair_style mlip/opt model.mtp 0 1` 加 `pair_coeff * *` 的方式仍受支持，
模型只声明一次。`fix mlip/maxvol` 使用独立的数字映射语法，见其专用说明。

## 5. 长程模型与原生模型

模型含长程贡献，或需要原生模型组合时，按下表选择。普通短程 MTP 无需配置电荷或
隐变量 compute。

| 模型与需求 | 接口与优势 | 限制与选择依据 |
| --- | --- | --- |
| 线性 MTP 短程头 + Coulomb LES v2 / SOG v3、v4，在 CPU 上运行 | [`mlip/opt-les`（英文）](USER-MLIP/opt-les/README.md)：专用 CPU/MPI 实现，不依赖 `libimrmlp` | 要求三维全周期边界；不支持非线性 MTP-M 头；restart 后需重新声明模型与 compute |
| 长程 MTP-M、原生神经网络、MTP 混合势，或需要长程 GPU | [`mlip/native`（英文）](USER-MLIP/native/README.md)：复用原生模型及其 CPU/GPU 路径 | 链接 MPI `libimrmlp`；保存格式与模型/设备组合必须受支持；不提供 CPU 回退 |

以已通过 CPU 安装脚本安装的 SOG 接口为例：

```text
pair_style mlip/opt-les
pair_coeff * * sog.mtp Ni Al
compute latent all mlip/sog/latent
```

Coulomb LES 使用 `compute q all mlip/les/charge`；SOG 的隐变量无量纲，不能解释为
物理电荷。这些模型已包含其长程贡献，不能用短程接口作等价替代。

在 native 插件中加载 GPM 的示例如下（直接编入 `lmp` 时省略 `plugin load`）：

```text
plugin load /path/to/mlip_native_plugin.so
# 在前述 units、建盒和 read_data 命令之后定义 pair：
pair_style mlip/native device cpu precision fp64 pottype auto
pair_coeff * * model.gpm Al Ni
run 0
```

GPM 可自动推断势类型；旧格式需显式给出 `pottype`，可能还需 `settings` 与配套资源。
电荷/隐变量 compute、GPU 设备分配与总电荷约束按 [native 说明（英文）](USER-MLIP/native/README.md)设置。
GPM 是模型文件格式，转换为 GPM 不会扩大接口的模型支持范围；详见
[GPM 模型说明](../docs/reference/gpm-v1_zh.md)。

## 6. 热流、主动学习与热力学积分

这些功能按任务需要添加，普通 MD 不需要。

| 任务 | 入口与优势 | 适用范围与限制 |
| --- | --- | --- |
| 热导率、全体系热流或静态界面能流 | [`mlip/transport`、`/omp`、`/spline`（英文）](USER-MLIP/transport/README.md)，配合专用 heat compute | CPU/MPI；仅支持线性短程 MTP 及其 spline 形式；不支持 MTP-M、LES/SOG 与 GPU；总热流 compute 不接受 pair hybrid 及额外的 bonded/kspace 能量项 |
| 在 MD 中识别陌生构型并收集候选 | [`fix mlip/maxvol`](USER-MLIP/maxvol/README_zh.md)，可附加在已有 pair 上 | 独立的 MPI `libimrmlp` 插件；需预先生成匹配的主动集；只打分并保存候选，不改变 pair 力，也不生成参考标签 |
| 简谐参考与热力学积分 | [`harmonic/reference` 工作流](../docs/advanced/harmonic-ti_zh.md) | CPU 复制安装已包含；需准备参考结构、力常数与 TI 设置 |

计算热流的最小声明如下，替换普通 `opt` 的 pair 声明：

```text
pair_style mlip/transport
pair_coeff * * model.mtp Al Ni
compute J all mlip/heat/flux
```

该 compute 输出全体系广延热流，体积归一化按具体分析方法处理。
加载 GPM 不会自动启用主动学习；MaxVol 的模型范围、主动集生成与候选输出见其专用说明。
GPU pair 与 CPU MaxVol 打分器混用的效果尚未确认，不建议这样组合。

## 7. 更换模型、续算与安装维护

更换已加载的模型前，应先删除依赖它的观测量 compute，再清空 pair 并重新声明：

```text
pair_style none
pair_style mlip/opt
pair_coeff * * other.mtp Al Ni
```

所需的 compute 随后重新创建。`opt` / `opt/omp` / `gpudirect` 的短程 MTP-M/GPM
restart 仍要求原模型路径可读且内容不变；native 还需保留所引用的模型文件与设置文件。
跨目录续算建议使用绝对模型路径。`opt-les` 续算须重新声明模型与 compute。
各风格的具体 restart 限制见对应说明。

CPU 安装脚本不覆盖已有文件。更新接口时，应先用哈希校验卸载，再安装同一源码版本的文件：

```sh
python3 LAMMPS/USER-MLIP/install.py --uninstall /path/to/lammps/src
```

安装或卸载后都需重新配置并重建 LAMMPS。同一份源码上不要混用本安装脚本与 transport
专用安装脚本。MaxVol 与 native 插件不包含在 CPU 复制安装中。

## 8. 模型元数据参考

模型通过单行 JSON 标签记录元素顺序，例如：

```text
potential_tag = {"schema":"gpml.mtp.metadata","version":1,"elements":["Al","Ni"],"type":"MTP","model":"linear"}
```

schema 名称有意保持共享，未针对 IMR-MLP 改名。MTP-M 使用 `type: "MTP_M"`，
在 `model` 中给出头名称，并给出整数 `q` 与 `model_coeffs_count`；神经头另用
`nn.activation` / `nn.hidden_layers`。LES/SOG 使用 `type: "MTP_LES"` 及
`long_range.kernel` / `long_range.channels`；长程 MTP-M 仍为 `type: "MTP_M"`，
另带独立的 `long_range` 字段。

既有的空标签、文本标签与 `imrmlp_meta_v1:` 标签仍可读取。保存时写入规范 schema，
旧文本保留在 `legacy_tag` 中，扩展字段同样保留；旧的 IMR q 名称/定义与头扩展保留在
`imrmlp_q` 与 `imrmlp_external_model` 中。旧版 GPU 读取器使用的 `# mtp_*` 页脚字段
仍然保留，并与 JSON 做一致性检查。版本非法、键或元素重复、维度不一致以及元素顺序冲突
都会导致失败。元素扩展必须显式给出每个输出元素：
`imrmlp adjustmtp ... --neworder ... --elements ...`。

## 9. 许可

源自 LAMMPS 的适配器保留其原始的 Sandia/LAMMPS 声明，并按
[GPLv2](COPYING.GPL-2.0) 分发，不会被重新许可为本仓库的 MIT 许可。
新增的集成改动使用 IMR-MLP contributors 版权，头文件会标明适用许可。
独立维护的 `runtime/mtp` 核心与安装脚本复制的项目辅助头文件保留其 MIT 声明。
合并后的 LAMMPS 可执行文件在再分发时必须与 GPL 及其他依赖许可保持一致，
详见[第三方许可说明](../THIRD_PARTY_NOTICES.md)。
