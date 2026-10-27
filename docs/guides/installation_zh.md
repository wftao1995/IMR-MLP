# 安装与构建 {#guide_install}

[English](installation.md) · [开始使用](getting-started_zh.md)

本页先带你获得可运行的 CPU 命令行程序，再按需要安装 Python、MPI 或 GPU 版本。
不修改 C++ 源码也可以使用 IMR-MLP。当前提供的 Make 构建面向 Linux x86_64；
Windows/macOS 用户可使用 Linux 计算机或相应 Linux 环境，本文不提供原生安装流程。

## 先选需要安装的部分

| 想做什么 | 选择 | 暂时不需要 |
| --- | --- | --- |
| 求能量/力、弛豫、NEB，或训练普通 MTP | CPU CLI：按下面步骤安装 | Python 扩展、MPI、GPU、文档编译工具 |
| 已有管理员提供的可执行程序 | 先运行 `imrmlp --version`，再做[入门示例](../../examples/quickstart/README_zh.md) | 自行重编译 |
| 在 Python / ASE 里计算 | 先准备本地依赖，再安装 Python 包 | LAMMPS |
| 用已有短程模型做 LAMMPS MD | 按 [LAMMPS 接口](../../LAMMPS/README_zh.md)的 CPU / standalone 路线 | 完整训练库通常不必安装 |
| 多进程或 GPU 计算 | 先把 CPU 示例跑通，再看后面的模式表 | 首次安装时配置全部后端 |

命令中的 `/path/to/...` 是要替换的路径，不是原样执行的目录。
系统依赖通常只安装一次；每次新开终端还需要启用相应的运行环境。

## 1. 准备依赖

核心 CPU 程序需要 C++23 编译器及配套标准库、GNU Make 4.3+、Eigen、oneMKL、
ALGLIB、FFTW、OpenMP 和默认启用的 zlib。下面的基础路线关闭可选 OpenKIM，
避免首次安装额外准备 KIM API。

### Ubuntu / Debian 上的系统开发包

有管理员权限时，可先安装常用开发包：

```sh
sudo apt update
sudo apt install build-essential git python3 python3-venv \
  libeigen3-dev libalglib-dev libfftw3-dev zlib1g-dev
```

这些是发行版包名示例，并不意味着该系统默认编译器已经通过本项目验证。
安装的是开发头文件和库；仅有运行库不够。ALGLIB 包名可在
[Ubuntu 软件包目录](https://packages.ubuntu.com/noble/libalglib-dev)核对。
其他 Linux 发行版请使用对应包管理器；集群上先询问管理员是否已有编译器和依赖模块，
无需为了使用程序修改系统目录。

### oneMKL：安装开发包，再启用环境

按 [Intel 官方安装说明](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html)
安装 oneMKL **开发包**。已配置 Intel APT 软件源的机器可使用
`sudo apt install intel-oneapi-mkl-devel`；仅运行库包不能替代开发包。
无需为本项目使用 oneMKL 而把 C++ 编译器换成 Intel 编译器。

常见组件目录布局可在当前终端运行：

```sh
source /opt/intel/oneapi/setvars.sh
```

私有安装需替换安装目录；统一目录布局使用对应版本的 `oneapi-vars.sh`。
具体路径见 [Intel 环境设置说明](https://www.intel.com/content/www/us/en/docs/oneapi-hpc-toolkit/get-started-guide-linux/2025-0/before-you-begin.html)。
检查 `MKLROOT` 是否指向包含 `include/mkl.h` 的目录：

```sh
echo "$MKLROOT"
test -f "$MKLROOT/include/mkl.h" && echo "MKL headers found"
g++ --version
make --version
```

若自行设置 `MKLROOT=/实际的/mkl/目录`，还需配置 oneMKL 的运行库搜索路径；
仅设置 `MKLROOT` 不会自动启用运行环境。
源码采用 `-std=c++23`。当前源码的 CPU 验证环境使用 GCC 16.2.1；尚未据此确定
所有旧版编译器的最低兼容版本。旧系统若报标准库特性或编译选项错误，应使用
支持所需 C++23 特性的编译器与标准库组合，不要只去掉标准选项。

## 2. 获取源码并构建 CPU 程序

已有源码时直接进入仓库根目录；尚未下载时：

```sh
git clone https://github.com/wftao1995/IMR-MLP.git
cd IMR-MLP
```

确认当前目录有 `Makefile`，然后构建：

```sh
make -j2 imrmlp USE_KIM=0
```

`-j2` 表示同时运行两个编译任务，内存充足时可提高；它不是程序计算时的线程数。
产物在 `build/release/`，不是源码根目录。不要运行 `sudo make`。

将当前构建加入本终端的程序与库搜索路径，再确认能够启动：

```sh
export PATH="$PWD/build/release:$PATH"
export LD_LIBRARY_PATH="$PWD/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
imrmlp --version
imrmlp --help
```

`PATH` 让终端找到 `imrmlp`，`LD_LIBRARY_PATH` 让它优先找到同一构建的共享库。
这些设置中的 `$PWD` 此时必须是仓库根目录；之后可以切换到自己的计算目录。
新开终端时重新启用 oneMKL，并用仓库的绝对路径设置这两项。

## 3. 跑通随仓库提供的示例

仍在仓库根目录运行：

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 examples/quickstart/run.py \
  --work build/quickstart
```

示例只需要 Python 标准库和刚编好的 CLI，会自行生成合成模型与数据，完成拟合和
验证。各阶段应显示 `OK`，输出在 `build/quickstart/`，与后续页面的默认路径一致。再次运行请换一个新的 `--work`
目录，脚本会拒绝覆盖非空目录。示例用途及结果解释见[快速入门](../../examples/quickstart/README_zh.md)。

完成后，先按[开始使用](getting-started_zh.md)中的入门数据练习查看单点输出；
无需重跑同一目录的训练脚本。还没有自己的计算输入时，可继续
[附带全部输入的连续计算练习](../../examples/basic-calculations/README_zh.md)，完成弛豫、NEB 和物性。
已有适用模型时，再按各任务页准备自己的结构与映射。
测试和文档构建工具不属于运行该示例的前提。

## 4. 可选：安装到用户目录

也可以一直从构建目录运行。若希望在任意工作目录直接使用，保持相同构建选项：

```sh
make -j2 install USE_KIM=0 PREFIX="$HOME/.local"
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
imrmlp --version
```

默认安装到个人目录，无需管理员权限。oneMKL 等外部运行库仍须可用。
Make 安装目标安装 CLI 与库，不安装 Python 包。迁移到其他机器时，除了程序与库，
还需兼容的系统依赖；跨 CPU 分发前应重新选择 `CPU_ARCH_FLAGS`。

## 5. 按需添加 Python、MPI 或 GPU

### Python / ASE

Python 包要求 Python 3.10+，还需 CMake 和 Python 开发头文件
（Ubuntu / Debian 可安装 `cmake python3-dev`）。在仓库根目录创建并激活虚拟环境：

```sh
python3 -m venv .venv
source .venv/bin/activate
USE_KIM=0 python -m pip install '.[ase]'
python -c 'import imrmlp; print(imrmlp.__version__)'
```

本地 C++ 依赖必须已准备好；pip 不会提供 oneMKL、ALGLIB 或 FFTW。
安装会通过 CMake 调用源码目录中的 Make release 构建，机器还需可用的 CMake。
命令沿用基础路线的 `USE_KIM=0`。无需 ASE 时将 `'.[ase]'` 换成 `.`。
目录外导入、元素映射和计算器示例见
[Python 指南](../../README-python_zh.md)。
源码目录内开发时也可 `python3 -m pip install numpy pybind11` 后执行
`make -j2 python PYTHON=python3 USE_KIM=0`，但新用户通常只需选择一种安装方式。

### MPI 与 GPU 模式

每个模式使用独立构建目录；主程序名和运行库应来自同一模式。
MPI 需要匹配的 `mpicxx`、启动器和运行库；GPU 需要对应 CUDA/HIP 工具链。

| `BUILD_MODE` | 目录中的主程序 | 用途 |
| --- | --- | --- |
| `release`（默认） | `imrmlp` | 普通 CPU 计算 |
| `mpi` | `imrmlp-mpi` | MPI 多进程；NEB 可分配图像给不同 rank |
| `cuda` | `imrmlp-cu` | NVIDIA CUDA 支持的模型与命令 |
| `hip` | `imrmlp-hip` | AMD HIP / DCU 支持的模型与命令 |
| `mpi-cuda`、`mpi-hip` | `imrmlp-mpi` | MPI 与对应 GPU 后端组合 |

例如构建 MPI，并运行一份已准备好的 NEB 配置：

```sh
make -j2 BUILD_MODE=mpi USE_KIM=0 imrmlp
imrmlp_root="$PWD"
cd /path/to/neb-run
LD_LIBRARY_PATH="$imrmlp_root/build/mpi${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  mpirun -np 4 "$imrmlp_root/build/mpi/imrmlp-mpi" neb neb.ini --num_threads 1
```

配置中的相对文件仍从启动目录解析，因此实际任务通常应在其输入目录启动，并使用
程序和库的绝对路径。MPI rank 与线程的规划见 [NEB 引导](neb_zh.md)。
GPU 架构设置和支持范围见 [GPU 指南](../development/cuda_zh.md)。
仅构建 GPU 版本不会让所有命令自动使用 GPU；例如 `trainmtp` 仍在 CPU 运行，
受支持的 GPU 拟合入口是 `trainnlpot`。

## 安装卡住时先查这里

| 症状 | 先检查 |
| --- | --- |
| `g++` / `make` 找不到，或不认识 `-std=c++23` | 编译器与 GNU Make 是否安装并在 `PATH` 中 |
| `Eigen/...`、`mkl.h`、`libalglib/...` 或 `fftw3.h` 缺失 | 是否安装开发包；Eigen、oneMKL、ALGLIB 路径是否匹配 |
| 链接报 `cannot find -l...` | 对应开发库和链接目录是否存在，是否初始化 oneMKL 环境 |
| 启动报动态库找不到 / undefined symbol | 程序与库来自同一构建，当前构建目录在库路径最前 |
| `cc1plus` 被杀死 | 降到 `-j1` 或 `-j2`，确认机器内存 |
| pip 构建失败 | 先确认原生依赖齐全，再看 pip 日志中的第一条实际编译/链接错误 |

更多按症状解释见[故障排查](troubleshooting_zh.md)。

## 构建参数与维护资料（按需查阅）

普通安装不需要修改 Makefile。非系统依赖路径可在命令行指定：

```sh
make -j2 imrmlp USE_KIM=0 EIGEN_DIR=/path/to/eigen3 \
  ALGLIB_INCLUDE_ROOT=/path/to/include \
  ALGLIB_LIBS='-L/path/to/lib -lalglib'
```

`ALGLIB_INCLUDE_ROOT` 是包含 `libalglib/` 的父目录。fmt、spglib、TinyXML-2、
Voro++ 和 LBFGS++ 随仓库提供；ALGLIB、FFTW 和 oneMKL 为外部依赖。

| 选项 / 目标 | 何时需要 |
| --- | --- |
| `USE_KIM=1` | 使用 OpenKIM；需 KIM API，Make 默认开启 |
| `USE_GZIP=0` | 关闭默认 gzip 支持，否则需要 zlib |
| `USE_ZSTD=1` | 读取压缩 GPM 数据块，另需 libzstd |
| `MKL_LIBDIR` | 覆盖 oneMKL 库目录，默认从 `MKLROOT` 查找 |
| `CPU_ARCH_FLAGS=` | 取消默认 `-march=native`，或设置实际目标 CPU 参数 |
| `BUILD_MODE=debug`、`asan`、`relwithdebinfo`、`relwithdebinfo-asan`、`mpi-debug` | 调试与错误检查；常规用户使用 release/mpi |
| `make all` | 库和全部 CLI/辅助程序 |
| `make main` | 不依赖 `libimrmlp` 的主程序；仍可依赖外部动态库 |
| `make clean-<mode>` | 清理指定模式；不要清理存放了计算结果的目录 |

`ccache` 安装后会自动使用。`make test` 需要 Catch2 / Catch2Main，运行既有测试时应
把匹配构建目录放在 `LD_LIBRARY_PATH` 最前。完整测试、其他维护目标和第三方许可见
[贡献说明](../../CONTRIBUTING.md)、[许可说明](../../THIRD_PARTY_NOTICES.md)。
阅读手册不需要 Doxygen；自行构建手册见[文档构建说明](../README.md)。
