# 将 native 推理编入标准 LAMMPS 可执行文件

[English](README.md)

此入口使用上游 LAMMPS 自己的 `lmp` 和 `main.cpp`，通过标准 style 工厂注册
`mlip/native`、`mlip/native/charge` 和 `mlip/native/latent`。无需 `plugin load`、
PLUGIN 包或自定义启动程序，也不修改上游源码树。
现有 [plugin 构建入口](../README.md) 仍可使用。

## CPU 构建

在 IMR-MLP 仓库根目录运行，使用干净的上游 LAMMPS 源码、C++23，
并保持 MPI 实现及 ABI 与 IMR-MLP 一致：

```sh
make BUILD_MODE=mpi -j2 libimrmlp.so
cmake -S LAMMPS/USER-MLIP/native/lammps -B build/lmp-native-cpu \
  -DLAMMPS_SOURCE_DIR=/path/to/lammps \
  -DIMRMLP_NATIVE_BACKEND=CPU -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MPI=ON -DPKG_PLUGIN=OFF
cmake --build build/lmp-native-cpu -j2
LD_LIBRARY_PATH="$PWD/build/mpi:$LD_LIBRARY_PATH" \
  mpiexec -n 4 build/lmp-native-cpu/lmp -in in.lammps
```

`LAMMPS_SOURCE_DIR` 接受上游根目录或 `src` 目录。可继续使用 `PKG_*`、
`BUILD_SHARED_LIBS`、`LAMMPS_SIZES` 和 `CMAKE_INSTALL_PREFIX` 等标准 LAMMPS
CMake 选项。即使只运行一个进程，native 仍需要 MPI IMR-MLP 库，因此拒绝
`BUILD_MPI=OFF`；也拒绝已安装同名 native style 的源码树。
`cmake --install build/lmp-native-cpu` 安装标准目标；运行时仍需保留 IMR-MLP 及其依赖。

## CUDA 和 HIP 构建

先构建 IMR-MLP 设备内核。按目标机器替换下面的工具链路径及 GPU 架构；
每个后端使用独立的 LAMMPS 构建目录：

```sh
make BUILD_MODE=mpi-cuda CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_90 -j2 libimrmlp.so
cmake -S LAMMPS/USER-MLIP/native/lammps -B build/lmp-native-cuda \
  -DLAMMPS_SOURCE_DIR=/path/to/lammps \
  -DIMRMLP_NATIVE_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MPI=ON -DPKG_PLUGIN=OFF
cmake --build build/lmp-native-cuda -j2

make BUILD_MODE=mpi-hip HIP_ROOT=/opt/rocm HIP_ARCH=gfx90a -j2 libimrmlp.so
cmake -S LAMMPS/USER-MLIP/native/lammps -B build/lmp-native-hip \
  -DLAMMPS_SOURCE_DIR=/path/to/lammps \
  -DIMRMLP_NATIVE_BACKEND=HIP -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MPI=ON -DPKG_PLUGIN=OFF
cmake --build build/lmp-native-hip -j2
```

默认分别选择 `build/mpi/libimrmlp-mpi.so`、`build/mpi-cuda/libimrmlp-mpi.so`、
`build/mpi-hip/libimrmlp-mpi.so` 及对应 `.flags`。外部构建需同时指定
`IMRMLP_LIBRARY` 和 `IMRMLP_FLAGS_FILE`。配置阶段拒绝后端与标志冲突、使用串行
IMR-MLP 库等情况。IMR-MLP ABI 标志只应用于 native 源文件，上游 LAMMPS 保持自己的标志。

设备内核已在 IMR-MLP 中构建，此入口只编译 C++，无需另外启用 CUDA/HIP 语言，
也无需 LAMMPS GPU/KOKKOS 包。运行 `lmp` 时仍需对应 GPU runtime 和 IMR-MLP 依赖。
GPU 构建也可在 CPU 上执行。模型与后端的支持范围、对不支持请求的显式报错
均与 plugin 一致。

## 输入及格式

```lammps
units metal
atom_style atomic
newton on
read_data structure.data
# 为所有原子类型设置质量。
pair_style mlip/native device cpu precision fp64 pottype auto
pair_coeff * * model.gpm Al Ni
run 0
```

设备计算将 `device cpu` 换成 `device cuda gpu_id auto` 或 `device hip gpu_id auto`。
auto 模式要求每个节点内 MPI rank 使用不同设备；显式 `gpu_id N` 允许有意共享设备进行诊断。

上例假设两个 LAMMPS 原子类型依次为 Al、Ni，且模型包含对应元素元数据；
请替换成自己的模型及元素顺序。`pair_coeff` 需要为每个 LAMMPS 原子类型给出
一个模型已知的元素符号。旧模型没有元素元数据时，改用零基模型物种索引
（例如 `0 1`）；不能混用符号与数字，也不能使用 `NULL`。

GPM 自动识别模型类型并内嵌资源；旧输入必须显式指定 `pottype`（如 `mtp`），
需要额外模型设置时使用 `settings`。普通短程模型不提供 charge/latent 观测量。
只有 Coulomb LES 可定义 `compute q all mlip/native/charge`；
只有启用 SOG 的模型可定义 `compute latent all mlip/native/latent`。
应在 `pair_style` 和 `pair_coeff` 之后定义相应 compute，更换模型后重新创建。
观测量前提及模型/GPU 支持范围见 [plugin 说明](../README.md)。restart 行为也与 plugin 一致；
恢复 restart 时需保留引用的模型和设置文件。加载 GPM 不会激活 AL。
详见 [GPM 规范](../../../../docs/reference/gpm-v1_zh.md)。

## 验证

```sh
LD_LIBRARY_PATH="$PWD/build/mpi:$LD_LIBRARY_PATH" \
  ctest --test-dir build/lmp-native-cpu --output-on-failure

python LAMMPS/USER-MLIP/native/tests/run_regression.py \
  --lmp build/lmp-native-cpu/lmp \
  --generator build/lmp-native-cpu/generate_native_test_models \
  --gpm --ranks 1 2 4 --work build/native-gpm-regression
```

CTest 覆盖几何、旧格式及 GPM 家族计算对照、restart、charge/latent 输出、NVE、
工作区复用和配置约束。配置检查使用合成 CUDA/HIP 标志文件验证后端选择，无需 GPU；
这不代表设备编译或运行验证。内置 style 的运行脚本省略 `--plugin`，提供该参数仍可测试 plugin。

即使链接 GPU 库，测试也默认使用 CPU。已分配设备时设置
`IMRMLP_NATIVE_TEST_DEVICE=cuda` 或 `hip`，并把对应 IMR-MLP 库目录放入
`LD_LIBRARY_PATH`。完整 MPI 测试最多需要四张卡。单卡可在上面的脚本命令追加
`--device cuda --ranks 1` 或 `--device hip --ranks 1`。
`IMRMLP_NATIVE_TEST_TIMEOUT` 控制各运行测试的超时秒数；`BUILD_TESTING=OFF` 关闭测试目标及 Python 查找。
