# LAMMPS 原生 MaxVol 采样

[English](README.md)

`fix mlip/maxvol` 在 LAMMPS 进程内调用原生 `ActiveSelector`，读取现有 `.sel`
active set，在 MD 中计算完整构型的外推等级并导出候选构型。无需 Python 回调、
中间轨迹文件或每步启动 `imrmlp`。评分与 `imrmlp maxvol_sample` 使用同一套核心。

支持普通 MTP、MTP-M、NN-MTP、LES/SOG 和混合 MTP，类型与
对应命令的 `--pottype` 或 `pot:pottype` 一致。支持 full/stream/online 生成的 active set。
模型、加载设置和 `index.map` 必须与生成 `.sel` 时一致；改变参数后应重新构建
active set。维数检查无法识别同维数但参数不同的模型。

## 构建

先在仓库根目录构建 MPI 库和离线示例所需的 release CLI；MKL 等依赖与主项目一致：

```sh
make -j4 BUILD_MODE=mpi libimrmlp.so
make -j4 imrmlp
cmake -S LAMMPS/USER-MLIP/maxvol -B build/lammps-maxvol-plugin \
  -DLAMMPS_HEADER_DIR=/path/to/lammps/src \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/lammps-maxvol-plugin -j4
```

必须使用目标 LAMMPS 可执行文件对应的源码头文件、MPI 实现和整数尺寸配置；默认
面向标准 SMALLBIG 构建。LAMMPS 需要启用 `PLUGIN` 包。插件复用
`build/mpi/.flags` 的 Eigen、CPU、MKL 和其他 ABI 编译选项。可用
`IMRMLP_LIBRARY`、`IMRMLP_FLAGS_FILE` 指定其他匹配的 **MPI** 构建。
串行 `libimrmlp.so` 含 MPI stubs，不能链接到此 MPI 插件。

运行前将匹配库目录放在 `LD_LIBRARY_PATH` 首位：

```sh
export LD_LIBRARY_PATH="$PWD/build/mpi${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mpiexec -n 4 /path/to/lmp -in in.md
```

也可将 `fix_mlip_maxvol.cpp` 和 `.h` 编入 LAMMPS 源码，使用同样的 C++23、
IMRMLP ABI 选项和 MPI 库；这种方式不需要 `maxvol_plugin.cpp` 或 `plugin load`。

## 输入示例

先用相同模型和物种顺序生成状态，例如：

```sh
build/release/imrmlp maxvol_select --pool train.xyz --potfile model.mtp \
  --dtype mtm --mode nbh --index index.map
```

在已有的 LAMMPS MD 输入中，`create_box` / `read_data` 和 pair 设置之后加入：

```text
plugin load /absolute/path/build/lammps-maxvol-plugin/mlip_maxvol_plugin.so

# units metal；index.map 的两行依次为 Al、Ni，模型物种为 0、1
fix mv all mlip/maxvol 10 model.mtp model.mtp.sel index.map 0 1 &
    save candidates.xyz threshold-save 2.1 threshold-break 11

thermo_style custom step temp pe f_mv f_mv[1] f_mv[2]
run 10000
```

完整语法：

```text
fix ID all mlip/maxvol NEVERY MODEL STATE INDEX TYPE_TO_SPECIES... &
    [pottype TYPE] [settings FILE] &
    [save FILE] [threshold-save VALUE] [threshold-break VALUE|none]
```

每个 LAMMPS atom type 必须有一个从零开始的模型物种索引；单物种传 `0`。
参数中的文件路径相对于 LAMMPS 的工作目录。`save` 默认为 `sampled.xyz`，
两个阈值默认分别为 `2.1`、`11.0`，要求 `0 <= threshold-save <= threshold-break`。
`threshold-break none` 禁用自动停止。只支持 `units metal`、三维、`group all`
和带唯一正 atom ID 的系统；非连续 ID、MPI 空 rank、正交／restricted triclinic
晶胞和周期／非周期边界均可使用。最小化不支持。

| 输出 | 含义 |
| --- | --- |
| `f_ID` | 最近一次评分的全局 MaxVol grade |
| `f_ID[1]` | 当前 fix 生命周期内已保存的构型数 |
| `f_ID[2]` | 最近一次评分的 timestep |

这些值均为 intensive，不随 thermo 的原子数归一化；采样间隔内保留上次结果。
在每次 `run` 的初始化阶段（包括 `run 0`）刷新 grade，以反映两次 run 之间的
坐标或晶胞变化。常规保存只在 timestep 为 `NEVERY` 的正向积分采样步发生，
避免连续 `run` 在交界步重复保存。

grade **严格大于** `threshold-save` 时追加一个 extxyz 构型，其中包含
`MV_grade`、`timestep`、晶胞、PBC 和物种／坐标。原子按 ID 排序，坐标减去
LAMMPS boxlo，得到以零为原点的等价晶胞。输出不带预测 EFS 标签，适合送往 DFT
重新标注。
所有输出在每个采样步 flush；已有文件始终追加。

grade **严格大于** `threshold-break` 时，先保存并 flush 触发构型，再用 LAMMPS
全局错误退出。初始化阶段也执行此检查和保存，因此高风险初始构型不会漏掉。
需要柔性停止时可设置 `threshold-break none`，再通过 LAMMPS `fix halt` 使用
`f_ID`。`.sel` 始终只读；采样不扩充 active set。外推等级取决于模型和主动集，不能直接当作预测误差。

## 势函数类型与设置

### cfg 与 nbh 模式

`dtype mtm` 可在生成 `.sel` 时通过 `--mode cfg` 或 `--mode nbh`（或
`builder:mode cfg/nbh`）选择。`cfg` 将各原子的 MTP 描述符求和作为构型行；
`nbh` 为每个原子邻域生成一行，构型 grade 取这些行的最大值。fix 自动恢复
状态中保存的模式，不能在 LAMMPS 输入中直接切换；切换后须重新生成 `.sel`。

`dtype pot_efs` 不使用 `mode` 切换行类型，而由 `w-energy`、`w-site`、
`w-force`、`w-stress` 决定。只启用 `w-energy` 对应完整构型能量梯度；
对支持 site 梯度的局域势，只启用 `w-site` 对应逐原子邻域评分。可组合多种行，
构型 grade 取所有行的最大值。长程势目前只支持完整构型的 E/F/S 梯度，
`mtm/nbh` 对这类模型仅衡量 MTP 基函数，不包含长程参数的外推信息。

### 模型加载

| `pottype` | `MODEL` | MAXVOL 行 |
| --- | --- | --- |
| `mtp` | 普通 MTP 文件 | `mtm` 或 `pot_efs`，支持 linear/all-coeffs 和 site 行 |
| `mtp_m` | MTP-M 文件 | `pot_efs` 完整参数梯度；`mtm` 仅描述 MTP 基函数 |
| `mtpnn1` | 文件前缀，对应 `.mtp`、`.nn`、`.sc` | `pot_efs`；自动读取 NN1/NN2 层数，支持 site 行 |
| `mtp_les` | LES 或 SOG 模型文件 | `pot_efs` 包含长程参数；`mtm` 仅描述短程基函数 |
| `hmtp` | 文件前缀，对应 `_sub1`、`_sub2`、可选 `_sub3` | `pot_efs`，合并各子势的参数梯度 |

除普通 MTP 外，`pot_efs` 始终使用模型的完整可训练参数布局，避免普通 MTP 的
线性快捷路径遗漏非线性或长程项；`pot` 中的参数冻结设置仍然生效。以前用线性
快捷路径为派生势生成的 `.sel` 需要重新生成。LES/SOG、带长程项的 MTP-M、
`hmtp` 只支持能量／力／应力行，要求 `w-site 0`；普通 MTP-M
支持 site 行。复合 NN-MTP、`hmtp` 不接受 `mtm` 状态。

`settings FILE` 读取与离线命令相同的设置文件，使用其中 `pot:` 子设置加载势。
类型优先级为 fix 的 `pottype`、`pot:pottype`、`builder:pottype`、默认 `mtp`。
位置参数 `MODEL` 和 `INDEX` 始终覆盖设置中的模型路径和类型映射；其他文件路径
相对于 LAMMPS 工作目录。评分行权重和归一化方式来自 `.sel`，不能通过此处的
`builder:` 设置改写。

例如 MTP-M 使用以下 `select.ini`，其中 `extmodel`、神经网络尺寸和参数冻结
选项应与训练时一致：

```text
builder:dtype pot_efs
builder:pottype mtp_m
builder:potfile model.mtp
builder:w-energy 1
builder:w-force 0.01
builder:w-stress 0.001
builder:w-site 0
pot:extmodel mtp_m_n2
pot:n_input 8
pot:n_neuron 16
pot:n_neuron2 8
```

```sh
build/release/imrmlp maxvol_select --settings select.ini --pool train.xyz --index index.map
```

```text
fix mv all mlip/maxvol 10 model.mtp model.mtp.sel index.map 0 1 &
    settings select.ini save candidates.xyz
```

MTP-M 的 `extmodel` 支持 `mtp_m_fs`、`mtp_m_q2`、`mtp_m_li`、`mtp_m_n1`、
`mtp_m_n2`（或 `mtp_m_nn2`），也支持模型中的 LES/SOG 长程项。
带长程项时在 `pot:` 中保留训练时的 `lr_les true` 等设置。
`hmtp` 在 `pot:potlevel` 中指定 `2` 或 `3`（默认 `2`）。

长程模型的周期边界限制与 IMRMLP 核心一致，不支持的 PBC 组合会报告全局错误。
完整参数及力／应力行通常比普通 MTP 线性评分更昂贵，建议根据体系大小调整采样间隔。

## MPI 与运行限制

MPI 各 rank 只发送 owned atoms；rank 0 排序并构造完整构型、计算 grade、写文件，
再广播结果。加载／评分／写文件失败会在下一个集合通信之前同步为全局错误。
这种实现支持与离线一致的 cfg、力和应力行，但描述符计算和全局构型内存仍在
rank 0，增加 MPI 进程数不会加速评分本身。可通过 `NEVERY` 控制开销。

Fix 不提供力、不修改现有 pair style 或积分器，可与独立安装的动力学 pair style
配合使用。回归用 `pair_style zero` 隔离验证评分；GPU／Kokkos pair 组合未在本回归中
验证。评分在 CPU 上独立进行。
当前不保存 LAMMPS restart 内部状态：`read_restart` 后重新声明 fix，继续读取
相同 `.sel`，候选文件仍追加，计数从零开始。同一输出文件不可由多个 fix 或
LAMMPS partition 同时写入。

## 回归测试

构建 release `imrmlp`，并为 CMake 提供带 PLUGIN 的 MPI LAMMPS：

```sh
make -j4 imrmlp
cmake -S LAMMPS/USER-MLIP/maxvol -B build/lammps-maxvol-plugin \
  -DLAMMPS_HEADER_DIR=/path/to/lammps/src \
  -DLAMMPS_EXECUTABLE=/path/to/lmp
cmake --build build/lammps-maxvol-plugin -j4
ctest --test-dir build/lammps-maxvol-plugin --output-on-failure
```

测试对照离线 `maxvol_sample`，覆盖 1/2/4 ranks（含空 rank）、反向类型映射、
非连续 ID、非零 boxlo、正交／三斜盒、PBC、描述符／EFS 权重、采样频率、
只读状态、追加输出、同一步几何变更、停止前保存和 MPI 统一错误处理。
类型回归动态生成小型确定性模型，覆盖 NN1/NN2、五种 MTP-M、LES/SOG、
MTP-M + LES/SOG、两层／三层混合 MTP，以及类型别名
和不兼容状态报错。核心 `ut_maxvol_mtp_family` 用参数数值差分独立校验 E/F/S
梯度。LAMMPS 回归使用 22 Jul 2025 Update 5 的 MPI/PLUGIN 构建。
