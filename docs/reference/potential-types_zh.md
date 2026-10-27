# 势类型与文件格式 {#reference_potential_types}

[English](potential-types.md)

势工厂根据 `pot:pottype`（部分命令使用扁平键 `pottype`）分派类型。
普通势文件的类型为空或 `auto` 时选择 `mtp`；GPM 从内嵌元数据推断类型。下表类型全部注册在
`potentials/potential_wrapper.cpp`；表格给出每种类型所需的输入文件，
以及已有专页时的链接。

## 类型选择表

| `pottype` | 家族 | 输入文件 | 说明 |
|---|---|---|---|
| `mtp` | 矩张量势 | `.mtp` 文本文件 | 默认类型；理论见 @ref theory_mtp |
| `mtpnn1` | 神经网络 MTP | `.mtp` 模板 + `.nn` 权重 | 用 `nnmtp` 生成模板，用 `trainnlpot` 训练 |
| `mtp_m` | 多通道 MTP 与可选能量头 | 含通道和能量头参数的 `.mtp` | 用 `initmtpm` 初始化；见 [MTP_M 理论](../theory/mtp-m_zh.md) |
| `hmtp` | 双分量混合 MTP | `pot:potfile` 给出源 MTP | 设置 `pot:potlevel`；分量读取 `source_sub1`/`source_sub2` |
| `shellfile` | 外部 shell 提供者 | 提供者配置 | 设置位于 `shellfile:`，见 shell 提供者小节 |
| `eamalloy` | EAM 合金势 | EAM alloy setfl 表 | 物种顺序跟随势文件 |
| `eamfs` | Finnis–Sinclair EAM | EAM FS 表 | — |
| `eampoly` | 多项式 EAM | EAM 参数文件 | 混合势示例中使用 |
| `openkim` | OpenKIM 模型 | `pot:kim_model` 中的模型名 | 需要 `USE_KIM` 构建 |
| `acelmp` | ACE/LAMMPS 势 | ACE 模型 | 需要 `WITH_ACELMP`；当前 Makefile 未暴露开关，需在 `CXXFLAGS` 中手动启用 |
| `mindist` | 最小距离惩罚 | 无 | 参数为 `pot:cutoff`、`pot:factor`、`pot:power` |
| `lj612` | Lennard–Jones 6-12 | 无 | 参数来自势配置段 |
| `zbl` | Ziegler–Biersack–Littmark | 无 | 通常作为 `hybrid` 的固定核心 |
| `mtp_les` | MTP + 长程静电 | 启用 LES 的 `.mtp` | 见 [MTP+LES v2](les-v2_zh.md) 与 [SOG](sog_zh.md) |
| `descriptor_linear` | 线性描述符势 | 描述符设置 | 用 `train_linear` 训练 |
| `pair_table` | 表格化 pair 势 | `pot2table` 写出的表 | pair 表包含 `-dE/dr` |
| `threebody_table` | 表格化三体势 | `pot2table` 写出的表 | 等键长采样；见 `pot2table` |
| `sw` | Stillinger–Weber | SW 参数文件 | — |
| `tersoff` | Tersoff | Tersoff 参数文件 | — |
| `hybrid` | 子势求和 | `pot1:`、`pot2:` 等配置 | 见 @ref guide_config |
| `hybrid_scale` | 平滑切换求和的子势 | `pot1:`、`pot2:` 及切换键 | 常用于 MTP + ZBL 核心 |

## 最小示例

下列各段是互相独立的势配置，选其中一段写入设置文件。

```text
# MTP
[pot]
pottype mtp
potfile potential.mtp

# OpenKIM
[pot]
pottype openkim
kim_model  Sim_LAMMPS_...

# MTP + ZBL 核心
[pot]
pottype hybrid
pot1:pottype mtp
pot1:potfile potential.mtp
pot2:pottype zbl
pot2:z 6 13
pot2:r0 0.8
pot2:r1 2.0
```

```bash
# 将训练好的 MTP 转换为 LAMMPS 样条表
imrmlp mtp2spline potential.mtp spline_output

# 查看内嵌有效势的 GPM 包
imrmlp model_info --input model.gpm
```

## shell 提供者（`shellfile`）

`shellfile` 把"能读结构并给出能量、力和应力"的程序变成一个势：DFT 程序、
其他包的模型，或者把它们包起来的脚本都可以。IMR-MLP 把结构写进作业目录，
在那里运行脚本，再读回结果——程序的其余部分看到的接口与 `mtp` 的 EFS 完全
一致。

### 设置项

| 键 | 默认值 | 含义 |
| --- | --- | --- |
| `shellfile:runscript` | — | 主脚本，必须在作业目录中产出 `read_outfile` |
| `shellfile:prescript` | 空 | 写入结构后执行，例如准备 INCAR/KPOINTS/POTCAR |
| `shellfile:postscript` | 空 | 读取结果后执行，例如清理临时文件 |
| `shellfile:index` | `index.map` | 元素映射文件 |
| `shellfile:write_posfile` | `POSCAR` | 写入作业目录的结构文件名 |
| `shellfile:write_posformat` | `POSCAR` | 结构格式（`poscar`、`xsf`、`cif`、`extxyz` 等） |
| `shellfile:read_outfile` | `output.cfg` | 从作业目录读取的结果文件名 |
| `shellfile:read_outformat` | `mlpcfg` | 结果格式（`extxyz`、`mlpcfg`、`poscar`、`xsf`、`cif` 等） |
| `shellfile:read_energy` | `false` | 改为从 `engfile` 读取总能量，而不是从结果文件读 |
| `shellfile:engfile` | `eng.dat` | 能量文件，其第一个数字为总能量（eV） |

### 脚本如何被调用

- 每个脚本按 `bash 脚本 作业目录绝对路径` 调用。`runscript` 必需，
  `prescript` 与 `postscript` 可选；任一脚本非零退出即中止本次计算。
- 顺序：写入结构 → `prescript` → `runscript` → 读取结果 → `postscript`。
- 脚本仍在提交目录中运行，因此继续使用自己的相对输入（POTCAR、
  可执行程序等），并通过作业目录参数访问结构与结果。
- 读取结果时取 `read_outfile` 的最后一帧。
- `index.map` 每行一个元素名，按 species 顺序排列——第一行为 species 0。
  文件缺失会打印警告并保留默认映射。
- 作业目录会被保留：同步计算使用 `run-job-omp-thread-<n>`（MPI 构建再追加
  `mpi-rank-<r>`），异步计算使用 `run-job-async-XXXXXX`。脚本出问题时直接
  打开对应目录——里面有 IMR-MLP 写下的全部内容和脚本留下的全部产物。

### 案例 1：用自包含脚本先跑通流程

先把计算器换成一个"把结构原样返回"的脚本。它能跑通，就说明路径、格式和
调用协议都是对的，接下来只需要换上真正的程序。

```text
work/
├── index.map
├── run.sh
└── frame.xyz                     # 待计算的结构
```

`index.map`——每行一个元素，按 species 顺序：

```text
Al
Ni
```

`run.sh`——唯一参数是作业目录：

```bash
#!/bin/bash
set -e
cp "$1/input.xyz" "$1/output.xyz"
```

IMR-MLP 写给脚本的 `input.xyz`：

```text
2
Lattice="10 0 0 0 10 0 0 0 10" pbc="T T T" Properties=species:S:1:pos:R:3
Al 0 0 0
Ni 2 0 0
```

元素名来自 `index.map`，因此计算器看到 species 0 是 Al、species 1 是 Ni。

```python
import numpy as np
from imrmlp import Atoms, Potential

potential = Potential.from_settings({
    'pottype': 'shellfile',
    'shellfile:index': 'index.map',
    'shellfile:runscript': 'run.sh',
    'shellfile:write_posfile': 'input.xyz',
    'shellfile:write_posformat': 'extxyz',
    'shellfile:read_outfile': 'output.xyz',
    'shellfile:read_outformat': 'extxyz',
})

atoms = Atoms(numbers=[13, 28], elements=['Al', 'Ni'],
              positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], cell=np.eye(3) * 10)
potential.calculate(atoms)
print(atoms.energy, atoms.forces, atoms.stress)

job = potential.submit(atoms)     # 同一次计算，但不阻塞
print(job.run_directory)          # 打开它：input.xyz 与 output.xyz 都在里面
print(job.result().energy)
```

回显脚本不写能量和力，因此这个案例只用于验证路径与格式：在真正写入结果
之前，`atoms.energy`、`atoms.forces` 与 `atoms.stress` 均为 `None`。

### 案例 2：用配置文件驱动 VASP

`shell.ini`：

```text
[pot]
pottype shellfile
shellfile:index index.map
shellfile:prescript prepare.sh
shellfile:runscript run_vasp.sh
shellfile:postscript clean.sh
shellfile:write_posfile POSCAR
shellfile:write_posformat poscar
shellfile:read_outfile result.xyz
shellfile:read_outformat extxyz
```

`prepare.sh`——结构之间不变的输入：

```bash
#!/bin/bash
set -e
cp INCAR KPOINTS POTCAR "$1"
```

`run_vasp.sh`——先跑程序，再转换输出：

```bash
#!/bin/bash
set -e
cd "$1"
mpirun -np 4 vasp_std
python3 /opt/tools/vasp2xyz.py vasprun.xml result.xyz
```

`clean.sh`——只留下还要看的东西：

```bash
#!/bin/bash
set -e
rm -f "$1"/CHG* "$1"/WAVECAR "$1"/vasprun.xml
```

像其他势一样使用：

```bash
imrmlp calc_efs --settings=shell.ini --input=frame.xyz --output=out.cfg
```

结果帧中携带 IMR-MLP 需要的数值——此处为 `extxyz`：

```text
2
Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" Properties=species:S:1:pos:R:3:forces:R:3 energy=-6.42 virial="0.12 0 0 0 0.12 0 0 0 0.12" pbc="T T T"
Al 0.0 0.0 0.0 0.02 0.0 0.0
Ni 2.0 0.0 0.0 -0.02 0.0 0.0
```

`virial` 按原值使用；`stress` 视为密度并按晶胞体积换算。

### 案例 3：只输出能量的程序

力和应力仍由结果文件给出，能量改从单独的文件读取：

```text
shellfile:read_outfile result.xyz
shellfile:read_outformat extxyz
shellfile:read_energy true
shellfile:engfile eng.dat
```

`eng.dat`——第一个数字是总能量（eV），其余被忽略：

```text
-6.4218   eV
```

`runscript` 返回前该能量文件必须存在；文件缺失或无法解析会中止本次计算。

### 案例 4：打包成单个 GPM 文件

先声明脚本需要的东西，再导出——脚本与资源都会随包带走：

```text
shellfile:dependencies vasp vaspkit
shellfile:resource:POTCAR/Al /opt/potentials/Al/POTCAR
shellfile:resource:POTCAR/Ni /opt/potentials/Ni/POTCAR
```

```bash
imrmlp model_convert --settings=shell.ini --output=deploy.gpm
```

接收方只需要在 `PATH` 中提供已声明的依赖。打包的文件与展开后的脚本在同一
位置，因此按脚本自身位置解析，而不是按工作目录：

```bash
#!/bin/bash
set -e
here="$(cd "$(dirname "$0")" && pwd)"
cp "$here/POTCAR/Al" "$1/POTCAR"
```

导出前必须声明依赖；资源必须是普通文件，其相对名不得逃出作业目录。
展开出来的脚本随加载的势对象管理生命周期。

### 并发求值

`submit()` 在各自的 `run-job-async-XXXXXX` 目录中启动计算并立即返回，这是
同时算多个结构的方式：

```python
jobs = [potential.submit(frame) for frame in frames]
results = [job.result() for job in jobs]
```

如果不需要轮询，`calculate_many()` 会同步计算一批结构并按顺序返回结果，
它也接受 ASE 的 `Atoms` 和 pymatgen 结构：

```python
results = potential.calculate_many(frames)
```

`ShellJob.ready()` 不阻塞地轮询，`result()` 等待并释放 GIL。每次调用返回
新的 `Atoms`；失败的计算会在后续调用中重复抛出同一错误。释放最后一个作业
句柄会等待其完成——没有取消机制。C++ 侧的等价接口是
`potentials/potential_shell.h` 中的 `PotShellCalc::submit()`。

## 如何选择

- 短程化学通常从 `mtp` 开始；需要多层级或神经网络头时使用
  `mtp_m`、`hmtp` 或 `mtpnn1`。
- 长程静电使用 `mtp_les`；用
  `hybrid`/`hybrid_scale` 将机器学习核心与短程排斥修正组合。
- 加载经典参数化时使用 `eamalloy`、`eamfs`、`eampoly`、`sw` 或
  `tersoff`；表格化模型使用 `pair_table`/`threebody_table`。
- 需要单一可部署文件时，用 `model_convert` 将任意受支持模型封装为 GPM。

## 相关

- @ref guide_commands "命令参考"：`pot2table`、`adjustmtp` 等命令的选项
- @ref guide_config "配置说明"：`pot:`、`hybrid` 与 `potN:` 键
- [Python API](python_zh.md)：`Potential` 类型常量与 `submit()` shell 作业
