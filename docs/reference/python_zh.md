# Python API {#api_python}

[English](python.md)

@tableofcontents

`imrmlp` Python 包通过 pybind11 绑定提供对 IMR-MLP C++ 框架的 Python 接口。
采用两层架构：

| 层 | 模块 | 描述 |
|-------|--------|-------------|
| **C++ 绑定** | `_imrmlp`（编译的 `.so`） | 针对 Atoms、势函数评估、I/O、弛豫、描述符、主动学习的轻量 pybind11 包装 |
| **Python 包装** | `imrmlp`（纯 Python 包） | Pythonic API，包含属性访问、类型提示、ASE 集成、dataclass 结果 |

## 安装

初次使用时先按[Python 与 ASE 入门](../../README-python_zh.md)安装并运行同一构型的求值，
确认模型来源与输出含义；本页用于按需查阅 API，不要求从头执行所有片段。

需要 Python 3.10+、C++23 编译器和[构建指南](../guides/installation_zh.md)中的
本地开发库。先激活要使用的虚拟环境，再在仓库根目录选择一种方式：

```bash
# 安装到当前 Python 环境，可在其他目录使用
USE_KIM=0 python3 -m pip install .

# 或仅在源码目录构建和使用
python3 -m pip install numpy pybind11
make python PYTHON=python3 USE_KIM=0
python3 -c "import imrmlp; print(imrmlp.__version__)"
```

`pip install .` 通过 CMake 调用 Make，构建 release 版扩展并随包安装
`libimrmlp.so`；它仍然需要系统中的本地依赖（如 ALGLIB、FFTW），不会自动安装。
可选的 oneMKL 后端只在该次构建的环境已启用时生效。
`make python` 把扩展写入源码目录的 `imrmlp/`，默认链接 `build/release` 中的库；
更换 Python 解释器后应使用相同的 `PYTHON=...` 重新构建。
`make install` 不安装 Python 包。

可选依赖按需安装；引号可避免 shell 把方括号展开为文件名：

```bash
USE_KIM=0 python3 -m pip install '.[ase]'       # ASE 计算器
USE_KIM=0 python3 -m pip install '.[phonopy]'   # Phonopy 和 ASE
USE_KIM=0 python3 -m pip install '.[pymatgen]'  # pymatgen 转换
USE_KIM=0 python3 -m pip install '.[interop]'   # ASE 和 pymatgen
USE_KIM=0 python3 -m pip install '.[dev]'       # 测试工具
```

## 快速开始

先运行 [CLI 入门示例](../../examples/quickstart/README_zh.md)，再从仓库根目录运行以下
代码。需先构建 Python 扩展。这个合成模型用于验证接口，不是实际 Al 材料势。

```python
import numpy as np
import imrmlp

# Run the CLI quickstart first, from the repository root.
pot = imrmlp.Potential("build/quickstart/fitted.mtp", "mtp")
atoms = imrmlp.Atoms(
    numbers=[13, 13, 13],
    elements=["Al"],  # model species 0 = Al, NOT atomic number 13
    positions=[[7, 7, 7], [9.5, 7, 7], [7.8, 9.0, 7.1]],
    cell=np.eye(3) * 20.0,
    pbc=True,
)
pot.calculate(atoms)
print(atoms.energy)  # total eV
print(atoms.forces)  # eV/Å
print(atoms.stress)  # extensive stress, eV
imrmlp.write("build/quickstart/python-predicted.cfg", atoms)
```

后续代码片段是需要自备模型和数据的用法模板。用于求值或弛豫的 CFG 文件必须
采用模型的数值物种索引；例如单元素 Al MTP 通常使用 0，而不是原子序数 13。
若从 POSCAR 等元素标记文件构造模型输入，可先用 ASE 读取，再通过
`Atoms(numbers=..., positions=..., cell=..., pbc=..., elements=[...])` 明确模型顺序。

## Atoms

`imrmlp.Atoms` 类包装了 C++ `Atoms` 结构，提供基于属性的访问和自动原子序数到物种索引的翻译。

```python
import numpy as np
import imrmlp

# 从头创建
atoms = imrmlp.Atoms(
    numbers=np.array([13, 13, 13, 13]),
    positions=np.array([[0,0,0], [1.4,1.4,0], [2.8,0,1.4], [1.4,2.8,1.4]]),
    cell=np.eye(3) * 4.05,
    elements=["Al"],
)

# 属性——读取和设置
atoms.positions          # (N, 3) numpy 数组，默认是笛卡尔坐标
atoms.cell               # (3, 3) numpy 数组
atoms.numbers            # (N,) 原子序数
atoms.energy             # 总能量（eV），未计算时返回 None
atoms.forces             # (N, 3) 力（eV/Å），未计算时返回 None
atoms.stress             # (3, 3) 广延应力/virial（eV），未计算时为 None
atoms.chemmap            # 原始物种索引列表
atoms.velocities         # (N, 3) 速度，Å/ps（可选）
atoms.site_energies      # (N,) 逐原子能量（可选）
atoms.natoms             # 原子数量
atoms.volume             # 晶胞体积（Å³）

# 方法
atoms.wrap()             # 将原子包裹回周期晶胞
atoms.copy()             # 深拷贝
atoms.to_fractional()    # 转换为分数坐标
atoms.to_cartesian()     # 转回笛卡尔坐标

# 与 ASE 互转
ase_atoms = atoms.to_ase()
imrmlp_atoms = imrmlp.Atoms.from_ase(ase_atoms)

# 与 pymatgen 互转（可选依赖）
pmg_structure = atoms.to_pymatgen()
imrmlp_atoms = imrmlp.Atoms.from_pymatgen(pmg_structure)
```

`from_ase()` 和 `from_pymatgen()` 不接受 `elements` 参数，返回的类型使用
原子序数。向使用连续物种索引的模型求值前，应显式构造映射后的结构，例如：

```python
model_atoms = imrmlp.Atoms(
    numbers=ase_atoms.get_atomic_numbers(),
    positions=ase_atoms.get_positions(),
    cell=ase_atoms.get_cell().array,
    pbc=ase_atoms.get_pbc(),
    elements=["Al"],
)
```

这个构造示例只导入几何；已有能量、力等标注应在训练流程中另行保留。
ASE 计算器可直接用 `IMRMLPCalculator(..., elements=[...])` 完成同样的类型映射。

**关于 chemmap 和 numbers 的说明：** C++ 层以整数索引（`chemmap_`）存储物种信息。Python `Atoms` 类自动在原子序数和物种索引之间进行转换。提供 `elements` 列表（按 chemmap 顺序）以实现可逆转换：

```python
atoms = imrmlp.Atoms(
    numbers=np.array([13, 12, 13, 12]),
    elements=['Mg', 'Al'],   # 索引 0 → Mg (Z=12), 索引 1 → Al (Z=13)
)
```

## Potential

`imrmlp.Potential` 通过 C++ `PotentialWrapper` 工厂加载势函数。
实际可用类型取决于 OpenKIM、ACELMP 等可选构建特性。

```python
import imrmlp
from imrmlp import Potential

# 默认 auto：GPM 从模型包识别类型；旧格式回退到 mtp
pot = Potential("Al.mtp")                  # 旧 MTP 文件
# 其他旧格式需要显式指定类型
pot = Potential("Al.eam.alloy", "eamalloy")
pot = Potential("potential.sw", "sw")
pot = Potential("Si.tersoff", "tersoff")
pot = Potential("ZBL", "zbl")
pot = Potential("model.kim", "openkim")

# 属性
pot.cutoff           # 截断半径
pot.species_count    # 化学物种数量
pot.pottype          # 势函数类型字符串
pot.potfile          # 文件路径

# 计算能量、力、应力（原地修改 atoms）
pot.calculate(atoms)

# 仅能量（更快）
energy = pot.calculate_energy(atoms)

# OpenMP 并行评估
pot.calculate(atoms, use_omp=True)

# 检查是否支持线性拟合
if pot.supports_linear_fit():
    print("该势函数支持线性训练")
```

### 势函数类型常量

`Potential` 类提供类型字符串的类级常量：

| 常量 | 类型字符串 |
|----------|-------------|
| `Potential.MTP` | `"mtp"` |
| `Potential.EAM_ALLOY` | `"eamalloy"` |
| `Potential.EAM_FS` | `"eamfs"` |
| `Potential.EAM_POLY` | `"eampoly"` |
| `Potential.LJ612` | `"lj612"` |
| `Potential.SW` | `"sw"` |
| `Potential.TERSOFF` | `"tersoff"` |
| `Potential.ZBL` | `"zbl"` |
| `Potential.OPENKIM` | `"openkim"` |
| `Potential.HYBRID` | `"hybrid"` |
| `Potential.HYBRID_SCALE` | `"hybrid_scale"` |
| `Potential.PAIR_TABLE` | `"pair_table"` |
| `Potential.THREEBODY_TABLE` | `"threebody_table"` |


## 结构 I/O

`read`/`read_one` 保留文件中的物种索引，不会给 Python 包装附加 `elements` 映射；
不要假定 `.numbers` 一定是原子序数，也不要在缺少映射时直接 `to_ase()`。
需要向模型求值时，请用 `Atoms(..., elements=[...])` 构造物种顺序明确的输入。


`imrmlp.io` 模块包装了 C++ `GPIO` 类，支持 20 多种文件格式，自动格式检测。

```python
import imrmlp

# 读取所有结构
structures = imrmlp.read("training_set.cfg")
for s in structures:
    print(s.natoms, s.energy)

# 读取单个结构
atoms = imrmlp.read_one("POSCAR")

# 写入结构
imrmlp.write("output.cfg", atoms)
imrmlp.write("trajectory.extxyz", [atoms1, atoms2, atoms3])

# 显式指定格式
atoms = imrmlp.read_one("input.txt", format="poscar")
imrmlp.write("output", atoms, format="mlpcfg")

# 从文件名自动检测格式（位于 imrmlp.io.guess_format；返回格式名称字符串，
# 例如 ".cfg" 对应 "mlpcfg"；未知文件名返回 ""）
from imrmlp.io import guess_format
fmt = guess_format("structure.cfg")
```

### 支持的格式

| 格式 | 显式 `format` | 可识别扩展名 / 名称 | 读取 | 写入 |
|--------|-------------------|-----------------------------|------|-------|
| VASP POSCAR | `poscar` | `.vasp`, `POSCAR` | ✓ | ✓ |
| VASP XDATCAR | `xdatcar` | `XDATCAR` | ✓ | ✓ |
| VASP OUTCAR | `outcar` | `OUTCAR` | ✓ | |
| VASP XML | `vasprun` | `vasprun.xml` | ✓（`read`） | |
| VASP ML_ABN | `vaspabn` | `ML_ABN` | ✓ | |
| LAMMPS 轨迹 | `lmptrj` | `.lammpstrj` | ✓ | ✓ |
| LAMMPS data | `lmpdat` | `.lmpdat`, `.lmp` | ✓ | ✓ |
| LAMMPS YAML | `yaml` | `.yaml` | ✓ | |
| MLIP CFG | `mlpcfg` | `.cfg` | ✓ | ✓ |
| MLIP binary CFG | `bin` | `.cfgbin`, `.bin` | ✓ | ✓ |
| Extended XYZ | `extxyz` | `.extxyz`, `.xyz` | ✓ | ✓ |
| CIF | `cif` | `.cif` | ✓ | ✓ |
| XSF | `xsf` | `.xsf`, `.axsf` | ✓ | ✓ |
| CP2K XYZ | `cp2kxyz` | `.cp2kxyz` | ✓ | ✓ |
| n2p2 data | `n2p2` | `.n2p2` | ✓ | ✓ |
| VTK | `vtk` | `.vtk` | | ✓ |
| DLPOLY4 | `dlpoly` | `.dlpoly` | | ✓ |
| GROMACS | `gro` | `.gro` | ✓ | |
| Materials Studio | `msi`, `xsd`, `car` | `.msi`, `.xsd`, `.car` | ✓ | |

`guess_format()` 检查文件名，不检查内容。`data.lammps` 或 `input.data` 需按实际内容
指定 `format="lmpdat"` 或 `format="n2p2"`；`trajectory.cp2k.xyz` 会被识别为
`extxyz`，CP2K 数据需指定 `format="cp2kxyz"`。内容探测是独立功能：
`imrmlp.io.detect_format(path)` 报告候选格式，`read(path, detect_content=True)`
使用唯一的内容匹配结果。上表描述文件读写；流会话仅支持其中一部分格式，见
[共享工作流](../advanced/shared-workflows_zh.md)。

### 流式与增量 I/O

轨迹无法整体载入内存，或帧是边生成边消费时（例如主动学习循环），使用下列接口。

```python
import imrmlp
from imrmlp import iread, write_iter, write_one, open_reader, open_writer

# iread: 惰性逐帧迭代，文件在首次迭代时打开
for i, atoms in enumerate(iread("trajectory.extxyz")):
    if atoms.energy is not None and atoms.energy < -100.0:
        write_one(f"picked-{i}.cfg", atoms)
    if i > 10000:
        break          # 允许提前关闭，不做整文件校验

# write_iter: 直接写可迭代对象，无需构造列表
def generated():
    for frame in iread("md.extxyz"):
        yield frame

write_iter("filtered.extxyz", generated())

# open_reader / open_writer: 独立会话，可显式关闭
with open_reader("md.cfg") as reader:
    for atoms in reader:
        process(atoms)

with open_writer("out.extxyz", append=True) as writer:
    for atoms in new_frames:
        writer.write(atoms)   # 元素映射必须与写入会话一致
```

提前退出时，使用 `with contextlib.closing(iread(path)) as frames:`，或显式调用
迭代器的 `close()`。关闭会释放资源，但不会检查尚未读取的帧。

行为与限制：

| 接口 | 用途 | 说明 |
|---|---|---|
| `iread(path, format="", *, detect_content=False, input_units=None)` | 惰性逐帧迭代 | 支持 `extxyz`/`xyz`、`cfg`/`mlpcfg`、`bin`/`mlpbin`、`n2p2`、`cp2kxyz`、`gro`、`xdatcar`、`xdatcar6`、`outcar`、`xsf`/`axsf`。按原生格式支持 gzip 流式解压，全部读完时检查压缩尾部。损坏帧在被消费时抛错。 |
| `write_one(path, atoms, format="")` | 只写一帧 | `write()` 的便捷封装，格式探测与限制完全一致。 |
| `write_iter(path, structures, format="", *, append=False)` | 由可迭代对象写入 | 支持 EXTXYZ、CFG、BIN、XDATCAR、XDATCAR6、CP2KXYZ 与 N2P2。所有帧必须使用同一元素映射；空可迭代对象既不创建也不截断目标文件；`append=True` 当前会抛出 `ValueError`；后期失败可能留下已写入的前缀。 |
| `open_reader(path, format="", *, detect_content=False, input_units=None)` | 有界读取会话 | 支持 EXTXYZ、CFG、BIN、N2P2、XSF、OUTCAR 与 XDATCAR/6。可作为迭代器或上下文管理器使用；不同会话可并发。 |
| `open_writer(path, format="", *, elements=None, append=False)` | 惰性写入会话 | 支持 EXTXYZ、CFG、BIN、N2P2 与 XDATCAR/6。`elements` 与 `Atoms` 的显式元素映射相同，`None` 表示原子序数类型。仅 EXTXYZ、CFG 和 BIN 支持 `append=True`；写入首帧前不创建文件。 |

`write_iter` 每次只保留一帧，输入保持不变且内存占用有界。数据规模较小时，
`read()` 返回结构列表，`write()` 接受单结构或列表，调用更直接。需要追加帧时，
使用 `open_writer(..., append=True)` 并选择 EXTXYZ、CFG 或 BIN；
`write()` 和 `write_iter()` 均不接受追加。上表未列出的格式传给流式接口会抛出
`ValueError`。

## 结构弛豫

```python
from imrmlp import Potential, read_one, Relaxation

pot = Potential("Al.mtp")
atoms = read_one("initial.cfg")

relax = Relaxation(
    potential=pot,
    driver="fire",         # "lbfgs", "fire", "cg", 或 "gd"
    max_iter=1000,
    gtol=1e-3,             # 活动优化坐标中的梯度分量阈值
    relax_cell=True,       # 变晶胞弛豫
    pressure=0.0,          # 外部压力（GPa）
    trajfile="relax.trj",  # 可选轨迹输出文件
)

result = relax.run(atoms)
print(f"收敛: {result.converged}")
print(f"迭代次数: {result.n_iter}")
print(f"最终优化目标: {result.final_energy:.4f} eV")
print(f"最终势能: {result.final_atoms.energy:.4f} eV")
print(f"初始能量: {result.init_energy:.4f} eV")

# 原地修改：atoms 现在包含弛豫后的位置
print(atoms.positions)
```

`gtol` 针对活动优化坐标中的最大梯度分量；固定晶胞的笛卡尔原子坐标对应
eV/Å，变胞或其他坐标模式下不能直接解释为最大原子力。可用优化器还包括
`lbfgspp`、`mdmin`、`quickmin`、`sqp` 和 `nlc`。

`RelaxResult` 字段：

- `converged` — `bool`
- `n_iter` — `int`
- `init_energy` — `float`（eV）
- `final_energy` — 最终优化目标（eV），包含外压和约束项；势能见 `final_atoms.energy`
- `final_atoms` — `Atoms`
- `stop_reason` — 停止原因字符串

## Nudged Elastic Band (NEB)

```python
from imrmlp import Potential, read_one, NEB

pot = Potential("Al.mtp")
initial = read_one("initial.cfg")
final = read_one("final.cfg")

neb = NEB(
    potential=pot,
    n_images=7,            # 包括端点
    spring=5.0,            # 弹簧常数（eV/Å²）
    climb=True,            # 爬升镜像 NEB
    optimizer="fire",      # 统一的 C++ 优化循环
    max_iter=1000,
    neb_ftol=1e-2,         # 力容差（eV/Å）
)

result = neb.run(initial, final)
print(result.converged, result.stop_reason, result.max_neb_force)
print(f"收敛: {result.converged}")

for i, img in enumerate(result.images):
    print(f"镜像 {i}: E = {img.energy:.4f} eV")
```

`NEBResult` 的常用字段：

- `converged` — `bool`
- `n_iter` — `int`
- `images` — `List[Atoms]`（包括端点）
- `max_perp_force` — 最大垂直力
- `max_neb_force` — 包含弹簧和爬山项的最大有效 NEB 力
- `stop_reason` — 停止原因；请与 `converged` 一起检查

## 势函数训练

### 线性训练器

`LinearTrainer` 要求模型支持通用线性拟合接口，`descriptor_linear` 实现了该接口。
普通 MTP 未暴露此接口；其线性系数应使用 `imrmlp trainmtp train.ini --linear-fit`
拟合。下例需要含标注的 `training.cfg`，且只有一个 Al 物种（索引 0）。
无需外部数据的可运行合成示例见 [06_train.py](../../examples-python/06_train.py)：
构建 Python 扩展后，在仓库根目录运行 `PYTHONPATH=. python3 examples-python/06_train.py`，
输出位于 `build/examples-python/06-train/`。

```python
from imrmlp import Potential, read, LinearTrainer

pot = Potential.from_settings({
    "pottype": "descriptor_linear",
    "descriptor_linear:species_count": "1",
    "descriptor_linear:descriptor:dtype": "behler",
    "descriptor_linear:descriptor:behler:cutoff": "5",
    "descriptor_linear:descriptor:behler:dispcount": "2",
    "descriptor_linear:descriptor:behler:dispselect": "1 2",
    "descriptor_linear:descriptor:behler:eta": "0.3",
})
training_set = read("training.cfg")

trainer = LinearTrainer(
    energy_weight=1.0,
    force_weight=0.01,
    stress_weight=0.0001,
    reg_L2=1e-10,          # L2 正则化
    save_to="fitted.linear",
)

# 一步训练
result = trainer.train(pot, training_set)
print(f"特征数: {result.feature_count}")
print(f"能量 RMSE: {result.train_erms:.4f} eV/atom")
print(f"力 RMSE:   {result.train_frms:.4f} eV/Å")

# 或分步进行以便检查
trainer.assemble(pot, training_set)
# ... 检查矩阵 ...
result = trainer.solve()
```

`TrainResult` 字段：

- `feature_count` — `int`
- `train_erms` — `float`
- `train_frms` — `float`
- `train_srms` — `float`

## 主动学习

### 矩阵级 MaxVol

```python
import numpy as np
from imrmlp import MaxVol

A = np.random.randn(100, 16)
mv = MaxVol()
selected = mv.select_rows(A, k=16)
scores = mv.compute_scores(A, selected)
```

### MaxVol 主动集选择

```python
from imrmlp import read, ActiveSelector, Potential

pool = read("pool.cfg")

selector = ActiveSelector(
    potfile="model.mtp", pottype="mtp", mode="cfg",
    threshold=1.001,
    weight=1.0,
    pow_scale=0.5,
    max_swaps=0,           # 无限交换（默认）
)

selector.add_candidates(pool)
selected_count = selector.select()
print(f"选择了 {selected_count} 个结构")
print(f"选择索引: {selector.selected_indices}")

# 对新结构打分
grade = selector.grade(new_structure)
print(f"grade: {grade:.3f}")
# 1.0 = 在主动集内
# 1–10 = 轻度外推
# >>10 = 极端外推

# 批量打分
grades = selector.grade_all(test_structures)
print(f"最大 grade: {grades.max():.3f}")

# 持久化
selector.save("active_set.sel")
selector2 = ActiveSelector.load("active_set.sel", potential=Potential("model.mtp"))
```

不需要中间检查时，`select_from_pool()` 把上述三步合并成一次调用：

```python
from imrmlp import read, select_from_pool

selector = select_from_pool(
    read("pool.cfg"),
    potfile="model.mtp",
    pottype="mtp",
    mode="cfg",
    threshold=1.001,
)
print(selector.selected_indices)
grade = selector.grade(new_structure)
```

它根据势文件构造 `ActiveSelector`，把每个结构加入候选集并执行 `select()`。
关键字参数传给选择器：`threshold`、`weight`、`pow_scale`、`max_swaps`、
`pottype`、`dtype`、`mode`、`params`、`w_energy`、`w_site`、`w_force`、
`w_stress` 与 `all_coeffs`。无法识别的关键字会原样转发给 `ActiveSelector`，
可能在那里被拒绝。需要分批加入候选或需要续跑选择时，请使用显式三步形式。

@note Python 中提供了矩阵级 `MaxVol` API。MTP B 矩阵构建也可通过
`build_bmatrix()` 使用；对于很大的生产级候选池，仍建议使用
`maxvol_select`、`maxvol_sample` 和 `maxvol_extend` CLI 命令。

### 最远点采样（FPS）

当前高层 `FPSSelector` 包装没有暴露描述符/B 矩阵构建器的初始化接口；
`metric` 和 `random_seed` 仅保存在 Python 层，没有传给 C++ 选择器。
结构选择请使用 `fps_select` CLI 并显式配置势/描述符，不要把
`FPSSelector().add_candidates(pool)` 当作可独立运行的流程。

### CUR 选择

```python
import numpy as np
from imrmlp import cur_select

# descriptor_matrix: (n_structures, n_features)
result = cur_select(
    descriptor_matrix,
    k=20,
    deterministic=True,
    compute_error=True,
)
print(f"选择索引: {result.selected_indices}")
print(f"杠杆分数: {result.scores}")
print(f"重构误差: {result.reconstruction_error:.4f}")
```

`cur_select()` 返回 `CURResult` 数据类：

| 字段 | 类型 | 含义 |
|---|---|---|
| `selected_indices` | `list[int]` | 从 `A` 中选出的行索引，按选择顺序排列。 |
| `scores` | `np.ndarray` | `A` 每行的杠杆分数。 |
| `reconstruction_error` | `float` | 相对重构误差；`compute_error=False` 时为 `-1.0`。 |

`deterministic=True` 按分数取前 `k` 行；默认的概率加权采样用 `seed` 保证可复现，
不同种子会选出不同行。`k` 不得超过行数。

## 描述符

```python
from imrmlp import Descriptor

# 描述符参数会传递给 C++ 描述符包装器，后者读取类型专属的子设置。
# 键使用描述符前缀（"behler:" 等）；SO3 使用下例中的平铺键。

# Behler ACSF
behler = Descriptor("behler", {
    "behler:cutoff": "6.0",
})

# 键序参数
bo = Descriptor("bondorder", {
    "bondorder:maxl": "8",
})

# SO(3) bispectrum
so3 = Descriptor("biso3", {
    "biso3_lmax": "3", "biso3_rsize": "2", "biso3_rmax": "4",
    "biso3_real_harmonics": "true",
})

# SO(4) bispectrum
so4 = Descriptor("biso4", {
    "biso4:twojmax": "3", "biso4:cutoff": "4",
    "biso4:weights": "1 1.3", "biso4:selfweights": "0.6 0.8",
    "biso4:average": "inner-density", "biso4:contraction_backend": "snap",
})
```

@note `mtm` 描述符无法通过 Python 的 `Descriptor` API 使用——它需要
`PotMTP` 实例，而绑定不会传入。请改用 C++ 的 `DescriptorMTM` 或
`maxvol_*` CLI 命令。

SO4 的 `inner-density` 先平均密度再构造指纹。`compute_averaged_with_gradient` 返回固定晶胞的原子 Jacobian；`compute_averaged_with_strain_gradient` 返回应变 Jacobian。`compute_averaged_with_cell_gradient` 则在固定分数坐标下求原始晶胞分量导数。`compute_weighted_global_hvp` 和 `compute_weighted_global_hessian` 提供全局加权二阶导数，后二阶接口及原始晶胞接口要求 `inner-density`。

用 `bo.compute(atoms)` 得到逐原子特征，形状为 `(原子数, 特征数)`；
`bo.compute_averaged(atoms)` 得到结构特征，形状为 `(特征数,)`。
以上描述符参数只控制特征构造，不能直接用于能量或力预测。

## ASE 计算器集成

先安装 ASE，显式传入模型物种顺序。计算器将原生广延应力（eV）除以体积，
转换为 ASE 应力（eV/Å³）。以下示例使用入门合成模型；实际弛豫和 MD 应使用
经过验证的材料势。

```python
from ase import Atoms
from imrmlp.calculator import IMRMLPCalculator

atoms = Atoms("Al3", positions=[[7, 7, 7], [9.5, 7, 7], [7.8, 9, 7.1]],
              cell=[20, 20, 20], pbc=True)
atoms.calc = IMRMLPCalculator(
    filename="build/quickstart/fitted.mtp", pottype="mtp", elements=["Al"],
)
print(atoms.get_potential_energy())  # eV
print(atoms.get_forces())           # eV/Å
print(atoms.get_stress())           # eV/Å³, xx yy zz yz xz xy
```

## Phonopy 接入 {#phonopy-integration}

安装可选依赖：`python -m pip install '.[phonopy]'`。
`imrmlp.phonopy.calculate_forces(phonon, calculator)` 接收已生成位移的 Phonopy 对象，
使用现有 `IMRMLPCalculator` 或其他 ASE calculator 计算并设置 `phonon.forces`，同时返回力数组。
接口要求 Phonopy 的 calculator 为默认值或 `vasp`，长度使用 Å、力使用 eV/Å；
保持位移与超胞原子顺序，所有帧成功后才替换力，不额外移除漂移或施加对称化。

```python
from imrmlp.phonopy import calculate_forces

# phonon: Phonopy object; calculator: configured ASE IMRMLPCalculator
phonon.generate_displacements(distance=0.01)
calculate_forces(phonon, calculator)
phonon.produce_force_constants()
phonon.save("phonopy_params.yaml", settings={"force_constants": True})
```

完整示例：[09_phonopy.py](../../examples-python/09_phonopy.py)。例如：

```sh
python examples-python/09_phonopy.py POSCAR fitted.mtp --elements Al Ni --supercell 3 3 3
```

`--elements` 必须按模型物种顺序填写，输入结构应已用该势弛豫。
Phonopy 负责原胞/超胞映射、力常数拟合及后处理；示例写出参数、DOS 和热力学文件。
CLI 的 `phonopyfc` 接口继续可用，详见[命令参考](commands_zh.md)。

[Phonopy 官方 API](https://phonopy.github.io/phonopy/phonopy-module.html)说明其余计算流程。

## 工具函数

```python
from imrmlp.utils import (
    Z_TO_SYMBOL,        # 字典：原子序数 -> 元素符号
    SYMBOL_TO_Z,        # 字典：元素符号 -> 原子序数
    build_type_mapping, # 构建物种索引映射
    chemmap_from_numbers, # 将原子序数转换为 chemmap
    numbers_from_chemmap, # 将 chemmap 转换为原子序数
)

# 示例：元素翻译
al_z = SYMBOL_TO_Z["Al"]     # 13
al_sym = Z_TO_SYMBOL[13]     # "Al"

# 二元系的类型映射
type_map = build_type_mapping(["Mg", "Al"])
# {12: 0, 13: 1}
```

## OpenMP 线程控制

```python
from imrmlp import set_num_threads, get_num_threads

print(f"当前线程数: {get_num_threads()}")
set_num_threads(4)            # 设置显式线程数
```

## 模型包与主动学习会话

`ModelBundle` 暴露 GPM 容器，且不会隐式启动主动学习：

```python
from imrmlp import ModelBundle

bundle = ModelBundle.load("resume.gpm")
print(ModelBundle.inspect("resume.gpm"))
ModelBundle.validate("resume.gpm")

session = bundle.create_al()          # 显式激活
result = session.evaluate(atoms)      # 返回 EFS 结果并收集候选
print(session.should_abort, len(session.pool))
session.save("continued.gpm")

bundle.potential.save("copy.gpm")                    # 保留 AL 会话
bundle.potential.save("deploy.gpm", profile="inference")
```

`ModelBundle.from_settings(settings)` 从扁平 `{key: value}` 映射构造模型包。
`attach_al()` 与 `attach_selector()` 可附加已有会话或选择器；
`ALSession.selector` 返回当前用于选择的活动集选择器。

## Shell 求值任务

`Potential.submit()` 仅用于 `shellfile` 势的异步求值。先用 shellfile 配置
构造 `potential`；提交时复制输入，原 `atoms` 不变：

```python
job = potential.submit(atoms)
while not job.ready():
    ...                      # 执行其他 Python 工作
result = job.result()        # 返回含计算结果的新 Atoms 对象
```

`ShellJob.result()` 等待时释放 GIL。可以提交多个独立作业，再分别读取结果；
`calculate(atoms)` 则原地修改输入并返回 `None`。作业不支持取消，释放最后一个作业
句柄时会等待计算完成。

## Dimer 鞍点搜索

```python
from imrmlp import Dimer

dimer = Dimer(potential, ddr=0.01, max_iter=200)
result = dimer.run(atoms)
print(result.converged, result.exit_code, result.stop_reason)
saddle = result.saddle_atoms
```

`DimerResult` 还提供 `energy`、`curvature`、`max_force`、
`max_rot_gradient` 与 `n_iter`。参见[鞍点工作流](../advanced/saddle-workflows_zh.md)。

## 非线性训练

```python
from imrmlp import NonlinearTrainer

trainer = NonlinearTrainer(
    energy_weight=1.0, force_weight=0.01, stress_weight=1e-4,
    max_iterations=200, trainable_groups=["radial"],
)
result = trainer.train(potential, training_set, validation_set)
print(result.loss, result.valid_erms, result.test_erms)
```

`NonlinearTrainResult` 报告训练/验证/测试 EFS 误差、可选的 Hessian 与 HVP
RMS、选中的 L2 乘子，以及使用固定核心时的 hybrid manifest 路径。
额外关键字参数使用不带前缀的 `fit:` 键名。

## 单位与 XSF 场

```python
from imrmlp.units import (
    convert_units, unit_conversion_factor, source_unit_system,
    canonicalize_atoms, pressure_to_virial_factor,
)
from imrmlp.xsf_fields import (
    GridLayout, ScalarField, DataGridBlock, VectorFieldView,
    read_xsf_fields, write_xsf_fields,
)

scale = unit_conversion_factor("energy", "eV", "hartree")
factor = pressure_to_virial_factor("GPa")
fields = read_xsf_fields("band.xsf")
```

`source_unit_system()` 解析 `canonical`、`metal`、`atomic` 与 `real` 标签；
`canonicalize_atoms()` 将 `Atoms` 转换为规范单位。

## Atoms 与 Potential 的扩展成员

`Atoms` 提供磁矩、电荷、质量、标签、初始值、总电荷、自由能、PBC 标志
以及选择性动力学掩码：

```python
atoms.charges, atoms.magmoms, atoms.masses
atoms.tags, atoms.initial_charges, atoms.initial_magmoms
atoms.total_charge, atoms.free_energy, atoms.pbc
atoms.selective_dynamics_mask
atoms.set_hessian(hessian); atoms.set_hvp(directions, targets)
atoms.clear_hessian(); atoms.clear_hvp()
```

`Potential` 增加物种与参数访问以及 Hessian/HVP 求值：

```python
potential.append_species([0, 1], ["Al", "Ni"])
potential.supports_species_append()
potential.coefficients, potential.parameter_groups
potential.calculate_many(structures, properties=("energy", "forces", "stress"))
potential.hessian(atoms)
potential.hessian_vector_product(atoms, direction)
potential.supports_analytical_hessian()
```

`hessian()` 返回 `(3*N, 3*N)` 的笛卡尔 Hessian，`hessian_vector_product()`
接受长度 `3*N` 的一维方向并返回同形状向量；顺序均为逐原子的 xyz，单位为
eV/Å²，不做质量加权。默认 `method="auto"` 优先解析导数，否则对力做中心差分；
`method="analytic"` 要求模型明确支持。两种查询均保留输入结构与已有结果。

`imrmlp.atoms` 还提供 `from_ase_list`、`to_ase_list`、
`from_pymatgen_list` 与 `to_pymatgen_list`；`imrmlp.has_mpi`
指示扩展是否以 MPI 构建。

## 参见

- @ref api_core — C++ Atoms 和邻域 API
- @ref api_io — C++ 结构 I/O API
- @ref api_potential — C++ 势函数评估 API
- @ref api_analysis — C++ 分析和描述符 API
- @ref guide_install — 构建要求与编译
- @ref guide_config — 配置文件格式与参考
