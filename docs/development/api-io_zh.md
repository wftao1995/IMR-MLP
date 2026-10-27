# I/O API：读取和写入结构 {#api_io}

[English](api-io.md)

本页面向 C++ 集成。无需编程的格式转换请用 `imrmlp convstr`，脚本处理见
[Python I/O API](../reference/python_zh.md)。请按 [API 示例](cpp-examples_zh.md)
构建并链接库。以下片段需要自行提供所列输入文件。



@tableofcontents

## 基本读写

### 单个结构

```cpp
#include "io/gpio.h"

GPIO io;
Atoms atoms;

// 从文件读取（根据扩展名自动检测格式）
io.read_one(atoms, "POSCAR");
io.read_one(atoms, "structure.cfg");
io.read_one(atoms, "structure.lmpdat");

// 明确指定格式
io.read_one(atoms, "input.file", "poscar");
io.read_one(atoms, "input.file", "mlpcfg");
io.read_one(atoms, "data.lammps", "lmpdat");
io.read_one(atoms, "trajectory.cp2k.xyz", "cp2kxyz");

// 写入文件（根据扩展名自动检测格式）
io.write_one(atoms, "output.vasp");
io.write_one(atoms, "structure.cfg");
io.write_one(atoms, "data.lammps", "lmpdat");  // 此文件名需显式指定格式
```

### 多个结构

```cpp
std::vector<Atoms> structures;

// 从文件读取所有结构
io.read(structures, "trajectory.cfg");
io.read(structures, "XDATCAR");
io.read(structures, "md.lammpstrj");

// 写入所有结构
io.write(structures, "output.cfg");
```

### 使用流

```cpp
std::ifstream ifs("input.cfg");
Atoms atom;
io.read_one(atom, ifs, "mlpcfg");

std::ofstream ofs("output.vasp");
io.write_one(atom, ofs, "poscar");
```

### 压缩文件

文件读写透明支持 gzip（`.gz`）：读取时通过文件 magic bytes 自动识别 gzip，写入时当文件名以 `.gz` 结尾时自动压缩。不支持 Zstandard（`.zst`）；识别到的 Zstandard 输入和 `.zst` 输出文件名会被拒绝。

```cpp
io.read_one(atoms, "POSCAR.gz");          // 读取压缩的 POSCAR
io.write(structures, "traj.xyz.gz");      // 写入 gzip 压缩轨迹

```

## 顺序I/O（大文件）

适用于无法全部装入内存的大型轨迹文件：

```cpp
GPIO io;
std::ifstream ifs("md.lammpstrj");
if (!ifs) throw std::runtime_error("Cannot open md.lammpstrj");
io.seq_read_init(ifs, "lmptrj");

Atoms atom;
while (io.seq_read(ifs, atom)) {
    // 逐帧处理
    process(atom);
}
```

顺序写入：

```cpp
GPIO io;
std::ofstream ofs("output.cfg");
io.seq_write_init(ofs, "mlpcfg", "output.cfg");

for (auto &atom : structures) {
    io.seq_write(ofs, atom);
}
```

## 支持的格式

### 读取

| 格式 | 扩展名/名称 | 模式 |
|--------|------------------|------|
| VASP POSCAR | `.vasp`, `POSCAR` | 单帧 |
| VASP XDATCAR | `XDATCAR` | 顺序 |
| VASP OUTCAR | `OUTCAR` | 单帧/顺序 |
| VASP vasprun.xml | `vasprun.xml` | 文件式 `read`（需要 USEXML） |
| VASP ML_ABN | `ML_ABN` | 顺序 |
| LAMMPS dump | `.lammpstrj` | 顺序 |
| LAMMPS data | `.lmpdat`, `.lmp` | 单帧 |
| LAMMPS YAML | `.yaml` | 顺序 |
| MLIP CFG | `.cfg` | 顺序 |
| MLIP binary CFG | `.cfgbin` | 顺序 |
| Extended XYZ | `.extxyz` | 顺序 |
| Simple XYZ（EXTXYZ 读取器） | `.xyz` | 顺序 |
| CIF | `.cif` | 单帧 |
| XSF / AXSF | `.xsf`, `.axsf` | 单帧 / 顺序 |
| CP2K XYZ | `.cp2kxyz` | 顺序 |
| GROMACS | `.gro` | 单帧 / 顺序 |
| n2p2 data | `.n2p2` | 顺序 |
| MSI（Materials Studio） | `.msi` | 单帧 / 多模型 |
| XSD（Materials Studio） | `.xsd` | 单帧 |
| CAR（Biosym） | `.car` | 单帧 |

### 写入

| 格式 | 扩展名/名称 | 模式 |
|--------|------------------|------|
| VASP POSCAR | `.vasp`, `POSCAR` | 单帧 |
| VASP XDATCAR / v6 | `XDATCAR`, `XDATCAR6` | 顺序 |
| LAMMPS data | `.lmpdat`, `.lmp` | 单帧 |
| LAMMPS dump | `.lammpstrj` | 顺序 |
| MLIP CFG | `.cfg` | 顺序 |
| MLIP binary CFG | `.cfgbin` | 顺序 |
| Extended XYZ | `.extxyz` | 顺序 |
| CIF | `.cif` | 单帧 |
| XSF / AXSF | `.xsf`, `.axsf` | 单帧 / 动画 |
| VTK | `.vtk` | 单帧 |
| CP2K XYZ | `.cp2kxyz` | 顺序 |
| n2p2 data | `.n2p2` | 顺序 |
| DLPOLY4 | `.dlpoly` | 单帧 |

## 格式说明

### 文件名识别与显式格式

表中列出的是可识别文件名，不涵盖所有惯用后缀。未显式启用内容探测时，检测
依据文件名。`data.lammps` 或 `input.data` 需按实际格式传入 `"lmpdat"` 或
`"n2p2"`；`trajectory.cp2k.xyz` 默认识别为 `"extxyz"`，CP2K 数据需指定
`"cp2kxyz"`。写入时也使用相同的显式格式名，例如：

```cpp
io.read(structures, "input.data", "n2p2");
io.write(structures, "output.data", "n2p2");
```

读写方向与文件名检测是不同的能力。显式格式名见
[Python 格式表](../reference/python_zh.md#支持的格式)；独立流会话支持的格式
更少，见[共享工作流](../advanced/shared-workflows_zh.md)。

### VASP OUTCAR

- 读取头部 `ions per type` 行时会读取该行上的全部计数：多元素计算（如 Fe2O3 的 `2 3`）可以得到正确的总原子数与元素映射。
- POTCAR 元素名可能带有价电子后缀（`Fe_sv`、`K_pv`、`O_d` 等）；映射为原子序数前会去掉该后缀。
- 从首行（` vasp.5...` / ` vasp.6...`）读取 VASP 版本号。行过短、不含版本数字时直接报错，而不会越界读取。
- 恰好在其第 NELM 次电子迭代达到收敛的离子步会被接受（VASP 仍会在最后一次迭代后写出 `aborting loop because EDIFF is reached` 行）。达到 NELM 仍未收敛的步仍会被丢弃。

### CIF

- `loop_` 块中的每个 `_atom_site_*` 标签独占一行，这是 ASE 的 CIF 解析器等严格读取器所要求的格式。

### Materials Studio 格式（`msi`、`xsd`、`car`）

- **MSI**（Cerius2 DataModel）：每个 `(N Model ...)` 块是一个结构；`read()` 返回全部模型。晶胞取自 `A3`/`B3`/`C3` 记录，元素取自 `ACL` 记录，坐标为笛卡尔坐标（埃，与 OpenBabel 读取器一致）。存在时读取键（Bonds）和 `Charge` 记录。
- **XSD**：支持所有布局变体（扁平、嵌套的 `SymmetrySystem`/`IdentityMapping`、非周期 `Molecule`）。晶胞来自 `SpaceGroup` 的 `AVector`/`BVector`/`CVector`；有晶胞时 `Atom3d` 的 `XYZ` 为分数坐标，`CartesianXYZ` 始终为笛卡尔坐标。存在时读取 `Bond` 元素和 `Charge` 属性。
- **CAR**（Biosym 归档）：晶胞来自 `PBC a b c alpha beta gamma` 行，9 列原子记录（`label x y z XXXX mol potential element charge`）；电荷列可选。支持非周期文件（`PBC=OFF`）。

### LAMMPS 轨迹（`lammpstrj`）

- 向量属性按 LAMMPS 从 1 开始的列下标写出（`desc[1] desc[2]`，而非 `desc_[0]`），与读取端一致，保证逐原子向量属性无损往返。

## 格式检测

```cpp
// 通过文件扩展名检测
GPIO::supported_format fmt = GPIO::guess_format("structure.cfg");
// 返回 GPIO::MLPCFG

// 通过格式名称字符串检测
GPIO::supported_format fmt2 = GPIO::format2enum("poscar");
// 返回 GPIO::POSCAR

// 格式枚举
switch (fmt) {
    case GPIO::POSCAR:    /* ... */ break;
    case GPIO::MLPCFG:    /* ... */ break;
    case GPIO::LMPTRJ:    /* ... */ break;
    case GPIO::EXTXYZ:    /* ... */ break;
    // ...
}
```

## 化学映射

GPIO维护一个整数<->字符串之间的化学物种映射：

```cpp
GPIO io;

// 默认映射：原子序数 -> 元素符号
io.setmap();

// 若模型的物种 0 = Al、1 = Ni，应在读取前设置映射
io.setmap({"Al", "Ni"});

// 或从文件加载（每行一个"id element"对）
io.readmap("index.map");

// 查询映射
std::map<int, std::string> id2name = io.getmap();
std::map<std::string, int> name2id = io.getrevmap();

std::string element = io.get_type(0);  // "H", "O"等

// 获取物种列表
std::vector<int> species;
io.report_chem(species);
```

## 配置选项

```cpp
GPIO io;

// 输出控制
io.setoutrate(1, 0, INT_MAX);  // 所有帧，从头到尾
io.outrate = 2;                 // 每2帧写入一次
io.out_startfrom = 10;          // 从第10帧开始
io.out_endto = 100;             // 在第100帧结束

// 格式特定选项
io.write_stress = true;          // EXTXYZ: 输出 -virial/体积，单位 eV/Å³
io.write_virtual = true;         // EXTXYZ: 输出广延 virial（历史变量名）
io.absolute_index_write = false; // 输出结构中的原始物种索引
io.write_symbol = true;          // 写入元素符号（而非数字）
io.header_line = "My calculation";  // 自定义标题

// 读取选项
io.read_startfrom = 0;
io.read_endto = INT_MAX;
io.read_irate = 1;               // 每N帧读取一次
io.drop = false;                 // 跳过未收敛的结构？
```

## 示例：格式转换

```cpp
#include "io/gpio.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: converter INPUT OUTPUT\n";
        return 1;
    }
    GPIO io;
    std::vector<Atoms> structures;

    // 读取所有结构
    io.read(structures, argv[1]);

    // 以不同格式写入
    io.write(structures, argv[2]);

    return 0;
}
```

用法：
```bash
./converter input.cfg output.lmpdat
./converter POSCAR output.extxyz
```

## 参见

- @ref io — I/O模块
- @ref api_core — Atoms API
- @ref guide_commands — CLI命令（包括`convstr`）
