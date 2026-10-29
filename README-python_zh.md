# IMR-MLP Python 与 ASE

[English](README-python.md) · [API 指南](docs/reference/python_zh.md)

需要 Python 3.10+ 以及[构建指南](docs/guides/installation_zh.md)中的本地依赖。
先按该指南创建并激活虚拟环境；若希望稠密线性代数更快，再启用可选的 oneMKL，
然后在仓库根目录选择一种安装方式。
新用户可直接安装包和 ASE：

```sh
USE_KIM=0 python3 -m pip install '.[ase]'
python3 -I -c "import imrmlp; print(imrmlp.__version__)"
```

无需 ASE 时将 `'.[ase]'` 换成 `.`。若只在源码树使用，也可选
`python3 -m pip install numpy pybind11`，再运行 `make python PYTHON=python3 USE_KIM=0`；
不必把两种方式都做一遍。两种方式仍需要本地开发库。
`make install` 只安装 CLI 和库，不安装 Python 包。

源码树中也有 `imrmlp` 包目录。使用 pip 安装时，下方示例使用 `python3 -I`，
避免源码目录遮蔽已安装的包及其扩展；使用 `make python` 源码树构建时，
改用 `PYTHONPATH=. python3`。两种方式都从仓库根目录运行，以解析示例中的相对模型路径。

先完成[CLI 训练入门](examples/quickstart/README_zh.md)，取得
`build/quickstart/fitted.mtp`；安装页已运行过时直接复用。若指定了其他 `--work`，
替换下方路径。将代码保存为 `build/python-single-point.py`，在仓库根目录执行
`python3 -I build/python-single-point.py`（源码树构建则改用 `PYTHONPATH=. python3`）：

```python
import numpy as np
import imrmlp

# 先在仓库根目录运行 CLI 快速入门示例
pot = imrmlp.Potential("build/quickstart/fitted.mtp", "mtp")
atoms = imrmlp.Atoms(
    numbers=[13, 13, 13],
    elements=["Al"],  # 模型 species 0 = Al，不是原子序数 13
    positions=[[7, 7, 7], [9.5, 7, 7], [7.8, 9.0, 7.1]],
    cell=np.eye(3) * 20.0,
    pbc=True,
)
pot.calculate(atoms)
print(atoms.energy)  # 总能量 eV
print(atoms.forces)  # eV/Å
print(atoms.stress)  # 广延应力，eV
imrmlp.write("build/quickstart/python-predicted.cfg", atoms)
```

`Potential(filename)` 默认使用 `pottype="auto"`：GPM 文件从模型包读取类型；
旧格式文件按 `mtp` 加载。EAM、SW 等旧格式需显式指定类型，不能依靠扩展名判断。
用 `Atoms(numbers=..., elements=[...])` 构造结构时，请按模型的 species
顺序提供 `elements`。ASE 计算器也使用该参数；`Atoms.from_ase()` 和
`Atoms.from_pymatgen()` 不接受它，返回的类型仍为原子序数。从 CFG 文件读入时
保留数值 species 索引，它们不一定是原子序数。

运行后应打印有限的总能量、3×3 力数组和 3×3 原生应力张量，并生成
`build/quickstart/python-predicted.cfg`。这一步不需要自己准备 POSCAR。

已安装 ASE 时，将以下代码另存为 `build/python-ase.py`，同样从仓库根目录运行
`python3 -I build/python-ase.py`（源码树构建则改用 `PYTHONPATH=. python3`）；
未安装时先执行 `python3 -m pip install ase`：

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
print(atoms.get_stress())           # eV/Å³，顺序 xx yy zz yz xz xy
```

合成快速入门示例只是接口演示，不是 Al 材料模型。原生应力是广延量（eV）；
ASE 应力按 `-atoms.stress / atoms.volume` 转换（eV/Å³），同时改变符号。
两段代码使用相同模型和几何，能量与原子力应一致；应力按上述约定比较。
弛豫、NEB 和物性输入的获取见[连续计算练习](examples/basic-calculations/README_zh.md)，
更多 Python 功能再看 [API 指南](docs/reference/python_zh.md)。

使用 Phonopy 时运行 `python3 -m pip install '.[phonopy]'` 安装可选依赖，再运行
[09_phonopy.py](examples-python/09_phonopy.py)。可选的
`imrmlp.phonopy.calculate_forces` 适配器把 ASE 力提供给 Phonopy，
力常数拟合与声子后处理由 Phonopy 完成。

编译出的扩展模块是 `imrmlp._imrmlp`。自研包装代码使用 MIT 许可，
被链接的依赖保留各自许可。加载或导入问题见
[第三方许可说明](THIRD_PARTY_NOTICES.md)与[故障排查](docs/guides/troubleshooting_zh.md)。

`imrmlp.Dimer` 支持显式 MODECAR 输入、可复现的种子和最终状态结果，
参见[鞍点工作流](docs/advanced/saddle-workflows_zh.md)。
