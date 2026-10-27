# Python examples / Python 示例

[English API guide](../docs/reference/python.md) · [中文 API 指南](../docs/reference/python_zh.md)

Build the extension with `make python PYTHON=python3` or install the package with
`python3 -m pip install .` after preparing the [native dependencies](../docs/guides/installation.md).
For in-tree use, run from the repository root with `PYTHONPATH=.` so Python finds
the source package. To use a pip-installed package while in the repository root,
run scripts with `python3 -I examples-python/02_build_atoms.py` (without
`PYTHONPATH`), so the source package does not shadow the installation.
Optional ASE and Phonopy dependencies are described in the
[Python setup guide](../README-python.md).

先按[构建指南](../docs/guides/installation_zh.md)准备本地依赖，再运行
`make python PYTHON=python3`，或用 `python3 -m pip install .` 安装。
直接使用源码包时，请从仓库根目录运行并设置 `PYTHONPATH=.`。
如需在仓库根目录使用 pip 安装的包，请运行
`python3 -I examples-python/02_build_atoms.py`（不设置 `PYTHONPATH`），
以免源码包遮蔽已安装版本。
ASE、Phonopy 等可选依赖见 [Python 安装说明](../README-python_zh.md)。

| Script / 脚本 | Purpose and required input / 用途与输入要求 |
|---|---|
| `02_build_atoms.py` | Construct and convert a small structure; no input files. ASE conversion is optional. / 构造结构并转换坐标，无需输入文件；ASE 互转可选。 |
| `06_train.py` | Fit and reload a descriptor linear model on generated synthetic labels; no input files. `--work` sets the output directory. / 用合成标注拟合、重载描述符线性模型；无需输入文件，`--work` 设置输出目录。 |
| `08_threading.py` | Inspect version/MPI support and set the OpenMP thread count; no input files. / 查看版本、MPI 支持并设置 OpenMP 线程数，无需输入文件。 |
| `01_read_write.py` | I/O template; provide `POSCAR` and `trajectory.xyz` in the working directory. Writes `output.cfg` and `output.extxyz`. / 读写模板；需在工作目录提供 `POSCAR` 和 `trajectory.xyz`，写出 `output.cfg` 与 `output.extxyz`。 |
| `03_calc_efs.py` | Evaluation template; replace `Al.mtp` with your trained one-species Al model. / 求值模板；需把 `Al.mtp` 替换为自己的单元素 Al 势。 |
| `04_relaxation.py` | Relaxation template; provide a compatible Al model and POSCAR; adapt `elements` to the model species order. / 弛豫模板；需提供匹配的 Al 势和 POSCAR，按模型物种顺序修改 `elements`。 |
| `05_ase_integration.py` | ASE optimization/MD template; provide a validated Al model and pass `elements=["Al"]` to its calculator. / ASE 优化、MD 模板；需可靠的 Al 势，并在计算器中传入 `elements=["Al"]`。 |
| `07_active_selection.py` | Matrix-level MaxVol and CUR selection on synthetic data; no input files. For structure-level FPS use `fps_select` with a configured potential or descriptor. / 基于合成数据演示矩阵级 MaxVol 和 CUR 选择，无需输入文件。结构级 FPS 使用已配置势或描述符的 `fps_select` 命令。 |
| `09_phonopy.py` | Command-line example requiring a relaxed structure, a compatible model, ASE and Phonopy. / 命令行示例；需弛豫后的结构、匹配模型、ASE 和 Phonopy。 |

Start with the examples that need no external data / 建议先运行无需外部数据的示例：

```sh
PYTHONPATH=. python3 examples-python/02_build_atoms.py
PYTHONPATH=. python3 examples-python/06_train.py
PYTHONPATH=. python3 examples-python/07_active_selection.py
PYTHONPATH=. python3 examples-python/08_threading.py
PYTHONPATH=. python3 examples-python/09_phonopy.py --help
```

Only `06_train.py` and `09_phonopy.py` accept command-line options. Other scripts
use file paths and parameters written in the source; edit a copy before using
your data. `06_train.py` writes under `build/examples-python/06-train/` by default
and replaces its own model files on repeated runs. The synthetic training and
[quickstart](../examples/quickstart/README.md) models demonstrate the interface;
use a validated material model for physical predictions, optimization and MD.

只有 `06_train.py` 和 `09_phonopy.py` 提供命令行选项。其余脚本直接使用源码中的
文件名与参数，处理自己的数据时请先复制再修改。`06_train.py` 默认输出到
`build/examples-python/06-train/`，重复运行会替换其中的模型文件。合成训练和
[快速入门](../examples/quickstart/README_zh.md)模型用于演示接口；材料预测、
结构优化和 MD 应使用已验证的材料模型。

`elements` always follows model species order, for example `["Mg", "Al"]` means
species 0 = Mg and species 1 = Al. This order is independent of atomic numbers.
File I/O does not automatically attach this mapping to the Python object.

`elements` 必须按模型物种顺序填写。例如 `["Mg", "Al"]` 表示物种 0 为 Mg、
物种 1 为 Al，与原子序数无关；从文件读取时，Python 对象不会自动附带此映射。
