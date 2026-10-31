# 训练、评估与部署 {#guide_workflows}

[English](workflows.md)

用自己的参考数据训练普通 MTP，划分验证集与最终测试集，评估后把同一模型部署到
CLI、LAMMPS 或 Python。

不涉及：已有模型时的能量、弛豫、路径与性质计算（见[单任务计算](single-point_zh.md)
各页）、基组候选池的生成与删减（见[基组选择](basis-selection_zh.md)）。

## 前置输入

| 文件 | 内容 | 来源 |
| --- | --- | --- |
| `index.map` | 模型元素顺序 | 约定，见下节 |
| `reference.cfg` 等 | 带 E/F/应力标签的参考数据 | 参考计算后用 `convstr` 转换 |
| `template.mtp` | 模板模型，含预期的元素、径向基、截断与矩基 | 自行准备或示例提供 |
| `train.xyz`、`valid.xyz`、`test.xyz` | 划分后的训练、验证、最终测试集 | 由参考数据划分 |

命令假定 `imrmlp` 位于 `PATH`，或已替换为对应构建版本的绝对路径。每次计算使用独立
工作目录；配置中的相对路径相对于进程工作目录解析，见[输入文件与配置约定](conventions_zh.md)。

## 元素映射与参考标签

`index.map` 按模型内部物种顺序，每行写一个元素符号。Al/Ni 分别对应物种索引 0、1：

```text
Al
Ni
```

预处理、训练和部署必须使用同一顺序。LAMMPS atom type 从 1 开始，而模型 species
索引从 0 开始；新的 LAMMPS 输入推荐按元素名映射。映射写错时数值仍在合理范围，只是
能量和力整体变差，核对方法见[输入文件与配置约定](conventions_zh.md)。

逐个转换参考计算，再拼接 CFG 记录：

```sh
imrmlp convstr OUTCAR reference.cfg
imrmlp convstr vasprun.xml reference-xml.cfg
cat reference.cfg reference-xml.cfg > dataset.cfg
```

上述文件名是占位示例。XDATCAR 等仅含几何的轨迹不会因格式转换自动获得能量、力或应力
标签。应核查参考计算的电子收敛与标签单位；离子优化的停止阈值只约束结构弛豫，电子步
收敛要单独确认。

位置和晶胞使用 Å，总能量为 eV，力为 eV/Å。原生广延应力张量使用 eV；`calc_err` 还会
报告按体积归一化后的 GPa 应力。从其他代码导入时应遵循 IO 规定的符号、分量次序和
归一化约定。

## 数据集划分

尽量按轨迹、相、成分或生成计算划分，避免相邻 MD 帧泄漏到不同集合。验证集用于选择
模型和设置，最终测试集应在这些选择结束后评估。一条命令生成 80% / 10% / 10% 三份：

```sh
imrmlp split_set dataset.cfg split_ --ratios=8,1,1 --shuffle yes
cp split_0.xyz train.xyz
cp split_1.xyz valid.xyz
cp split_2.xyz test.xyz
```

`split_set` 写出 extxyz 文件。比例模式在可选打乱后按连续分段分配，整数子集大小
采用最大余数法取整，因此不加 `--shuffle` 时文件中相邻的结构会整块落入同一子集，
与按轨迹划分的要求一致。若只需要 N 份等长子集（并行标注、逐份检查），改用
`--nsplit N` 且不给 `--ratios`：它按轮转交错分配，默认 N=10。

当前 CLI 没有随机种子参数，复现时需保存实际划分结果。不要将 extxyz 直接改名为
`.cfg` 后再依赖扩展名自动识别格式。

自包含入门示例只提供训练与验证两份 CFG，不含已准备好的最终测试集；它用于核验流程
而非代替本节的三集合划分。

## 模板选择

模板应包含预期的元素数与顺序、径向基、截断和矩基，并保留未经修改的副本。用
`mtmbasis` 生成候选池、再用 `fastfit` 精简的过程见[基组选择](basis-selection_zh.md)；
生成器与径向基设置见[配置参考](../reference/configuration_zh.md)。生成器配置本身
不是完整可部署的模型。

模板不必从零构造。[MTM-new](https://github.com/wftao1995/MTM-new) 是一组可选的
预设基组，按基组规模分目录存放，每个目录是一份已经优化好的 `potential.mtp_0`，
与 `mtmbasis` 的输出同格式——单个元素、`RBChebyshev`、2–5 Å、8 个径向基函数，
不含任何拟合系数。

```sh
git clone https://github.com/wftao1995/MTM-new
```

挑一个规模合适的目录，复制为 `template.mtp` 即为训练起点。目录分两类：

| 目录 | 内容 |
| --- | --- |
| `ps-1/`、`ps-2/` | 两套缩并搜索参数下的结果；目录名中的第二个数字等于文件里的 `alpha_scalar_moments` |
| `shapeev/N` | Shapeev 原始基在层级 N 的定义，`settings.ini` 中 `max_lev = N` |

这些文件的 `species_count = 1`。多元素体系要在未训练的副本上改这一行，并与 `index.map`
的元素顺序对齐。目录里的 `settings.ini` 是 `mtmbasis` 的缩并搜索设置，不是训练配置。
名称中的标量数是基组规模而不是精度，`shapeev/30` 约 4000 个标量，代价要在 `fastfit`
基准里自己衡量。

`initmtpm` 用于 MTP-M 非线性头初始化，不是普通 MTP 训练必需的初始化步骤，不要直接
套用于普通模板。

## 训练配置与命令

保存 `train.ini`，键和值用空白分隔：

```text
# ===== Input/Output =====
index               index.map
trainset            train.xyz
validset            valid.xyz

# ===== Potential =====
potfile             template.mtp

# ===== Loss Weights =====
fit:force_weight    0.05
fit:stress_weight   0.001
fit:weight_L2       1e-8

# ===== Output =====
fit:save_to         checkpoint.mtp
```

这些权重以默认能量权重 1 为基准，仅为示例；应结合标签质量、归一化和目标物理量选择。

```sh
# ALGLIB L-BFGS：联合优化径向与线性系数
imrmlp trainmtp train.ini --do-lbfgs
cp _Trained_Potential_.mtp fitted.mtp

# 另一种选择：固定径向函数，仅拟合线性系数
imrmlp trainmtp train.ini --linear-fit
cp _Trained_Potential_.mtp fitted-linear.mtp
```

**必须显式选择优化器或 `--linear-fit`**：裸命令 `trainmtp train.ini` 不会自动选择
非线性优化方法。最终快照固定写入工作目录的 `_Trained_Potential_.mtp`；
`fit:save_to` 指定训练器检查点，不会重命名这个命令级最终输出。使用独立运行目录可避免
覆盖之前的模型。

非线性优化中可选的线性重拟合见[线性精修](../advanced/linear-refit_zh.md)。
`trainnlpot` 支持更多模型与设备路径，`train_linear` 是另一套通用线性拟合入口；
这些命令接受的训练设置不同，复用配置前应查看所选命令的 `--help`。

## 评估与复现记录

```sh
imrmlp calc_err --input valid.xyz --potfile fitted.mtp \
  --output-summary validation.txt
imrmlp calc_err --input test.xyz --potfile fitted.mtp \
  --output-summary test.txt
imrmlp calc_efs test.xyz predicted.xyz --potfile fitted.mtp
imrmlp --version > version.txt
sha256sum fitted.mtp template.mtp train.xyz valid.xyz test.xyz > inputs.sha256
```

分别报告每原子能量、力和应力误差，检查异常构型、不同相/成分和目标性质。在预期体积、
温度和缺陷范围验证；静态 E/F 误差只说明单帧精度，MD 稳定性要在 NVE/NVT 下跑一段，
看能量漂移和结构是否保持。保留元素映射、配置、模型、划分后的数据、日志、构建模式和
依赖版本。

### 误差报告读法

在入门示例的 `errors.txt` 中，能量统计覆盖 4 个验证构型，力统计覆盖 12 个原子；
误差接近数值精度，是因为这个合成问题能被所选模型精确表示。真实材料的误差反映拟合
残差与数据覆盖，读到的数字不会接近零。先核对参与统计的数量和标签是否齐全，再读各类误差。

`Energy per atom` 的 RMS 为 `0.004 eV` 即 `4 meV/atom`；`Forces` 的 RMS 为
`0.08 eV/Å` 即 `80 meV/Å`。报告中的 `Forces` 按力差向量统计，`Forces components`
按分量统计，比较不同模型时应使用同一行、同一验证集。平均值较低但最大误差很大时，
还需找出对应的相、缺陷或高温构型，决定是否补充参考数据；这一步靠数据覆盖，压低训练
误差解决不了。

## 部署

### LAMMPS

先[安装 LAMMPS 接口](../../LAMMPS/README_zh.md)，再使用：

```text
units metal
newton on
pair_style mlip/opt
pair_coeff * * fitted.mtp Al Ni
```

这里按 LAMMPS 类型顺序写 Al/Ni；单元素 Al 模型仅需 `Al`。旧模型无元素元数据时改用
已确认的数字 species 映射。该片段放在建盒/读入原子之后，完整输入见接口指南。
先用 `run 0`，在同一几何、晶胞、映射和边界条件下与 CLI 比较 E/F。样条模型使用
`mlip/spline`；LES/SOG 使用专用样式；热流计算需采用 transport pair style。

### Python 与 ASE

使用 [Python 接口](../reference/python_zh.md)，显式指定模型顺序的 `elements`。
主动学习的选样循环见[主动学习](active-learning_zh.md)，原理见 [MaxVol](../theory/maxvol_zh.md)。

## 常见错误

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| 训练结束但没有预期迭代输出 | 未指定 `--do-lbfgs` 或 `--linear-fit` | 显式选择优化方式，裸命令不会自动挑选 |
| 找不到自己命名的模型文件 | 最终快照固定写入 `_Trained_Potential_.mtp` | 训练后复制并按用途命名，或改用独立运行目录 |
| 训练误差很低但计算明显偏差 | 元素映射错误或数据划分泄漏 | 核对 `index.map`，按轨迹/相/成分划分集合 |
| 验证误差突然变好 | 验证集参与了模型或超参数选择之后的反复调整 | 用未参与选择的 `test.xyz` 报告最终指标 |
| 格式识别错误 | 把 extxyz 改名成 `.cfg` | 保留扩展名，或在配置中显式指定格式 |
| LAMMPS 能量与 CLI 不一致 | 类型顺序、映射或边界条件不同 | 先 `run 0` 在同一条件下逐项比对 |

## 相关

- [基组选择](basis-selection_zh.md)、[fastfit 成本选基](../advanced/fastfit-cost-selection_zh.md)
- [HVP 标签与拟合](../advanced/hvp-training_zh.md)、[MTP 变量投影](../advanced/mtp-native-varpro_zh.md)
- 训练后计算：[单点计算](single-point_zh.md)、[结构弛豫](relaxation_zh.md)、[NEB](neb_zh.md)、[晶体性质](properties_zh.md)
