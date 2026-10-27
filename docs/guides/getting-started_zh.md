# 开始使用：选任务与准备输入 {#guide_getting_started}

[English](getting-started.md)

第一次使用不需要先理解 C++ 类或全部参数。先确认自己有没有可用的势函数，再选择
下面的一条路线。本手册的命令示例在 Linux 终端运行；`imrmlp` 应已加入 `PATH`。
尚未安装时先看[安装指南](installation_zh.md)。

## 我应该从哪里开始？

| 你现在的情况 | 下一步 | 完成后得到什么 |
| --- | --- | --- |
| 还没有模型，只想确认程序能用 | [自包含入门示例](../../examples/quickstart/README_zh.md) | 自动生成的小型模型、数据、训练日志和验证结果 |
| 已有适用于目标材料的模型 | 本页的单点计算，然后按下表选择任务 | 带能量、力和应力的结构文件 |
| 有参考计算数据，想训练自己的模型 | [训练、评估与部署](workflows_zh.md) | 经独立验证的模型和可复现的训练记录 |
| 只想用已有模型做 LAMMPS MD | [LAMMPS 接口](../../LAMMPS/README_zh.md) | 安装接口、选择 pair style 并做 `run 0` 核对 |
| 想在 Python / ASE 中编写工作流 | [Python 安装与示例](../../README-python_zh.md) | 可导入的包与 ASE 计算器 |

自包含示例中的 `Al` 只演示元素映射，其合成模型不能替代真实铝材料势。
真实任务需取得或训练覆盖目标元素、结构和条件的模型；模型文件不包含所有材料的通用参数。

## 没有自备输入时，按这条顺序练习

1. [安装](installation_zh.md)后运行[训练入门](../../examples/quickstart/README_zh.md)，得到 `build/quickstart/`。
2. 用本页“接着入门示例练习”的命令查看单点输出；已完成第 1 步时不必再次运行生成脚本。
3. 用 [LJ 连续计算](../../examples/basic-calculations/README_zh.md)直接练习计算，或用 [MTP 连续练习](../../examples/mtp-workflow/README_zh.md)从基组生成、训练与验证开始，再完成弛豫、NEB 和物性。
4. 用[主动学习练习](../../examples/active-learning/README_zh.md)学习评分、阈值和候选保存，再了解参考标注与重训。

各练习有独立目录和模型；后续真实任务使用下面的普通 MTP 文件约定。
不需要把上一练习的势或输出逐个改名来凑齐下一页的占位文件。

## 做自己的计算前，准备三个文件

为每次任务新建一个目录，把以下文件放在里面，再进入该目录运行命令：

```text
my-calculation/
    model.mtp      已训练的普通 MTP 模型
    POSCAR         待计算的结构，含元素符号
    index.map      模型的元素顺序
```

这里和后续任务页的文件名都是占位名称，需换成自己的文件。
`model.mtp` 来自模型提供方或训练输出（如 `fitted.mtp`，可修改配置中的文件名）；
`POSCAR` 来自自己的建模或参考计算结构；`index.map` 来自模型的物种顺序记录。
这些输入不会由单点命令自动创建。
同一目录还可保存 `relax.ini`、`neb.ini` 等设置及日志。**相对路径从运行命令时的
当前目录解析，不从设置文件所在目录解析。** 输出文件名应与输入文件名区分。

`index.map` 每行写一个模型元素，例如模型 species 0 = Al、species 1 = Ni：

```text
Al
Ni
```

顺序应来自模型或其训练记录。POSCAR 中元素块的顺序可以不同，读取时借助元素名
映射；CFG 中的数值类型已经是模型 species 索引，不能把原子序数 13 当成 species 0。
不要仅因交换 `index.map` 后程序能运行，就认为映射已正确。

## 第一次单点计算

在上述任务目录运行：

```sh
imrmlp calc_efs POSCAR predicted.xyz --potfile model.mtp --index index.map
```

本页及后续基础示例沿用普通 MTP 默认类型，文件格式按文件名识别；
只有使用其他势类型或文件命名无法识别格式时，才需显式指定相应选项。

单点计算保持输入几何不变，用势函数预测总能量、每个原子的力和应力。
`predicted.xyz` 是扩展 XYZ 文本，保留结构及预测属性，可用文本编辑器或支持 EXTXYZ
的软件检查。看程序是否正常退出、输出原子数和元素是否正确、能量和力是否有限，
并与相同结构的参考结果比较。单纯生成文件不能证明模型在目标材料上准确。

有带参考 E/F/应力标签的数据时，另外评估误差：

```sh
imrmlp calc_err --input reference.xyz --potfile model.mtp \
  --index index.map --output-summary errors.txt
```

无参考标签的 POSCAR 可以做预测，不能凭空生成验证误差。

### 没有自己的输入时，接着入门示例练习

先按[自包含入门示例](../../examples/quickstart/README_zh.md)生成默认目录
`build/quickstart/`，再从仓库根目录进入它：

若安装页已生成该目录，直接使用；若自定义了 `--work`，把下面的目录改成实际输出路径。

```sh
cd build/quickstart
imrmlp calc_efs valid-input.cfg my-prediction.xyz --potfile fitted.mtp --index index.map
head -n 5 my-prediction.xyz
```

每帧第一行的 `3` 是原子数，第二行的 `energy` 是该三原子体系的总能量；
`Properties` 声明后续每列的含义，`forces:R:3` 表示每个原子的三个力分量。
文件共有四帧，不是一个十二原子结构。`valid-input.cfg` 只含几何；对应的
`valid.cfg` 才有参考标签，可用它运行 `calc_err` 并与 `errors.txt` 对照。
练习结束后用 `cd ../..` 返回仓库根目录，再从那里进入其他示例或自己的任务目录。

## 设置文件如何写？

设置文件使用空白分隔键和值，**不写等号**。例如：

```text
input POSCAR
output relaxed.xyz
index index.map
pot:potfile model.mtp
```

这只是语法示例，完整弛豫配置见[弛豫指南](relaxation_zh.md)。`pot:` 表示势函数设置，
`relax:` 表示弛豫设置，`neb:` 表示路径设置。为避免 section 与空行规则混淆，
任务页统一使用显式前缀。重复键保留第一次出现的值；修改设置时替换原行。
`.mtp` 模型文件有自己的语法，不应把它当作 `.ini` 配置编辑。

## 单位、模型类型和结果含义

| 项目 | 本项目的常用约定 |
| --- | --- |
| 位置、晶胞 | Å |
| 总能量 | eV；每原子能量需再除以原子数 |
| 力 | eV/Å |
| 原生广延应力 / 维里 | eV；转为应力需按体积及符号约定处理 |
| 外压 | 设置中的 `pressure` 使用 GPa，正值表示压缩 |
| 模型类型 | 本页用普通 `mtp`；MTP_M、EAM、LES 等使用各自的势类型和设置 |

CLI 各命令的模型选项不完全相同，不能把 `calc_efs` 的选项直接套给其他命令。
模型家族及旧格式要求见[势类型总览](../reference/potential-types_zh.md)，
详细单位和格式见[配置参考](../reference/configuration_zh.md)。

## 下一步按任务走

| 目标 | 使用引导 |
| --- | --- |
| 固定晶胞优化原子，或同时优化晶胞 | [结构弛豫](relaxation_zh.md) |
| 已知初态、终态，求迁移或反应势垒 | [NEB 最小能量路径](neb_zh.md) |
| 声子、弹性常数、状态方程 | [晶体性质计算](properties_zh.md) |
| 发现陌生结构、补充参考数据并重训 | [主动学习与候选选样](active-learning_zh.md) |
| 查某个选项的定义 | [命令参考](../reference/commands_zh.md)、[速查表](../reference/cheatsheet_zh.md) |

遇到错误先查[故障排查](troubleshooting_zh.md)。保留输入、配置、模型、元素映射、
日志和 `imrmlp --version` 输出，方便复现计算。
