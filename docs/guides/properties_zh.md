# 晶体性质：声子、弹性与状态方程 {#guide_properties}

[English](properties.md)

本页用于已经有模型、并已完成[结构弛豫](relaxation_zh.md)的用户。
同一材料的单点误差较小，不代表其力常数、弹性或体积响应一定可靠；应分别检查目标性质。
所有示例都需要自己的 `model.mtp`、`relaxed.vasp` 和 `index.map`，在独立目录运行。

**还没有配套输入？** [连续计算练习](../../examples/basic-calculations/README_zh.md)先生成并
弛豫教学 LJ 晶体，再明确把 `bulk.vasp` 复制为各目录的 `relaxed.vasp`，可跑通本页各类计算。
练习已使用 LJ 专用配置；本页正文则用自己的普通 MTP。真实任务的 `relaxed.vasp` 来自
[弛豫指南](relaxation_zh.md)最终结果的格式转换；EOS/零压弹性还应先完成相应晶胞优化。
从基组生成和拟合开始，并检验同一模型物性时，使用 [MTP 连续练习](../../examples/mtp-workflow/README_zh.md)。

## 先选择要计算的量

| 目标 | 工具 | 开始前确认 |
| --- | --- | --- |
| 原子位移的二阶响应、谐波力常数 | `calc_fc` | 平衡结构、超胞大小、Hessian 后端 |
| 声子色散、DOS、振动热力学 | IMR-MLP 求力/力常数，Phonopy 后处理 | 原子顺序、超胞定义和质量一致 |
| 应变下的刚度与机械响应 | `calc_elastic` | 平衡应力、离子弛豫约定和应变幅度 |
| 平衡体积、体模量、E–V 曲线 | `calc_eos` | 同一相的稳定体积区间和每点弛豫设置 |

## 1. 导出力常数

从常规晶胞构建 2×2×2 超胞并输出完整力常数：

```sh
imrmlp calc_fc --potfile model.mtp \
  --input relaxed.vasp --index index.map --supercell 2 2 2 \
  --reduced=false > fc.log 2>&1
```

检查 `FORCE_CONSTANTS`、`supercell_created.xsf` 和 `sup_debug_efs.xyz`。
力常数块的单位为 eV/Å²，顺序对应**实际导出的超胞**。
`2 2 2` 只是起点，需增加超胞检查收敛。默认解析后端在模型不支持时会报错；
可用 `--hessian-method auto` 在支持时用解析，否则回退到力的有限差分，
或显式选择 `force_fd` 并检查 `--hessian-nudge-pos` 位移步长。

`--lowest-modes N` 是另一种矩阵无关的最低 Gamma 模诊断：输出本征模，不写完整
`FORCE_CONSTANTS`，不能用来替代整个布里渊区的声子色散。
更多后端及输出定义见[命令参考](../reference/commands_zh.md)。

## 2. 用 Phonopy 得到色散或 DOS

也可采用 Phonopy 生成位移、IMR-MLP 计算受力的路线。与上一节二选一，
在新目录复制同一平衡结构、模型和映射，避免混入另一套 `FORCE_CONSTANTS`。
以下采用 Phonopy 4 的命令拆分：

```sh
phonopy-init -d --dim="2 2 2" -c relaxed.vasp
imrmlp phonopyfc --potfile model.mtp --disp phonopy_disp.yaml \
  --index-file index.map
```

上例是普通 MTP，需另外安装 Phonopy。`phonopy-init` 生成 `phonopy_disp.yaml`，
`phonopyfc` 写出 `FORCE_SETS`。随后可计算一个示例 DOS 网格：

```sh
phonopy --mesh="10 10 10" --dos phonopy_disp.yaml
```

检查 `total_dos.dat` 和日志，再调整网格或使用适合材料的能带路径。Phonopy 3 的
旧教程使用 `phonopy -d`；Phonopy 4 将准备步骤移至 `phonopy-init`，后处理读取 YAML，
不再重新提供 `-c` / `--dim`。版本区别见[官方迁移说明](https://phonopy.github.io/phonopy/migration-v4.html)。
`phonopyfc` 使用 `--index-file`，不是其他命令常用的
`--index`；若目录里有 `settings.ini`，应检查其中的 `pot:` 设置是否与本次模型一致。
Phonopy 的网格、能带路径和绘图选项见其[命令说明](https://phonopy.github.io/phonopy/command-options.html)
和[设置标签](https://phonopy.github.io/phonopy/setting-tags.html)。

也可通过[Python Phonopy 适配器](../reference/python_zh.md#phonopy-integration)
在内存中计算位移超胞的力。若直接导入解析 `FORCE_CONSTANTS`，必须使用 Phonopy
自己的超胞作为 `calc_fc` 输入，省略再次扩胞，并确保矩阵两个原子轴与
`phonon.supercell` 的顺序完全一致；仅原子数相同不够。

虚频可能来自真实失稳，也可能来自结构未收敛、差分/超胞误差、模型或单位/顺序问题。
先检查平衡力、声学模和数值收敛，再解释物理。声子力常数不等同于有限温度有效力常数；
后者另见[简谐参考与 TI](../advanced/harmonic-ti_zh.md)。

## 3. 计算弹性常数

普通 MTP 可先使用默认的自动后端：

```sh
imrmlp calc_elastic --potfile model.mtp --input relaxed.vasp \
  --index index.map --elastic-output elastic.out > elastic.log 2>&1
```

默认的 `auto` 选择受支持的计算方式；需要解析模式时使用 `--method analytical` 并确认模型
支持相应原子/晶胞 Hessian。需要有限差分核对时，可建立 `elastic.ini`：

```text
relax:cartesian true
relax:gtol 0.0001
relax:maxiteration 500
```

```sh
imrmlp calc_elastic --potfile model.mtp \
  --input relaxed.vasp --index index.map --method numerical \
  --settings elastic.ini --strain-magnitude 0.005 --elastic-output elastic-numerical.out
```

比较不同应变幅度以及更紧的原子弛豫容差。不要把“离子固定”和“允许内部坐标弛豫”
的刚度混为一谈；解析模式的 `--relax-correction` 控制相关修正。
输出中的刚度/模量按表头单位读取，非零外压下还需区分材料刚度和用于机械稳定性的
Born/Wallace 切线张量。若报内部软模或刚度奇异，先检查结构和模型的稳定性；
关闭离子修正会改变所求物理量，不能把它当作通用修复。完整定义见[弹性命令说明](../reference/commands_zh.md)。

### 小练习：把应变幅度减半

保留上面的 `elastic-numerical.out`（应变幅度 `0.005`，即 0.5%），用相同输入和
内部原子弛豫设置再算一次：

```sh
imrmlp calc_elastic --potfile model.mtp \
  --input relaxed.vasp --index index.map --method numerical \
  --settings elastic.ini --strain-magnitude 0.0025 \
  --elastic-output elastic-half-strain.out > elastic-half-strain.log 2>&1
```

例如，若两次输出的同一 `C11` 分别为 `162.4`、`162.3 GPa`，差值为 `0.1 GPa`，
约为 0.06%；这只是读法示例，不是某种材料的预期值或统一验收标准。
同时比较其他独立分量、模量及各应变点的原子收敛情况。若步长减半后结果明显
漂移，检查弛豫容差和有限差分稳定性；步长越小并不总越准确。要检验声子超胞
或 DOS 网格收敛，也采用这种“一次只改一个数值设置、分别保存结果”的做法。

## 4. 计算状态方程

保存 `eos.ini`，指定与平衡结构相同的势和弛豫精度：

```text
pot:potfile model.mtp
relax:gtol 0.0001
relax:maxiteration 500
```

```sh
imrmlp calc_eos --settings eos.ini --input relaxed.vasp --index index.map \
  --output ev.dat --output-eos eos.dat
```

默认扫描参考体积附近约 ±5%，得到曲线和 EOS 拟合参数。
查看采样点是否处于同一相、是否覆盖能量极小值、拟合体积和体模量是否合理，
再改变体积窗口与点数检查稳定性。没有稳定正体积极小值时，不应把拟合数字当作
平衡性质。压力扫描使用 `--curve pv`，与 E–V 路径的输入选项不同。

### EOS 输出的列和单位

| 文件 / 字段 | 含义与单位 |
| --- | --- |
| `ev.dat`：`Volume/at`、`Energy/at`、`Enth/at` | Å³/atom、eV/atom、eV/atom |
| `ev.dat`：`Press` | GPa |
| `eos.dat`：`V0`、`E0` | 拟合平衡体积与能量，Å³/atom、eV/atom |
| `eos.dat`：`B0` | eV/Å³；乘 160.217662 转为 GPa |
| `eos.dat`：`B0'` | 体模量对压力的导数，无量纲 |

以上针对本页直接扫描结构的 E–V 命令。若改用外部曲线文件拟合，体积/能量归一化
随输入数据，不能自动假定为每原子。单点 `calc_efs` 的总能量不能直接与 `Energy/at` 比较。

## 如何记录和比较结果

保存平衡结构、模型、元素映射、所有配置和日志，并记录超胞、位移/应变步长、
外压以及离子弛豫约定。先做数值收敛，再与参考数据比较目标性质。
无需先阅读 C++ API；特殊的高阶导数、HVP 和有限温度流程可再进入[进阶使用](../advanced/index_zh.md)。
