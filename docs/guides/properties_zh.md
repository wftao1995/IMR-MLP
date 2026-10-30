# 晶体性质 {#guide_properties}

[English](properties.md)

用已训练的模型计算力常数、声子、弹性常数和状态方程。同一材料的单点误差较小，
不代表其力常数、弹性或体积响应一定可靠，每类性质都需要单独检查。

不涉及：平衡结构的求解（见[结构弛豫](relaxation_zh.md)）、模型训练（见
[训练、评估与部署](workflows_zh.md)）。

## 前置输入

| 文件 | 内容 | 来源 |
| --- | --- | --- |
| `model.mtp` | 已训练的普通 MTP 模型 | 训练输出或模型提供方 |
| `relaxed.vasp` | 平衡结构 | 弛豫结果用 `convstr` 导出的 POSCAR |
| `index.map` | 模型元素顺序 | 模型或其训练记录 |

每个计算在独立目录中进行；声子与弹性还要另装 Phonopy（仅声子后处理需要）。
`relaxed.vasp` 的具体路径与命名规则见[输入文件与配置约定](conventions_zh.md)。
EOS 与零压弹性要求先完成对应的晶胞优化。

## 计算项与工具

| 目标 | 工具 | 开始前确认 |
| --- | --- | --- |
| 原子位移的二阶响应、谐波力常数 | `calc_fc` | 平衡结构、超胞大小、Hessian 后端 |
| 声子色散、DOS、振动热力学 | IMR-MLP 求力/力常数，Phonopy 后处理 | 原子顺序、超胞定义和质量一致 |
| 应变下的刚度与机械响应 | `calc_elastic` | 平衡应力、离子弛豫约定和应变幅度 |
| 平衡体积、体模量、E–V 曲线 | `calc_eos` | 同一相的稳定体积区间和每点弛豫设置 |

## 力常数

从常规晶胞构建 2×2×2 超胞并输出完整力常数：

```sh
imrmlp calc_fc --potfile model.mtp \
  --input relaxed.vasp --index index.map --supercell 2 2 2 \
  --reduced=false > fc.log 2>&1
```

检查 `FORCE_CONSTANTS`、`supercell_created.xsf` 和 `sup_debug_efs.xyz`。
力常数块的单位为 eV/Å²，顺序对应**实际导出的超胞**。`2 2 2` 只是起点，
需增加超胞检查收敛。默认解析后端在模型不支持时会报错；可用 `--hessian-method auto`
在支持时用解析、否则回退到力的有限差分，或显式选择 `force_fd` 并检查
`--hessian-nudge-pos` 位移步长。

`--lowest-modes N` 是另一种矩阵无关的最低 Gamma 模诊断：输出本征模，不写完整
`FORCE_CONSTANTS`，不能替代整个布里渊区的声子色散。更多后端及输出定义见
[命令参考](../reference/commands_zh.md)。

## 声子：色散与 DOS

与上一节二选一。采用 Phonopy 生成位移、IMR-MLP 计算力时，在新目录复制同一平衡
结构、模型和映射，避免混入另一套 `FORCE_CONSTANTS`。以下采用 Phonopy 4 的命令拆分：

```sh
phonopy-init -d --dim="2 2 2" -c relaxed.vasp
imrmlp phonopyfc --potfile model.mtp --disp phonopy_disp.yaml \
  --index-file index.map
```

`phonopy-init` 生成 `phonopy_disp.yaml`，`phonopyfc` 写出 `FORCE_SETS`。
随后可计算一个示例 DOS 网格：

```sh
phonopy --mesh="10 10 10" --dos phonopy_disp.yaml
```

检查 `total_dos.dat` 和日志，再调整网格或使用适合材料的能带路径。Phonopy 3 使用
`phonopy -d`；Phonopy 4 将准备步骤移至 `phonopy-init`，后处理读取 YAML，不再重复
提供 `-c` / `--dim`，区别见[官方迁移说明](https://phonopy.github.io/phonopy/migration-v4.html)。
`phonopyfc` 使用 `--index-file`，不是其他命令常用的 `--index`；若目录里有
`settings.ini`，应检查其中的 `pot:` 设置是否与本次模型一致。网格、能带路径和绘图
选项见 Phonopy 的[命令说明](https://phonopy.github.io/phonopy/command-options.html)
和[设置标签](https://phonopy.github.io/phonopy/setting-tags.html)。

也可通过 [Python Phonopy 适配器](../reference/python_zh.md#phonopy-integration)
在内存中计算位移超胞的力。若直接导入解析 `FORCE_CONSTANTS`，必须使用 Phonopy 自己的
超胞作为 `calc_fc` 输入，省略再次扩胞，并确保矩阵两个原子轴与 `phonon.supercell`
的顺序完全一致；仅原子数相同不够。

虚频可能来自真实失稳，也可能来自结构未收敛、差分/超胞误差、模型或单位/顺序问题。
先检查平衡力、声学模和数值收敛，再解释物理。声子力常数不等同于有限温度有效力常数，
后者见[简谐参考与 TI](../advanced/harmonic-ti_zh.md)。

## 弹性常数

普通 MTP 可先使用默认的自动后端：

```sh
imrmlp calc_elastic --potfile model.mtp --input relaxed.vasp \
  --index index.map --elastic-output elastic.out > elastic.log 2>&1
```

默认的 `auto` 选择受支持的计算方式；需要解析模式时使用 `--method analytical` 并确认
模型支持相应原子/晶胞 Hessian。需要有限差分核对时，建立 `elastic.ini`：

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

比较不同应变幅度以及更紧的原子弛豫容差。不要把“离子固定”和“允许内部坐标弛豫”的
刚度混为一谈；解析模式的 `--relax-correction` 控制相关修正。输出中的刚度/模量按表头
单位读取；非零外压下还需区分材料刚度和用于机械稳定性的 Born/Wallace 切线张量。
若报内部软模或刚度奇异，先检查结构和模型的稳定性；关闭离子修正会改变所求物理量，
不能当作通用修复。完整定义见[命令参考](../reference/commands_zh.md)。

### 应变幅度核对

保留应变幅度 `0.005`（即 0.5%）得到的 `elastic-numerical.out`，用相同输入和内部
原子弛豫设置再算一次：

```sh
imrmlp calc_elastic --potfile model.mtp \
  --input relaxed.vasp --index index.map --method numerical \
  --settings elastic.ini --strain-magnitude 0.0025 \
  --elastic-output elastic-half-strain.out > elastic-half-strain.log 2>&1
```

例如两次输出的同一 `C11` 分别为 `162.4`、`162.3 GPa`，差值 `0.1 GPa`，约 0.06%；
这只是读法示例，不是某种材料的预期值或统一验收标准。同时比较其他独立分量、模量及各
应变点的原子收敛情况。若步长减半后结果明显漂移，检查弛豫容差和有限差分稳定性；
步长越小并不总越准确。检验声子超胞或 DOS 网格收敛也用同样办法：一次只改一个数值
设置，分别保存结果。

## 状态方程

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

默认扫描参考体积附近约 ±5%，得到曲线和 EOS 拟合参数。查看采样点是否处于同一相、
是否覆盖能量极小值、拟合体积和体模量是否合理，再改变体积窗口与点数检查稳定性。
没有稳定正体积极小值时，不应把拟合数字当作平衡性质。压力扫描使用 `--curve pv`，
与 E–V 路径的输入选项不同。

### EOS 输出的列和单位

| 文件 / 字段 | 含义与单位 |
| --- | --- |
| `ev.dat`：`Volume/at`、`Energy/at`、`Enth/at` | Å³/atom、eV/atom、eV/atom |
| `ev.dat`：`Press` | GPa |
| `eos.dat`：`V0`、`E0` | 拟合平衡体积与能量，Å³/atom、eV/atom |
| `eos.dat`：`B0` | eV/Å³；乘 160.217662 转为 GPa |
| `eos.dat`：`B0'` | 体模量对压力的导数，无量纲 |

以上针对直接扫描结构的 E–V 命令。改用外部曲线文件拟合时，体积/能量归一化随输入
数据，不能自动假定为每原子。单点 `calc_efs` 的总能量不能直接与 `Energy/at` 比较。

## 结果记录

保存平衡结构、模型、元素映射、所有配置和日志，并记录超胞、位移/应变步长、外压以及
离子弛豫约定。先做数值收敛，再与参考数据比较目标性质。

## 常见错误

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| 声子谱出现虚频 | 真实失稳、结构未收敛、差分/超胞误差或模型问题 | 先检查平衡力、声学模和超胞收敛，再解释物理 |
| 刚度值与文献差一个量级 | 混用了离子固定与内部坐标弛豫两种约定 | 明确 `--relax-correction` 与 `--method` 的含义后比较 |
| 内部软模或刚度奇异 | 结构或模型本身不稳定 | 先检查稳定性；关闭离子修正不是修复手段 |
| 应变步长减半后结果漂移 | 弛豫容差不足或有限差分不稳定 | 收紧 `relax:gtol`，并按一次只改一个设置的方式复核 |
| EOS 拟合数字不合理 | 采样跨越相变或没有正体积极小值 | 缩小体积窗口、增加点数，确认同一相 |
| `Energy/at` 与单点能量对不上 | 总能量与每原子能量混用 | 按原子数换算后再比较 |

## 相关

- [结构弛豫](relaxation_zh.md)：平衡结构的来源
- [命令参考](../reference/commands_zh.md)、[Python Phonopy 适配器](../reference/python_zh.md#phonopy-integration)
- [简谐参考与 TI](../advanced/harmonic-ti_zh.md)
