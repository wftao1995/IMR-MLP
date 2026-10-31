# 晶体性质 {#guide_properties}

[English](properties.md)

用已训练的模型计算力常数、声子、弹性常数和状态方程，每类性质各有自己的收敛判据。

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
  --input relaxed.vasp --supercell 2 2 2 \
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
  --elastic-output elastic.out > elastic.log 2>&1
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
  --input relaxed.vasp --method numerical \
  --settings elastic.ini --strain-magnitude 0.005 --elastic-output elastic-numerical.out
```

比较不同应变幅度以及更紧的原子弛豫容差。不要把“离子固定”和“允许内部坐标弛豫”的
刚度混为一谈；解析模式的 `--relax-correction` 控制相关修正。输出中各段的分工、单位与
常见误读见下文[弹性输出文件的读法](#elastic-output-format)，其中特别要留意非零外压下
材料刚度与 Born/Wallace 切线张量的区别。若报内部软模或刚度奇异，先检查结构和模型的
稳定性；关闭离子修正会改变所求物理量，不能当作通用修复。完整定义见
[命令参考](../reference/commands_zh.md)。

### 弹性输出文件的读法 {#elastic-output-format}

`--elastic-output` 指定的文件（默认 `elastic.out`）按固定顺序分段写出，段与段之间没有空行：

| 段 | 内容与单位 |
| --- | --- |
| `EPS: ...` | 仅数值模式写出，记录实际使用的应变幅度；解析模式没有这一行 |
| `Stiffness tensor (GPa)` | 6×6 材料刚度 C，已按 `--symprec` 找到的空间群做对称化 |
| `Born/Wallace stiffness tensor (GPa)` | 机械稳定性判据所用的切线张量 B；零压时与上一段完全相同 |
| `Compliance tensor (GPa^-1)` | 由 C 求逆得到的柔度 S，不是 B 的逆 |
| `Elastic properties` | Voigt / Reuss / Hill 三种平均下的 `B`、`G`、`E`、`v` 与 `G/(9B)` |
| `1.0 / S[0][0]` | 沿第一个笛卡尔轴的单轴模量，与表中的多晶平均 `E` 不是一个量 |
| `# BORN/WALLACE STIFFNESS EIGENSYSTEM` | B 的六个本征值（GPa）与稳定性结论 |

矩阵不带行列标签，行列都按 Voigt 顺序 `11 22 33 23 31 12`：第 4、5、6 行/列依次对应
yz、xz、xy。顺序认错会把 `C44`、`C55`、`C66` 与 `C12`、`C13` 张冠李戴。另外，
`Elastic properties` 里 Reuss 那一列拼作 `Ruess`，按列名检索时要用这个拼写。

读的时候注意三点：

1. 通常说的“弹性常数”就是第一段 C。解析模式在 `--relax-correction` 打开（默认）时、
   数值模式对每个应变点弛豫离子后，得到的都是离子弛豫后的刚度；解析模式关掉该项得到
   的是钳制离子（clamped-ion）刚度。两者相差可能很大，不能混比。
2. 只有外压非零时 B 才与 C 不同：对角元减 P，正应力交叉项加 P。例如同一结构在 5 GPa 下
   `C11` = 97.90、`B11` = 92.90，`C12` = 54.57、`B12` = 59.57。本征值和
   “Born elastic stability conditions meet / NOT meet” 只看 B；柔度张量与模量表由 C 得到。
   两套数字不要交叉引用。
3. 低于内部阈值（解析 `1e-10`、数值 `1e-5`）的元素直接打印成 `0`。对称性禁戒的分量本应为
   零，若打印出可观数值，多半是 `--symprec` 与结构不匹配或结构没弛豫干净；反过来，打印成
   `0.000000` 也不表示该分量被约束住了。

其余输出文件：

| 文件 | 何时写出 | 内容 |
| --- | --- | --- |
| `eigensys.out` | 仅解析模式，总是写出（`--eigensys-output` 改名） | 9×9 `cellhess`（eV/Å²）、其本征系统、9×9 `C_ijkl` 与 6×6 `C_ij`（均为 eV，未除体积）及各自本征系统 |
| `eigtotal.dat` | 仅解析模式且给出 `--compute-eigsys` | 合并的 (3N+9)×(3N+9) 晶胞+原子 Hessian 与本征系统，前 9 个自由度是晶胞 |
| `debuginfo.log` | 仅数值模式且给出 `--debug` | 每个应变点的应变向量与对应应力向量，可自己复算拟合 |

`eigensys.out` 里的张量单位是 eV 而不是 GPa（同一例中 `C11` = 35.77 eV 对应 97.90 GPa，
只差体积因子与 160.217662 的换算），不能与 `elastic.out` 的数字直接对照。它的用途是看
晶胞 Hessian 的本征值有没有软化；真要定位软模对应的位移模式，用 `eigtotal.dat`，而不是
只看 6×6 的本征值列表。

数值模式还会把对称化之前的拟合结果以 `Original fitted C:` 打印到 stdout（不在
`elastic.out` 里）。它与文件里投影后的张量差得多，说明应变点或离子弛豫有问题。

最后，`elastic.out` 的 `B` 与 `eos.dat` 的 `B0` 不是同一个量：前者是当前结构在有限应变下的
刚度，后者是 EOS 拟合的平衡体模量，只有结构确实落在能量极小值附近时二者才接近。

### 应变幅度核对

保留应变幅度 `0.005`（即 0.5%）得到的 `elastic-numerical.out`，用相同输入和内部
原子弛豫设置再算一次：

```sh
imrmlp calc_elastic --potfile model.mtp \
  --input relaxed.vasp --method numerical \
  --settings elastic.ini --strain-magnitude 0.0025 \
  --elastic-output elastic-half-strain.out > elastic-half-strain.log 2>&1
```

例如两次输出第一段刚度矩阵的同一 `C11` 分别为 `162.4`、`162.3 GPa`，差值 `0.1 GPa`，约
0.06%；这只是读法示例，不是某种材料的预期值或统一验收标准。同时比较其他独立分量、模量及各
应变点的原子收敛情况。若步长减半后结果明显漂移，检查弛豫容差和有限差分稳定性；两档步长
给出同一个 `Cij` 时才是平台段，继续减小步长只会放大舍入噪声。检验声子超胞或 DOS 网格
收敛也用同样办法：一次只改一个数值设置，分别保存结果。

## 状态方程

保存 `eos.ini`，指定与平衡结构相同的势和弛豫精度：

```text
pot:potfile model.mtp
relax:gtol 0.0001
relax:maxiteration 500
```

```sh
imrmlp calc_eos --settings eos.ini --input relaxed.vasp \
  --output ev.dat --output-eos eos.dat
```

默认在参考体积 ±5% 内取点，得到曲线和 Birch–Murnaghan 拟合参数。`calc_eos` 只检查拟合
出的 `V0`、`B0` 为正且有限，不检查极小值是否落在扫描窗口内：驻点落在窗口外时，`eos.dat`
仍会写出外推的 `V0`、`B0`，换一个窗口重算就会换一组数。判据看 `ev.dat`——`Energy/at`
是否在窗口内先降后升、`Press` 是否在窗口内过零，过零点就是 `V0`。不满足时先用
`--lower`/`--upper` 收窄窗口或重新弛豫平衡结构，再改变 `--npos`/`--nneg` 检查稳定性；
窗口内确实没有正曲率极小值，程序报 `EOS fit has no stable finite equilibrium point`
并以非零码退出，不写 `eos.dat`。压力扫描用 `--curve pv`，选项是 `--press-lb`/`--press-ub`，
只输出 `Pressure`、`Volume`、`Energy/at`、`H/at` 四列，不做 EOS 拟合。

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
| 输出里两段刚度张量数值不同 | `--pressure` 非零，第二段是 Born/Wallace 切线张量 | 材料常数读第一段，稳定性判据读第二段 |
| `eigensys.out` 的数值比 `elastic.out` 大一个量级 | 前者单位是 eV 且未除体积 | 按体积与 160.217662 换算后再比较 |
| 内部软模或刚度奇异 | 结构或模型本身不稳定 | 先检查稳定性；关闭离子修正不是修复手段 |
| 应变步长减半后结果漂移 | 弛豫容差不足或有限差分不稳定 | 收紧 `relax:gtol`，并按一次只改一个设置的方式复核 |
| EOS 拟合数字不合理 | 采样跨越相变，或极小值落在窗口外 | 用 `--lower`/`--upper` 收窄窗口、增加 `--npos`/`--nneg` 点数，确认同一相 |
| `Energy/at` 与单点能量对不上 | 总能量与每原子能量混用 | 按原子数换算后再比较 |

## 相关

- [结构弛豫](relaxation_zh.md)：平衡结构的来源
- [命令参考](../reference/commands_zh.md)、[Python Phonopy 适配器](../reference/python_zh.md#phonopy-integration)
- [简谐参考与 TI](../advanced/harmonic-ti_zh.md)
