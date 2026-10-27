# LJ 连续计算练习：输入准备、弛豫、NEB 与物性

[English](README.md) · [安装](../../docs/guides/installation_zh.md)

这条路线适合已经装好 CPU CLI、但还没有自己的模型和结构的读者。
所有输入由随仓库提供的 Python 脚本生成，只需要 Python 标准库与 CLI；
最后的 Phonopy 步骤是可选项。先运行本页，再进入各任务指南理解参数。

想从基组生成、参考标注和训练开始，再用同一个 MTP 做这些计算时，使用
[MTP 连续练习](../mtp-workflow/README_zh.md)。两条路线都有完整输入，分别保存在独立目录。

本练习使用参数明确的 **Lennard–Jones（`lj612`）教学势**，不是训练好的 MTP。
`Al` 只用于元素映射和质量，所有数值都不能作为铝材料性质。
[训练入门](../quickstart/README_zh.md)中的合成 MTP 用于学习拟合；不要把那个
微型模型直接当作这里的稳定晶体势。两条路线的输入和模型分别保存。

## 1. 在仓库根目录准备环境与输入

先完成[安装指南](../../docs/guides/installation_zh.md)的 CPU 构建和 oneMKL 环境设置。
下面第一条命令必须在含 `Makefile` 的仓库根目录执行，路径变量会记录绝对路径：

```sh
imrmlp_root="$PWD"
export PATH="$imrmlp_root/build/release:$PATH"
export LD_LIBRARY_PATH="$imrmlp_root/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
imrmlp --version
python3 examples/basic-calculations/prepare.py
imrmlp_practice="$imrmlp_root/build/basic-calculations"
```

`prepare.py` 只写输入，不运行计算，不覆盖非空目录。重做时用
`--work build/basic-calculations-2`，并把 `imrmlp_practice` 改成对应的绝对路径。
之后每一步都用此变量切换目录，不依赖上一步停在哪里。新开终端时需要重新设置环境和变量。

| 子目录 | 已生成的输入 | 后续需要的输入 |
| --- | --- | --- |
| `relax/` | 扰动的四原子 FCC `POSCAR`、`relax.ini`、`bulk.ini` | 无 |
| `neb/` | 31 原子空位跳跃的两端点初猜、端点弛豫与 `neb.ini` | 先生成弛豫后的端点 |
| `fc/`、`elastic/`、`eos/`、`phonopy/` | 各自的模型、映射；需要时附设置 | 从弛豫结果复制 `relaxed.vasp` |

每个目录都有 `lj.ini` 和 `index.map`。`lj.ini` 指定单一物种、
`epsilon=0.1 eV`、`sigma=2.5 Å`、截断 6 Å。配置已经指定 `lj612`，无需手动编辑。
各任务指南正文中的普通 MTP 配置用于自己的 MTP 计算；本页使用已生成的 LJ 配置。

**每次主计算后紧接着的 `echo $?` 应显示 `0`，再继续下一步。** 非零时先查看同名日志；
弛豫或 NEB 返回 `2` 表示未收敛，应停在该步骤检查，不能仅凭已有输出继续。

## 2. 单点与固定胞弛豫

```sh
cd "$imrmlp_practice/relax"
imrmlp calc_efs POSCAR initial.xyz --potfile lj.ini --pottype lj612 --index index.map
echo $?
imrmlp relax relax.ini > relax.log 2>&1
echo $?
tail -n 5 relax.log
```

`initial.xyz` 是原始扰动结构的预测，`relaxed.xyz` 是固定胞优化后的结构；
两者都应有四个原子和 E/F 属性。`relax-trajectory.xyz` 可逐帧检查优化过程。
本例仅优化原子，晶胞边长仍为 3.9 Å，暂时不是零压平衡晶胞。
输出列的含义见[单点指南](../../docs/guides/getting-started_zh.md)，
收敛判断见[弛豫指南](../../docs/guides/relaxation_zh.md)。

## 3. 优化零压晶胞，并交给物性计算

`bulk.ini` 明确读取上一步的 `relaxed.xyz`，同时优化原子和晶胞：

```sh
cd "$imrmlp_practice/relax"
imrmlp relax bulk.ini > bulk.log 2>&1
echo $?
imrmlp convstr bulk.xyz bulk.vasp --index index.map
echo $?
cp bulk.vasp ../fc/relaxed.vasp
cp bulk.vasp ../elastic/relaxed.vasp
cp bulk.vasp ../eos/relaxed.vasp
cp bulk.vasp ../phonopy/relaxed.vasp
```

`bulk.vasp` 是四原子零压平衡结构，晶胞边长约 3.883 Å；各物性目录中的
`relaxed.vasp` 都是这个文件的副本。不要把固定胞输出误当作已经优化了晶格。
格式转换只传递几何，不会再优化，也不会生成参考标签。

## 4. 弛豫两个端点，再运行 NEB

这里是独立的固定胞空位跳跃任务：从 2×2×2 FCC 的 32 个晶格位中删除原点原子，
并让相邻原子移入空位，两个端点保留相同的逐行原子身份。
端点初猜、模型和晶胞都已生成，不需要从上面的四原子结构手工造空位。

```sh
cd "$imrmlp_practice/neb"
imrmlp relax initial.ini > initial.log 2>&1
echo $?
imrmlp convstr initial.xyz initial.vasp --index index.map
imrmlp relax final.ini > final.log 2>&1
echo $?
imrmlp convstr final.xyz final.vasp --index index.map
imrmlp neb neb.ini > neb.log 2>&1
echo $?
cat neb.dat
```

`neb.ini` 读取转换得到的 `initial.vasp`、`final.vasp`，共七张图像、每张 31 个原子。
摘要应为 `# status converged`、`# converged true`，完整 `# max_neb_force` 不超过
本例的 `0.02 eV/Å`。`ent_diff` 列最高值约 0.527 eV，两端等能，正反向势垒近似相同。
这是该 LJ 小体系的教学结果，不是铝空位迁移能。用 `mep.xyz` 检查路径，
用[NEB 指南](../../docs/guides/neb_zh.md)学习势垒、图像数收敛和续算。

## 5. 依次计算力常数、弹性与 EOS

这些是彼此独立的物性任务，都读取第 3 步复制的四原子结构。

```sh
cd "$imrmlp_practice/fc"
imrmlp calc_fc --potfile lj.ini --potential-type lj612 \
  --input relaxed.vasp --index index.map --supercell 2 2 2 \
  --reduced=false > fc.log 2>&1
echo $?
head -n 3 FORCE_CONSTANTS
```

`FORCE_CONSTANTS` 第一行应为 `32 32`，表示完整矩阵的两个原子轴均有 32 个原子；后续每对原子对应
一个 3×3 力常数块，单位 eV/Å²。它还不是声子色散图。

```sh
cd "$imrmlp_practice/elastic"
imrmlp calc_elastic --potfile lj.ini --pottype lj612 --input relaxed.vasp \
  --index index.map --settings elastic.ini \
  --elastic-output elastic.out > elastic.log 2>&1
echo $?
head -n 8 elastic.out
```

`Stiffness tensor (GPa)` 下的 6×6 矩阵使用 Voigt 次序 `xx yy zz yz xz xy`。
本例 `C11` 约 97.9 GPa，`C12`、`C44` 约 54.6 GPa。后续模量按其表头读取；
数值会随势参数和结构改变，不能跨模型照抄。

```sh
cd "$imrmlp_practice/eos"
imrmlp calc_eos --settings eos.ini --input relaxed.vasp --index index.map \
  --output ev.dat --output-eos eos.dat > eos.log 2>&1
echo $?
head -n 5 ev.dat
cat eos.dat
```

`ev.dat` 的体积、能量和焓按**每原子**输出，压力为 GPa；这与单点总能量不同。
`eos.dat` 中 `V0` 约 14.63 Å³/atom、`E0` 约 −0.788 eV/atom，`B0` 约
0.432 eV/Å³，乘 160.217662 约为 69.1 GPa；`B0'` 无量纲。
完整物性流程和数值收敛检查见[物性指南](../../docs/guides/properties_zh.md)。

## 6. 可选：用 Phonopy 计算 DOS

需要在当前环境额外安装 Phonopy 4（如在已激活的虚拟环境执行
`python3 -m pip install phonopy`）。不做声子后处理时可跳过本节。

```sh
cd "$imrmlp_practice/phonopy"
phonopy-init -d --dim="2 2 2" -c relaxed.vasp
echo $?
imrmlp phonopyfc --potfile lj.ini --disp phonopy_disp.yaml \
  --index-file index.map --settings-file settings.ini > forces.log 2>&1
echo $?
phonopy --mesh="10 10 10" --dos phonopy_disp.yaml > dos.log 2>&1
echo $?
head -n 5 total_dos.dat
```

本目录的 `settings.ini` 用 `pot:pottype lj612` 选择 LJ；不能省略后误按默认 MTP
读取 `lj.ini`。这里由 Phonopy 定义位移超胞，`phonopyfc` 写 `FORCE_SETS`；
不从 `fc/` 复制 `FORCE_CONSTANTS`。`total_dos.dat` 的两列是频率（默认 THz）和 DOS；
查看整个谱并收敛超胞与网格后再作物理解读，文件开头的少量负频率采样点本身不代表虚频模。

## 接下来到哪里？

| 目标 | 输入从哪里来 | 继续阅读 |
| --- | --- | --- |
| 学习训练与误差评估 | 自包含脚本生成模板、参考标签和验证集 | [训练入门](../quickstart/README_zh.md)、[训练指南](../../docs/guides/workflows_zh.md) |
| 学习选样 | 独立脚本生成模型、已有覆盖与候选池 | [主动学习练习](../active-learning/README_zh.md) |
| 使用真实材料模型 | 自己取得或训练模型，准备目标结构和元素映射 | [开始使用](../../docs/guides/getting-started_zh.md)及各任务指南 |
| 换用 Python / ASE 或 LAMMPS | 先用训练入门生成 `fitted.mtp`，再按接口页准备同一构型 | [Python](../../README-python_zh.md)、[LAMMPS](../../LAMMPS/README_zh.md) |

不用把教学 LJ 势转换成 MTP。换成真实任务时，成套替换模型、势类型、元素映射和
结构，保留自己的验证记录；物性和迁移路径还需分别验证。
