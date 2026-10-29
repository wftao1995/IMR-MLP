# MTP 连续练习：基组、训练、验证与计算

[English](README.md) · [CPU 安装](../../docs/guides/installation_zh.md) · [已有势的 LJ 练习](../basic-calculations/README_zh.md)

本页从零生成两个普通 MTP 候选基组，用合成参考数据拟合，按验证误差选模型，
最后把**同一个训练后的 MTP**用于弛豫、NEB、力常数、弹性和 EOS。
除可选 Phonopy 后处理外，只需 Python 标准库和已编译的 CPU CLI；无需下载模型、
私人数据或安装 Python 扩展。所有文件来自本仓库的 [prepare.py](prepare.py)。

这是针对已知双体参考函数的受控练习：使用普通 MTP 的**二体子集**和固定径向参数，
不是通用金属多体势，也不是铝材料模型。`Al` 仅用作单物种标签和质量。
参考问题本身是双体势，因此选择这个受限子集有明确依据。
先跑通整条链，再看最后一节如何换成自己的参考计算。

## 1. 输入与参考标签从哪里来？

| 内容 | 来源与用途 |
| --- | --- |
| 参考势 `reference/force-shifted-lj.table` | 脚本将力平移的 LJ 函数写成 `pair_table` 能量表；CLI 对其插值求 E/F/应力，**不是 DFT 标签** |
| `train-input.cfg`、`valid-input.cfg`、`test-input.cfg` | 分别用种子 1701、1702、1704 独立生成，只有几何，尚无参考标签 |
| `basis-8/`、`basis-16/` | 基组生成设置、训练设置；拓扑由 `mtmbasis` 产生，系数随后拟合 |
| `calculations/` | 另行生成的四原子 FCC、31 原子空位端点及各任务配置；稍后才部署训练模型 |
| `provenance.json` | 势参数、表格网格、种子、各类结构数量、单位和候选基组记录 |

参考函数采用 `epsilon=0.1 eV`、`sigma=2.5 Å`、`rc=6 Å`：

@f$ U(r)=4\epsilon[(\sigma/r)^{12}-(\sigma/r)^6] @f$，
@f$ \phi(r)=U(r)-U(r_c)-(r-r_c)U'(r_c) @f$（@f$r<r_c@f$），截断外为零。
能量和一阶导数在截断处为零；二阶导数在截断处不保证连续。
表格从 1.8 到 6 Å、步长 0.001 Å，共 4201 点，原生表格插值提供导数。
它与 LJ 练习中直接截断的 `lj612` 不是同一个参考势，不能照搬那页的数值。

| 数据集 | 双原子 | 扰动 FCC（4 原子） | 扰动空位路径（31 原子） | 总构型数 |
| --- | --- | --- | --- | --- |
| 训练 | 60 | 24 | 16 | 100 |
| 验证 | 16 | 8 | 4 | 28 |
| 最终测试 | 16 | 8 | 4 | 28 |

双原子间距为 2.1–5.95 Å；FCC 晶格常数为 3.72–4.08 Å，原子扰动每分量 ±0.04 Å。
训练双原子使用包含两个端点的 60 点等距网格，确保整个区间都有覆盖；验证/测试双原子随机取点。
空位结构从 2×2×2 FCC 删除原点原子，再移动相邻原子填入空位；晶格常数为
3.84–3.96 Å，沿路径随机取点并加入每分量 ±0.02 Å 扰动。
三个集合覆盖同一类受控结构，测试的是插值，不是未知相、温度或元素的外推能力。

## 2. 准备环境、几何和配置

完成安装（oneMKL 可选）后，在含 `Makefile` 的仓库根目录运行：

```sh
imrmlp_root="$PWD"
export PATH="$imrmlp_root/build/release:$PATH"
export LD_LIBRARY_PATH="$imrmlp_root/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 examples/mtp-workflow/prepare.py
imrmlp_mtp="$imrmlp_root/build/mtp-workflow"
```

脚本拒绝覆盖非空目录。重做时加 `--work build/mtp-workflow-2`，并相应修改
`imrmlp_mtp`。新终端需要重新启用环境和设置绝对路径变量。
以下各计算应正常返回 `0`；紧随计算的 `echo $?` 非零时先查日志再继续。

## 3. 用参考势标注三个集合

```sh
cd "$imrmlp_mtp"
imrmlp calc_efs train-input.cfg train.cfg --potfile reference/force-shifted-lj.table \
  --pottype pair_table --index index.map > label-train.log 2>&1
echo $?
imrmlp calc_efs valid-input.cfg valid.cfg --potfile reference/force-shifted-lj.table \
  --pottype pair_table --index index.map > label-valid.log 2>&1
echo $?
imrmlp calc_efs test-input.cfg test.cfg --potfile reference/force-shifted-lj.table \
  --pottype pair_table --index index.map > label-test.log 2>&1
echo $?
```

输出 `train.cfg`、`valid.cfg`、`test.cfg` 才有参考 E/F/应力。
划分已在标注前固定；训练只读取训练集和验证集，最终测试集留到选型完成后评估。
这些 CFG 的 `type=0` 是模型 species 0，`index.map` 的第一行是 `Al`。

## 4. 生成并补全两个基组模板

生成命令是 `mtmbasis`；名称相近的 `mtpbasis` 用于基组比较和参数映射，不是生成器。

```sh
cd "$imrmlp_mtp/basis-8"
imrmlp mtmbasis basis.ini > basis.log 2>&1
echo $?
cd "$imrmlp_mtp/basis-16"
imrmlp mtmbasis basis.ini > basis.log 2>&1
echo $?
python3 "$imrmlp_root/examples/mtp-workflow/prepare.py" \
  --work "$imrmlp_mtp" --complete-templates
```

每个目录首先产生 `potential.mtp_0`。它含拓扑，**还不是训练模型**。
最后的命令将两份拓扑补全为各自的 `template.mtp`，保留原文件；不会标注或拟合。
它检查本例预期的标量数量，并拒绝覆盖已有模板。

16 项候选的 `basis.ini` 为：

```text
max_lev 32
max_dim 1
max_mu 15
max_nv 0
mode 3
seed 17
maxiter 100
```

8 项候选对应 `max_lev 16`、`max_mu 7`，其他相同。这些都是**顶层键**，不写 `pot:`。
`max_nv=0`、`max_dim=1` 只保留 `M(mu,0)`，因此分别得到 8、16 个标量项。
这不是通常含角向与高体阶项的完整 level-16 / level-32 基组规模。
`mode`、`maxiter` 控制缩并布局搜索，不是训练优化器或训练迭代数。

补全模板时设置一个物种、`RBChebyshev`、径向区间 2–6 Å，径向基数量分别为 8、16；
径向系数为单位矩阵，初始物种偏移与线性系数全为零。即
@f$g_\mu(r)=(6-r)^2T_\mu((r-4)/2)@f$，在截断内使用。
本例固定这些径向函数，只拟合物种偏移和 8/16 个标量系数，共 9/17 个线性参数。
这里没有把参考势的系数预先写入 MTP。基组定义见 [MTP 理论](../../docs/theory/mtp_zh.md)，
通用生成设置见[配置参考](../../docs/reference/configuration_zh.md)。

## 5. 拟合、验证并选择基组

在两个独立目录中使用各自的 `fit.ini`，显式选择线性拟合：

```sh
cd "$imrmlp_mtp/basis-8"
imrmlp trainmtp fit.ini --linear-fit > train.log 2>&1
echo $?
cp _Trained_Potential_.mtp fitted.mtp
imrmlp calc_err --input ../valid.cfg --potfile fitted.mtp \
  --index ../index.map --output-summary validation.txt
echo $?
cd "$imrmlp_mtp/basis-16"
imrmlp trainmtp fit.ini --linear-fit > train.log 2>&1
echo $?
cp _Trained_Potential_.mtp fitted.mtp
imrmlp calc_err --input ../valid.cfg --potfile fitted.mtp \
  --index ../index.map --output-summary validation.txt
echo $?
```

两份配置使用同一数据与权重：`energy_weight=1`、`force_weight=0.1`、
`stress_weight=0.001`、`weight_L2=1e-12`，关闭随机初始化与径向正交化。
完整配置已生成，不要用其他训练器的设置替换。最终模型固定写入
`_Trained_Potential_.mtp`；脚本设置的 `checkpoint.mtp` 是另一份检查点。

预先为本教学问题规定：验证集每原子能量 RMS 不超过 1 meV/atom，力向量 RMS
不超过 0.001 eV/Å，选择满足两项的较小候选。典型结果为：

| 候选 | 线性参数数 | 验证能量 RMS（meV/atom） | 验证力 RMS（eV/Å） | 达标？ |
| --- | --- | --- | --- | --- |
| 8 标量 | 9 | 约 8.03 | 约 0.0952 | 否 |
| 16 标量 | 17 | 约 0.024 | 约 0.000324 | 是 |

读取 `validation.txt` 中 **Energy per atom** 和 **Forces** 的 RMS 行；
后者不是 `Forces components` 的 RMS。每份报告应覆盖 28 个构型、188 个原子。
这些阈值用于本练习选型，不是材料计算的通用精度标准。
若自己的结果不满足条件，先检查输入和日志，不要不看验证就进入后续计算。

选定 16 项后，才计算训练与最终测试误差，并部署同一个文件：

```sh
cd "$imrmlp_mtp"
cp basis-16/fitted.mtp model.mtp
imrmlp calc_err --input train.cfg --potfile model.mtp \
  --index index.map --output-summary training.txt
echo $?
imrmlp calc_err --input test.cfg --potfile model.mtp \
  --index index.map --output-summary test.txt
echo $?
for task in relax neb fc elastic eos phonopy; do
  cp model.mtp "calculations/$task/model.mtp"
done
sha256sum model.mtp calculations/*/model.mtp
```

哈希应相同；循环只把选定模型分发到各任务目录，不会重训。
`test.txt` 同样应检查 28 个构型、188 个原子，典型能量 RMS 约 0.024 meV/atom、
力 RMS 约 0.000297 eV/Å。最大力误差约 0.00142 eV/Å；RMS 达标不等于每个原子都
低于同一阈值，还应检查最坏构型。
不要根据测试误差再反复调基组；若这样做，需要新的最终测试集。

## 6. 用训练 MTP 弛豫，并准备物性输入

```sh
cd "$imrmlp_mtp/calculations/relax"
imrmlp relax relax.ini > relax.log 2>&1
echo $?
imrmlp relax bulk.ini > bulk.log 2>&1
echo $?
imrmlp convstr bulk.xyz bulk.vasp --index index.map
echo $?
cp bulk.vasp ../fc/relaxed.vasp
cp bulk.vasp ../elastic/relaxed.vasp
cp bulk.vasp ../eos/relaxed.vasp
cp bulk.vasp ../phonopy/relaxed.vasp
```

四原子 FCC 初始晶格常数为 3.9 Å，第一个原子扰动 `(0.03, −0.02, 0.01) Å`。
`relax.ini` 固定胞优化原子，`bulk.ini` 再读取 `relaxed.xyz` 优化零压晶胞。
最终 `bulk.vasp` 边长约 3.90045 Å；各物性输入都是它的副本。
弛豫返回 `2` 表示未收敛，应停下来检查；判断方式见[弛豫指南](../../docs/guides/relaxation_zh.md)。

## 7. 用同一模型弛豫端点并运行 NEB

```sh
cd "$imrmlp_mtp/calculations/neb"
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

这里另行生成 2×2×2 FCC、删除原点原子，得到 31 原子的空位初态；同一行的相邻
原子移入空位构成终态，两端保持 7.8 Å 的共同晶胞。先弛豫两端，再使用七张图像的
固定胞 CI-NEB。摘要应为 `converged`，完整 `max_neb_force <= 0.02 eV/Å`，
最高 `ent_diff` 约 0.551 eV，两端等能。
`mep.xyz` 是七帧最终路径；原子对应、收敛与续算见 [NEB 指南](../../docs/guides/neb_zh.md)。

## 8. 力常数、弹性与 EOS

```sh
cd "$imrmlp_mtp/calculations/fc"
imrmlp calc_fc --potfile model.mtp \
  --input relaxed.vasp --index index.map --supercell 2 2 2 \
  --reduced=false > fc.log 2>&1
echo $?
head -n 3 FORCE_CONSTANTS
cd "$imrmlp_mtp/calculations/elastic"
imrmlp calc_elastic --potfile model.mtp --input relaxed.vasp \
  --index index.map --settings elastic.ini \
  --elastic-output elastic.out > elastic.log 2>&1
echo $?
head -n 8 elastic.out
cd "$imrmlp_mtp/calculations/eos"
imrmlp calc_eos --settings eos.ini --input relaxed.vasp --index index.map \
  --output ev.dat --output-eos eos.dat > eos.log 2>&1
echo $?
cat eos.dat
```

`FORCE_CONSTANTS` 首行为 `32 32`，为扩胞后完整力常数矩阵，块单位 eV/Å²。
弹性输出使用 GPa，典型 `C11≈88.24`、`C12≈C44≈49.50`。
EOS 的 `V0≈14.835 Å³/atom`、`E0≈−0.6415 eV/atom`、
`B0≈0.3905 eV/Å³`（约 62.56 GPa）；`ev.dat` 的能量按每原子输出。
数值设置的收敛与各输出定义见[物性指南](../../docs/guides/properties_zh.md)。

验证 E/F 后仍要检查目标性质。本环境将参考势独立跑过同一流程，得到
晶格约 3.90040 Å、`C11≈88.25 GPa`、空位势垒约 0.55086 eV，与上述 MTP 接近。
这说明本例的拟合能复现这一参考问题，不说明真实铝材料正确。
也可在 MTP 最终路径上重新取得参考标签，再检查模型误差：

```sh
cd "$imrmlp_mtp"
imrmlp calc_efs calculations/neb/mep.xyz path-reference.xyz \
  --potfile reference/force-shifted-lj.table --pottype pair_table --index index.map
echo $?
imrmlp calc_err --input path-reference.xyz --potfile model.mtp \
  --index index.map --output-summary path-errors.txt
echo $?
```

该报告应覆盖七帧、217 个原子；参考标签来自参考势，不是用 MTP 预测验证 MTP 自己。

## 9. 可选：Phonopy DOS

额外安装 Phonopy 4 后，在已准备好的独立目录运行：

```sh
cd "$imrmlp_mtp/calculations/phonopy"
phonopy-init -d --dim="2 2 2" -c relaxed.vasp
echo $?
imrmlp phonopyfc --potfile model.mtp --disp phonopy_disp.yaml \
  --index-file index.map --settings-file settings.ini > forces.log 2>&1
echo $?
phonopy --mesh="10 10 10" --dos phonopy_disp.yaml > dos.log 2>&1
echo $?
head -n 5 total_dos.dat
```

检查 `FORCE_SETS`、`total_dos.dat` 与日志，不从 `fc/` 复制另一套超胞力常数。
DOS 列、负频率采样点和超胞收敛的解释见 [LJ 连续练习](../basic-calculations/README_zh.md)。

## 换成自己的材料时，哪些部分要改？

用自己收敛的 DFT 等参考计算替换合成标注，并按轨迹/相/成分划分训练、验证和测试；
不要用待部署的 MTP 自己产生“参考标签”。核对单位、应力约定和模型物种顺序。
真实多体材料通常还需要角向和更高体阶项，不能仅增加本例的径向项数。
本例的模板补全脚本只支持上述 8/16 项设置，不是任意 MTP 模板的通用初始化器。
`min_dist=2` 也不是防止原子进入短距离的硬墙。

基组、截断和径向参数应通过验证误差、目标性质及成本共同选择；需训练径向参数时
再用适配的非线性配置。完整流程见[训练指南](../../docs/guides/workflows_zh.md)、
[MTP 理论](../../docs/theory/mtp_zh.md)和[线性精修](../../docs/advanced/linear-refit_zh.md)。
需要继续扩充数据时，用[主动学习](../../docs/guides/active-learning_zh.md)发现候选，
取得新参考标签后重训，并重建匹配主动集。
