# 命令速查表 {#cheatsheet}

[English](cheatsheet.md)

最常用 `imrmlp` 命令的快速参考。完整说明请参见 @ref guide_commands "命令参考"。
下列文件名是用户输入的占位，应在输入所在目录运行，并始终使用同一份 `index.map`。
各命令支持的选项和短别名不同，修改示例前先查看 `imrmlp COMMAND --help`。

完整输入和结果解读见[单点计算](../guides/single-point_zh.md)、
[弛豫](../guides/relaxation_zh.md)、[NEB](../guides/neb_zh.md)、
[性质计算](../guides/properties_zh.md)与[主动学习](../guides/active-learning_zh.md)。

---

## 训练与拟合

| 目标 | 命令 | 配置节 |
|------|------|---------|
| 完整 MTP 训练（径向参数 + 线性系数） | `imrmlp trainmtp train.ini --do-lbfgs` | `[fit]`、顶层 `potfile` |
| 仅拟合 MTP 线性系数 | `imrmlp trainmtp train.ini --linear-fit` | `[fit]`、顶层 `potfile` |
| 线性描述符势 | `imrmlp train_linear --settings=linear_train.ini -in train.cfg -o linear.pot` | `[fit]`、`[pot]` |
| 基组选择与快速拟合 | `imrmlp fastfit settings.ini` | 顶层 `mode`：0 计算谱，1 拟合，其余模式用于选基 |
| NN-MTP 训练 | `imrmlp trainnlpot train.ini --do-lbfgs` | `[fit]`、`[pot]` |
| 初始化 MTP-M 非线性头 | `imrmlp initmtpm --settings=init.ini --dataset=train.cfg` | MTP-M `[pot]` 设置和训练数据 |
| 创建 NN-MTP 模板 | `imrmlp nnmtp -pot base.mtp -d train.cfg -n 10 -o nn` | — |

`nnmtp` 的 `nn` 是输出前缀，会生成 `nn.mtp`、`nn.nn`、`nn.sc` 和归一化统计。
`initmtpm` 仅用于 MTP-M；普通 MTP 训练从匹配的模板开始。

## EFS 与验证

| 目标 | 命令 | 示例 |
|------|------|---------|
| 计算能量、力和应力 | `imrmlp efs -pot potential.mtp -in str.cfg -out out.cfg` | — |
| 对比预测值与参考值 | `imrmlp err -pot potential.mtp -in ref.cfg` | — |
| 检查势函数导数 | `imrmlp testpot --settings=test.ini` | — |
| 计算描述符 | `imrmlp calc_descriptors --settings=settings.ini` | 顶层 `dtype` 和各描述符专用设置 |

## 弛豫与鞍点搜索

| 目标 | 命令 | 配置节 |
|------|------|---------|
| 几何结构弛豫 | `imrmlp relax relax.ini` | `[relax]`、`[pot]` |
| NEB（最小能量路径） | `imrmlp neb neb.ini` | `[neb]`、`[pot]` |
| 模式跟随 | `imrmlp modefollowing --settings=mf.ini --input=str.cfg` | — |

## 主动学习

| 目标 | 命令 | 示例 |
|------|------|---------|
| MaxVol 选择子集 | `imrmlp maxvol_select --pool=pool.cfg --potfile=potential.mtp` | `--threshold=1.001` |
| MaxVol 对结构评分 | `imrmlp maxvol_sample --potfile=potential.mtp --input=pool.cfg --state-file=potential.mtp.sel` | — |
| MaxVol 扩充主动集 | `imrmlp maxvol_extend --potfile=potential.mtp --state-file=potential.mtp.sel --candidates=pool.cfg` | — |
| FPS / CUR 多样性选择 | `imrmlp diverse_select --n-select=50 --pool=pool.xyz --potfile=potential.mtp` | `--method=cur` |

MaxVol 选择将可复用的主动集写到 `potential.mtp.sel`。多样性选择默认使用 MTP 描述符，因此也需要匹配的势文件。
选择分数只排序相对当前主动集的外推程度，误差要用 `calc_err` 在同一批构型上量。

## 弹性常数与声子

| 目标 | 命令 | 示例 |
|------|------|---------|
| 解析计算弹性常数 | `imrmlp elastic --method=analytical --potfile=potential.mtp --input=POSCAR` | — |
| 数值计算弹性常数 | `imrmlp elastic --method=numerical -pot potential.mtp -in POSCAR` | `--strain-magnitude=0.005` |
| 力常数 | `imrmlp fc -pot potential.mtp -in POSCAR -dim 3 3 3` | `--reduced=false --hessian-method=analytic` |
| Phonopy FORCE_SETS | `imrmlp mlpphdisp -pot potential.mtp -d phonopy_disp.yaml` | — |
| EOS（能量-体积） | `imrmlp ev --input=POSCAR --settings=eos.ini -ub 0.05 -lb -0.05` | `[pot]`、`[relax]`（或默认 `input.ini`） |

## 分析

| 目标 | 命令 | 示例 |
|------|------|---------|
| RDF、键角、SRO 等 | `imrmlp calc_prop --rdf --bondangle --input=str.cfg` | `--cutoff=5.0` |
| 轨迹平均 | `imrmlp calc_trjavg traj.cfg avg.cfg` | `--start=100 --end=500` |
| 自由能误差 | `imrmlp calc_ferr --input=dft.cfg --prediction=mlp.cfg --temperature=300` | — |
| 查找间隙位点 | `imrmlp findint --input=POSCAR --insert=Li` | `--min-dist=1.0` |

## 结构操作

| 目标 | 命令 | 示例 |
|------|------|---------|
| 转换格式 | `imrmlp convstr POSCAR output.xyz` | `-if POSCAR -of extxyz` |
| 创建超胞 | `imrmlp mk_super --input=POSCAR --supercell 2 2 2 --output=POSCAR-2x2x2` | — |
| 将晶胞变为下三角形式 | `imrmlp mk_lower --input=POSCAR` | — |
| 寻找近似正交晶胞 | `imrmlp mk_orth --input=str.cfg` | — |
| 搜索正交晶胞 | `imrmlp mk_orth_search --input=str.cfg` | `--tol=1e-4` |
| 约化到初基晶胞 | `imrmlp primi_reduce --input=POSCAR` | — |
| 约化平移多重性 | `imrmlp trans_reduce --input=POSCAR` | `--tol=1e-4` |
| 交换晶格轴 | `imrmlp swap_axis --order=132 --input=POSCAR` | — |
| 检查结构等价性 | `imrmlp sameq --input1=s1.cfg --input2=s2.cfg` | `--tol=0.1` |
| 划分数据集 | `imrmlp split_set ds.cfg split_ --ratios=8,1,1 --shuffle yes` | `--nsplit=2`（等分） |
| 删除重叠原子 | `imrmlp rmdups --input=overlap.cfg --rtol=0.5` | — |
| 删除重复结构 | `imrmlp unique --input=ds.cfg --sym-prec=1e-3` | — |

## 势函数工具

| 目标 | 命令 | 示例 |
|------|------|---------|
| MTP → 样条势（LAMMPS） | `imrmlp mtp2spline potential.mtp spline_out` | — |
| 重排物种顺序 | `imrmlp adjustmtp --input=in.mtp --output=out.mtp --neworder 0 2 1` | — |
| 比较/映射 MTP 基组 | `imrmlp mtpbasis -s source.mtp -t target.mtp [-o remapped.mtp]` | 精确逐项/空间判等 |
| 生成 MTP 基组 | `imrmlp mtmbasis --settings=mtmbasis.ini` | 写入 `potential.mtp_<iteration>` |
| 导出表格化势 | `imrmlp pot2table potential.sw sw --pair --threebody` | — |

## 其他

| 命令 | 描述 |
|---------|-------------|
| `imrmlp smooth_traj --input=traj.cfg --output=smooth.cfg` | 对轨迹进行低通滤波 |

---

## 常见选项名称

| 选项 | 作用 |
|------|------|
| `--settings` | 支持此选项的命令读取设置；`phonopyfc` 使用 `--settings-file` / `-i` |
| `--potfile` / `-pot` | 势函数文件 |
| `--input` / `-in` | 输入结构文件 |
| `--output` / `-o` | 输出文件 |
| `--input-format` / `-if` | 输入文件格式 |
| `--output-format` / `-of` | 输出文件格式 |
| `--index` / `-id` | 物种索引文件 |
| `--verbose` | 详细输出（仅少数命令支持） |
| `--help` | 命令专用帮助 |
