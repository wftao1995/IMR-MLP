# 主动学习与候选选样 {#guide_active_learning}

[English](active-learning.md)

主动学习的目的，是找出当前训练覆盖不足、值得再做参考计算的结构，减少重复标注。
一个完整循环是：**训练模型 → 探索结构 → 评分选样 → 参考标注 → 合并数据重训 → 独立验证**。
MaxVol 评分和 FPS/CUR 选样都不会自动提供 DFT 标签，也不会仅凭选出结构就改善模型。

## 先选离线还是在线

| 场景 | 入口 | 适合什么时候用 |
| --- | --- | --- |
| 已有候选结构文件，先离线评分 | `maxvol_select` + `maxvol_sample` | 第一次建立可检查的选样循环 |
| 希望更新既有主动集 | `maxvol_extend` | 理解模型/主动集的一致性后使用 |
| 主要按几何/描述符多样性减少冗余 | `diverse_select` 的 FPS / CUR | 构建代表性子集；分数含义不同于 MaxVol |
| LAMMPS MD 中实时发现陌生构型 | [`fix mlip/maxvol`](../../LAMMPS/USER-MLIP/maxvol/README_zh.md) | 已跑通模型与接口，且准备好匹配主动集 |

本页先用普通 MTP 演示离线循环。MTP_M、LES/SOG 等派生模型可能需要完整参数
梯度的 `pot_efs` builder，不能直接套用普通 MTP 的线性快捷设置。

## 先跑一个不需要自备数据的小例子

编译好 CLI 后，在仓库根目录运行：

```sh
export LD_LIBRARY_PATH="$PWD/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 examples/active-learning/run.py
```

脚本只用 Python 标准库，自动准备模型、映射、配置及双原子构型，输出到新的
`build/active-learning/` 目录。已有覆盖的原子间距为 3.8–4.0 Å，探索池得到：

| 探索间距（Å） | Grade（约） | 超过保存阈值 2.1？ |
| --- | --- | --- |
| 3.9 | 0.854 | 否 |
| 3.5 | 1.488 | 否 |
| 3.0 | 2.425 | 是 |
| 2.5 | 3.445 | 是 |

查看 `grades.tsv`、`candidates.xyz` 和 `sample.log`，应看到 `Saved 2 / 4`。
这说明同一主动集下，不同探索结构会得到不同分数；不是说间距越短就必然外推。
模型是合成教学模型，`Al` 仅演示映射。候选文件仅含几何和评分，尚未完成参考标注
或重训。手动命令、文件解释及调低阈值的小练习见[完整示例](../../examples/active-learning/README_zh.md)。
下面再把同一过程换成自己的模型与数据。

练习中的 `coverage.cfg` 对应下文的已有训练覆盖 `train.xyz`，`pool.cfg` 对应 `pool.xyz`；
手动重做练习时沿用其 CFG 文件名即可，无需改扩展名。自己的 `train.xyz` 来自模型的训练数据，
`pool.xyz` 可由 MD、结构扰动或缺陷构型转换得到；选样阶段只要求几何，不会自动创建这个探索池。

## 1. 从现有训练覆盖建立主动集

准备 `model.mtp`、已有训练结构 `train.xyz`、待探索结构 `pool.xyz` 和
`index.map`。不要用最终测试集建立选样统计。保存 `select.ini`：

```text
builder:potfile model.mtp
```

其余沿用普通 MTP 的默认逐原子环境评分和精确流式选样设置。

```sh
imrmlp maxvol_select select.ini --pool train.xyz --index index.map \
  --output selected.xyz --score-file training-grades.txt
```

输出包括选中的结构和 `model.mtp.sel` 主动集状态。它不是新的势函数。
检查日志是否完成选择、是否仍有虚拟方向或未收敛说明；达到交换/扫描预算时保存
state 不等于选择已收敛。小数据或相关参数方向还可能造成秩不足。

## 2. 用同一配置筛选探索结构

```sh
imrmlp maxvol_sample select.ini --input pool.xyz --index index.map \
  --state-file model.mtp.sel --output candidates.xyz \
  --threshold-save 2.1 --threshold-break 1000000000
```

示例提高了**离线输出**的 break 阈值，便于收集较高 grade 的帧；`2.1` 等数值只是
初始设置，需结合模型、活动集和参考误差校准。查看日志中的保存数量及 `BREAK` 提示。
CLI 的默认 break 为 11；遇到超过 break 的构型时会提前停止输出，且该极端帧本身
不会先写入，因此不要把不完整输出误当作整个候选池的结果。
LAMMPS fix 的阈值/保存行为另按其专用说明处理。

`candidates.xyz` 可以包含输入中的既有属性和 `MV_grade`，但预测值或原有 MD 属性
不是新取得的参考标签。grade 衡量相对于特定活动集的外推程度，不是 eV/Å 误差，
低 grade 也不能单独证明模型可靠。

## 3. 标注、重训并重新建立状态

1. 检查候选的几何与重复性，再用与旧训练集一致的方法取得参考能量、力和应力。
2. 核对单位、元素映射和电子收敛，把有参考标签的新数据合并到训练池。
3. 按[训练指南](workflows_zh.md)重训，并在独立验证集及目标性质上比较旧模型和新模型。
4. 为新模型重新建立匹配的主动集，再进入下一轮探索。

例如，原训练集为带标签的 `train.xyz`，某个新候选完成参考计算后得到
`reference-001/OUTCAR`，可转换并合并为下一轮数据：

```sh
imrmlp convstr reference-001/OUTCAR new-reference.xyz --index index.map
cat train.xyz new-reference.xyz > train-next.xyz
```

这里两个 `.xyz` 都应为带参考标签的 EXTXYZ；不能直接把几何候选当作已标注数据
拼进去。核对 `new-reference.xyz` 的标签后，在新的训练配置中将 `trainset` 改为
`train-next.xyz`，保留原验证集，按[训练指南](workflows_zh.md)运行。多个参考计算需逐一
确认收敛再合并，原始参考文件和每轮数据都应保留。

保持每一轮模型、builder 设置、映射、主动集、候选和标签的对应关系。
修改模型参数后，不应因旧 `.sel` 仍可读取或维度相同就沿用旧状态。
`maxvol_extend` 可更新状态，但不会替代重训与验证；对重要状态先留副本。

## 4. 扩展到生产采样

离线循环稳定后，再考虑在线运行。LAMMPS 的 `fix mlip/maxvol` 需要单独安装 MPI
插件，并使用匹配的模型、builder 和主动集；它只打分/保存候选，不改变 pair 力。
GPU pair 与 CPU 打分器组合需按接口说明核验。CLI 中读取 GPM 也不会自动启动采样。

候选池很大时，`stream` + `exact` 可通过缓存行完成全池扫描；`online` 不保留所有
历史候选，结果与到达顺序有关，不能当作同一种精确选择。内存和 MPI 控制见
[命令参考](../reference/commands_zh.md)，评分原理见 [MaxVol 理论](../theory/maxvol_zh.md)。

## 什么时候需要调整流程？

- 几乎没有候选：先确认实际读入的结构、模型及映射，再检查阈值和探索是否覆盖目标条件。
- 几乎所有结构都高分：核对模型/主动集是否匹配、训练覆盖与 builder 尺度，不要只把阈值调大。
- 候选很多但重复：先按机制或描述符多样性去重，再分批参考标注。
- 重训后目标性质未改善：比较参考标签质量和数据覆盖，避免仅用训练误差或 grade 判断收益。
