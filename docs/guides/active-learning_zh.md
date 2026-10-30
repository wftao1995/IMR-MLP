# 主动学习与候选选样 {#guide_active_learning}

[English](active-learning.md)

主动学习找出当前训练覆盖不足、值得再做参考计算的结构，减少重复标注。
一个完整循环是：**训练模型 → 探索结构 → 评分选样 → 参考标注 → 合并数据重训 → 独立验证**。
MaxVol 评分和 FPS/CUR 选样都不提供 DFT 标签，也不会仅凭选出结构就改善模型。

不涉及：参考计算与拟合本身（见[训练、评估与部署](workflows_zh.md)）、
MaxVol 的数学推导（见 [MaxVol 理论](../theory/maxvol_zh.md)）。

## 离线与在线入口

| 场景 | 入口 | 前提 |
| --- | --- | --- |
| 已有候选结构文件，离线评分 | `maxvol_select` + `maxvol_sample` | 可检查的选样循环 |
| 更新既有主动集 | `maxvol_extend` | 模型与主动集一致 |
| 按几何/描述符多样性减少冗余 | `diverse_select` 的 FPS / CUR | 构建代表性子集；分数含义不同于 MaxVol |
| LAMMPS MD 中实时发现陌生构型 | [`fix mlip/maxvol`](../../LAMMPS/USER-MLIP/maxvol/README_zh.md) | 模型与接口已就绪，且有匹配的主动集 |

下面以普通 MTP 的离线循环为例。MTP_M、LES/SOG 等派生模型可能需要完整参数梯度的
`pot_efs` builder，不能直接套用普通 MTP 的线性快捷设置。

## 前置输入

| 文件 | 内容 | 来源 |
| --- | --- | --- |
| `model.mtp` | 已训练的普通 MTP 模型 | 训练输出 |
| `train.xyz` | 已有训练覆盖，用于建立主动集 | 模型的训练数据 |
| `pool.xyz` | 待探索结构，只需几何 | MD、结构扰动或缺陷构型转换 |
| `index.map` | 模型元素顺序 | 模型或其训练记录 |
| `select.ini` | builder 设置 | 见下节 |

不要用最终测试集建立选样统计。选样阶段不会自动创建探索池。

## 建立主动集

`select.ini` 只需指定 builder：

```text
builder:potfile model.mtp
```

其余沿用普通 MTP 的默认逐原子环境评分和精确流式选样设置：

```sh
imrmlp maxvol_select select.ini --pool train.xyz --index index.map \
  --output selected.xyz --score-file training-grades.txt
```

输出包括选中的结构和 `model.mtp.sel` 主动集状态；它不是新的势函数。检查日志是否完成
选择、是否仍有虚拟方向或未收敛说明。达到交换/扫描预算时保存 state 不等于选择已收敛；
小数据或相关参数方向还可能造成秩不足。

## 筛选探索结构

```sh
imrmlp maxvol_sample select.ini --input pool.xyz --index index.map \
  --state-file model.mtp.sel --output candidates.xyz \
  --threshold-save 2.1 --threshold-break 1000000000
```

示例提高了**离线输出**的 break 阈值，便于收集较高 grade 的帧；`2.1` 等数值只是初始
设置，需结合模型、活动集和参考误差校准。查看日志中的保存数量及 `BREAK` 提示。
CLI 的默认 break 为 11；遇到超过 break 的构型时会提前停止输出，且该极端帧本身不会先
写入，因此不完整输出不代表整个候选池的结论。LAMMPS fix 的阈值与保存行为按其专用说明处理。

`candidates.xyz` 可以包含输入中的既有属性和 `MV_grade`，但预测值或原有 MD 属性不是新
取得的参考标签。grade 衡量相对于特定活动集的外推程度，不是 eV/Å 误差，低 grade 也不能
单独证明模型可靠。

### 用入门示例核验

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

查看 `grades.tsv`、`candidates.xyz` 和 `sample.log`，应看到 `Saved 2 / 4`。同一主动集下
不同探索结构会得到不同分数，这不表示间距越短就必然外推。模型是合成教学模型，`Al` 仅
演示映射；候选文件只含几何和评分，尚未完成参考标注或重训。手动命令与调低阈值的做法见
[完整示例](../../examples/active-learning/README_zh.md)。

练习中的 `coverage.cfg` 对应上面的 `train.xyz`，`pool.cfg` 对应 `pool.xyz`；手动重做
时沿用其 CFG 文件名即可，无需改扩展名。

## 标注、重训与状态更新

1. 检查候选的几何与重复性，再用与旧训练集一致的方法取得参考能量、力和应力。
2. 核对单位、元素映射和电子收敛，把带标签的新数据合并到训练池。
3. 按[训练流程](workflows_zh.md)重训，并在独立验证集及目标性质上比较旧模型和新模型。
4. 为新模型重新建立匹配的主动集，再进入下一轮探索。

原训练集为 `train.xyz`，某个候选完成参考计算后得到 `reference-001/OUTCAR`，可转换并
合并为下一轮数据：

```sh
imrmlp convstr reference-001/OUTCAR new-reference.xyz --index index.map
cat train.xyz new-reference.xyz > train-next.xyz
```

两个 `.xyz` 都应为带参考标签的 EXTXYZ；不能把几何候选当作已标注数据拼进去。核对
`new-reference.xyz` 的标签后，在新的训练配置中将 `trainset` 改为 `train-next.xyz`，
保留原验证集。多个参考计算需逐一确认收敛再合并，原始参考文件和每轮数据都应保留。

保持每一轮模型、builder 设置、映射、主动集、候选和标签的对应关系。修改模型参数后，
不应因旧 `.sel` 仍可读取或维度相同就沿用旧状态。`maxvol_extend` 可更新状态，但不替代
重训与验证；重要状态先留副本。

## 生产采样的扩展

LAMMPS 的 `fix mlip/maxvol` 需要单独安装 MPI 插件，并使用匹配的模型、builder 和主动集；
它只打分/保存候选，不改变 pair 力。GPU pair 与 CPU 打分器组合需按接口说明核验。
CLI 中读取 GPM 也不会自动启动采样。

候选池很大时，`stream` + `exact` 可通过缓存行完成全池扫描；`online` 不保留所有历史
候选，结果与到达顺序有关，不能当作同一种精确选择。内存与 MPI 控制见
[命令参考](../reference/commands_zh.md)。

## 常见错误

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| 几乎没有候选 | 结构、模型或映射与实际不符，或探索未覆盖目标条件 | 先确认读入的结构与映射，再检查阈值和探索范围 |
| 几乎所有结构都高分 | 模型与主动集不匹配，或 builder 尺度不同 | 核对三者一致性，不要只把阈值调大 |
| 候选很多但重复 | 探索池本身冗余 | 先按机制或描述符多样性去重，再分批标注 |
| 重训后目标性质未改善 | 标签质量或数据覆盖不足 | 比较参考标签质量与覆盖，不要只看训练误差或 grade |
| 把 grade 当作误差 | grade 是相对特定主动集的外推度量 | 用参考标签另行评估 E/F 误差 |
| 沿用旧 `.sel` 状态 | 模型参数已改 | 为新模型重建主动集 |

## 相关

- [训练、评估与部署](workflows_zh.md)：重训与验证
- [MaxVol 理论](../theory/maxvol_zh.md)、[命令参考](../reference/commands_zh.md)
- [自包含的 MaxVol 选样练习](../../examples/active-learning/README_zh.md)
- [LAMMPS `fix mlip/maxvol`](../../LAMMPS/USER-MLIP/maxvol/README_zh.md)
