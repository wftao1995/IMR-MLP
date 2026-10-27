# fastfit 按 E/F/S 计算成本选基 {#doc_docs_fastfit_cost_selection}

[English](fastfit-cost-selection.md)

`selection-objective error-cost` 为 mode 5/6 增加计算成本目标。搜索复用现有 E/F/S 二次型评分，以压缩模型的依赖结构估算计算量，最后实测少量候选。默认 `legacy` 保留原有选择逻辑。

普通 MTP 的生成、拟合、模式选择和输出文件见[基组生成、选择与验证](basis-generation-selection_zh.md)；本页说明模式 5/6 的误差与成本目标。

## 使用

先准备势文件、训练集和带 E/F/S 标签的验证集。在配置中加入：

```ini
selection-objective    error-cost
max-valid-loss         0.02
cost-benchmark-topk    8
cost-benchmark-configs 16
cost-archive-size      128
cost-benchmark-rounds  5
cost-benchmark-seconds 0.01
cost-report           fastfit-cost.tsv
```

`0.02` 仅为示例，必须根据数据、权重及所允许的误差填写。执行：

```sh
imrmlp fastfit fastfit.ini -m 6 -np 1 --profile-output fastfit-profile.tsv
# mode 5 还需要 selected_ones.dat 和 removed.dat
imrmlp fastfit fastfit.ini -m 5 -np 1 --profile-output fastfit-profile.tsv
```

`max-valid-loss` 是必填的非负数，定义为**完整验证集的加权 E/F/S 平方损失除以构型数**，不含 L2。沿用现有 fitness、atomic_fitness、原子数幂和力归一化；virial 使用完整 3×3 张量。允许的数值容差为 `1e-12 * max(1, max-valid-loss)`。它约束综合损失，不分别约束 E/F/virial 的 RMSE。验证集用于选模型，独立测试集应留作最终评估。

mode 6 的双目标为平均验证损失与相对计算量，使用可行性优先的支配规则；完整基集加入初始种群。mode 5 保留固定数量交换，使用可行性优先的成本分数；SA 可探索不可行区域，跨代档案保留可行解。新模式中退火温度对应无量纲成本分数。NSGA-II 的误差排序及档案误差平局使用固定二进制精度网格（`2^(frexp_exponent(max(1,abs(loss)))-42)`），抑制微小归约误差导致的排序翻转；可行性判断、报告和最终直接复核均使用未舍入误差。

### NSGA-II 搜索范围和变异率

mode 6 提供可选控制项，默认保留原有搜索行为：

```ini
population-init stratified
mutation-rate -1
```

`population-init half`（默认）使随机个体以 50% 概率保留每个标量；`stratified` 按种群位置将保留概率从 5% 分布到 95%，同时保留已有的文件种子和 `error-cost` 完整基组种子。这是概率分层，不保证每个个体的确切标量数。

`mutation-rate` 是逐标量独立翻转概率，允许 `[0,1]`，默认 `0.05`；`-1` 表示 `1/nscalar`，即每个子代**平均**翻转一项，并非每次恰好一项。对于 756 项基组，默认值平均翻转 37.8 项。更小的概率有利于局部调整，但可能减慢大范围探索，应在相同种群、代数和多个 seed 下比较。日志及成本报告记录实际概率；这两个参数只影响 NSGA-II。

也可以直接搜索基本矩依赖分支：

```ini
population-init dependency
mutation-rate -1
group-mutation-rate 0.5
```

每个依赖组包含使用某个基本矩的**所有**标量输出，由与模型压缩相同的依赖闭包生成；相同组去重，空组和覆盖整个基集的组不参与。删除整组会使相应基本矩成为可压缩的孤立节点，而逐标量剪枝通常仍保留这些共享依赖。

`dependency` 初始化从完整模型逐组删除，按 seed 打乱组顺序；最后一个个体保留完整模型，已有 `selected_ones.dat` 仍优先作为第一个种子。种群超过组数时循环使用组，没有可用组时回退到 `half` 随机个体。`group-mutation-rate` 默认 `0`，允许 `[0,1]`：在逐标量变异后，以该概率从能够继续删除且不会清空模型的组中均匀选择一个并删除。交叉和逐标量变异仍可恢复此前删除的标量。这些操作不增加每代拟合候选的数量，日志报告去重后的组数；最终仍需通过原有误差门限与保存重载复核。

初始化与变异是独立开关，可用 `dependency` 加 `group-mutation-rate 0` 单独检验初始化收益。它们只影响 mode 6，组删除本身不保证满足误差约束，也不保证实测加速。

## 成本口径

共享依赖闭包与 `remove_orphans()` 使用相同分析接口。对选中标量的依赖取并集，统计基本矩、全部矩、乘积规则、径向通道和最大张量阶数；同一目标的所有累加规则都会计入。通道数按压缩后的模型统计，支持多元素与 pair-split 径向基。

`efs-ops-v1` 为当前 CPU E/F/S 循环的运算和写入次数估计：

```
q_active * (6 * channels * rb_size + 2 * channels + 8 * degree + 39 * basics)
+ q_stored * 12 * basics
+ 12 * products + 2 * moments + 4 * scalars + 2 * channels + 2 * rb_size
```

径向组合的两条乘加及写入为每通道每径向项 6；基本矩部分为 32 次算术和 7 次累加结果写入；Jacobian 清零、收缩和写入为每保存邻居每基本矩 12；DAG 前后向合计每规则 12。FMA 按乘、加分别计数。它不模拟缓存、向量化和真实指令时延，**相对计算量不是加速比**。共同的径向基求值、邻居构建和力归并成本由实测覆盖。

代表构型按元素组成、原子数和邻居密度分层，固定 seed 抽样；`q_active`、`q_stored` 按抽中构型的原子总数归一化。分层准备会在 rank 0 对验证构型各建立一次邻居表。搜索阶段不会因成本目标额外逐候选遍历 E/F/S 数据。

候选系数缓存仍限制为 256 项；跨代档案由 `cost-archive-size` 限制，只保留 ID、损失和结构统计。档案按成本排序位置保留分散代表，并保留最低成本与最低误差；它不是完整 Pareto 前沿。实测最多 `cost-benchmark-topk` 个档案候选，完整基集另作基线，同一基集去重。

## 实测与最终选择

入围模型都经过压缩、`.mtp` 格式序列化和重载，再对完整训练/验证集直接计算 E/F/S，与缓存损失和结构统计核对。序列化通过内存流完成，避免 MPI 和并发任务共享临时文件。误差复核失败会终止，超出上限的模型不能被选中。

线程数通过 imrmlp 的全局 `-np` / `--num_threads` 参数指定；仅设置 `OMP_NUM_THREADS` 不会覆盖该程序的默认线程数。报告中的 `omp_threads` 是实际配置值。

rank 0 使用 `PotMTP::getEFSOMP` 预热并交错测量模型，记录两种路径的 `ns/atom`：

- `kernel_ns_atom`：复用已有邻居表。
- `total_ns_atom`：每次重新构造邻居表，默认选择依据。

每条路径每轮至少运行 `cost-benchmark-seconds` 秒（单次计算较长时可以超出），至少 3 轮，默认 5 轮。报告中位数和总耗时的 25%/75% 分位数。先找可行模型中总耗时中位数最小者，再将与其四分位区间重叠的模型视为无法区分快慢，以验证误差、原始 ID 顺序处理平局。

mode 6 可以选择完整基集作为结果；mode 5 的完整基集只作速度基线，最终标量数量保持初始数量。最快的结论仅适用于本次实际测量的候选、构型和线程/后端环境；不保证全局最优，也不保证每个模型均能加速。固定 seed 的结构目标可复现，最后的实测选择可能受机器负载影响。

MPI 各 rank 使用同一负载统计和目标函数，rank 0 完成实测并广播最终选择。其他 rank 不进行候选计算，但运行库等待线程仍可能影响测量；正式比较性能建议另用独立单进程、固定 CPU 绑定与线程数运行。

## 输出与失败行为

- mode 6：`__NSGA2__.mtp`、`selected_ones_nsga2.dat`、`removed_nsga2.dat`。
- mode 5：`__TEMPERED__.mtp`、`selected_ones_tempered.dat`、`removed_tempered.dat`。
- `fastfit-cost.tsv`：基线、入围候选、跨代档案及测量环境，原始标量 ID 按压缩后顺序列出，因此列表位置就是新标量 ID。

报告记录输入模型验证损失、相同设置重拟合的完整基集、误差上限、可行性、结构计数、直接 E/F/virial RMSE、计算量和实测速度。F RMSE 沿用当前每原子力向量范数口径；virial RMSE 沿用 `ErrorsReport` 的构型原子数归一化，不是压力张量的逐分量 RMSE。

找不到可行模型时写出失败报告并返回错误，已有势文件保留。诊断报告必须成功写出才发布最终结果；输出文件写出失败也会向所有 rank 传播。

`cost_analysis`、`cost_benchmark` 已接入可选的分段 profile。`scratch_bytes_estimate` 只是指定最大邻居数与线程数下主要 E/F/S 临时数组的估计，排除分配器容量、径向基内部空间、原子力缓冲区和训练矩阵；它不是进程 RSS。进程峰值继续使用原有 profile 的 RSS 历史峰值统计。

## Assembly cache

同一输入上重复搜索时，可通过 [`apps/fastfit_cache.h`](../../apps/fastfit_cache.h)
中的 `func_fastfit::fastfit_cached()` 传入调用方管理的 `AssemblyCache`，最多保存
两个已组装系统，通常为训练和验证系统。缓存逐字节比较输入模型，以及解析后的
构型、标签、每帧/每原子权重、归一化和 MPI 分区；全部 rank 都命中才复用。
全局 E/F/S 权重与 L2 在查找后应用，不修改缓存块。释放 MPI communicator 前应清空
缓存。普通命令行行为不变，缓存不会跨进程或构建持久化。

回归测试见 [`unitest/ut_fastfit_regression.cpp`](../../unitest/ut_fastfit_regression.cpp)。

## 统一 E/F/V 损失与 L2（显式选择）

旧默认 `objective-convention legacy` 保持不变：fastfit 的正规方程对应数据损失总和，并惩罚物种偏置和 moment 系数；普通 MTP 非线性训练使用帧平均损失，L2 只惩罚 moment 系数。仅把 L2 数值写成一样，不能保证目标一样。

新选项 `objective-convention mean-moments` 让两条路径对应同一个目标：所有已加载训练帧的加权 E/F/完整 9 分量 virial 残差平方和除以全局帧数 N，再加 `lambda_mean * ||moment_coeffs||²`；物种偏置、径向参数不受这项 L2 惩罚。权重、原子数指数和参考力归一化仍须显式对齐。缺失标签不改变帧数分母。

`l2-reference mean`（新口径的默认解释）直接接收 `lambda_mean`；`l2-reference sum` 接收总和口径的 lambda，并用实际全局 N 换算。fastfit 内部继续求解总和正规方程，实际矩阵对角增量为 `N * lambda_mean`。例如 N=8410、总和口径 lambda=1e-8，对应平均口径 `1.1890606420927468e-12`。这会移除旧 fastfit 的偏置惩罚，所以不应把新实验写成旧实验的原样复现。

```ini
# fastfit 设置
objective-convention mean-moments
l2-reference sum
weight-l2 1e-8
objective-report objective.tsv
```

非线性 `trainnlpot` 使用对应的 `fit:objective-convention mean-moments`、`fit:l2-reference sum`、`fit:weight_L2 1e-8`；E/F/V 权重配置名为 `fit:energy_weight`、`fit:force_weight`、`fit:stress_weight`。当前统一口径限定普通 CPU MTP、`radial_mode=0`；需要 `fit:weight_L1 0`、`fit:penalty_weight 0`，以及零电荷/HVP 权重。其他势或额外惩罚应使用 legacy 并明确说明目标，不会被静默当成等价。标准输出和可选的全新 `objective-report` 路径记录转换前后 L2、参数组、全局帧数及权重语义。

## 多档模型导出

在 `selection-objective error-cost` 的模式 5/6 下，可选 `cost-archive-rule pareto`，维护有容量上限的非支配档案；默认 `spread` 保持原先的分散成本探索。代理成本有偏差时，保留 `spread` 也可以导出实测前沿。容量裁剪与 shortlist 都保留最低成本、最低验证损失端点，但有限容量不保证全局完整 Pareto 集。

设置 `cost-export-dir pareto-models` 后，保存并重载每个 shortlist 模型、全量 EFS 复核通过，再以临时目录整体发布导出包。目标目录及同名 `.tmp` 必须不存在；原先单赢家文件仍输出。包内含输入母势的精确副本 `parent.mtp`、所有已核验的 `candidate-*.mtp`、`objective.tsv`、`manifest.tsv` 和逐轮 `rounds.tsv`。

manifest 分开标记 `proxy_pareto` 与 `measured_pareto`，并包含原始标量 ID、压缩后图规模、验证误差、代理成本、实测延迟和 IQR。只表示本次有限 shortlist 的前沿，不表示所有搜索候选或全局前沿。力误差栏为旧接口的向量 RMSE，不能直接当力分量 RMSE；完整分量误差可用独立评估器报告。IQR 和计时重叠规则只描述波动，不是统计显著性检验。模型、数据与二进制 SHA256 应在冻结实验时一并记录；导出目录本身不包含私有数据，也不自动提交模型。
