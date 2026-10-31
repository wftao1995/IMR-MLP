# 普通 MTP 的原生变量投影训练 {#doc_docs_mtp_native_varpro}

[English](mtp-native-varpro.md)

`trainnlpot --do-varpro` 在每次试探中求解线性系数，同时优化径向参数；当仅做终点线性精修仍有较大误差时，可考虑此方法。支持普通 MTP 的单进程和 MPI 多进程 CPU。每次试探先调用现有 QR/SVD 线性求解器，将物种偏置和线性矩系数求到条件最优，再调用现有完整 E/F/virial 梯度，取其径向部分交给 ALGLIB L-BFGS。

训练复用数据权重、线性 L2 和径向惩罚。训练数据和邻居表只加载一次；依赖径向参数的设计矩阵在每个试探点重新组装。若只想固定全部非线性参数求解一次，见[线性精修](linear-refit_zh.md)。

## 使用

准备普通 `mtp` 模型、训练集和可选验证集，`index.map` 的元素顺序须与模型一致。在仓库根目录构建，为每次训练指定独立的输入和输出路径：

```sh
make -j8 imrmlp
LD_LIBRARY_PATH=$PWD/build/release:$LD_LIBRARY_PATH \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 \
build/release/imrmlp trainnlpot \
  --settings /absolute/fit.ini --do-varpro --output /absolute/exported.mtp --num_threads 2
```

`fit.ini` 示例（将 `/absolute/...` 替换为实际绝对路径）：

```text
# ===== Input/Output =====
index               /absolute/index.map
trainset            /absolute/train.cfg
trainset-format     mlpcfg
validset            /absolute/valid.cfg
validset-format     mlpcfg

# ===== Potential =====
pot:pottype         mtp
pot:potfile         /absolute/initial.mtp
pot:typemap         /absolute/index.map

# ===== Runtime =====
fit:device          cpu

# ===== Initialization =====
fit:random_init     false
fit:force_random    false

# ===== Training Schedule =====
fit:training_schedule single

# ===== Parameter Scaling =====
fit:parameter_scaling none
fit:scaled_step_max 0

# ===== Loss Function =====
fit:energy_weight   1
fit:force_weight    0.05
fit:stress_weight   0.001
fit:energy_pow      1
fit:force_pow       0
fit:stress_pow      1
fit:weight_L2       1e-10
fit:weight_L1       0
fit:penalty_weight  1e-6

# ===== L-BFGS =====
fit:bfgs_m          10
fit:bfgs_maxiteration 40
fit:bfgs_grad_tol   1e-8
fit:bfgs_conv_tol   1e-12
fit:bfgs_step_tol   1e-12

# ===== Diagnostics =====
fit:varpro_report   varpro.jsonl
fit:validation_interval 5
fit:validation_metric force

# ===== Output =====
fit:save_to         fitted.mtp
fit:save_every      5
```

示例权重与容差是起点：先用它们跑通，再按报告里的验证 RMSE 调整。不需要额外开启 `linear_refit`；VP 内部总是执行 `sr` 线性求解。现有 `linear_refit_solver/rtol/atol/tolerance` 设置也用于内层求解。默认要求完整梯度中的线性部分无穷范数不超过 `fit:varpro_linear_gradient_tolerance=1e-7`。首次不满足时，在同一径向参数下，以当前线性解为起点重新拟合残差，最多额外执行 `fit:varpro_linear_refinement_steps=2` 次；设置为 `0` 可关闭。修正不改变损失、正则项、秩截断或梯度阈值。超过次数仍不达标就停止并恢复原参数，避免使用精度不足的包络梯度；目标不一致或非有限值直接拒绝，不重试。MPI 用所有进程线性梯度范数的最大值决定是否修正，避免分支不一致。

这里的残差修正是在同一次径向试探中，把线性模型 `A c` 的当前预测误差重新作为右端，求解增量 `δc`，再令 `c ← c + δc`。正则项按更新后的 `c + δc` 计算。这样可以减轻第一次大增量相减留下的浮点误差；不会重启外层 L-BFGS，秩截断或模型病态造成的停滞仍要靠外层重启或换模型解决。它仍受每次线性求解的完整目标检查和最终梯度阈值约束。

`bfgs_maxiteration=0` 表示只做初始线性投影和报告。其余三项停止容差分别对应径向梯度范数、相对函数变化、步长。最大步数停止不标记为收敛。

## 保存和选择模型

设 `fit:save_to` 为 `fitted.mtp`：

- `fitted.mtp`：最近接受的完整参数，每个接受点更新；初始线性投影也保存。
- `fitted.mtp.iter-000005.mtp`：按 `save_every` 保存的接受点快照。
- `fitted.mtp.best-training.mtp`：本次运行中原生总目标最低的接受点，包含正则项。
- `fitted.mtp.best-data.mtp`：本次运行中加权 E/F/virial 数据损失最低的接受点，不包含径向惩罚和线性 L2。初始投影和每个接受步都检查，与 `save_every`、验证间隔及验证上限无关。
- `fitted.mtp.best-validation.mtp`：本次运行中满足约束且验证指标最低的已检查接受点。
- CLI 的 `--output`：训练结束时导出的最终接受点，与 `fit:save_to` 对应同一模型。

初始点和最终点总是验证，中间按 `validation_interval` 验证；“最佳验证”只涵盖实际检查过的点。验证集不参与线性拟合、梯度或线搜索。无验证集时不生成最佳验证模型。

`validation_metric` 支持 `energy`、`force`、`stress`。可同时设置绝对上限：

| 设置 | 单位 | 默认 |
|---|---|---|
| `fit:validation_max_energy` | eV/atom | -1，不限制 |
| `fit:validation_max_force` | eV/Å | -1，不限制 |
| `fit:validation_max_stress` | GPa | -1，不限制 |

指标或约束所需标签缺失时明确报错。所有验证点均不满足上限时，不生成最佳验证模型，报告中的 `best_validation_iteration` 为 -1。上限只影响验证模型选择，不影响训练目标。

每次运行须使用新的 `varpro_report` 文件和 `save_to` 前缀，防止混入旧报告或旧最佳模型。续训时将已有检查点作为 `pot:potfile`，并设置 `fit:bfgs_iteration_offset`。这是**系数重启**：径向和线性参数完整恢复，L-BFGS 历史重建；不承诺与不中断训练的路径相同。最佳模型记录也从本次运行的初始投影重新开始；跨运行选择可以比较各自报告。

## 结构化报告

JSONL 包含 `settings`、`initial`、`accepted`、`final`、`termination`，发生残差修正时还记录 `linear_refinement`（求值编号及每次线性梯度范数，包括未被外层接受的成功试探）；求值异常时尽可能追加含具体错误的 `evaluation_failed`，训练失败时追加 `failed`。试探点不会更新接受点检查点或最佳模型。

每个接受点记录：

- `loss.total/data/energy/force/virial/radial_penalty/linear_penalty`；数据损失按现有权重归一化，总目标等于数据损失与两类惩罚之和。
- 原物理径向梯度、优化器坐标梯度、线性梯度无穷范数，内层秩、最优性残差、组装/求解/核验时间。
- `linear_refinements` 和 `linear_gradient_initial_inf`：当前点的额外修正次数和首次线性梯度范数；内层时间累计全部求解，秩和归一化最优性残差来自最后一次求解。
- 每个径向通道跨全部元素对和径向基的系数范数。
- 训练集及当次验证集的独立物理 RMSE：能量为逐构型每原子误差，力为全部笛卡尔分量，应力为 virial 除以体积后的完整 3×3 张量（与 `calc_err` 的 GPa 分量 RMS 一致）。应力指标为 `sqrt(sum(||预测应力 - 参考应力||_F²) / (9 * N))`，`N` 为有应力标签的构型数，满足整体旋转不变性。单位分别为 eV/atom、eV/Å、GPa，**不施加训练权重**。缺失标签为 `null`。
- 当前最佳总目标、最佳纯数据损失、最佳验证模型所在的累计迭代编号，分别为 `best_training_iteration`、`best_data_iteration`、`best_validation_iteration`。

`settings` 记录实际生效的数据权重、幂次、力归一化参数、径向惩罚和线性正则权重、内外层求解容差，以及验证选择规则。`termination.type` 采用 ALGLIB 编号；1/2/4 标记停止条件满足，5 表示达到迭代上限。0 表示只做了初始投影，L-BFGS 一步未走；此时判断模型可用性看 JSONL 里的验证 RMSE 与 `best_validation_iteration`。

JSONL 为 `schema_version=3`，保留原字段，并在 `settings` 中记录 `stress_metric="full_tensor_rmse_9"`。此前版本的 `stress_GPa` 按六个独立分量等权计算；旧报告数值和 `validation_max_stress` 阈值需按新口径重新核对，不能用一个固定比例换算。`best-data` 仍使用训练设置中的权重与归一化，不等同于最低未加权力 RMSE。改变正则权重时，可以比较相同数据权重下的 `loss.data`；不同正则权重的 `loss.total` 不适合直接排名。

## 可选的固定径向坐标缩放

设置 `fit:varpro_radial_scale_file /absolute/radial-scales.txt` 启用。文件只包含恰好 `radial_coeff_size` 个有限正数，以空白分隔，排列顺序与 MTP 径向系数一致（元素对、径向通道、径向基索引）。未设置时全部为 1，保留原物理坐标。文件只由 MPI 主进程读取、验证，再广播尺度向量；缺失、数目不符、非正或非有限值均拒绝。

优化器使用 `r_i = s_i z_i`，梯度按 `g_z,i = s_i g_r,i` 转换。每次试探仍在物理参数 `r` 上执行原来的线性求解、E/F/virial 目标和两类正则项，模型保存的也是物理参数。它不是对描述符或惩罚公式的修改。

尺度在整次运行中固定，报告完整向量。L-BFGS 的梯度和步长停止条件应用于 `z` 坐标，因此不同尺度下不能仅靠相同停止阈值宣称相同物理收敛精度；报告同时提供 `radial_gradient_inf` 与 `optimizer_gradient_inf`。系数续训仍重建优化器历史，需明确设置本次采用的尺度文件。旧的 `parameter_scaling`/`set_scale`/`scaled_step_max` 选项仍不与 VP 混用。

结构化报告及总目标、纯数据损失、验证误差三类最佳检查点随 `--do-varpro` 启用；已有其他优化器的日志行为保持原样。

## 范围和验证

支持普通 `PotMTP`、`radial_mode=0`、单进程或 MPI CPU、E/F/virial、线性 L2 和上述固定对角径向坐标缩放。LES/其他派生势、GPU、HVP/电荷损失、线性 L1、非空 `trainable_groups` 设置及其他参数尺度模式不在此路径的支持范围，明确拒绝不兼容设置。


`unitest/ut_varpro.cpp` 覆盖：重新求解内层的径向有限差分、独立线性拟合与梯度比对、完整参数回滚、接受点模型保存、验证选择、系数续训、缺失标签与不支持目标的拒绝。

## MPI 多进程

```sh
make BUILD_MODE=mpi -j6 imrmlp
LD_LIBRARY_PATH=$PWD/build/mpi:$LD_LIBRARY_PATH \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
mpirun --bind-to none -n 4 \
  build/mpi/imrmlp-mpi trainnlpot \
  --settings fit.ini --do-varpro --output exported.mtp --num_threads 1
```

使用 MPI 编译的 `build/mpi/imrmlp-mpi`；release 构建仍按单进程工作。所有进程使用同一组输入和设置。相对路径以各进程的工作目录为基准。

主进程统一运行 L-BFGS，广播试探点、接受点和结束指令。各进程按既有构型分配参与线性拟合与梯度计算；初始系数以主进程为准。验证误差汇总全局平方误差及分量数，再计算 RMSE。没有本地训练或验证构型的进程仍参与通信，不会被当成零误差样本。最佳模型选择、JSONL 和模型写入只在主进程执行。

受控的本地计算异常在下一个通信阶段前同步。主进程的文件写入或优化异常也会通知工作进程退出，保留最近接受的完整系数。内层 QR/SVD 的根进程求解与设计矩阵重新组装仍然存在；这项改动提供数据并行能力，加速比随构型规模与通信开销变化，用同一设置在 `-np 1` 与 `-np N` 下各跑一次比较。当前 CLI 仍由每个进程读取输入后分配构型。

MPI 回归测试：

```sh
make BUILD_MODE=mpi build/mpi/unitest/ut_varpro_mpi.test
# 在仓库根目录执行；分别将 -n 改为 1、2、4。
LD_LIBRARY_PATH=$PWD/build/mpi:$LD_LIBRARY_PATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
mpirun --bind-to none -n 4 build/mpi/unitest/ut_varpro_mpi.test
```

测试包括与 `MPI_COMM_SELF` 全数据参考的完整径向梯度比较、空数据进程、只有工作进程持有验证标签、单进程计算报错、主进程报告和检查点写入失败。MPI 测试显式固定 Catch2 的顺序和随机种子，确保所有进程进入相同测试。
