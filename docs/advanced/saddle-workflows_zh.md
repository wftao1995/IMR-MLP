# IMR-MLP 过渡态与驻点计算 {#advanced_saddle_workflows}

[English](saddle-workflows.md)

新用户先按[弛豫](../guides/relaxation_zh.md)和 [NEB 引导](../guides/neb_zh.md)完成基础计算。
本页进一步介绍 `nroptimize`、Dimer、mode following 和 NEB/CI-NEB/VCNEB。
Dimer 同时提供 Python API。坐标约定与数值测试见 [验证说明](saddle-numerics_zh.md)。

## 构建和输入

在仓库根目录按 [安装说明](../guides/installation_zh.md) 配置依赖（oneMKL 可选），再运行
`make -j2 imrmlp`。只有使用后面的 Python API 时才另外构建 Python 包。执行时让 `LD_LIBRARY_PATH` 优先指向本次构建的
`build/release`，防止载入旧共享库。以下示例使用 CPU/OpenMP 构建。

需要用户提供适用于目标材料的势函数、结构和 `index.map` 元素映射。
NR 和 mode following 直接使用 `PotMTP` 的解析 Hessian；不能传入任意势类型。
Dimer/NEB 经通用势接口计算，具体模型仍须分别核验能量、力及变胞所需的应力。
所有相关任务应在独立目录运行，因为原生程序会写入固定名称的中间文件。

## NR 驻点精修

```sh
imrmlp nroptimize --potfile=potential.mtp --input=POSCAR --output=CONTCAR \
  --index=index.map --no-cell --gtol=0.001 --max-iteration=100
```

去掉 `--no-cell` 可优化对称晶胞应变；`--pressure` 单位为 GPa。
固定晶胞时 `gtol` 对应原子梯度分量（eV/Å）；变胞时使用混合广义坐标，
不能把单一数值直接解释成所有分量的 eV/Å。`--fixed-atoms` 接收从 0 开始的原子索引；
固定晶胞时固定笛卡尔位置，变胞时固定分数坐标。受约束 Hessian 在自由子空间内求逆。
`example-inputs/nroptimize.ini` 是参数参考，不是可直接加载的配置文件。

退出 0 表示达到梯度条件，2 表示迭代耗尽；两者均保存最终结构。NR 可以收敛到极小点
或任意阶驻点，仍需检查 Hessian 负模数，不能仅以退出 0 宣称找到一阶鞍点。

## MODECAR 和 Dimer

```sh
# mode.ini 可以为空；若已有 Hessian，可加 --hessian=hessianfull。
imrmlp modefollowing mode.ini --potfile=potential.mtp --input=POSCAR \
  --index=index.map --nocell --mode=9 --imode=0 --modecar-out=MODECAR
imrmlp dimer dimer.ini
```

`dimer.ini` 使用空白分隔的键值，最小示例如下：

```text
structure POSCAR
input-format POSCAR
index index.map
output CONTCAR
pot:pottype mtp
pot:potfile potential.mtp
dimer:modecarf MODECAR
dimer:maxiteration 200
dimer:gtol 0.001
dimer:rot_gtol 0.001
dimer:trn_gtol 0.001
```

不指定 `modecarf` 时用随机方向，`dimer:random_seed 0` 是可复现默认值，-1 为随机种子。
显式指定的方向文件缺失、截断、非有限、零向量、维度或坐标不匹配均报错。
`ddr` 是中心到单端点的距离；固定晶胞单位 Å。平移梯度使用两端平均；旋转梯度与曲率
目标一致。旧版参数可能需要重新调整，尤其是旋转阈值。

固定晶胞 MODECAR 采用笛卡尔原子位移；新版以 `GPML_MODECAR 1 N convention` 开头，
VC 文件还保存参考晶胞，并在格式标识中记录坐标约定。旧纯数值文件仍可读，
但必须与当前坐标设置一致；它不是 VASP 原始 MODECAR 的无条件替代品。
变胞导出会提示所需设置：对称模式需要 `vcflag true`、`symm_strain true`、
`restore_F_cholesky true`、`recast_atomic_force true`，并使用同一参考晶胞。
`--car-rep` 的变胞导出使用 `symm_strain false`。详见完整 `example-inputs/dimer.ini`。

Dimer 退出 0 为收敛、1 为未收敛；输入/数值/输出错误会产生错误诊断并非零退出。
Python 对错误抛出异常，对正常耗尽返回 `converged=False`。
`dimer.dat` 含停止状态、迭代数、中心能量/焓、中心及旋转梯度、曲率；还输出最终结构、
端点和 `dimer_modecar_final`。负方向曲率不能单独证明只有一个负模。

```python
from imrmlp import Dimer
# pot、atoms 分别为已加载的 Potential 和 Atoms。
result = Dimer(pot, modecar="MODECAR", max_iter=200, write_traj=True).run(atoms)
print(result.converged, result.stop_reason, result.max_force, result.curvature)
saddle = result.saddle_atoms  # 独立副本，可在释放驱动后使用
```

## NEB、VCNEB 与 mode following

NEB 的原有设置和 Python `NEB` 保留；VCNEB 的 `pressure` 单位 GPa，
镜像能量已经包含 pV，后处理不要再次累加。应对目标材料的模型、晶胞尺寸及图像数
做收敛检查。MPI、路径对称性和释放检查见 [NEB 并行与路径对称性](neb-parallel-symmetry_zh.md)。

`modefollowing` 模式 1 是 Hessian/动画，模式 2 才是迭代模式跟随，模式 4 只做单步 NR，
模式 9 导出方向。模式 2 通过 `--max-loop` 限制每阶段次数；重复初始化不会读取
并未指定的 `VIB_CONSTRAINT`。预算耗尽或最终力/单负模检查未通过返回 2。
固定胞笛卡尔模式应同时设置 `--nocell --car-rep` 和配置文件中的
`relax:relax_cell false`、`relax:car_mode true`、`relax:driver fire`。
变胞 mode following 的 Hessian/投影坐标组合需要额外验证；在两个不同晶胞之间
搜索时，优先使用经过核验的 VCNEB 路径。

## 核验候选过渡态

确认收敛并检查路径；一阶鞍点应恰有一个非平移的负 Hessian 模。应检查力容差、
所用有限差分步长、晶胞尺寸，并以独立参考数据验证势函数精度。数值回归测试不能
证明某一材料迁移或相变势垒的物理精度。
