# NEB 并行计算与路径对称性

[English](neb-parallel-symmetry.md)

`neb` 和 `calc_neb` 共用同一个控制器。MPI 模式下只有 rank 0 运行优化器、调整
图像数量、写轨迹和结果；图像求值按顺序分配给各 rank。端点、有限差分探测、
被拒绝的试探步和变图像求值均使用同一接口。允许空闲 rank，求值异常统一报告，
各 rank 使用相同退出状态。每批几何在各 rank 上复制，每张图像只由其所属 rank
求值，结果按图像顺序汇总。MPI 传输要求相同构建，不引入新的 restart 格式。

## MPI 与 OpenMP

```bash
make BUILD_MODE=mpi imrmlp
MKL_NUM_THREADS=1 OMP_NUM_THREADS=2 mpiexec -n 4 \
  build/mpi/imrmlp-mpi neb neb.ini --num_threads 2
```

运行时按需要将匹配的构建目录放在 `LD_LIBRARY_PATH` 最前。所有 rank 必须能访问
相同势文件和外部计算器输入。输入解析和输出仅由 rank 0 完成。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `neb:omp_mode` | `images` | `images`：同一 rank 上的图像间 OpenMP；`atoms`：逐张处理本 rank 的图像，在势内部使用 OpenMP。 |
| `--num_threads` / `num_threads` | `4` | 每个 rank 的线程数；显式设置覆盖环境中的线程数。 |

多张小图像可用 `images`；大图像可尝试 `atoms`，但取决于势是否实现内部并行，
否则回退到串行求值。两层 OpenMP 不嵌套。另行控制 BLAS 线程和外部计算器进程，
避免资源超额使用。MPI 通信在 OpenMP 区域外执行，多线程运行要求至少支持
`MPI_THREAD_FUNNELED`。

外部 shell 计算器使用 rank/thread 独立目录，但脚本必须自行管理计算资源；
从 MPI worker 启动外部 MPI 计算器并不会被自动配置。在线主动学习暂限单 rank，
因为模型更新尚未分发。Python 的 `NEB(omp_mode="images"|"atoms")` 只控制本地
OpenMP；MPI 图像调度由 CLI 提供。

## 对称性诊断

```ini
neb:symmetry_diagnostics neb-symmetry.dat
neb:symprec 1e-5
```

报告列出初始端点的空间群操作、原子置换、共同操作索引、所选子群及允许的原子／
晶胞模式数。每张图像列出空间群、操作数量和相对所选仿射子空间的残差。残差是在
NEB 缩放坐标中的广义范数，不应当作 Å 单位位移。初始共同子群用 `initialized_path` 标记，后续段另外列出当前共同操作索引。不同段区分初始化路径、扰动或
约束处理后的路径及最终路径。`symprec` 是 Å 单位的位置／晶格容差；接近容差的
对称性判定可能随其变化。

共同操作必须保持整条**有固定原子标签且已展开周期边界的路径**，使用相同原子
置换和整数晶格偏移。相同空间群编号并不足够。程序保留输入的原子顺序、晶胞基和
原点，不自动重排等价原子、匹配原点或等价晶胞，也不搜索反转图像顺序的路径
对称性；这些变换可能改变研究的扩散机制。操作采用行向量约定 `s' = s W + t`。

空间群诊断暂支持三维全周期体系；VC 路径还要求 `symm_strain=true`。不支持的
路径会在报告中明确标记，普通 NEB 和平滑扰动仍可使用。

## 可复现的破对称扰动

```ini
neb:perturb_amplitude 0.02
neb:perturb_seed 17
neb:symmetry_diagnostics neb-symmetry.dat
```

在线性／IDPP 初始化后，对中间图像施加以 `sin(pi*i/(n_images-1))` 调制的随机
原子位移，固定端点和晶胞。整条路径中的最大原子笛卡尔位移等于设置的 Å 单位
幅度。存在非平凡共同子群时先去掉其不变分量；多原子体系去掉整体平移，并遵守
selective-dynamics 掩码。如果过滤后没有可用位移则报错。P1 或不支持空间群分析的
体系仍可生成平滑扰动，但不承诺进一步降低对称性。

幅度默认 0，种子默认 0，二者均写入报告。可比较不同种子生成的收敛路径和势垒；
初始扰动本身不能证明存在更低的势垒。

## 可选的保对称约束

```ini
neb:keep_symm true
neb:symmetry_diagnostics neb-symmetry.dat
# 可选：初始端点报告中的操作索引，作为子群生成元。
# neb:symmetry_operations 0,3
```

默认保持整条初始化路径的共同子群；指定生成元时，通过现有 relax 子群选择工具
补齐单位元并闭合。闭合后任何操作若不满足整条路径都会报错。索引属于初始端点
的操作列表，不是每张图像各自搜索出的列表。只有单位元时不会减少自由度。

原子轨道模式与对称晶胞模式在 NEB 的缩放度量下正交投影；坐标解码、广义力、
切线、弹簧／爬升力、受保护的更新方向和收敛判据使用同一允许子空间。优化期间
操作及映射保持固定，支持图像插入和删除。端点保持输入值，中间图像可在检测
容差内进行对称化修正。被约束禁止的原始力不影响受约束收敛；这样的收敛不代表
对被排除模式也稳定，应与破对称的无约束路径对比。

约束要求三维全周期，VC-NEB 使用对称应变坐标；支持晶胞度量权重及两种原子坐标
表示。不能与 selective dynamics 或非零扰动幅度同时使用。`keep_symm` 默认关闭，
单独设置 `symm_strain` 仅处理晶胞旋转，不会开启空间群约束。

Python `NEB` 构造函数以仅限关键字参数提供相同的本地选项：`keep_symm`、
`symprec`、`symmetry_operations`、`symmetry_diagnostics`、`perturb_amplitude`、
`perturb_seed`。

## 数值语义与 restart

`climb_delay` 在 `stable-v1` 中按接受步计数，在 `legacy` 中按力评估计数。
收敛采用完整活动 NEB 残差；固定胞原子力单位为 eV/Å，VC 残差与广义坐标度量
有关。顶层 `gtol` 覆盖 `neb:gtol`，NEB 不读取 `relax:` 设置。
退出码：0 收敛；2 正常结束但未收敛；1 错误（原有的设置文件缺失状态仍为 128）。

restart 文件格式和现有路径／阶段恢复行为保持不变，仍重置优化器历史。由于现有
格式不保存固定对称操作和映射，`--next_step` 不接受新增对称约束或扰动；应把已有
图像明确提供给一次新计算。仅输出对称性诊断可以与 restart 一起使用。

## MPI+OMP 分阶段计时（默认关闭）

```ini
neb:profile profile.json
```

空值默认不增加计时通信或文件。CLI JSON 记录控制进程的初始化（解析设置之后）、
主优化、结果输出和对称性释放检查耗时；各 rank 的 `main`、`symmetry_release`
分别记录 EFS、序列化/反序列化、MPI 集合通信及其中的等待、批次数和尝试计算的
image 数。端点计入 image 数。EFS 时间是本地 OpenMP 批次的墙钟时间，
不是各线程 CPU 时间之和。势加载归入初始化，不计入 rank EFS；报告汇总/写入、
MPI 启动与终止不计入这些阶段。

整体阶段已包含 EFS 和通信，各 rank 的时间又互相重叠，不能相加。
`batch_wall_seconds` 包含批次内计时分类和调度开销。
便宜的 images 或空闲 ranks 会使通信占比增大。

## 保对称计算后自动释放检查（默认关闭）

```ini
neb:keep_symm true
neb:symmetry_release release-check
neb:symmetry_release_trials 3
neb:symmetry_release_seed 0
neb:symmetry_release_steps 200
neb:symmetry_release_evaluations 2001
neb:symmetry_release_amplitude 0.01
neb:symmetry_release_strain 0.001
neb:symmetry_release_energy_tol 0.0001
```

`symmetry_release` 默认为空；非空值指定一个新目录，不覆盖已有目录。
CLI 先保存主计算，再在独立副本上检查。Python 的
`NEB(symmetry_release="release-check", …)` 调用同一实现，结果字段
`NEBResult.symmetry_release_status` 给出检查状态；C++ 提供计算后调用的
`check_symmetry_release(RunOptions)`。

只检查已收敛、启用 `keep_symm` 且存在被排除自由度的路径；其他情况明确跳过。
暂不支持在线主动学习和外应力。每次试算保留势、压力、参考晶胞、坐标度量和端点，
解除共有子群约束，固定 image 数，不重新 MIC 展开或 IDPP 初始化。
诊断使用有预算限制的 `stable-v1` 控制器和所选优化器，不改变主计算的数值策略。
已进入爬山阶段时不重复爬山延迟。每次试算有独立迭代和整带 EFS 预算；
两端点评估单独计入 image 数，不占整带预算。

第 0 次无扰动释放，其余使用连续种子，在原子群投影的补空间中产生平滑扰动，
多原子体系去除整体平移，端点扰动为零。晶胞和原子分量都在解码后检查幅度：
晶胞采用 `||H_old^{-1} H_new - I||_F`；原子 Å 上限指晶胞仿射变形后的额外位移，
不包括晶胞携带原子的运动。将某项幅度置零可禁用该分量，没有剩余扰动的试算明确跳过。

`report.json` 保存主路径与试算能垒（总 eV）、收敛情况、残余力、实际扰动、
已完成评估次数、含失败批次的尝试 image 数和耗时。主路径保存在 `image-*.vasp`，各试算初始和最终路径分别保存在
`trial-N/image-*.vasp` 和 `trial-N/final/`。固定胞比较能量，变胞比较焓。
只有已收敛试算降低能垒超过阈值才报告 `lower_barrier_found`；其他状态分别是
`no_lower_barrier_observed`、`inconclusive`（预算不足、停滞或试算跳过）、
`failed`、`skipped`。诊断失败不替换主路径、计数和 CLI 收敛退出码；
报告本身无法写入时给出警告，需单独检查诊断状态。

这是有限次局部试算，不是鞍点阶数或全局最低路径认证。残差求解器也可能将受扰动路径
带回原驻点，即使某个被排除方向不稳定。因此应比较多个种子和幅度，检查未完成的试算；
需要鞍点认证时另做独立验证。restart 行为保持原样。

## 显式多路径搜索

`python scripts/benchmark/neb/search_paths.py --help` 提供独立批量工具说明。
输入为已有 NEB 运行目录：`neb.ini`、`neb.dat` 和全部有序的
`nebimage_N_final.vasp`。源路径可以未收敛；`--frames initial` 改用完整初始路径。
输出目录必须不存在。工具冻结输入、模型和程序，记录哈希、动态库、物理核心和
线程绑定信息，不修改源数据。

```sh
PYTHONPATH="$PWD/build/neb-python:$PWD" \
LD_LIBRARY_PATH="$PWD/build/release:$LD_LIBRARY_PATH" \
python scripts/benchmark/neb/search_paths.py --source build/previous-neb \
  --output build/path-search --binary build/release/imrmlp --threads 8
```

默认 **25 次**：无扰动基线，加四个 seed（0、1、2、3）× 三种模式 × 正负方向。
每次使用 `stable-v1`、源优化器和力阈值，独立限制为 200 步、2001 次整带 EFS、
3600 秒；端点 EFS 单独计数。可用 `--steps`、`--evaluations`、`--timeout` 修改。
候选串行编排，单次可通过 `--ranks`、`--threads`、`--omp-mode` 使用现有 MPI+OMP。
计时仍默认关闭，需显式 `--profile`。某次失败不影响其余候选继续执行。

共享生成器不需要势函数，也不执行 EFS：

```python
from imrmlp.neb import make_path_seed
candidate, info = make_path_seed(images, settings={"vcneb": False},
    mode="mixed3", seed=7, amplitude=0.01, atoms=[12, 13])
```

`settings` 使用不带 `neb:` 的原生键。输入和端点保持不变。`sine1`、`sine2`
分别为 sin(πs)、sin(2πs)；`mixed3` 使用最多前三个可解析正弦模，各自取独立随机
方向并乘 1/k² 权重。图像数不足时忽略不可解析模式，零扰动明确跳过。seed、模式
和符号确定几何，不依赖 MPI rank 数或调度。

`--atoms` 显式指定零基原子编号；`--center` 和 `--radius` 在源路径中间 image
选择余弦径向衰减邻域，与显式编号取交集，随后冻结原子集合。方向同时满足局域
支持、固定自由度和共同对称投影的补空间；不存在可破缺对称模式时使用全部自由
方向并记录。交集为空就跳过，不扩大为全局扰动。局域单原子运动不会因去除质心
平移而消失。默认幅度在解码物理几何上限制：原子为晶胞仿射运动之外的 0.01 Å，
晶胞为 `||H_old^-1 H_new-I||_F ≤ 0.001`。晶胞仿射运动可影响局域集合之外的原子。
`--amplitudes`、`--strain` 可显式改变幅度；开始前显示展开后的试算数量。

VC 搜索要求完全周期、对称应变表示，以及原始 `neb_reference_cell.data`、六个
有效晶胞度量尺度、压力和 `neb.dat` 中的坐标元数据；缺失时停止。支持分数坐标
和参考笛卡尔坐标，以及 Cholesky/对称平方根恢复，但源路径必须属于对应坐标图。
固定自由度遵循原坐标表示。一般变形坐标、部分周期 VC、在线主动学习和非静水
外应力暂不支持；固定胞可使用部分周期边界。

普通 CLI 新计算可用 `neb:reference_cell FILE` 传入 3×3 原始参考晶胞，必须提供
所有 VC 图像；它保留坐标框架并禁用 MIC、IDPP 和晶胞对齐，与 restart、
`keep_symm`、初始化扰动不兼容。`neb:path_pbc 110` 恢复 POSCAR 未保存的周期标志。
`neb:restore_F_cholesky` 默认 true，false 使用已有对称平方根恢复。
这些选项不恢复优化器状态。批量工具释放空间约束、固定图像数与端点，不使用
`--next_step`。已有单次初始化扰动、释放检查的随机序列及 restart 行为保持不变。

### 报告判读

每次保存初始/最终路径、扰动诊断和 `result.json`；`manifest.json`、增量
`results.json`、最终 `report.json` 保存来源、独立开销和分类。仅收敛结果进入
能垒排名和正式重复组。固定胞比较总能量，变胞比较总焓 E+pV。超时、预算耗尽、
停滞、失败和跳过分别保留。与基线比较“更低”要求双方收敛；接近能量容差或力
阈值的结果标为 `needs_refinement`，不能据此宣称新机制。

去重保留原子编号和端点对应，允许整条路径共用一次原点平移、每个原子固定的
晶格表示偏移，以及一次行列式 +1 的整数幺模晶胞变换。禁止逐 image 重排、包裹、
独立对齐或路径反转。仅在比较时，用同一源度量的归一化弧长和双方折点并集重采样；
原始路径不改写。默认 RMS 0.01 Å、最大差 0.03 Å、相对晶胞差 1e-4、收敛能垒差
1e-4 eV。确定顺序的完全链接聚类要求组内任意两条都满足条件，避免链式误合并。
整条路径共用一个空间操作和原子映射得到的几何等价只记录关联，不合并编号。

这是有限局域搜索，不认证全局最低路径或鞍点。包含 image 反转的路径群及不可约
表示扰动留待独立群作用实现。回归覆盖平移、整体编号变换、幺模基底变换后的
几何/EFS/投影；距离仅在同一物理度量下比较。焓梯度以 1e-4、1e-5、1e-6 步长和
截断/舍入误差判据独立验证；弹簧、切向和 climbing 残余仍按代数关系另测。
