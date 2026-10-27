# MTP GPU 求值与训练（CUDA/HIP） {#development_cuda}

[English](cuda.md)

本页前半部分面向使用 GPU 求值、训练的用户，后半部分说明开发实现。
需要可用的 GPU、匹配的 CUDA 或 HIP 工具链，以及[构建指南](../guides/installation_zh.md)
中的本地依赖。CPU 构建不能通过添加运行参数获得 GPU 支持。

## 构建与后端选择

从仓库根目录选择与硬件匹配的一种构建：

```sh
make BUILD_MODE=cuda -j2 main  # NVIDIA CUDA，生成 build/cuda/imrmlp-cu
make BUILD_MODE=hip -j2 main   # HIP/DCU，生成 build/hip/imrmlp-hip
```

CUDA 构建可设置 `NVCC`、`CUDA_ARCH`；HIP 构建可设置 `HIPCC`、`HIP_ARCH`，
具体值取决于安装的工具链和设备。MPI 构建分别使用 `mpi-cuda`、`mpi-hip`，
两者的主程序都叫 `imrmlp-mpi`，请通过完整构建路径区分。

运行时 `--device` 接受 `cpu`、`cuda`、`hip` 和 `dcu`；`dcu` 是 `hip` 的别名。
训练中的 `fit:device` 使用相同取值。每个构建只包含一种 GPU 后端，
请求与构建不匹配的设备会报错。下文以 CUDA 为例，HIP 对应使用
`build/hip/imrmlp-hip` 和 `--device=hip`，训练则设 `fit:device hip`。

## 当前推理实现

`runtime/mtp/gpu-direct` 中的公共核心支持 CUDA 和 HIP。批求值适配器服务两个前端：

- LAMMPS `pair_style mlip/gpudirect`，使用完整邻接表和 LAMMPS ghost 原子；
- `imrmlp-cu calc_efs --device=cuda` / `imrmlp-hip calc_efs --device=hip` 及其底层的
  `MTPGpuEvaluator`/`MTPGpuBatch` 适配器，使用连续 CSR 邻接表和显式周期位移。

```bash
build/cuda/imrmlp-cu calc_efs --input=POSCAR --output=efs.cfg \
  --index=index.map --potfile=potential.mtp --pottype=mtp \
  --device=cuda --gpu-id=0 --gpu-memory-fraction=0.75
```

CLI 保持结构边界，并把输入打包为有界 GPU 块。`--gpu-batch-atoms` 和
`--gpu-batch-structures` 可显式限制规模；零表示根据当前空闲显存和
`--gpu-memory-fraction` 自动定尺寸。`--gpu-reuse-topology` 在完全兼容的连续单结构
轨迹帧之间复用静态 CSR 数组。`--energy-only` 选择真正的设备能量路径，
不启动力或应力计算。CPU 仍是默认后端。`--gpu-id=-1` 使用当前 GPU 设备；
显式非负 id 选择可见设备。完整 Al-Ni 示例见 `examples/mtp-gpu-alni`。

前向路径分三个阶段：基本矩按 tile 累加；复合矩及其能量伴随按依赖层级传播；
对导数重新计算并直接缩并到力。初始化时快速路径按依赖层级把乘积规则按前向输出和
反向目的分组。每个目的由一个线程持有，在寄存器中求和并只写一次，
因此生产路径的矩与伴随传播没有竞争的 FP64 原子操作。原始调度顺序仍保留给
单线程有序诊断路径。对导数重算避免了全局
`basic_moment x neighbor x xyz` Jacobian。中心原子和矩 tile 在并行 GPU 块中执行。

FP64 仍是默认和参考模式。实验性 LAMMPS 选项 `precision mixed` 把经典线性
`moment_vals` 和基本矩部分工作区改为 FP32；能量伴随（`dmom`）、全部公开接口、
径向计算、系数、归约、能量、力和 virial 仍为 FP64。它覆盖单 Chebyshev、
ChebyCos、现代参数化 Poly5、split/type-pair 和可配置径向表示。
不支持的非线性模型和 GPU 拟合会明确拒绝混合精度。另一个实验选项
`precision mixed-aggressive` 还把 `dmom` 存为 FP32，但保留 FP64 的依赖组归约和最终输出。
两种混合模式均为实验性，且仅限经典线性推理。
LAMMPS `deterministic yes` 启用现有有序诊断内核，该模式下同设备重复必须逐字节一致。
默认 `deterministic no` 保留并行归约，并以严格数值重复容差而非逐字节相等来检查。

当前支持带 `RBChebyshev`、`RBChebyCos`、`RBChebyshevPoly5` 和
`RBChebyshevConfigurable` 的经典线性 MTP。IMRMLP 适配器还支持 split/type-pair 径向基，
每个中心-邻居物种对一个有序基，各对可独立选择最小距离、截断、缩放和基类型。
可配置描述符显式携带二次/余弦/Poly5 截断、线性/指数坐标变换、Poly5 `cutoff_b`
和指数 `shift_lambda`；所有对仍保留 `PotMTP` 要求的公共径向基大小。
IMRMLP 适配器还支持 MTP-M 的二次 `mtp_m_q2`、ArcSinh `mtp_m_fs`、
线性 `mtp_m_li`、单隐藏层 `mtp_m_n1` 和双隐藏层 `mtp_m_n2` 外部模型，用于 E/F/S 和完整参数梯度。
神经 GPU 路径接受任意正描述符和隐藏维数，保留原生扁平系数布局，
实现 tanh、ReLU、sigmoid 以及 NN1 线性激活。standalone LAMMPS 势读取器现在接受
相同的单基和有序对径向表示，包括可配置截断/变换参数以及自描述 q2/fs/NN1/NN2 布局。
IMR-MLP 批适配器还支持 FP64 SOG E/F/S 和完整参数梯度（线性 MTP 及这些 MTP-M 头），
包括多通道和冻结参数组。SOG 要求完整周期结构；倒空间工作区计入推理和训练分块内存预算。
此处的批求值适配器不支持 Coulomb LES，Hessian/HVP 训练目标和电荷损失也需要 CPU。
LAMMPS `mlip/gpudirect` 仅支持短程模型。另一个
[`mlip/native` 接口](../../LAMMPS/USER-MLIP/native/README.md)具有独立的分布式 GPU
长程求值路径，可处理受支持的 LES/SOG 模型；它的能力与本页批适配器不同。

不可变批几何、标签和 SOG 网格以 identity/revision token 缓存。刷新几何、改变标签、
切换求值器或前向调用失败都会使相关状态失效。训练验证使用所选后端及其自身批次；
`fit:validation_interval` 控制已接受 L-BFGS 迭代上的验证（默认 1），
存在验证集时总会检查初始和最终点。

### 运行时契约

- 所有线程执行相同的邻居 tile 屏障序列，包括不完整的末尾 tile。
- 在使用邻居索引前屏蔽 LAMMPS 特殊键位。
- 启动前校验 GPU 维度、二次幂归约、网格限制、动态共享内存、整数索引、
  原子类型、中心和邻居边界。
- 力输出按设备增量计算并在主机累加，因此保留 `pair_style hybrid` 的贡献，
  而无需上传已有力数组。
- 逐原子 virial 在中心与邻居间均分，与 CPU MLIP 核心一致；返回本地与 ghost 原子贡献。
- 可复用静态 center/offset/neighbor 数组；每次调用仍检查行元数据，并刷新周期位移。
- 公开批请求可禁用力/virial 构造，因此仅能量调用跳过方向导数内核及其输出。
- `gpu_id auto` 使用节点本地 MPI rank 分配，并强制执行一 rank 一 GPU 策略。
- 同一 GPU 核心可通过公开连续批请求使用；系数数组可在不重建矩调度的情况下更新。

## GPU MTP 训练

经典线性 MTP 具备能量、力和应力损失的完整 FP64 GPU 梯度。
在 `trainnlpot` 中启用：

```ini
fit:device cuda
fit:gpu_id -1
```

`gpu_id=-1` 选择 `本地 MPI rank % 可见设备数`；显式非负 id 覆盖该选择。
CPU 仍是默认。主机侧 L-BFGS/CG/SGD、系数/检查点布局、正则化、可训练掩码和
MPI rank 归约保持不变。

每个 rank 确定性地把所选结构划分为保持结构完整的 CSR 块。自动分块尺寸考虑可见空闲显存、
原子、有向邻居、矩维度、标签、梯度分块和安全系数。也可显式限制原子数和结构数。
结构永不拆分；超出预算的单个结构会带索引和内存估计报错。SGD 仅从所选 mini-batch
构造块，不评估零权重结构。

默认把 E/F/S 标签与权重上传到 GPU，并在设备上构造残差、损失伴随、七项损失/误差指标
和梯度。标签上传使用传输流和事件，以便与常驻前向内核重叠。第二个事件保护常驻标签/中心
缓冲，直到全部反向消费者完成。梯度与指标累加器跨块保留在设备上，只把最终指标向量和
原生顺序梯度（`radial_coeffs`、`species_coeffs`、`moment_coeffs`）复制回主机。
`fit:gpu_device_loss false` 保留主机残差路径作为诊断回退。

对固定全数据集优化器回调，`fit:gpu_cache_batches true` 在首次回调后缓存打包的主机 CSR
批次与标签；后续回调复用并只更新系数。随机或部分 SGD 选择绕过该缓存。
设备分配在兼容块之间保留容量，两个输入槽允许构造/上传下一块而不使当前槽失效。
缓存减少重复主机构造，代价是与打包数据集成比例的主机内存。

力/应力伴随为每个有向邻居位移定义一个方向算子，四个矩数组足够：

- `m`：前向矩；
- `q = Dm`：方向矩；
- `a = dU/dm`：能量伴随；
- `h = Da`：方向伴随。

对乘积规则 `m_k += c m_i m_j`，`q` 前向传播，`h` 反向经过同一依赖层级。
线性梯度为 `lambda_E*m + q`；径向梯度组合 `(lambda_E*a+h)*db/dc`
与对局部混合项 `a*D(db/dc)`。这避免了完整的坐标×参数 Jacobian。

### 内核顺序

1. 复用推理的基本/复合矩和能量伴随。
2. 由力与应力损失权重构造每个基本矩的方向导数，再经复合矩 DAG 传播。
3. 在设备上构造 E/F/S 残差与伴随，并累加损失/误差指标。
4. 并行归约线性梯度。生产径向反向中，每个线程负责一个物种局域系数，
   只访问该径向函数下的基本矩，并在寄存器中累加；写入紧凑的逐中心行，
   再由第二个内核按中心物种映射并归约。这避免了全局原子×参数、
   共享 FP64 原子操作以及完整四维 Jacobian。
5. 跨所有块保留指标与梯度，每个 rank 复制一次最终结果，并在主机优化器步之前
   使用训练器现有的 MPI reduce/broadcast 序列。
6. 下一迭代调用 `MTPGpuDirect::update_coefficients()`；不要重建拓扑或静态矩调度。

有序诊断路径保留原始单线程径向累加，使浮点顺序稳定；快速路径使用并行归约。
`fit:gpu_deterministic_reduction true` 会排序前向矩传播、力/virial 累加、
设备损失指标、方向传播、径向前向分块和最终参数归约，用于逐字节重复性诊断，
但明显慢于默认路径。

### 训练设置

```ini
fit:device cuda
fit:gpu_id -1
fit:gpu_batch_atoms 0
fit:gpu_batch_structures 0
fit:gpu_memory_fraction 0.75
fit:gpu_device_loss true
fit:gpu_deterministic_reduction false
fit:gpu_radial_parameter_tile 128
fit:gpu_cache_batches true
```

零批限制表示自动定尺寸；正限制在内存预算之外额外生效。
径向 tile 设置是部分行块中物种局域系数线程数的上限；
启动时会按模型和设备限制裁剪，并向上取整到 warp 大小块。

### 验证与当前限制

GPU 单元测试对每个系数的能量/力/应力联合梯度与中心差分进行比较，
还覆盖设备损失、主机/设备损失等价性、跨两块保留梯度、有序归约、系数更新和推理导数。
未发现匹配的可见 GPU 设备时，运行测试会跳过。
