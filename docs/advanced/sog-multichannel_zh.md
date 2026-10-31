# 多通道 SOG：训练与部署

[English](sog-multichannel.md)

多通道 SOG 在共享短程模型和描述符的基础上，加入多个独立长程潜变量通道；每个通道有独立的潜变量头、Gaussian 权重和带宽。单通道表达能力不足时可尝试此模型。增加通道数只提高长程部分的成本，短程开销不变。

\[
q_{ic}=b_{s_i,c}+\sum_a h_{ac}B_a(i),\qquad
E=E_{\mathrm{SR}}+\sum_{c=1}^{C} E_{\mathrm{SOG}}[q_c,G_c],
\quad G_c(k^2)=\sum_{m=1}^{M}w_{cm}\exp[-k^2e^{-2u_{cm}}].
\]

每个通道沿用[单通道的能量、自能与零模约定](../reference/sog_zh.md)。通道之间没有交叉核项；共享描述符参数的梯度累加所有通道的贡献。支持完整的能量、力、virial 及训练所需的混合参数导数。

## 配置与训练

在原有 `mtp_les` 配置中设置：

```text
pot:les_kernel sog
pot:sog_channels 3
pot:sog_terms 8
fit:charge_weight 0
```

`sog_channels` 默认为 1，范围为 1–16；`sog_terms` 是**每通道**的 Gaussian 数量，范围为 1–256。所有通道共用带宽上下界、倒空间截断、taper、零模和自能模式。潜变量没有物理电荷单位，不能用于总电荷损失或 BEC。

完整示例见 [fit-sog-multichannel.ini](../../example-inputs/fit-sog-multichannel.ini)。先复制到运行目录，替换数据与模板路径后运行：

```sh
imrmlp trainnlpot --settings fit-sog-multichannel.ini --do-lbfgs \
  --output fitted-sog-multichannel.mtp
```

`mtp_sog_from_scratch` 与 `mtp_sog_resume` 均支持多通道。零初始化检查逐通道进行：保留已有非零头，为整个头为零的通道确定性随机初始化，避免该通道停在零梯度状态。新增通道初始 Gaussian 网格相同，潜变量头的随机初始化使各通道能够学习不同响应。

训练参数组仍为 `les_species`、`les_moment`、`sog_weights` 和 `sog_log_bandwidths`，每组包含所有通道；当前没有逐通道冻结配置。L2 正则覆盖所有通道的头与核权重，带宽继续使用有界映射。通道数不是连续优化变量；恢复训练时应保持模型布局和带宽上下界一致。

MTP-M 同样支持多通道以及冻结头的核参数训练。其通道数来自匹配的 MTP-M 模板元数据。模型分配后不允许直接改变通道布局。MTP-M 使用普通单阶段训练，不使用 `mtp_sog_*` 自动阶段。

CUDA/HIP 求值与训练见 [GPU 指南](../development/cuda_zh.md)及 [SOG 参考](../reference/sog_zh.md)。下述独立 LAMMPS 适配器使用 CPU/MPI。

## 模型格式与潜变量输出

单通道继续读写 SOG v3，Coulomb LES 继续使用 v2。多通道写入 `les_version = 4`；旧版本程序会拒绝 v4。每个系数块内部按通道排列，设物种数为 S、标量描述符数为 A：

| 字段 | 长度 | 顺序 |
| --- | ---: | --- |
| `les_species_coeffs` | C×S | 通道 0 的全部物种，再通道 1，依次排列 |
| `les_moment_coeffs` | C×A | 通道 0 的全部描述符，再通道 1，依次排列 |
| `sog_weights` | C×M | 通道 0 的全部 Gaussian，再通道 1，依次排列 |
| `sog_log_bandwidths` | C×M | 同上，文件保存物理 log 带宽 u |

线性 MTP 文件中，`les_kernel = sog` 和 `sog_channels` 必须先于多通道系数块，`sog_terms` 必须先于 Gaussian 数组。使用模型的 Save 接口生成文件即可满足顺序约束。MTP-M 有自己的短程模型格式，不能作为线性 MTP 模型部署。

预测后的 `PotMTP_LES::LES_all_latents`、`PotMTP_M::LES_all_latents` 是 N×C 矩阵；原有 `LES_all_qs` 保留第一通道。`Atoms::charges_` 在 SOG 预测时清空。


## LAMMPS 输出

CPU `mlip/opt-les` 支持线性 MTP + SOG v4。三通道模型示例：

```text
pair_style mlip/opt-les
pair_coeff * * fitted-sog-multichannel.mtp 0 1
compute q all mlip/sog/latent
dump latent all custom 10 trajectory.lammpstrj id type x y z c_q[1] c_q[2] c_q[3]
```

映射按 LAMMPS 原子类型逐个给出；模型带元素元数据时可直接写元素名，旧文件用上例的
从零开始的索引。单通道输出仍为向量 `c_q`；多通道输出为逐原子数组 `c_q[1]` 到 `c_q[C]`。更换通道数不同的势时须重新创建 compute 与引用它的 dump。物理电荷 compute `mlip/les/charge` 拒绝 SOG。

## 计算成本与验证范围

CPU 倒空间通道求和目前逐通道执行，长程部分的时间随通道数近似线性增长。倒空间网格与谐波递推机制继续复用；LAMMPS 缓存所有通道的核值，但各通道分别计算结构因子和进行 MPI 归约。IMRMLP 的核值缓存按当前通道更新。因此选择通道数，就是在长程分辨率与这部分开销之间取舍。

`unitest/ut_sog_channels.cpp` 覆盖独立通道求和、参数与坐标导数、v4 读写、
非法布局、冻结参数及阶段训练。MPI 测试使用专用初始化入口：

```sh
make -j8 BUILD_MODE=mpi build/mpi/unitest/ut_sog_channels.test
OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 \
  LD_LIBRARY_PATH=$PWD/build/mpi:$LD_LIBRARY_PATH \
  mpirun -np 2 build/mpi/unitest/ut_sog_channels.test --rng-seed 42
```

LAMMPS 测试覆盖多通道输出、零模/自能组合、空 MPI 分区及非法元数据，
执行方法见 [LES/SOG 适配器](../../LAMMPS/USER-MLIP/opt-les/README.md)。
