# MTP + 可训练 SOG（v3/v4） {#reference_sog}

[English](sog.md)

SOG 支持周期求值、非线性能量/力/virial 训练、MTP-M 集成以及 LAMMPS CPU/MPI 部署。
独立多通道 SOG 也已实现，参见[多通道指南](../advanced/sog-multichannel_zh.md)和
[训练示例](../../example-inputs/fit-sog-multichannel.ini)。使用现有 `mtp_les`
势类型并设置 `pot:les_kernel sog`。普通 Coulomb LES 模型仍为 v2，泛函不变。

## 泛函

对无量纲潜变量 `q_i`，Fourier 乘子为

\[
G(k^2)=\sum_m w_m\exp[-k^2/\exp(2u_m)],\qquad
E=\frac{1}{2V}\sum_k T(k^2)G(k^2)|S(k)|^2
-\frac12 C\sum_i q_i^2,
\]

其中 `S(k)=sum_i q_i exp(ik.r_i)`，`w_m` 单位为 eV Å³，
`exp(u_m)` 是 Å⁻¹ 单位的倒空间带宽。在位系数为
`C=sum_m w_m exp(3u_m)/(8*pi^(3/2))`。权重可正可负。
能量及全部导数都包含 q 对 MTP 描述符的依赖。

默认包含 k=0，只减去解析在位自能。保存的模型可选择
`sog_zero_mode = include | omit` 与 `sog_self_mode = onsite | none`。
倒空间 C2 渐隐不修改解析在位系数。零频贡献能量、virial 和描述符响应，
但其固定 q 的位置力为零。

仅支持完全周期、有限右手的晶胞。直接倒空间求和支持三斜晶胞。
本实现不使用 NUFFT。

## 训练

参见 [fit-sog.ini](../../example-inputs/fit-sog.ini)。需要提供兼容的 MTP 模板
或已拟合的短程模型、物种映射和训练数据：

```sh
imrmlp trainnlpot --settings fit-sog.ini --do-lbfgs --output fitted-sog.mtp
```

`pot:les_kernel sog` 使用 `pot:sog_terms` 个高斯（默认 8）。初始权重为 `1/M`；
对数带宽在 `pot:sog_log_bandwidth_min` 与 `pot:sog_log_bandwidth_max`
（默认 -4 与 1）之间严格均匀分布。边界必须位于 [-30,30]，M 位于 [1,256]。
这些只是起始设置，不保证精度；应结合目标晶胞和截断收敛性选择带宽范围与 `pot:les_dl`。

`fit:training_schedule mtp_sog_from_scratch` 依次执行 SR 拟合、固定带宽预热和联合训练；
`mtp_sog_resume` 跳过 SR 阶段。两者沿用 `sr_stage_iterations`、
`les_stage_iterations` 与 `joint_stage_iterations`。完全为零的潜变量头用
`fit:sog_zero_init_range`（默认 0.1）和 `fit:les_zero_init_seed`（默认 5489）
确定性初始化；非零头保持不变。显式 `fit:random_init` 会把核重置为确定性初始网格。

参数组为 `sr_radial`、`sr_linear`、`les_species`、`les_moment`、
`sog_weights` 与 `sog_log_bandwidths`。单阶段调度也支持显式
`fit:trainable_groups`；自动 SOG 调度会自行选择参数组。
仅能量以及能量/力/virial 联合梯度均为解析计算，包括描述符响应对核参数的混合导数。

优化器为每个带宽使用无约束变量 z：
`u = (u_min+u_max)/2 + (u_max-u_min)*atan(z)/pi`，训练中包含其链式导数。
模型文件保存物理 u 而非 z。无法严格表示在边界内的系数会被拒绝，而不是静默截断。
改变边界会改变优化器坐标；resume 时应保持不变。

现有 `fit:penalty_weight` 除 SR 惩罚外，还会对潜变量头和核权重施加 L2，
有助于控制缩放自由度 `q -> c*q, w -> w/c²`。带宽有界但不施加 L2 惩罚。
应根据损失量级选择参数缩放与容差；初始长程能量很小时，可能在核学到东西之前
就满足停止容差。选择正则化时应监控验证误差。

## 序列化与观测量

单通道 SOG 文件使用 `les_version = 3`、`les_kernel = sog`、`sog_channels = 1`、
`sog_terms`、两个数组、带宽边界、零模/自能模式以及现有 LES 潜变量头块。
`sog_terms` 必须位于其数组之前。不完整数组、非有限系数、重复 SOG 字段、
非法模式和非法带宽都会被拒绝。往返读写保留完整可训练状态。
旧 LES 文件仍以 v2 加载和保存。
多通道文件使用 v4，`sog_channels` 位于 [2,16]，系数数组以通道为主序。
`sog_terms` 是每通道的高斯数量。

通用 SOG 潜变量没有固定物理电荷尺度。SOG 预测会清空 `Atoms::charges_`；
内部 q 可通过 `PotMTP_LES::LES_all_qs` 获取。多通道时
`LES_all_latents` 包含完整 N×C 矩阵；预测后 `LES_all_qs` 保留第一通道。
SOG 不支持总电荷训练、电荷平衡和 BEC，应将 `fit:charge_weight` 设为 0。

外部、物理标定的 BEC 模型可作为 SOG 的零场势组合使用。
[LAMMPS 示例](../../examples/les-response/README_zh.md)使用固定晶胞、固定 BEC 的
DFT 参考张量；它不从 SOG 通道推导电荷，属于局域线性响应近似。

`mtp_les` 与兼容的 MTP-M+SOG 支持解析笛卡尔 Hessian 和 HVP；
解析晶胞 Hessian 与晶胞–位置混合导数要求平滑倒空间截断。
详见 [LES 导数支持](les-v2_zh.md)。

## GPU 求值与训练

兼容的 MTP+SOG 和 MTP-M+SOG v3/v4 模型可通过 CUDA 或 HIP 后端求值及训练。
求值使用 `calc_efs --device cuda`（或 `hip`），训练在设置文件中指定
`fit:device cuda`（或 `hip`）。使用对应的构建模式；模型/径向基限制和多 GPU
配置见 [GPU 指南](../development/cuda_zh.md)。下述独立 `mlip/opt-les` 是 CPU/MPI
接口，不因模型格式支持 GPU 就自动转为 GPU 计算。

## MTP-M 与 LAMMPS 部署

`PotMTP_M<T>` 在自身短程和潜变量头系数块之外读写 SOG v3/v4 元数据。
模板必须具有匹配的 MTP-M 头布局；线性 MTP 文件不是 MTP-M 文件。
`set_sog_definition(SOGModel)` 在加载兼容模板前设置核定义。文件中的元数据会自动启用 SOG。
共享的 `potentials/sog_model.h` 提供可移植元数据校验和独立高斯公式。

MTP-M 使用普通单阶段非线性训练器。参数组为 `sr_radial`、`sr_species`、
`sr_moment`、`sr_model`、`les_species`、`les_moment`、`sog_weights` 与
`sog_log_bandwidths`。`pot:req_les false` 冻结潜变量头；
`pot:req_sog false` 冻结核权重/带宽。关闭全部 SR/头请求标志时支持仅拟合核。
有界带宽映射与完整力/virial 响应梯度与 MTP 一致。上述自动 `mtp_sog_*`
调度仍专用于 `mtp_les`；MTP-M 随机初始化会保留模板中提供的核。

[LAMMPS CPU 适配器](../../LAMMPS/USER-MLIP/opt-les/README.md)接受线性 MTP 短程模型，
搭配 Coulomb LES v2 或 SOG v3/v4。它支持 SOG 零模/自能模式、三斜晶胞、
逐原子能量与 virial 以及 MPI 区域分解。无量纲 q 使用
`compute ID GROUP mlip/sog/latent`；物理电荷 compute `mlip/les/charge` 拒绝 SOG。
该 standalone pair style 不支持非线性 MTP-M 短程头，会被拒绝。

倒空间求值使用网格/核缓存、谐波递推和批量 MPI 归约。有限 SOG 在任意大距离上
不强制严格 Coulomb 尾部。

## 验证

`ut_sog`、`ut_sog_training`、`ut_sog_channels` 与 `ut_sog_mtp_m` 测试覆盖
独立参考求和、坐标与参数导数、序列化、冻结参数组和分阶段拟合。
MPI 测试命令见[多通道指南](../advanced/sog-multichannel_zh.md)，
部署回归见 [LAMMPS 适配器](../../LAMMPS/USER-MLIP/opt-les/README.md)。
