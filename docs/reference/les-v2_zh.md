# MTP+LES v2 {#reference_les_v2}

[English](les-v2.md)

可训练的高斯和（SOG）核（v3/v4）见 [MTP + SOG](sog_zh.md)。下文 v2 Coulomb 泛函保持不变。

## 用途与输入

LES 在局域 MTP 能量之外加入依赖环境电荷的长程项，适用于三维全周期体系。
已有 LES 模型可直接求值：

```sh
imrmlp calc_efs --pottype mtp_les --potfile model.mtp \
  --input structure.xyz --input-format extxyz --output predicted.xyz
```

输入必须使用模型的元素顺序；输出包含预测能量、力及 virial。
训练使用 `trainnlpot`，见下方电荷约束设置。潜电荷由模型学习，单靠能量/力拟合
不能把它当作唯一可辨识的物理原子电荷。`les_sigma` 和 `les_dl` 的长度单位为 Å。
本页的 `字段 = 值` 属于势文件语法；设置文件仍使用 `pot:键 值`。

## 能量泛函

LES v2 计算周期高斯倒空间泛函：

\[
E_\mathrm{LES}=\frac{2\pi k_e}{V}\sum_{\mathbf k\ne0}
\frac{e^{-\sigma^2 k^2/2}}{k^2}w(k)|S(\mathbf k)|^2
-\frac{k_e}{\sigma\sqrt{2\pi}}\sum_i q_i^2,
\quad S(\mathbf k)=\sum_i q_i e^{i\mathbf k\cdot\mathbf r_i}.
\]

自作用项由 `remove_self_interaction` 控制。省略的零模具有均匀背景语义，
潜电荷不会投影为零均值。由于 `q_i` 是 MTP 描述符模型，力始终包含显式倒空间
导数与 `(dE/dq)(dq/dr)` 两部分。

## 势字段

```text
les_version = 2
les_sigma = 5.0e-1
les_dl = 2.0
les_cutoff_mode = smooth
les_taper_fraction = 1.0e-1
les_charge_treatment = uniform_background
remove_self_interaction = 1
les_response_parity = 1
les_species_coeffs = { ... }
les_moment_coeffs = { ... }
```

`smooth` 在倒空间球最后 `les_taper_fraction` 范围内施加 C2 五次渐隐，
是训练和变胞模拟的默认值；`reference_hard` 复现旧版/参考硬球以便比较。
截断为 `k_max = 2*pi/les_dl`。

`les_response_parity` 必须为 `+1` 或 `-1`。它固定类电荷观测量本来无法确定的
全局符号，并作用于报告的潜电荷、总电荷目标与梯度、周期偶极以及 Born 有效电荷。
改变它不会改变能量、力或 virial。

## 周期性、广义偶极与 BEC

LES v2 要求有限、右手、三个晶格方向都启用的周期晶胞。非周期与部分周期结构会被拒绝；
尚未实现 slab/wire 长程求和。位置可以位于主晶胞外，构造邻域时内部会折回，
但不修改输入结构。

周期广义偶极是复数。周期 BEC 先对完整复数分数偶极求导，去掉被位移原子的参考相位，
再取实部并变换到笛卡尔轴。张量的行是响应方向，列是位移方向。

BEC 通过局域电荷的解析导数计算，无需逐原子有限差分。
电荷中心化也包含平均电荷的导数；输出不会自动投影到声学求和规则。

固定 Coulomb LES 头时，`trainnlpot --linear-refit-only` 还支持
`fit:linear_refit sr`：只重拟合短程线性系数，保留全部 LES 电荷与响应。
`sr_sog` 仍然只用于 SOG。

这是**有限波矢响应诊断**，不是精确的 Berry 相位极化导数。环境相关电荷下，
结果通常依赖原胞倒格矢选择和超胞尺寸；电荷中心化不会强制声学求和规则。
具体而言，原子相关的退相位不保证存在可积的极化 Jacobian，
因此这些张量乘以均匀电场不保证给出保守力。固定 BEC 场计算请使用独立标定的
参考张量，参见 [LAMMPS 响应示例](../../examples/les-response/README_zh.md)。

LES 系数要求显式的受支持格式版本。无版本 LES 文件和过时 LES 字段会被拒绝。

## LAMMPS

[LAMMPS LES/SOG 适配器](../../LAMMPS/USER-MLIP/opt-les/README.md)支持受限三斜晶胞、
MPI 归约结构因子、全局与逐原子能量/virial、完整描述符反作用力以及潜电荷输出。
v2 不包含 GPU LES。

## 电荷约束训练

`trainnlpot` 可以惩罚每个构型解析预测的总潜电荷。默认目标为中性；
单个构型可用 `GlobalDouble` 或 `Feature total_charge VALUE` 覆盖。

```text
fit:charge_weight 1.0
fit:charge_pow 1
fit:charge_target_attrib total_charge
fit:charge_default_target 0
fit:training_schedule mtp_les_from_scratch
fit:sr_stage_iterations 100
fit:les_stage_iterations 50
fit:joint_stage_iterations 200
fit:charge_warmup_factor 10
fit:les_zero_init_range 1e-3
fit:les_zero_init_seed 5489
fit:parameter_scaling group_auto
fit:scale_floor 1e-3
fit:scale_ceiling 0.1
fit:scaled_step_max 1
```

损失为 `charge_weight*(Q-Q_target)^2/N^charge_pow`。它对 LES 物种系数、
LES 矩系数和 MTP 径向系数的梯度均为解析计算；有限差分只用于梯度验证测试。
从已拟合的短程模型开始时，可用 `mtp_les_resume` 跳过仅短程阶段。
仅靠总电荷中性并不能使单个潜电荷具有物理可辨识性；仍需偶极、Born 有效电荷
或原子电荷标签。

对 `mtp_les_from_scratch`，全零 LES 块会按 `les_zero_init_range` 和
`les_zero_init_seed` 确定性扰动，再将其物种偏置拟合到每个构型的总电荷目标。
已有非零系数和 resume 训练保持不变。

## Hessian 支持

MTP+LES 和兼容的 MTP-M+LES/SOG 模型支持解析笛卡尔 Hessian 与 Hessian–向量积
（HVP），包含电荷/潜变量随结构变化的完整响应。晶胞 Hessian 及晶胞–位置混合
导数要求 `les_cutoff_mode smooth`；硬倒空间截断不支持解析晶胞导数。
旧参数 `pot:les_hessian_fd_nudge` 仍可读取，但不控制上述解析路径。

`lr_les=true` 的 MTP-M 在力、virial 和参数梯度中包含完整描述符电荷响应，
LES 系数冻结时仍包含径向梯度。GPU 求值/训练目前支持 SOG，Coulomb LES v2
使用 CPU；具体后端范围见 [GPU 指南](../development/cuda_zh.md)。
