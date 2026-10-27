# HVP 监督训练与数据格式 {#doc_docs_hvp_training}

[English](hvp-training.md)

训练可使用少量坐标 Hessian–向量积标签，无需为每个结构存储完整的 `3N × 3N` Hessian。
这里 `H = ∂²E/∂R²`，不是损失对模型参数的 Hessian，也不是质量加权动力学矩阵。

## 支持路径

| 入口 | HVP 处理 |
| --- | --- |
| `imrmlp trainmtp --linear-fit` | 解析计算每个线性基函数的 `H_j v`，支持普通及 `lfit_fastalgo` 线性拟合 |
| `imrmlp trainmtp --do-lbfgs` / SGD / CG | 通过中心力差分计算 HVP，复用力对参数的梯度；支持固定径向或线性参数 |
| `imrmlp trainnlpot --do-lbfgs` / SGD / CG | 同样使用中心力差分；适用于已有正确力及力参数梯度的模型，包括 LES |

`trainnlpot` 的 HVP 训练要求 `fit:device cpu`。模型提供受支持的仿射参数块时，
条件线性精修也可包含相同 HVP 观测，见[线性精修](linear-refit_zh.md)。
`trainmtp` 不允许同时使用 `fit:lfit_eng_only true`，其 E/F/S 分块矩阵导出也不支持 HVP。
这些组合会报错，不会忽略启用的 HVP 项。
通用 `train_linear` / Python `LinearTrainer` 支持 CPU 固定晶胞的 HVP-only 与 E/F/S 混合拟合，
通过对力设计矩阵作中心差分装配 HVP 行；适用于实现 `supports_linear_fit()` 的模型，
包括 `descriptor_linear`。普通 MTP 仍使用 `trainmtp --linear-fit`。
通用线性训练的 `force_normalization` 为 1，不额外乘 `force_weight` 或 `atomic_fitness`。
只用 HVP 无法识别常数能量偏置；需要能量标签时应同时开启能量权重。
Python `NonlinearTrainer` 也允许仅 HVP 的正权重，仍须使用 CPU 并遵守所选模型的限制。

## 配置与权重

在现有训练配置中加入：

```text
fit:hvp_weight   0.01
fit:hvp_pow      0
fit:hvp_fd_step  0.0001
```

`hvp_weight` 默认是 `0`（关闭）。`hvp_pow` 默认是 `0`。
`hvp_fd_step` 默认 `1e-4 Å`，必须为有限正数。
示例权重仅用于演示，应结合 E/F/S 和 HVP 的误差量级调整。

单个结构的损失为

```text
L_hvp = hvp_weight * fitness * force_normalization / (N^hvp_pow * m)
        * sum_k || predicted_Hv[k] - target_Hv[k] ||²
```

这里 `m` 是该结构的方向数。沿用训练器已有的构型力归一化因子；
不额外使用 `atomic_fitness`，也不隐含除以 `3N`。
日志中 `HVP monitor` 单独报告损失和逐分量 RMS，验证集也报告 HVP RMS。
普通 E/F/S 标签和 HVP 标签可以混合，缺少 HVP 的结构不贡献此项。

MTP 专用线性拟合使用解析 `Hv`。通用线性拟合对力设计矩阵使用同一差分公式，
非线性损失和反传则对力使用

```text
Hv ≈ -(F(R + epsilon*v) - F(R - epsilon*v)) / (2*epsilon)
epsilon = hvp_fd_step / max_i ||v_i||
```

因此最大单原子位移为 `hvp_fd_step`，方向和标签本身不会被重新归一化。
每个方向需要两个扰动构型。求梯度时，为刷新带状态模型的缓存，还会重算其中一个构型的力，
并执行两次力梯度回传。该实现不构造完整 Hessian 或完整参数 Jacobian。
非线性差分的截断误差随位移减小而下降，但过小位移会放大数值噪声；应检查步长收敛。
线性解析拟合后的训练误差也通过该力差分路径报告，可能留下相应的差分误差。


Python 通用线性入口示例（训练结构应已包含 HVP 标签）：

```python
trainer = imrmlp.LinearTrainer(energy_weight=0, force_weight=0, stress_weight=0,
                             hvp_weight=1.0, hvp_pow=0, hvp_fd_step=1e-4)
result = trainer.train(potential, structures)
print(result.train_hvp_rms, result.train_hvp_loss)
```

`assemble()` / `solve()` 同样支持 HVP。训练和验证监控使用相同的差分步长；
MPI 允许空训练分片和仅部分 rank 有验证数据，并集体传播装配/监控错误。

## 数据约定

C++ `Atoms` 保存两个 `Eigen::MatrixXd`：

- `hvp_directions_`：形状 `(3N, m)`，每列为无量纲方向，行顺序为 `atom0.x,y,z, atom1.x,y,z, ...`。
- `hvp_targets_`：相同形状，每列为该方向的 `Hv`，单位 `eV/Å²`。

IO 支持 `extxyz`、文本 `cfg`、二进制 `bin`，以及 GPIO 原有的压缩文件读写。
文本将每列依次展平，即先方向编号，再原子编号，再 xyz；不存在隐含的质量加权。
两个标签必须成对出现，长度必须是 `3N` 的正整数倍，方向数必须相同。
非有限数值、零方向、重复标签、坏长度和不完整数字会被拒绝。零响应允许存在。
输出到不支持 HVP 的格式会报错；确实只需几何时先调用 `clear_hvp()`。

例如两个原子、一个方向：

```text
2
Lattice="8 0 0 0 8 0 0 0 8" hvp_directions="1 0 0 -1 0 0" hvp_targets="2 0 0 -2 0 0" Properties=species:S:1:pos:R:3
H 0 0 0
H 1.2 0 0
```

CFG 使用同样的展平顺序，在 `END_CFG` 前写入：

```text
Feature hvp_directions 1 0 0 -1 0 0
Feature hvp_targets 2 0 0 -2 0 0
```

二进制 CFG 使用保留名称的 Feature 字符串记录，数值以 17 位精度往返；
不改变既有二进制版本。旧读取器可把这些记录作为普通 Feature 读取，但旧训练器不会使用它们。
这两个名称不可同时出现在用户自定义属性中。

## Python 写入与读取

```python
import numpy as np
import imrmlp

atoms = imrmlp.read("structure.xyz")[0]
# 例：一个方向；实际可提供多列。
v = np.zeros((3 * atoms.natoms, 1))
v[0, 0], v[3, 0] = 1.0, -1.0
# 从参考计算获得 F_plus / F_minus，均为 (N, 3)。
epsilon = 1e-3
hv = -(F_plus - F_minus).reshape(-1, 1) / (2 * epsilon)
atoms.set_hvp(v, hv)
imrmlp.write("train-hvp.xyz", [atoms], format="extxyz")

restored = imrmlp.read("train-hvp.xyz")[0]
print(restored.hvp_directions.shape, restored.hvp_targets.shape)
```

参考力必须来自同一个结构的 `R ± epsilon*v`，保持晶胞、原子顺序和计算设置一致。
应移除整体平移方向，避免把方向预算用于零模；方向数少时约束的是采样子空间，不能保证完整 Hessian 都准确。

## 从已有完整 Hessian 生成标签

`attach_hessian` 增加三个选项：`--hvp-probes`、`--hvp-seed`、`--hvp-only`。
生成高斯随机方向、去除整体平移并归一化到单位二范数，然后计算 `H*v`。
种子默认 `42`；不指定 `--hvp-probes` 时维持原有完整 Hessian 输出。

```sh
imrmlp attach_hessian --input structure.xyz --hessian FORCE_CONSTANTS --source fd \
    --output train-hvp.xyz --output-format extxyz \
    --hvp-probes 4 --hvp-seed 42 --hvp-only
```

`--hvp-only` 省去输出中的完整 Hessian；上述命令仍需读取已有完整 Hessian。
若目标是降低参考计算成本，直接按前面的两次力计算方法生成每个方向的标签。

## 验证

`unitest/ut_hvp_training.cpp` 覆盖解析基函数 HVP 与完整 Hessian 的一致性、线性拟合、
非线性与 LES 参数梯度、权重及缓存行为，以及 IO 往返和坏数据拒绝。
`unitest/python/test_hvp_io.py` 验证 Python 标签设置和三个格式的往返。
