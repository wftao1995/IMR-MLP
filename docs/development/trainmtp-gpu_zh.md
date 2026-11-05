# trainmtp GPU 训练 {#development_trainmtp_gpu}

[English](trainmtp-gpu.md)

FP64 GPU 训练支持普通 `PotMTP`，要求 `radial_mode=0`。
CPU 是默认后端；显式请求不匹配的后端会报错。

```sh
make BUILD_MODE=cuda -j4 imrmlp
# 先按 quickstart 生成公开合成数据与初始模型。
OMP_NUM_THREADS=1 python3 examples/quickstart/run.py
cp examples/quickstart/gpu.ini build/quickstart/gpu.ini
cd build/quickstart
LD_LIBRARY_PATH=../cuda:$LD_LIBRARY_PATH ../cuda/imrmlp-cu trainmtp gpu.ini \
  --do-lbfgs --output _Trained_Potential_.mtp
```

也可选择 `--do-cg`、`--do-sgd` 或 `--linear-fit`；线性拟合可加 `--do-rescale`。
`--calc-grad`、`--check-grad`、`--check-grad-all` 使用相同 GPU 非线性回调。
检查点由 `fit:save_to` 指定；显式传 `--output _Trained_Potential_.mtp`
可保留原生最终模型（默认是 GPM `_Trained_Potential_.gpm`）。
本例标签来自合成势，适合验证接口，不能代表真实材料精度。

| 设置 | 默认值 | 含义 |
| --- | --- | --- |
| `fit:device` | `cpu` | `cpu`、`cuda`、`hip` 或 `dcu`（HIP 别名） |
| `fit:gpu_id` | `-1` | 自动选择节点内设备；显式编号允许有意共享 |
| `fit:gpu_batch_atoms` / `fit:gpu_batch_structures` | `4096` / `32` | 正整数，完整构型批次上限 |
| `fit:gpu_memory_mib` / `fit:gpu_workspace_mib` | `256` / `64` | 线性阶段为非观测/观测预算，非线性 owner 使用二者之和 |
| `fit:gpu_curvature_host_mib` | `128` | CUDA 线性曲率的主机所有权预留上限 |
| `fit:gpu_device_loss` | `true` | 设备残差与损失；false 使用主机残差，预测和梯度仍在 GPU |
| `fit:gpu_deterministic_reduction` | `false` | 有序归约诊断，不保证跨设备逐位一致 |

MiB 为 2²⁰ 字节。预算约束受跟踪设备分配，排除驱动/上下文；不代表进程 RSS 或整卡显存上限。
非线性分配超限时缩小完整构型批次，单构型仍超限则失败。SGD 使用实际打乱后的选择索引。

能量、力、九分量原始 virial 的权重、缺失标签、fitness 和全局构型平均沿用专用 CPU 训练器。
GPU 负责观测、预测、数据损失和参数梯度；CPU 保留优化器、正则化、线性求解及 MPI 归约。
内部线性重拟合、正交化、rescale 和 radial/linear/species 冻结继续生效。
切换到线性阶段或改变模型定义/缩放会释放并重建非线性设备状态。

CUDA `--linear-fit` 支持固定晶胞的解析 Hessian/HVP 观测；完整 Hessian 只拟合上三角，
HVP 使用保存的原始方向，线性求解前后使用独立 CPU `H*v` 验收。
非线性 GPU 训练拒绝曲率标签/损失；HIP 线性曲率尚不支持。
派生/LES/SOG/MTP-M、ladder、修正项和 additive EFS 不在本入口 GPU 支持范围内，
可用的更多模型族见 [trainnlpot GPU 支持](cuda_zh.md)。DF、随机搜索、species-only、
矩阵导出和曲率诊断拒绝 GPU 请求。

MPI 使用 `BUILD_MODE=mpi-cuda` 或 `mpi-hip`。完整构型属于单个 rank，支持不均与空本地分片；
所有 rank 必须使用相同数学目标、模型和优化器设置，设备及资源预算可不同。
本地错误在归约前集体传播；非线性失败恢复系数、缩放、有效正则化和 SGD 更新状态。
检查点和最终模型先暂存并重载，再替换目标。成功后输出 rank 0 的 JSON 工作量和显存报告，
其中时间不等于完整命令墙钟，不构成普遍加速承诺。

HIP/DCU 改用 `BUILD_MODE=hip HIP_ARCH=<设备架构>`、`fit:device hip` 与
`build/hip/imrmlp-hip`。本地实机验证使用 CUDA；HIP 需在对应硬件验证。

## 相关

- [MTP GPU 求值与训练（CUDA/HIP）](cuda_zh.md)
- [训练、评估与部署](../guides/workflows_zh.md)
- [配置系统](../reference/configuration_zh.md)
