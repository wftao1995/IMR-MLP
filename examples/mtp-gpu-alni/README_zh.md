# Al-Ni MTP GPU 拟合示例

[English](README.md)

本示例使用 L1₂-Ni3Al 结构演示 GPU 能量、力、应力计算和合成标签拟合。
需要自行提供元素顺序与 `index.map` 一致的 Al/Ni MTP 模型。

## 文件

- `POSCAR.L12-Ni3Al`：参考结构。
- `POSCAR.L12-Ni3Al-displaced`：带位移的结构，用于生成参考标签。
- `index.map`：模型元素顺序，Al、Ni。
- `train/fit-gpu.ini`：CUDA 拟合设置。
- `train/make_synthetic_training_set.py`：将参考 E/F/S 标签乘以 1.02，构造非零拟合损失。
- `train/submit-gpu.slurm`：在独立运行目录中完成标注与拟合。

## 运行

在仓库根目录构建 CUDA 程序：

```bash
make -j8 BUILD_MODE=cuda CUDA_HOME=/path/to/cuda CUDA_ARCH=sm_80 imrmlp
```

按实际 GPU 选择 `CUDA_ARCH`，使用 `sm_80` 这样的 nvcc 架构名；
程序生成在 `build/cuda/imrmlp-cu`。
导出程序和模型路径，进入示例目录，并按集群配置调整 Slurm 脚本中的分区、节点和 GPU 资源：

```bash
export GPMAIN="$PWD/build/cuda/imrmlp-cu"
export MTP_POTENTIAL=/absolute/path/to/alni.mtp
cd examples/mtp-gpu-alni
```

提交任务：

```bash
sbatch train/submit-gpu.slurm
```

脚本先用 `calc_efs --device=cuda` 计算参考标签，再构造合成训练集，最后运行
`trainnlpot --do-lbfgs`。每次运行写入 `MTP_RUN_ROOT/train-JOB_ID`，默认根目录是
提交目录下的 `mtp-gpu-runs`。`fit-gpu.ini` 只训练三次迭代，应检查 `train.log`
和保存的模型，不能把任务成功退出视为拟合收敛。所附脚本面向 CUDA；HIP 需要
使用 HIP 构建，并相应修改评估命令和训练设置中的设备选项。

合成标签用于验证计算和训练流程。实际模型拟合应使用独立的标注训练集与验证集。
GPU 后端接口和配置见 [GPU 开发说明](../../docs/development/cuda_zh.md)，LAMMPS
部署见 [GPU runtime（英文）](../../runtime/mtp/gpu-direct/README.md)。
