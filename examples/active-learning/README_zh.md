# 自包含的 MaxVol 选样练习

[English](README.md) · [主动学习指南](../../docs/guides/active-learning_zh.md)

本例使用仓库提供的小型合成 MTP，自动生成双原子结构。已有覆盖的原子间距为
3.8、3.9、4.0 Å，探索池的间距为 3.9、3.5、3.0、2.5 Å。通过这个小问题，可以
观察主动集如何表示已有覆盖，以及哪些新结构会通过选样阈值。
`Al` 只是元素映射标签，这个模型不描述真实铝材料。

## 运行

编译好 CPU CLI 后，在仓库根目录运行：

```sh
export LD_LIBRARY_PATH="$PWD/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 examples/active-learning/run.py
```

只需要 Python 标准库和 CLI，不需要 Python 扩展、ASE、LAMMPS 或参考数据集。
默认输出到 `build/active-learning/`，非空目录不会被覆盖。再次运行可指定
`--work build/active-learning-2`；也可用 `--imrmlp /absolute/path/to/imrmlp`
指定其他可执行程序，并配置与其匹配的运行库路径。

## 查看结果

脚本会打印以下近似分数，并写入 `grades.tsv`：

| 探索池帧号（从 0 开始） | 间距（Å） | Grade | grade > 2.1 时保存？ |
| --- | --- | --- | --- |
| 0 | 3.9 | 0.854 | 否 |
| 1 | 3.5 | 1.488 | 否 |
| 2 | 3.0 | 2.425 | 是 |
| 3 | 2.5 | 3.445 | 是 |

在 `select.log` 中确认 `converged` 和 `virtual_rows=0`，在 `sample.log` 中确认
`Saved 2 / 4`。这个刻意简化的模型只有一个活动 builder 列，选中结构数量不能
推广为其他 MTP 的固定数量。

| 文件 | 含义 |
| --- | --- |
| `coverage.cfg`、`pool.cfg` | 仅含几何的已有覆盖与探索输入 |
| `model.mtp`、`index.map`、`select.ini` | 模型、元素顺序与评分配置 |
| `model.mtp.sel`、`selected.xyz` | 主动集状态及其选中行所对应的结构 |
| `graded-pool.xyz`、`grades.tsv` | 全部四帧的评分与便于阅读的汇总 |
| `candidates.xyz` | 两个待考虑参考标注的构型，**尚无参考 E/F/应力标签** |

## 自己重复一步

在生成目录中，确保 `imrmlp` 位于 `PATH`，核心命令为：

```sh
imrmlp maxvol_select select.ini --pool coverage.cfg --index index.map \
  --output selected.xyz --score-file coverage-grades.txt
imrmlp maxvol_sample select.ini --input pool.cfg --index index.map \
  --state-file model.mtp.sel --output candidates.xyz \
  --threshold-save 2.1 --threshold-break 1000000000
```

试着改为 `--threshold-save 1.2 --output candidates-1.2.xyz`，应保存四帧中的三帧。
这改变的是待标注数量，不是模型精度。很高的 break 阈值用于收集这个有限的离线池，
不能直接作为在线 MD 的安全设置。只改变输出阈值时，保持模型和 `.sel` 的对应关系。

实际工作中，还要检查候选几何，用与旧数据一致的参考方法标注，再重训、验证并
建立匹配的新状态，步骤见[指南](../../docs/guides/active-learning_zh.md)。本练习只完成
选样阶段。grade 依赖模型、builder 和主动集，既不是原子间距阈值，也不是物理误差。
