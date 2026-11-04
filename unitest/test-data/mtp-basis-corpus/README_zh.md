# 真实 MTP 基组回归语料

这 8 份无系数模板来自用户提供的 `MTM-correct` 集合，包含拓扑和径向定义，不含原子
数据集、训练参数或生成日志。`sources.json` 记录全部 27 份源文件的相对路径、原始
SHA-256 和结构统计；仓库内样本与源模板逐字节一致。局部 `.gitattributes` 保留原始
空 `potential_tag` 行的尾随空格；来源校验仍逐字节检查所有样本。

| 样本 | 标量数 | 乘积规则数 | 依赖深度 | 覆盖目的 |
|---|---:|---:|---:|---|
| `shapeev-02.mtp` | 1 | 0 | 0 | 无乘积规则 |
| `shapeev-06.mtp` | 5 | 5 | 2 | 小图与不连续标量索引 |
| `shapeev-16.mtp` | 92 | 741 | 3 | 六阶角张量与交错累加 |
| `shapeev-20.mtp`、`shapeev-20-t.mtp` | 各 288 | 4665 / 4526 | 3 | 相同标量空间、不同收缩图与顺序 |
| `ps1-7b-517.mtp` | 517 | 842 | 5 | 深层依赖 |
| `ps1-5b-5323.mtp` | 5323 | 16715 | 4 | 大规模标量与系数置换 |
| `ps2-7b-4464.mtp` | 4464 | 12172 | 5 | 深层依赖与更高角阶 |

`ut_mtp_basis_corpus.test` 初始化确定性的双元素普通 MTP、两层 NN 的 MTP-M、MTP+ZBL。
通过唯一系数标签检查每个标量列及全部潜变量通道，并比较非对称、周期性和应变构型
的能量、力、应力。覆盖重复及交替执行 `level`/`level-group`、排序后的层级缓存、
元素克隆与置换、GPM/传统格式重载。所选构型上 ZBL 有非零贡献。初始化后删除输入
模板。不同收缩实现之间使用相同径向参数和均匀标量权重，独立于标量编号比较逐原子
标量值集合及能量、力、应力。

在仓库根目录构建和运行（按环境设置 `MKLROOT`）：

```sh
make build/release/unitest/ut_mtp_basis_corpus.test
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
LD_LIBRARY_PATH="$PWD/build/release:$LD_LIBRARY_PATH" \
build/release/unitest/ut_mtp_basis_corpus.test
```

完整外部集合的扩展验证：

```sh
python3 scripts/validation/audit_mtm_basis_corpus.py \
  /home/atao/workspace/MTM-correct --output build/mtm-correct-audit.json

MTP_BASIS_CORPUS_ROOT=/home/atao/workspace/MTM-correct \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
LD_LIBRARY_PATH="$PWD/build/release:$LD_LIBRARY_PATH" \
build/release/unitest/ut_mtp_basis_corpus.test
```

外部验证使用全部 27 份模板，包括 `shapeev/30` 两种各约 19 万规则的收缩图，并额外
比较它们的标量值与能量、力、应力。显式指定不存在或空目录时明确失败。常规
`make test` 只使用仓库内样本，无须外部路径。它们用于数值回归，不是已拟合的材料势。

## 可选 MPI/CUDA 检查

指定对应版本、最新构建的 CLI 路径。MPI 每个用例使用两个进程。CUDA 用例同时比较
重排后模型的 CPU 预测与 GPU 预测，并验证输入 GPM 逐字节未改动。指定外部语料时
额外运行 `shapeev/30` 最大规模用例。

```sh
MTP_BASIS_CORPUS_ROOT=/home/atao/workspace/MTM-correct \
MTP_WORKFLOW_MPI_imrmlpCUTABLE="$PWD/build/mpi/imrmlp-mpi" \
MTP_WORKFLOW_GPU_imrmlpCUTABLE="$PWD/build/cuda/imrmlp-cu" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
LD_LIBRARY_PATH="$PWD/build/release:$LD_LIBRARY_PATH" \
pytest -q unitest/python/test_mtp_basis_corpus_cli.py
```
