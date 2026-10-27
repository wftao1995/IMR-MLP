# 鞍点与路径坐标及数值测试

[English](saddle-numerics.md)

这些检查使用解析势和小型合成 MTP 夹具，验证导数与求解器行为；
材料精度取决于计算所使用的模型和数据。

## 坐标与求解器约定

- NR 使用 `xx yy zz yz xz xy` 顺序的对称线性应变，即 `H = H_ref (I + strain)`。
  坐标、梯度和 Hessian 使用同一参考晶胞，并包含应变-位置混合导数。
- 分数坐标变胞 mode following 使用同顺序的度规分量 `C-I`，即 `H = H_ref chol(C)`。
  Hessian 包含胞映射的二阶导数。`--car-rep` 则使用独立的原始晶胞与笛卡尔原子坐标。
  参见[命令参考](../reference/commands_zh.md)。
- NR 约束在求逆前对 Hessian 做降维。`--no-cell` 下固定原子按笛卡尔固定，
  否则按分数坐标固定。固定自由度不计入收敛残差。
- 回溯会拒绝非有限、晶胞翻转和能量升幅过大的步长；回溯预算耗尽会返回错误。
- Dimer 平移使用端点平均，转动使用其曲率的导数，并包含方向归一化的 Jacobian。
  静水压变胞目标一致地使用 `E+pV`。对称 VC-Dimer 与 VCNEB 通过 Cholesky
  或对称矩阵平方根使用解析拉回。
- Dimer 结果描述最终中心点，并包含转动残差、中心梯度与负曲率检查。
  仅凭负的方向曲率不能证明结构恰好有一个负 Hessian 特征值。
- `random_seed=0` 在同一构建下可复现；`-1` 请求非确定性初始化。
  显式给出的非法 MODECAR 会报错。
- `modefollowing --mode=2` 在预算耗尽或最终力/单一负模检查失败时返回 2。
  `--max-loop` 限制每个阶段的循环数。

## 测试

| 测试源文件 | 覆盖范围 |
|---|---|
| `unitest/ut_saddle_numerics.cpp` | 解析鞍点、独立的能量/梯度差分、约束求逆与 Dimer 状态 |
| `unitest/ut_saddle_inputs.cpp` | NR 收敛与约束、严格输入、输出错误与 MODECAR 切线 |
| `unitest/ut_modefollowing.cpp` | 固定/变胞 Hessian、远离参考晶胞的度规导数、模态投影与鞍点搜索 |
| `unitest/ut_neb.cpp` | VCNEB 压力导数、CI-NEB 双阱势垒与收敛 |
| `unitest/python/test_saddle_api.py` | 所有权、重复运行、不收敛、显式方向与轨迹写出 |
| `unitest/python/test_saddle_cli.py` | 命令行输出失败路径 |

按[安装说明](../guides/installation_zh.md)配置依赖后，在仓库根目录运行：

```sh
export LD_LIBRARY_PATH="$PWD/build/release:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1
make -j8 imrmlp test python
python3 -m pytest
```

## 适用范围

NR 与 mode following 需要 MTP 解析 Hessian。其稠密 Hessian 面向较小的候选结构。
NR 寻找驻点时不选择 Hessian 索引。测试以 CPU/OpenMP 以及定胞或静水压导数
为目标，不用于验证各向异性外应力或任意材料迁移路径。
仅靠导数与 MODECAR 检查不能证明变胞 mode following 已收敛。
