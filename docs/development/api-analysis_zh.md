# 分析与描述符 API {#api_analysis}

[English](api-analysis.md)

本页面向添加分析方法或描述符的 C++ 用户。使用已有分析功能时，请先看
[命令参考](../reference/commands_zh.md)。请先构建库，再按
[API 示例](cpp-examples_zh.md)编译下文完整的键序程序。


@tableofcontents

## 结构与轨迹分析

| 类 | 头文件 | 用途 |
|---|---|---|
| `ComputeRDF` | `analyse/compute_rdf.h` | 对分布计数和径向分布函数 |
| `ComputeVoro` | `analyse/compute_voro.h` | Voronoi 体积、面及配位 |
| `ComputeAngles` | `analyse/compute_bondangle.h` | 键角和二面角分布 |
| `ComputeMSD` | `analyse/compute_msd.h` | 均方位移 |
| `ComputeBondOrder` | `analyse/compute_boo.h` | 键取向序 |
| `ComputeSRO` | `analyse/compute_sro.h` | 短程有序 |
| `ComputeCNA` | `analyse/compute_cna.h` | 公共近邻分析 |
| `ComputeHA` | `analyse/compute_ha.h` | Honeycutt–Andersen 指数 |
| `ComputeVACF` | `analyse/compute_vacf.h` | 速度自相关 |

可配置的分析通过 `GPSettings` 构造。命令行选项和输出文件见
[calc_prop](../reference/commands_zh.md) 和[配置参考](../reference/configuration_zh.md)。

`ComputePosavg`（`analyse/compute_posavg.h`）通过 `init`、`accumulate`、`compute`
执行流式平均，可选择变胞轨迹的分数坐标平均，或固定晶胞下的质量加权漂移消除。
`calc_prop --avgpos` 和 `calc_trjavg` 共用该实现，各自的漂移约定见
[命令参考](../reference/commands_zh.md)。

`ComputeSmoothFFT`（`analyse/compute_fftfilter.h`）实现 `smooth_traj`。
`compute2(trajectory, timestep_fs, window)` 要求帧均匀采样；
`ComputeSmoothFFT::SUPERGAUSSIAN` 选择超高斯窗口。

## 模型 Hessian 与 Phonopy

`ComputeHessian`（`analyse/compute_hessian.h`）计算模型 Hessian。
`compute_fc` 返回稠密原子 Hessian，`compute_fc_blocks` 返回
`Potentials::HessianMatrix` 分块存储。通过 `set_method` 选择 `auto`、`analytic`、
`force_fd`、`energy_fd` 或 `hybrid`。

`calc_fc` 导出力常数，`phonopyfc` 从 Phonopy 位移文件生成 `FORCE_SETS`。
[Python 适配器](../reference/python_zh.md#phonopy-integration)将 ASE calculator
的力交给 Phonopy，由其完成力常数拟合、色散、DOS 和热力学。

## 原子描述符

`Descriptors`（`descriptors/descriptors.h`）定义 `feature_size()`、
`compute(Neighborhood&, Eigen::VectorXd&)` 和 `clone()`。实现包括 ACSF
（`DescriptorBehler`）、键序（`DescriptorBondOrder`）和 MTP 标量矩（`DescriptorMTM`）。

以下完整程序计算一个原子的键序特征：

```cpp
#include "base/neighborhood.h"
#include "descriptors/descriptor_BondOrder.h"
#include <iostream>

int main() {
    Atoms atoms;
    atoms.cell_ = 20 * Eigen::Matrix3d::Identity();
    atoms.pos_ = {{1, 1, 1}, {2.2, 1.4, 1.8}, {1.7, 3.1, 2.5}};
    atoms.chemmap_ = {0, 0, 0};
    Neighborhoods neighbors(atoms, 5.0);
    DescriptorBondOrder descriptor(8);
    Eigen::VectorXd features;
    descriptor.compute(neighbors[0], features);
    std::cout << features.transpose() << '\n';
}
```

键序输出为 `[Q_2, Q_4, ..., Q_maxl, W_2, W_4, ..., W_maxl]`，`maxl` 必须是
不小于 2 的偶数。ACSF 和 MTP 描述符还接受带 `DescriptorGradient` 的
`ComputeOptions`，梯度形状为 `(feature_size, neighbor_count, 3)`。
`DescriptorMTM` 引用已加载的 `Potentials::PotMTP` 模型，模型须比描述符存活更久。

## 对称性与结构比较

`external/spglib_driver.h` 提供 `Symmetry3D::spacegroup_number`、
`standardize_cell` 和 `get_sym_dataset`。返回的 `symdataset` 持有自身的
对称操作、Wyckoff 编号与原子映射。

比较结构时，先用相同容差对两个输入调用 `prepare_comparison_structure`，再通过
`StructureComparisonOptions` 和 `equivalent_structures` 判等。共享内核保留物种，
在周期平移与晶胞对称操作下寻找一一原子映射，供 `sameq` 和结构去重共同使用。

## 参见

- @ref analyse — 分析模块
- @ref descriptors — 描述符模块
- @ref symmetry — 对称性模块
