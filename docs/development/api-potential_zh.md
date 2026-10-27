# 势函数API：能量、力、应力 {#api_potential}

[English](api-potential.md)

本页面向集成势函数求值或自定义训练的 C++ 用户。日常计算请使用 `calc_efs`
或 [Python API](../reference/python_zh.md)。请先构建库，并参考
[C++ 编译说明](cpp-examples_zh.md)。示例需提供有效 MTP 模型，结构的物种索引
须与模型一致。能量单位为 eV，力为 eV/Å，原生应力为广延 virial，单位 eV。


`Potentials::PotMTP` 定义于
`potentials/potential_mtp.h`。


@tableofcontents

## 加载和使用势函数

### 加载MTP势函数

```cpp
#include "potentials/potential_mtp.h"

// 从文件加载（构造函数接受文件名）
Potentials::PotMTP pot("potential.mtp");

// 或通过流加载
std::ifstream ifs("potential.mtp");
pot.Load(ifs);
```

### 基本EFS计算

```cpp
#include "io/gpio.h"
#include "potentials/potential_mtp.h"

GPIO io;
io.setmap({"Al"});  // 模型物种 0 = Al
Atoms atoms;
io.read_one(atoms, "POSCAR");

Potentials::PotMTP pot("potential.mtp");

// 计算能量、力、应力（单线程）
pot.getEFS(atoms);

// 访问结果
double energy = atoms.energy_;
std::vector<Eigen::Vector3d> &forces = atoms.forces_;
Eigen::Matrix3d &stress = atoms.stress_;

std::cout << "Energy: " << energy << " eV" << std::endl;
std::cout << "Max force: " << atoms.report_maxforce() << " eV/A" << std::endl;
```

### OpenMP并行EFS

```cpp
// 线程安全的OpenMP评估
pot.omp_init();  // 初始化OpenMP缓冲区（调用一次）

#pragma omp parallel for
for (int i = 0; i < structures.size(); ++i) {
    pot.getEFSOMPthread(structures[i]);
}

// 或使用内置的并行包装器
pot.getEFSOMP(atoms);  // 单个结构的并行评估
```

### 使用预构建邻域的EFS

```cpp
Neighborhoods nbhs(atoms, pot.cutoff);
pot.getEFS(atoms, nbhs);  // 跳过邻域构建

// 仅在几何不变、邻域截断覆盖第二个势的截断时重用
Potentials::PotMTP pot2("potential2.mtp");
pot2.getEFS(atoms, nbhs);
```

### 仅能量计算

```cpp
// 更快：跳过力和应力计算
pot.getE(atoms);
double e = atoms.energy_;
```

### 可加EFS

```cpp
// 评估多个势函数并累加
Potentials::PotMTP pot_short("short_range.mtp");
Potentials::PotMTP pot_long("long_range.mtp");

// 第一个势函数
pot_short.getEFS(atoms);

// 第二个势函数：累加到现有EFS值上
pot_long.add_efs = true;
pot_long.getEFS(atoms);
```

## Hessian计算

### 完全Hessian

```cpp
#include "potentials/potential.h"

Potentials::HessianMatrix hessian;
pot.getEFSHessian(atoms, hessian);

// 访问：hessian(i_atom, a, j_atom, b)
double d2E_dxdy = hessian(0, 0, 1, 1);  // d2E / d(x0)d(y1)

// 原子力常数矩阵（原子对的3x3块）
Eigen::Matrix3d &fc = hessian(0, 1);
```

### 晶胞Hessian

```cpp
Eigen::MatrixXd cellhess(9, 9);   // d2E / d(strain)2
Eigen::MatrixXd lrhess(9, 3 * natoms);  // d2E / d(strain)d(pos)

pot.getEFSHessianCell(atoms, hessian, cellhess, lrhess);
```

### 对称性感知的Hessian

```cpp
// 利用晶体对称性提高速度
double symprec = 1e-4;
pot.getEFSHessianSymm(atoms, hessian, symprec);
```

## 系数访问

### 获取系数

```cpp
// 总系数数量
int ncoeffs = pot.get_coeff_size();

// 获取所有系数
Eigen::VectorXd coeffs(ncoeffs);
pot.get_coeffs(coeffs);

// 以原始数组方式获取
std::vector<double> coeff_arr(ncoeffs);
pot.get_coeffs(coeff_arr.data());

// 访问系数段
double *radial = pot.get_inter_pointer_radial_coeffs();
double *linear = pot.get_inter_pointer_linear_coeffs();
double *all = pot.get_inter_pointer_coeffs();

// 单个线性系数
double linear_coeff = pot.get_linear_coeff(0);
```

### 设置系数

```cpp
// 修改并更新
Eigen::VectorXd new_coeffs = coeffs;
new_coeffs[0] *= 1.1;  // 缩放第一个系数
pot.set_coeffs(new_coeffs);

// 仅设置线性系数（固定径向系数）
// 注意: set_linear_coeff 接受 double*, 使用 .data() 传递 Eigen vector
pot.set_linear_coeff(new_linear_only.data());
```

## 以编程方式创建势函数

### 从完整模板初始化

MTP 需要径向基、矩基和系数布局，仅设置 `cutoff` 和 `species_count` 不会创建
这些内容。应先加载完整模板，再随机化系数；`init_basic()` 仅为已有基准备工作
缓冲区，加载模板时已自动调用。

```cpp
// 先运行 CLI 快速入门，然后从仓库根目录运行本例。
// 模板提供径向基、矩基和系数布局。
Potentials::PotMTP pot("build/quickstart/template.mtp");
pot.random_init();
```

这只是用于拟合的模型初始化，不代表真实材料势。处理实际数据时，应选择元素
顺序、截断和基函数均匹配的模板。

### 使用设置

```cpp
#include "utils/gputils.h"

GPSettings settings;
settings["potfile"] = "template.mtp";

Potentials::PotMTP pot(settings);
```

## 使用AnyPotential

### 多态接口

```cpp
#include "potentials/potential.h"

// 直接使用具体类型，或包装在 unique_ptr 中实现多态
auto pot = std::make_unique<Potentials::PotMTP>("potential.mtp");
Potentials::AnyPotential *anypot = pot.get();

Atoms atoms;
// ... 填充原子数据 ...

// 使用虚接口
pot->getEFS(atoms);
pot->getEFSOMP(atoms);  // 自动构建邻域
```

### 保存和加载

```cpp
// 保存到文件
pot.Save("potential_saved.mtp");

// 保存到流
std::ofstream ofs("potential_stream.mtp");
pot.Save(ofs);

// 从流加载
std::ifstream ifs("potential.mtp");
pot.Load(ifs);
```

## 训练接口

标准 MTP 的矩系数在固定径向基时线性；同时优化径向系数仍是非线性训练。
以下是自定义训练循环的接口片段。常规拟合优先使用 `trainmtp`、`trainnlpot`
或 Python 训练器。

### 梯度累积

```cpp
// 用于非线性势函数训练
int ncoeffs = pot.get_coeff_size();                   // 系数数量
double dLdE = 2.0 * (atoms.energy_ - ref_energy);     // MSE梯度
Eigen::MatrixXd dLdF(force_rows, 3);                  // 默认 ColMajor
// ... 用力的差值填充dLdF ...
Eigen::Matrix3d dLdS = 2.0 * (atoms.stress_ - ref_stress);

std::vector<double> gradient(ncoeffs);
Neighborhoods nbhs(atoms, pot.cutoff);
pot.AccumulateEFSCombinationGard(atoms, nbhs,
    dLdE, dLdF, dLdS,
    gradient.data());   // 就地累加梯度
```

### 正则化

```cpp
double loss = 0.0;
std::vector<double> loss_grad(ncoeffs);   // ncoeffs 来自之前的 get_coeff_size()
pot.AddPenaltyGard(1.0, loss, loss_grad.data());
// loss: 正则项对总损失的贡献
// loss_grad: 正则项的梯度
```

## 案例：多阶 MTP (PotMTP\_M)

`PotMTP_M` 是标准 MTP 的多阶泛化。它不是单一标量位点能量，而是生成 @f$n@f$ 个
描述符量 @f$Q_k = c_{s,k} + \sum_\alpha c_{k,\alpha} B_\alpha@f$，
然后通过外部模型 @f$F(Q_1, \dots, Q_n)@f$ 传递——支持在 MTP 描述符之上的多项式、
神经网络或自定义回归器。

### 第 1 步：创建势函数

```cpp
#include "potentials/potential_mtp_m.h"
#include "base/structures.h"
#include "io/gpio.h"

using namespace Potentials;

// 通过设置配置外部模型类型
GPSettings settings;
settings["potfile"]    = "basis_template.mtp";  // 预构建的 MTP 基组
settings["extmodel"]   = "mtp_m_q2";            // 二次模型
// 其他模型选项: "mtp_m_n1"（单隐藏层 NN）, "mtp_m_li"（线性）,
//              "mtp_m_fs"（ArcSinh 微调）

// 创建势函数 — 加载基组 + 初始化模型
PotMTP_M<m_model_mtpq2> pot(settings);
```

### 第 2 步：随机初始化（从零训练）

```cpp
// 在 [-0.1, 0.1] 范围内随机播种系数
pot.random_init();

// 或控制范围
pot.random_lower = -0.05;
pot.random_upper =  0.05;
pot.random_init();
```

### 第 3 步：评估能量和力

```cpp
Atoms atoms;
GPIO io;
io.setmap({"Al"});  // 按实际模型的物种顺序修改
io.read_one(atoms, "structure.vasp");

// 构建邻域（EFS 必需）
Neighborhoods nbhs(atoms, pot.get_cutoff());

// 计算能量、力、应力
pot.getEFS(atoms, nbhs);

std::cout << "能量: "   << atoms.energy_  << " eV"    << std::endl;
std::cout << "最大力: " << atoms.report_maxforce() << " eV/Å" << std::endl;
```

### 第 4 步：训练（梯度累积）

多阶 MTP 训练通过矩张量树逐原子累积梯度。
在训练循环中使用基于梯度的优化器（L-BFGS、SGD 等）：

```cpp
int ncoeffs = pot.get_NL_coeffs_count();
std::vector<double> gradient(ncoeffs, 0.0);

// 对每个训练结构:
for (auto &train_atoms : training_set) {
    Neighborhoods nbhs(train_atoms, pot.get_cutoff());
    pot.getEFS(train_atoms, nbhs);  // 前向传播存储中间数据

    double dLdE = 2.0 * (train_atoms.energy_ - ref_energy);  // MSE 导数

    for (int ii = 0; ii < nbhs.size(); ++ii) {
        pot.AccumulateFPGard(nbhs[ii], gradient.data(), dLdE, nullptr);
    }

}

// 每个总目标只加一次正则化
double loss = 0.0;
pot.AddPenaltyGard(1.0, loss, gradient.data());

// 使用梯度进行优化器步进（L-BFGS、SGD 等）
// ... optimizer.update(coeffs_ptr, gradient.data()) ...

// 将更新后的系数应用回势函数
pot.nl_set_coeffs(coeffs_ptr);
```

### 第 5 步：保存和重用

```cpp
// 保存训练好的势函数
pot.Save("trained_mtp_m.mtp");

// 稍后重新加载
PotMTP_M<m_model_mtpq2> loaded_pot("trained_mtp_m.mtp");
// 注意：纯文件加载构造函数适用于模型不需要 init() 的情况
```

### 与 PotMTP 的关键区别

| 方面 | PotMTP | PotMTP\_M |
|--------|---------|------------|
| **位点能量** | 线性: `sum c_a * B_a` | 模型化: `F(Q_1, ..., Q_n)` |
| **系数** | species + moment 系数 | + model 系数（NN 权重、多项式项） |
| **模型类型** | 无额外外部模型；固定径向基时对矩系数线性 | 模板参数：多项式、NN、自定义 |
| **适用场景** | 标准 MTP | MTP 描述符上的非线性回归 |
| **训练** | `AccumulateEFSCombinationGard` | `AccumulateFPGard`（通过模型前向+反向传播） |

## 参见

- @ref potentials — 势函数模块
- @ref theory_mtp — MTP理论
- @ref theory_training — 训练方法
