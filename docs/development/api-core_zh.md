# 核心API：Atoms和邻域 {#api_core}

[English](api-core.md)

本页面向扩展或嵌入 IMR-MLP 的 C++ 用户。日常结构操作建议先看
[Python API](../reference/python_zh.md)。请先构建 `libimrmlp.so`，编译和链接方法见
[API 示例](cpp-examples_zh.md)。除非另有说明，片段沿用前面构造的 `atoms`。



@tableofcontents

## 创建和操作Atoms

### 基本构造

```cpp
#include "base/structures.h"

// 创建一个空的Atoms对象
Atoms atoms;

// 设置立方晶胞（每行是一个晶格矢量）
atoms.cell_ << 5.0, 0.0, 0.0,
               0.0, 5.0, 0.0,
               0.0, 0.0, 5.0;

// 添加带物种索引的原子
atoms.pos_.push_back({0.0, 0.0, 0.0});  // 原子0在原点
atoms.pos_.push_back({2.5, 2.5, 2.5});  // 原子1在体心
atoms.chemmap_ = {0, 1};                 // 物种0和1

// 坐标为笛卡尔坐标；周期性默认在三个方向均开启
atoms.cartesian = true;
atoms.pbc_ = {true, true, true};
```

### 从文件读取

```cpp
#include "io/gpio.h"

GPIO io;
Atoms atoms;

// 读取单个结构
io.read_one(atoms, "POSCAR");

// 读取多个结构
std::vector<Atoms> structures;
io.read(structures, "trajectory.cfg");
```

### 晶胞操作

```cpp
// 创建2x2x2超胞
Eigen::Matrix3i super_mat;
super_mat << 2, 0, 0,
             0, 2, 0,
             0, 0, 2;
atoms.mksuperL(super_mat);

// 获取晶胞属性
double volume = atoms.get_volume();
Eigen::Matrix3d metric = atoms.get_metric();
Eigen::Matrix3d recip = atoms.get_recip();  // 包含2*pi

// 晶胞参数 (a, b, c, alpha, beta, gamma)
std::vector<double> params = atoms.get_cell_parameters();
double a = params[0], b = params[1], c = params[2];
double alpha = params[3], beta = params[4], gamma = params[5];

// 标准化晶胞取向
atoms.to_lower_standard();  // 下三角矩阵，右手系，a || x
```

### 坐标变换

```cpp
// 分数坐标 <-> 笛卡尔坐标转换
atoms.to_fractional();    // pos_现在为分数坐标
atoms.to_cartesian();     // pos_回到笛卡尔坐标

// 直接转换（不修改状态）
Eigen::RowVector3d frac = atoms.car2dir(cart_pos);
Eigen::RowVector3d cart = atoms.dir2car(frac_pos);

// 绕轴旋转原子
atoms.rotate(Eigen::Vector3d(0, 0, 1), M_PI/4);        // 绕z轴旋转45°
atoms.rotate<true, false>(1, 0, 0, M_PI/2);             // 仅旋转原子
                                                         // 绕[100]晶格方向

// 平移
atoms.translate(Eigen::Vector3d(1.0, 0.0, 0.0));

// 将原子包裹回晶胞
atoms.wrap();
```

### 几何查询

```cpp
int natoms = atoms.size();                    // 原子数量
double d = atoms.get_distance(0, 1);          // 最小镜像距离
Eigen::RowVector3d rab = atoms.get_rab(0, 1); // 最小镜像矢量
Eigen::RowVector3d cm = atoms.get_cm();       // 质心
bool orth = atoms.isorth();                    // 检查是否正交晶胞
bool rh = atoms.is_right_hand();              // 检查右手系
int handed = atoms.get_handedness();           // +1 = 右手, -1 = 左手
```

### 属性访问

```cpp
// 能量、力、应力
double energy = atoms.energy_;
std::vector<Eigen::Vector3d> &forces = atoms.forces_;
Eigen::Matrix3d &stress = atoms.stress_;

// 检查可用数据
bool has_forces = atoms.has_forces();
bool has_stress = atoms.has_stresses();
bool has_velocities = atoms.has_velo();

// 清零和重置
atoms.resetEFS();    // 将能量、力、应力设为零
atoms.clearEFS();    // 移除能量、力、应力数据
```

### 物种管理

```cpp
// 获取唯一物种
std::set<int> types = atoms.get_types_set();

// 统计各类型的原子数量
std::map<int, int> counts = atoms.types_count_map();
for (auto &[type, count] : counts) {
    std::cout << "Type " << type << ": " << count << " atoms" << std::endl;
}
```

### 附加属性

```cpp
// 逐原子标量属性
atoms.alloc_ld("site_energy");
atoms.local_attribs_d_["site_energy"][0] = -3.14;

// 全局标量属性
atoms.global_attribs_d_["temperature"] = 300.0;

// 逐原子矢量属性（每个原子3个分量）
atoms.alloc_lvd("dipole", 3);
atoms.local_attribs_v_["dipole"](0, 0) = 0.1;  // 分量0，原子0
atoms.local_attribs_v_["dipole"](1, 0) = 0.2;  // 分量1，原子0
atoms.local_attribs_v_["dipole"](2, 0) = 0.3;  // 分量2，原子0

// 检查是否存在
if (atoms.has_l_attrib_d()) { /* ... */ }
```

### 拟合权重

```cpp
// 逐结构权重
atoms.fitness = 2.0;  // 该构型权重更高

// 逐原子权重
atoms.atomic_fitness = {1.0, 0.5, 1.0, 1.0};  // 原子1权重较低
```

## 构建邻域

### 基本用法

```cpp
#include "base/neighborhood.h"

Atoms atoms;
// ... 填充原子数据 ...

double cutoff = 5.0;
Neighborhoods nbhs(atoms, cutoff);  // 构建近邻列表

// 或者，两步初始化
Neighborhoods nbhs2;
nbhs2.init(atoms, cutoff);
```

### 访问近邻信息

```cpp
// 遍历中心原子
for (int i = 0; i < atoms.size(); ++i) {
    // 访问原子i的邻域
    auto &nbh = nbhs[i];              // 通过 operator[] 访问
    int centertype = nbh.centertype;   // 中心原子的物种
    int n_neighbors = nbh.nbhcount_;   // 近邻数量

    for (int j = 0; j < n_neighbors; ++j) {
        double rij = nbh.distances_[j];    // 距离
        int jtype = nbh.chemmap_[j];      // 近邻物种
        Eigen::Vector3d rvec = nbh.pos_[j]; // 相对位置矢量
    }
}
```

### 选择性动力学

显式初始化每个分量，`1` 表示允许移动，`0` 表示固定。弛豫驱动还需开启
选择性动力学选项；此处不设置晶胞约束。

```cpp
atoms.sel_dynpos_.assign(atoms.size(), Eigen::RowVector3d::Ones());
if (atoms.size() > 0) atoms.sel_dynpos_[0](0) = 0;  // 固定首个原子的 x 分量
```

## 合并和提取结构

```cpp
// 合并两个结构（拼接原子）
Atoms combined = merge_atoms(struct1, struct2);

// 通过掩码提取子结构
GPMask mask;  // （在其他地方定义）
Atoms subset = atoms.extract_substructures_from_mask(mask);
```

## 序列化

```cpp
// 二进制序列化
std::ofstream bofs("atoms.bin", std::ios::binary);
atoms.write_to_bin(bofs);

```

## 参见

- @ref core — 核心数据类型模块
- @ref api_io — 文件I/O API
- @ref api_potential — 势函数评估API
