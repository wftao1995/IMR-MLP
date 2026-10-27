# API 示例 {#p02_api}

[English](cpp-examples.md)

@tableofcontents

本页面向需要把 IMR-MLP 嵌入自己程序的 C++ 用户。日常任务优先使用
[命令行](../reference/commands_zh.md)或 [Python API](../reference/python_zh.md)。
完整程序包含 `main()`，其余片段使用所在页面说明的输入结构和模型。

---

## API 层次

C++ API 与[架构](@ref p04_architecture)各层对应：

| 层次 | 页面 | 可学习内容 |
|-------|------|-------------------|
| **数据** | @ref api_core | 构造 `Atoms`、构建 `Neighborhoods`、坐标变换、访问力/能量/应力、管理物种和附加属性 |
| **I/O** | @ref api_io | 读写结构、大文件流式顺序 I/O、格式检测和物种映射 |
| **计算** | @ref api_potential | 加载势函数、计算 EFS（能量/力/应力）、OpenMP 并行评估、Hessian 计算、系数访问、训练接口 —— *包含 PotMTP\_M 案例* |
| **分析** | @ref api_analysis | RDF、Voronoi、力常数、键角、ACSF/BondOrder 描述符、轨迹平滑/退卷绕、空间群检测 |
| **Python** | @ref api_python | Python 绑定：`imrmlp` 包 API、Atoms、势函数、I/O、弛豫、NEB、训练、描述符、主动学习、ASE 计算器 |

---

## 快速开始：最小完整程序

先运行 `make libimrmlp.so` 和 `python3 examples/quickstart/run.py`，再将以下
代码保存为仓库根目录的 `my_program.cpp`。它使用快速入门生成的合成模型和
验证结构，仅演示调用接口，不代表真实 Al 材料模型。

```cpp
#include "base/structures.h"
#include "io/gpio.h"
#include "potentials/potential_mtp.h"

#include <iostream>

int main() {
    // 1. 加载结构
    GPIO io;
    io.setmap({"Al"});  // 模型物种 0 = Al
    Atoms atoms;
    io.read_one(atoms, "build/quickstart/valid-input.cfg");

    // 2. 加载势函数
    Potentials::PotMTP pot("build/quickstart/fitted.mtp");

    // 3. 计算能量、力、应力
    pot.getEFS(atoms);

    // 4. 访问结果
    std::cout << "能量: " << atoms.energy_ << " eV\n";
    std::cout << "最大力: " << atoms.report_maxforce() << " eV/Å\n";

    // 5. 写入结果
    io.write_one(atoms, "build/quickstart/cpp-predicted.cfg");
    return 0;
}
```

从仓库根目录编译。`@build/release/.flags` 是编译器响应文件，复用该库实际的
头文件路径和编译宏，避免头文件与库的 ABI 不一致。下例使用默认 release/GCC
构建；自定义构建应使用相同编译器以及对应 `build/<模式>/.flags` 和链接选项：
```bash
g++ @build/release/.flags my_program.cpp \
  -Lbuild/release -limrmlp -o build/release/my_program
LD_LIBRARY_PATH=$PWD/build/release:$LD_LIBRARY_PATH build/release/my_program
```

---

## Python 快速开始

先按 [Python API](../reference/python_zh.md) 安装扩展，仍从仓库根目录运行：

```python
import numpy as np
import imrmlp

# 1. 读取结构
atoms = imrmlp.read_one("build/quickstart/valid-input.cfg")

# 2. 加载势函数
pot = imrmlp.Potential("build/quickstart/fitted.mtp", "mtp")

# 3. 计算能量、力、应力
pot.calculate(atoms)

# 4. 访问结果
print(f"能量: {atoms.energy:.4f} eV")
print(f"最大力: {np.linalg.norm(atoms.forces, axis=1).max():.4f} eV/Å")

# 5. 写入结果
imrmlp.write("build/quickstart/python-predicted.cfg", atoms)
```

---

## 子页面

\subpage api_core
\subpage api_io
\subpage api_potential
\subpage api_analysis
\subpage api_python

---

## 参见

- @ref p04_architecture — 带数据流图的架构概述
- @ref guide_install — 构建要求与编译
- @ref guide_config — 配置文件格式与 API
- [类索引](annotated.html) — 所有已文档化类的完整列表
- [模块索引](topics.html) — 按子系统组织的类
