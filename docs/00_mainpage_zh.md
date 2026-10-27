# 简介 {#mainpage}

[English](00_mainpage.md)

\image html IMR-MLP-logo.png "" width=400px
\image latex IMR-MLP-logo.png "" width=0.6\\textwidth

**矩张量机器学习势框架**

一个用于构建、训练和部署机器学习原子间势的 C++23 框架。
实现了矩张量势（MTP）形式体系，支持非线性
神经网络模型、主动学习以及广泛的结构分析工具。

由中国科学院金属研究所材料人工智能研究部开发。

## 文档分区

| 分区 | 用途 |
|---|---|
| [使用指南](guides/index_zh.md) | 按任务完成安装、单点、弛豫、NEB、性质计算、训练与选样。 |
| [参考手册](reference/index_zh.md) | 查询命令、配置、Python 和模型格式。 |
| [进阶使用](advanced/index_zh.md) | 选基、HVP、variable projection、长程模型与鞍点搜索。 |
| [理论与算法](theory/index_zh.md) | MTP 与 MTP_M 模型、能量导数、拟合及主动学习。 |
| [开发文档](development/index_zh.md) | C++ 架构、API 与集成模块。 |

---

## 开始使用

第一次使用请按[任务引导](guides/getting-started_zh.md)选择路线：没有模型时先跑
[自包含示例](../examples/quickstart/README_zh.md)，已有模型时从单点计算开始。
[安装指南](guides/installation_zh.md)解释基础 CPU 构建及验证，Python、MPI 和 GPU 按需添加。

| 任务 | 使用引导 |
| --- | --- |
| 准备输入、理解元素映射并计算 E/F/应力 | [开始使用](guides/getting-started_zh.md) |
| 优化原子或晶胞 | [结构弛豫](guides/relaxation_zh.md) |
| 计算迁移或反应势垒 | [NEB 路径](guides/neb_zh.md) |
| 声子、弹性、状态方程 | [晶体性质](guides/properties_zh.md) |
| 训练与验证自己的模型 | [训练流程](guides/workflows_zh.md) |
| 选样、参考标注与重训循环 | [主动学习](guides/active-learning_zh.md) |
| 在 LAMMPS / Python 中运行模型 | [LAMMPS](../LAMMPS/README_zh.md) / [Python](../README-python_zh.md) |

完整选项见[命令参考](reference/commands_zh.md)，遇到问题见[故障排查](guides/troubleshooting_zh.md)。
普通计算无需先阅读以下 C++ 模块与 API 索引。

---

## 模块参考

| 模块 | 描述 |
|--------|-------------|
| @ref core "核心数据类型" | Atoms、Neighborhoods 和基础数据结构 |
| @ref potentials "势函数" | 原子间势：MTP、EAM、LJ、KIM 等 |
| @ref active_learn "主动学习" | MaxVol 外推评分、ALPotential、ALModifier 和在线弛豫 |
| @ref descriptors "描述符" | 机器学习描述符：ACSF/Behler、Bond Order、MTM |
| @ref io "输入/输出" | VASP、LAMMPS、XYZ、CFG 等格式的读写器 |
| @ref analyse "分析" | RDF、键角、Voronoi、MSD、SRO、力常数 |
| @ref cheatsheet "命令速查表" | 快速参考的单页命令摘要 |
| @ref mindriver "最小化" | L-BFGS、FIRE、共轭梯度、最速下降、NEB |
| @ref basis "基函数" | 径向基函数（Chebyshev、多项式、GTO）和截断函数 |
| @ref maxvol "MAXVOL" | 用于训练配置筛选的主动集选择 |
| @ref train "训练" | MTP 训练、非线性势拟合、SGD 调度 |
| @ref symmetry "对称性" | 通过 spglib 进行空间群检测和 Wyckoff 位置分析 |
| @ref nnbase "NN 原语" | 神经网络层（1 隐藏层和 2 隐藏层） |
| @ref tools "工具" | CLI 解析、化学辅助函数、物理常量 |
| @ref apps "命令" | imrmlp 二进制的 CLI 子命令实现 |
| @ref trajectory "轨迹" | MD 轨迹的平滑、解卷绕和 FFT 滤波 |

---

## API 参考

- [类索引](annotated.html) — 所有已文档化类的完整列表
- [文件索引](files.html) — 带有文档的源文件
- [待办列表](todo.html) — 已知缺失功能和计划改进
- [缺陷列表](bug.html) — 已知问题

---

## 参考文献

- **Wang, J. et al.** "Efficient moment tensor machine-learning interatomic potential
  for accurate description of defects in Ni-Al Alloys."
  *Phys. Rev. Materials* 9, 053805 (2025).
  DOI: [10.1103/PhysRevMaterials.9.053805](https://doi.org/10.1103/PhysRevMaterials.9.053805)
- Shapeev, A. "Moment Tensor Potentials: A Class of Systematically Improvable Interatomic Potentials."
  *Multiscale Model. Simul.* 14, 1153–1173 (2016).
- Bartok, A. et al. "On representing chemical environments." *Phys. Rev. B* 87, 184115 (2013).
- Behler, J. "Atom-centered symmetry functions for constructing high-dimensional neural network potentials."
  *J. Chem. Phys.* 134, 074106 (2011).
