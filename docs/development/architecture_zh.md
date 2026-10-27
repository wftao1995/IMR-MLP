# 架构概述 {#p04_architecture}

[English](architecture.md)

本页面向开发者和高级 C++ 用户，帮助定位各模块。普通用户可从
[工作流指南](../guides/workflows_zh.md)开始，运行命令不需要了解类的继承关系。



## 设计理念

IMR-MLP 遵循分层架构：

1. **数据层** — Atoms、Neighborhoods、掩码和 I/O 格式定义了*我们处理的内容*
2. **计算层** — 势函数和描述符从数据中计算物理量
3. **驱动层** — 最小化驱动程序和训练算法协调计算
4. **应用层** — CLI 命令向最终用户暴露功能

## 数据流

```
输入文件 (POSCAR, CFG, XYZ, ...)
    │
    ▼
┌──────────┐    ┌──────────────┐
│   GPIO   │───▶│    Atoms     │
│  (I/O)   │    │ （结构数据） │
└──────────┘    └──────┬───────┘
                       │
              ┌────────▼────────┐
              │  Neighborhoods  │
              │ （近邻列表）    │
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   ┌──────────┐ ┌───────────┐ ┌──────────┐
   │ Potential│ │Descriptor │ │ Analysis │
   │  (EFS)   │ │（特征）   │ │  (RDF,   │
   │          │ │           │ │ Hessian, │
   │          │ │           │ │  ...)    │
   └────┬─────┘ └─────┬─────┘ └────┬─────┘
        │             │            │
        ▼             ▼            ▼
   能量、力、      特征向量     分析结果
   应力、Hessian
```

## 关键基类

| 类 | 模块 | 作用 |
|-------|--------|------|
| `Atoms` | \ref core | 核心数据结构：晶胞、位置、种类、属性 |
| `AnyNeighborhoods` | \ref core | 类型擦除的近邻列表接口 |
| `AnyPotential` | \ref potentials | 所有势能面的虚基类 |
| `Descriptors` | \ref descriptors | 原子环境描述符的虚基类 |
| `AnyBasis` | \ref basis | 径向基函数接口 |
| `AnyCutoff` | \ref basis | 截断函数接口 |
| `GPIO` | \ref io | 文件 I/O 调度器 |
| `SimpleFireF` | \ref mindriver | FIRE 优化器 |
| `NEBDriver` | \ref mindriver | Nudged Elastic Band (NEB) 驱动 |
| `ALPotential` | \ref active_learn | 持有势函数和 AL 评分器/会话 |
| `ALModifier` | \ref active_learn | 弛豫过程中的即时主动学习修改器 |

## 模块关系

```
core ──────────────────────────────────────────────────────
  │  (Atoms, Neighborhoods — 被所有模块使用)
  │
  ├── io ── (读写 Atoms)
  │
  ├── basis ── (径向/截断函数，被势函数和描述符使用)
  │
  ├── potentials ── (虚基类 AnyPotential 接口)
  │      ├── PotMTP (MTP 实现)
  │      ├── PotEAM (EAM 实现)
  │      └── PotLJ, ... (其他实现)
  │
  ├── descriptors ── (ACSF, BondOrder, MTM)
  │      └── 使用 basis 提供截断函数
  │
  ├── analyse ── (RDF, Voronoi, Hessian, 轨迹分析/平滑, ...)
  │
  ├── mindriver ── (L-BFGS, FIRE, CG, NEB)
  │
  ├── train ── (MTP 训练、非线性拟合、线性最小二乘)
  │      └── 使用 mindriver + potentials + maxvol
  │
  ├── active_learn ── (MaxVol 评分器、ALPotential、ALModifier)
  │      └── 使用 potentials + train + mindriver
  │
  ├── maxvol ── (主动集选择：MaxVol、FPS、CUR)
  │
  ├── NNBase ── (神经网络层，被支持 NN 的势函数使用)
  │
  ├── symmetry ── (空间群、Wyckoff 位置)
  │
  ├── tools ── (CLI 解析、化学辅助函数)
  │
  └── apps ── (CLI 入口点、调度)
```

## 计算热点路径

对于能量/力/应力评估，关键路径如下：

```
PotMTP::getEFS()
    │
    ├── Neighborhoods::construct()
    │       └── 周期镜像与空间分箱（默认 Neighborhoods）
    │
    ├── CalcSiteEFS（逐原子；getEFSOMP 选择并行求值）
    │       ├── 径向基函数评估 (rb_calcder)
    │       ├── 矩张量构建（基本 → 缩并）
    │       ├── 模型传播（线性或 NN）
    │       └── 反向传播（通过 Jacobian 计算力和应力）
    │
    └── 累加：单位点能量求和 → 总能量、力、应力
```

## 配置模型

大多数非平凡类继承自 `ConnectSettings`，它提供了一种 ini 风格的
键值配置系统。配置文件使用带命名空间前缀（例如，
`relax:`、`mtp:`、`neb:`）来将参数路由到相应的子系统。

## 模块详情

### `active_learn/` — 主动学习

- `al_scorer.h`：`ALScorer`、`MaxVolScorer` 和 `ALSetup` 根据已保存的主动集进行外推评分。
- `al_potential.h`：`ALPotential` 管理评分、候选收集及导出
  （`create`、`evaluate`、`check`、`finalize`），支持 `maxvol` 和 `none`，不支持的模式会报错。
- `mindriver/al_modifier.h`：`ALModifier` 通过弛豫驱动的可选 AL 接口，
  将 `ALPotential` 接入 `relax`。

### `mtp_kim/` — KIM API 模型驱动

独立的 KIM API 兼容模型驱动，将 MTP 打包为标准的 KIM 模型，
供 LAMMPS、ASE 或任何 KIM 感知的 MD 代码使用。包含：

- `kim_model_driver.cpp`：KIM API v2 模型接口（Create、ComputeArgumentsCreate、
  Compute、Destroy、Refresh）。
- `mtp_kim_core.{cpp,h}`：自包含的 MTP 计算核心，依赖最少 —
  实现了 `Vec3`、Chebyshev 径向基、矩张量计算、近邻列表构建和 `.mtp` 文件解析。

### `potentials/potential_openkim.{cpp,h}` — PotKIM

`PotKIM` 将任何 KIM API 模型包装为 `AnyPotential` 接口，使 KIM 兼容势可作为
即插即用的替代品在整个框架中使用。支持自动检测支持的物种类型，并与标准
`Neighborhoods` 基础设施集成以构建近邻列表。
