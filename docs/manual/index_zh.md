# IMR-MLP 文档

```{raw} html
<div class="imr-home-hero">
  <div class="imr-home-logo">
    <img src="_static/IMR-MLP-logo.png" alt="IMR-MLP — Institute of Metal Research, Machine Learning Potential" width="200" height="200">
  </div>
  <div class="imr-home-intro">
    <p class="imr-home-tagline">矩张量机器学习势框架</p>
    <p>用机器学习原子间势开展原子尺度计算：训练与验证模型，完成结构弛豫、迁移路径和晶体性质计算，并通过 Python、ASE 或 LAMMPS 使用模型。</p>
    <div class="imr-home-actions">
      <a class="imr-home-primary" href="guides/index.html">使用指南</a>
      <a href="guides/quickstart.html">运行示例</a>
      <a href="reference/index.html">参考手册</a>
    </div>
  </div>
  <div class="imr-home-developers">
    <p>由中国科学院金属研究所材料人工智能研究部开发。</p>
  </div>
</div>
```

## 使用入口

| 要做的事 | 页面 |
| --- | --- |
| 构建 CPU 程序并验证环境 | [安装与构建](guides/installation.html) |
| 了解输入文件、元素映射、配置语法与单位 | [输入文件与配置约定](guides/conventions.html) |
| 计算能量、力和应力，评估误差 | [单点计算](guides/single-point.html) |
| 没有自备模型与数据时核验流程 | [自包含示例](guides/quickstart.html) |

完整的任务分类见[使用指南](guides/index.html)，可运行示例列在其末尾。

## 我想完成一个计算

| 目标 | 从这里开始 |
| --- | --- |
| 固定胞 / 变胞结构优化 | [结构弛豫](guides/relaxation.html) |
| 扩散、反应或相变势垒 | [NEB 最小能量路径](guides/neb.html) |
| 声子、弹性、体模量 | [晶体性质](guides/properties.html) |
| 准备参考数据、拟合并验证模型 | [训练工作流](guides/workflows.html) |
| 找出陌生构型并补充训练数据 | [主动学习与选样](guides/active-learning.html) |
| LAMMPS MD / Python 与 ASE | [LAMMPS](guides/lammps.html) / [Python](guides/python-installation.html) |

遇到问题先看[故障排查](guides/troubleshooting.html)，参数名称可查[命令速查](reference/cheatsheet.html)。

## 按用途查找

| 分区 | 在这里做什么 |
|---|---|
| [1. 使用指南](guides/index.html) | 回答“怎样完成一个任务” |
| [2. 参考手册](reference/index.html) | 回答“某个命令、参数或格式如何定义” |
| [3. 进阶使用](advanced/index.html) | HVP、SOG、fastfit 和专题案例 |
| [4. 理论与算法](theory/index.html) | 回答“模型和方法为什么这样工作” |
| [5. 开发文档](development/index.html) | 面向修改代码和扩展接口的开发者 |

C++ 符号与源码索引见[API 站点](../../build_zh/html/annotated.html)。

```{toctree}
:hidden:
:maxdepth: 3

guides/index
reference/index
advanced/index
theory/index
development/index
```
