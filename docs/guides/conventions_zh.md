# 输入文件与配置约定 {#guide_conventions}

[English](conventions.md)

所有任务共用一套输入约定：一个独立目录、一个势模型、一个结构文件、一份元素映射，
再加若干 `key value` 配置。本页只列这些共享约定；具体计算流程见各任务页。

不涉及：`.mtp` 模型文件自身的语法、Python 接口的调用参数。

## 任务目录

每次计算使用一个独立目录，输入、配置、日志和输出都放在其中：

```text
my-calculation/
    model.mtp      已训练的普通 MTP 模型
    POSCAR         待计算的结构，含元素符号
    index.map      模型的元素顺序
    relax.ini      该任务的配置（按任务命名）
```

`model.mtp`、`POSCAR` 等是占位名称，需换成自己的文件。`model.mtp` 来自模型提供方
或训练输出（如 `fitted.mtp`，也可修改配置中的文件名）；`POSCAR` 来自自己的建模或
参考计算结构；`index.map` 来自模型的物种顺序记录。这些文件不会被任何命令自动创建。

**相对路径从运行命令时的当前目录解析，不从配置文件所在目录解析。** 输入与输出
应使用不同文件名，避免覆盖。

## 模型、结构与元素映射

`index.map` 每行写一个模型元素。若模型 species 0 = Al、species 1 = Ni：

```text
Al
Ni
```

顺序取自模型或其训练记录。POSCAR 中元素块的顺序可以不同，读取时借助元素名映射；
CFG 中的数值类型已经是模型 species 索引，原子序数 13 不等于 species 0。
交换 `index.map` 后程序仍能运行，不能据此认为映射正确。

## 配置文件语法

键和值用空白分隔，**不写等号**：

```text
input POSCAR
output relaxed.xyz
index index.map
pot:potfile model.mtp
```

`pot:` 表示势函数设置，`relax:` 表示弛豫设置，`neb:` 表示路径设置。任务页统一使用
显式前缀，避免 section 与空行规则混淆。重复键保留第一次出现的值，修改时替换原行。

`.mtp` 模型文件有自己的语法，不应按 `.ini` 配置编辑。

## 单位与模型类型

| 项目 | 常用约定 |
| --- | --- |
| 位置、晶胞 | Å |
| 总能量 | eV；每原子能量需再除以原子数 |
| 力 | eV/Å |
| 原生广延应力 / 维里 | eV；转为应力需按体积及符号约定处理 |
| 外压 | 配置中的 `pressure` 使用 GPa，正值表示压缩 |
| 模型类型 | 默认普通 `mtp`；MTP_M、EAM、LES 等使用各自的类型与配置 |

CLI 各命令的模型选项不完全相同，不能把 `calc_efs` 的选项直接套给其他命令。
模型家族及旧格式要求见[势类型总览](../reference/potential-types_zh.md)，
完整配置键与单位见[配置参考](../reference/configuration_zh.md)。

## 相关

- [安装与构建](installation_zh.md)
- [单点计算与误差评估](single-point_zh.md)
- [配置参考](../reference/configuration_zh.md)
