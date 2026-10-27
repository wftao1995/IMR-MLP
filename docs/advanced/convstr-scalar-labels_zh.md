# convstr 中的标量元数据表达式

[English](convstr-scalar-labels.md)

使用 `--label-expr` 在转换结构时计算数值标签：

```sh
imrmlp convstr --inputfiles trajectory.xyz --input-format extxyz \
  --outputfile labelled.xyz --output-format extxyz \
  --label-expr 'energy_per_atom=energy/size' 'energy_density=energy/volume'
```

每个赋值要作为一个 shell 参数加引号。多个赋值依次跟在选项之后。
目标名是包含字母、数字和下划线的标识符，且以字母或下划线开头。
物理量与内建字段名保留为变量名，如 `energy`、`fenergy`（含输入别名
`free_energy`）、`fitness`、`volume`、`size`、`id`、`forces` 和
`total_charge`。重复目标，以及同一目标同时使用 `--label` 与
`--label-expr`，都会报错。

表达式支持数字（含科学计数法）、括号、一元 `+`/`-`、`+ - * /`、
用 `^` 表示的常量幂，以及单参数函数 `sqrt`、`exp`、`log`、`sin`、`cos`。
幂运算右结合，且比前导符号结合更紧：`-2^2` 为 `-4`，`2^-2` 为 `0.25`。
指数中不能出现帧变量：`energy^2` 合法，`2^size` 不合法。

变量读取当前帧的 `energy`、`fenergy`、`fitness`、`volume`、`size`、`id`，
以及全局 double/integer 属性。缺少能量或自由能数据会报错。
文本标签不会被解析为数字，即使其内容看起来是数字。
数组和逐原子属性不在本功能范围内。

所有赋值读取的都是同一个原始帧，且发生在显式源单位换算之后、
wrap、位移或排序之前。已存在的普通元数据键可以被重新计算。例如原始
`offset=3` 时，`'offset=offset+1' 'saved=offset'` 得到 `offset=4`、`saved=3`。
同一次调用中新建的输出不能被另一次赋值使用。目标原有的整数/文本别名
会被 double 覆盖；已存在的数组目标会被拒绝。排序可以使用派生出的数值属性：

```sh
imrmlp convstr --inputfiles trajectory.xyz --outputfile sorted.xyz \
  --label-expr 'energy_per_atom=energy/size' --sort-by energy_per_atom
```

批处理、顺序输入、顺序输出与完全流式路径使用同一个赋值函数。
排序仍要求保留帧在内存中的路径。`--label note --value 'energy/size'`
保持其原有字面文本行为，不求值表达式。

EXTXYZ 与 BIN 把计算出的标签保存为数值标量属性。CFG 通过已有的
`Feature name number` 文本表示写出。读取任意 CFG Feature 文本不会
提升为数值元数据，因此后续算术运算应借助 EXTXYZ 或 BIN 中间文件。
本选项不会为本来不支持标签的格式增加元数据保留能力。

语法错误与目标冲突在打开输出前检查。变量缺失或类型非数值、除数为零、
函数定义域失败以及非有限结果都会报告帧 ID 与赋值。流式模式下
后续失败会保留已写入的有效前缀，不回滚整个文件。
