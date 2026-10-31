# 单点计算与误差评估 {#guide_single_point}

[English](single-point.md)

单点计算在固定几何下预测总能量、每个原子的力和应力，不做结构优化。
有带参考标签的数据时，用 `calc_err` 量化同一几何下的预测误差。

不涉及：改变原子位置或晶胞（见[结构弛豫](relaxation_zh.md)）、沿反应路径的
能量（见 [NEB](neb_zh.md)）。

## 前置输入

| 文件 | 内容 | 来源 |
| --- | --- | --- |
| `model.mtp` | 已训练的普通 MTP 模型 | 模型提供方或训练输出 |
| `POSCAR` | 待预测结构，含元素符号 | 自己的建模或参考计算 |
| `index.map` | 模型元素顺序 | 模型或其训练记录 |
| `reference.xyz`（可选） | 带 E/F/应力标签的结构 | 参考计算；仅 `calc_err` 需要 |

命名与路径规则见[输入文件与配置约定](conventions_zh.md)。预测命令不会创建上述输入。

## 命令

### 预测能量、力与应力

```sh
imrmlp calc_efs POSCAR predicted.xyz --potfile model.mtp
```

元素映射默认读 `index.map`，文件名不变时不必加参数；势类型默认普通 MTP，文件格式
按文件名识别。只有使用其他势类型、映射文件改了名字，或文件名无法识别格式时，
才需显式指定相应选项。各命令的选项不完全相同，完整列表见
[命令参考](../reference/commands_zh.md) 的 `calc_efs` 一节。

### 评估误差

```sh
imrmlp calc_err --input reference.xyz --potfile model.mtp \
  --output-summary errors.txt
```

`reference.xyz` 必须带参考标签。只有 POSCAR 时可以预测，但没有可比较的参考值，
不能凭空生成验证误差。

## 输出与判据

`predicted.xyz` 是扩展 XYZ 文本，保留输入几何并写入预测属性，可用文本编辑器或
支持 EXTXYZ 的软件检查。每帧第二行的 `energy` 是该结构的**总**能量；
`Properties` 声明后续列的含义，`forces:R:3` 表示每个原子的三个力分量。

检查项：程序正常退出、原子数与元素正确、能量和力为有限值，且与同结构参考结果的偏差
在可接受范围。`calc_err` 直接给出这个偏差；没有参考标签时只能确认数值有限。

## 用入门示例核验

没有自备输入时，可用[自包含入门示例](../../examples/quickstart/README_zh.md)
生成默认目录 `build/quickstart/`，再从仓库根目录进入：

```sh
cd build/quickstart
imrmlp calc_efs valid-input.cfg my-prediction.xyz --potfile fitted.mtp
head -n 5 my-prediction.xyz
```

`valid-input.cfg` 只含几何，`valid.cfg` 才有参考标签，可用它运行 `calc_err` 并与
`errors.txt` 对照。该文件共四帧三原子结构，不是单个十二原子结构。示例中的 `Al`
仅用于演示元素映射，其合成模型不能替代真实的铝势。

自定义了 `--work` 时，把上面的目录换成实际输出路径；练习结束后用 `cd ../..`
返回仓库根目录。

## 常见错误

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| 能量或力量级明显异常 | `index.map` 元素顺序与模型不一致 | 按元素名逐行核对模型顺序；“能运行”说明不了这一项 |
| 找不到输入文件 | 相对路径按当前工作目录解析 | 在任务目录内运行命令，或改用绝对路径 |
| 输出被上一轮覆盖 | 输入输出同名 | 输出使用不同文件名 |
| 选项被忽略或报错 | 把 `calc_efs` 的选项套给其他命令 | 按[命令参考](../reference/commands_zh.md)逐条核对 |
| `calc_err` 得不到误差 | 输入没有参考标签 | 使用带 E/F/应力标签的数据 |

遇到其他症状见[故障排查](troubleshooting_zh.md)。

## 相关

- [输入文件与配置约定](conventions_zh.md)
- [结构弛豫](relaxation_zh.md)、[NEB](neb_zh.md)、[晶体性质](properties_zh.md)
- [命令参考](../reference/commands_zh.md)、[速查表](../reference/cheatsheet_zh.md)
