# GPM v1 模型文件 {#reference_gpm_v1}

[English](gpm-v1.md)

GPM 在一个文件中保存实际生效的势函数、依赖资源和可选的主动学习会话。加载势函数不会自动收集候选。旧势文件和 `.sel` 输入仍然可用。


普通使用只需转换、加载和验证模型，无需编辑二进制内容。转换会保留模型行为，
不会增加训练或运行后端支持。部署前仍须准备模型需要的外部运行库和程序。

## CLI 与 Python

```sh
imrmlp model_convert --settings input.ini --output model.gpm
imrmlp model_convert --input old.mtp --type mtp --index index.map --output model.gpm
imrmlp model_convert --settings with-al.ini --output resume.gpm --profile resume
imrmlp model_info --input model.gpm
imrmlp model_validate --input resume.gpm
```

`with-al.ini` 使用已有 `pot:` 和 `al:` 配置。设置 `al:source embedded` 从同一模型恢复 AL，它与 `al:state_file`、`al:committee_prefix` 互斥。模拟在使用 AL 前显式针对势函数初始化。

```python
from imrmlp import Potential, ModelBundle

potential = Potential("model.gpm")
potential.calculate(atoms)              # 仅推理
potential.save("copy.gpm")              # 默认保留内嵌 AL
potential.save("deploy.gpm", profile="inference")

bundle = ModelBundle.load("resume.gpm")
session = bundle.create_al(export_file="candidates.xyz")
session.check(atoms)                     # 显式评分和收集
session.save("continued.gpm")

# 从现有配置创建，并附加已选样的状态。
bundle = ModelBundle.from_settings({"pottype": "mtp", "potfile": "old.mtp"})
bundle.attach_selector(selector)
bundle.save("selected.gpm", profile="resume")
```

默认保存配置为 `preserve`；`inference` 显式删除 AL；`resume` 要求存在 AL。模型改变后，不能用 `preserve` 沿用旧 AL；须重建并附加选样，或显式选择 `inference`。附加操作保存状态快照，后续使用 `session.save` 保存最新候选池和计数。

LAMMPS native 使用统一工厂并从 GPM 推断类型。Standalone 共享 C++17 读取器，支持已有短程 MTP/MTP-M 执行器，拒绝其他家族及长程贡献。`MTP_ZSTD=ON` 开启压缩块支持。格式接入不增加 LAMMPS AL 执行能力。

## 二进制契约

除数据块类型另有声明外，整数均为无符号小端编码。64 字节前导如下：

| 字节偏移 | 长度 | 含义 |
|---|---:|---|
| 0 | 8 | ASCII `GPMLMODL` |
| 8 | 2 | 主版本，1 |
| 10 | 2 | 次版本，0 |
| 12 | 4 | 必需标志，目前为 0 |
| 16 | 8 | UTF-8 JSON 字节数 |
| 24 | 8 | 二进制数据区的绝对起点 |
| 32 | 32 | JSON 原始字节的 SHA-256 |

JSON 紧接前导。数据区起点为 `align64(64 + json_length)`，填充字节为零。每个块的偏移相对数据区，且为 64 的倍数。空块可以共用偏移，非空块不得重叠。

`schema` 为 `gpml.model`。顶层必需成员为 `units`、`species`、`root_model`、`models` 和 `blocks`。可选成员包括 `active_learning`、`structures`、`resources`、`provenance` 和 `extensions`。`model_fingerprint` 标识实际生效的根模型图。未知的 `required_features` 或前导标志导致加载失败；无修改转存保留可选扩展。

单位为 `{ "length": "angstrom", "energy": "eV", "charge": "e", "stress": "virial_eV" }`。stress 保存项目当前符号约定下的内部 virial，不是压力或除以体积后的量。

每个块目录项包含 `id`、`dtype`、`shape`、`offset`、`stored_size`、`raw_size`、`codec` 和 `sha256`。数据类型为 `f64`、`i32`、`u32`、`i64`、`u64`、`u8`。FP64 保留 IEEE-754 位模式，矩阵按行存储。维度、字节数和偏移均使用规范的无符号十进制**字符串**。维度乘积必须精确等于 `raw_size / sizeof(dtype)`。校验覆盖解压后的数据。`codec` 为 `none` 或 `zstd`，默认使用 `none`；底层 writer 支持逐块指定。未启用 zstd 的构建在需要读取压缩块时明确报错。

数值引用格式为 `{ "block": "unique/id" }`。构造模型之前检查 JSON 校验、重复键、维度、偏移、重叠和必需版本。每块在读取时校验 SHA-256。普通推理不读取 AL 历史和候选数据，完整验证读取所有块。默认 JSON 上限为 64 MiB，单块解码上限为 8 GiB。

writer 在目标目录暂存数据，校验并 fsync 后原子重命名。重命名前的失败保持旧目标完整。C++ writer 为本地操作，适用于已有的主进程保存位置；转换 CLI 由主进程执行并向其他进程广播失败结果。调用方不得对同一路径并发启动独立 writer。


这里的 `zstd` 是 GPM 内部数据块的编码；与普通结构/设置/势文本文件的外层
`.zst` 压缩不同，后者不受支持。通常使用 CLI 默认的未压缩 GPM 即可。

## 模型节点

`models` 将 ID 映射到包含 `type`、`version: 1`、`species`、`species_mode`、`common` 和 `children` 的节点。各类型按需增加 `configuration`、`parameters`、`state` 或 `resources`。子节点引用文件内 ID，不引用输入路径。遍历模型图时检查缺失引用和环。

元素类型从零编号，每个类型对应 `{ "id": 0 }`，仅在已知时增加 `symbol`。`mindist` 声明 `species_mode: "type_independent"`。显式外部映射必须与已知元素一致；未提供化学符号时仍保留明确的类型顺序。

v1 codec 覆盖 `mtp`、`mtp_m`、`mtpnn1`、`mtp_les`、`hmtp`、`hybrid`、`hybrid_scale`、`eamalloy`、`eamfs`、`eampoly`、`lj612`、`zbl`、`sw`、`tersoff`、`pair_table`、`threebody_table`、`mindist`、`descriptor_linear`、`openkim`、`acelmp` 和 `shellfile`。

MTP 节点保存径向基定义、各基缩放、矩规则、明确的系数顺序和具名的头部/核字段。`mtp_m` 元数据区分 li/q2/fs/n1/n2，并记录网络维度和激活函数。数值数组不依赖 Eigen 或对象内存布局。现有 MTP 解析器读取这些字段在内存中生成的表达，不重新打开原模型。

其他 codec 保存有效样条表、函数参数、描述符配置、NN 归一化与权重或有序子模型。MTM 描述符内嵌依赖的 MTP；混合势保留训练标记和特征过渡参数，计算和训练能力边界保持不变。

`EAMZhou2004::SaveParameterSet` / `LoadParameterSet` 保存独立的 `eam_zhou2004_parameters` 对象，包含活跃元素和具名参数。它是参数集文件，不是新增的可执行势工厂类型；`model_info` 和 `model_validate` 支持它。

KIM 节点声明模型标识并保存有效参数数组。ACE 内嵌已加载的基资源，仍需 PACE 运行库。Shell 内嵌捕获的脚本和显式列出的输入资源；用 `shellfile:dependencies` 声明程序依赖，用 `shellfile:resource:relative/name source/path` 声明输入。资源名不得逃出任务目录。展开的脚本随加载对象管理生命周期；显式计算 shell 模型时仍会运行这些脚本。

## 主动学习

`active_learning` 独立保存 `version: 1`、`method`、`target_model` 和 `model_fingerprint`。公开版支持 `maxvol` 方法。`policy` 包含选择/中止阈值、检查间隔、步数、中止标记、稀疏化方法和目标数量。导出路径属于运行时覆盖项，`candidate_pool` 引用带版本的构型记录。

构型记录保留晶胞、坐标、类型、PBC、物理属性、标签、掩码、拓扑、逐原子/全局类型属性和字符串元数据。目录本身是 `u8` JSON 块，数值字段使用独立数据块，不保存原生 `Atoms` 对象或指针。

MaxVol 保存 builder 配置、E/F/virial/site 权重、行缩放、初始化方式、活跃行、行来源、已选构型、有序历史、进度、选样模式和终止状态。builder 绑定指纹同时覆盖配置和数值块。full 和 exact streaming 保留重扫所需历史。B 矩阵、数值分解、线程缓冲及 mmap 文件均重新建立。导入旧 `.sel` 时使用给定模型重建来源行，与活跃行比较后绑定。


目标模型的配置、参数或元素顺序改变后，已绑定会话失效，参数数量相同不代表兼容。恢复后评分和保存均检查绑定。优化器、MD 积分器和标注调度器检查点不属于本格式。
