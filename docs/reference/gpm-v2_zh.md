# GPM v2 模型文件 {#reference_gpm_v2}

[English](gpm-v2.md)

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

GPM v2 拒绝主版本 1；不提供旧 GPM 兼容读取或隐式转换。
魔数仍为 `GPMLMODL`，JSON schema 仍为 `gpml.model`。
二进制整数采用小端编码，64 字节前导如下：

| 偏移 | 字节数 | 含义 |
|---|---:|---|
| 0 | 8 | `GPMLMODL` |
| 8 | 2 | 主版本 2 |
| 10 | 2 | 次版本 0 |
| 12 | 4 | 必需标志，当前为 0 |
| 16 | 8 | JSON 字节数 |
| 24 | 8 | 二进制数据区绝对起点 |
| 32 | 32 | 保留，写零；不存校验和 |

顶层 schema 为 `gpml.model`，必需成员为 `units`、`species`、`root_model`、`models`、`blocks`。
单位约定为 angstrom、eV、电荷 e，stress 表示内部 virial_eV，不是压力或体积归一化的应力。
容器主版本 2 与模型节点自身的 `version: 1` 独立。未知必需能力和必需标志会被拒绝。

JSON 紧接前导，数据区起点为 `align64(64 + json_length)`。
块偏移相对数据区，并按 64 字节对齐；非空块不能重叠。
目录项只包含 `id`、`dtype`、`shape`、`offset`、`stored_size`、`raw_size`、`codec`，
没有 `sha256`。类型为 `f64`、`i32`、`u32`、`i64`、`u64`、`u8`；矩阵按行存储。
维度、偏移和字节数使用无符号十进制字符串，维度乘积必须符合原始字节长度。
压缩方式为 `none` 或 `zstd`，默认 `none`。

普通数组使用 `{ "block": "id" }`；构型属性切片使用
`{ "block": "id", "start": "0", "shape": ["16", "3", "1"] }`，
`start` 为元素偏移。空数组只记录 `dtype`、`shape`，不分配块。
有限标量直接写入 JSON；负零、无穷和 NaN 分别用
`{ "float": "-0" }`、`inf`、`-inf`、`nan` 显式表示。
二进制数组保留 FP64 数据。矩阵向量合并成 `[数量, 行数, 列数]` 数组，
形状不一致时保留各自形状。

每个构型集合按属性和类型合并连续数组，覆盖全部构型的坐标、力和晶胞。
逐构型的小型 JSON 记录切片、存在标志、属性、标签和拓扑；目录本身为一个 `u8` 块。
解码时仅在恢复该集合期间缓存其数值块。选中构型快照与历史仍独立保存。

读取只检查版本、重复键、引用、类型、维度、长度、切片范围、偏移、重叠及解压结果。
不计算或验证文件、JSON 或数据块校验和；不会检测仍满足结构约束的任意位变化。
推理不读取 AL 数据；完整 `model_validate` 解码全部块和模型并恢复嵌入的 AL。
默认 JSON 上限为 64 MiB，单个解码块上限为 8 GiB。

写入使用目标目录下的临时文件，检查写入和关闭错误，fsync 后原子替换；
不再重新打开成品文件计算摘要。发布失败时保留旧目标。
同一目标只能有一个写入者，继续沿用 rank-zero 保存及 CLI 集合错误传播。

## 模型节点

`models` 将 ID 映射到包含 `type`、`version: 1`、`species`、`species_mode`、`common` 和 `children` 的节点。各节点按需增加 `configuration`、`parameters`、`state` 或 `resources`。子节点引用文件内 ID，不引用输入路径。遍历模型图时检查缺失引用和环。

元素类型从零编号，每个类型对应 `{ "id": 0 }`，仅在已知时增加 `symbol`。`mindist` 声明 `species_mode: "type_independent"`。显式外部映射必须与已知元素一致；未提供化学符号时仍保留明确的类型顺序。

v1 codec 覆盖 `mtp`、`mtp_m`、`mtpnn1`、`mtp_les`、`hmtp`、`hybrid`、`hybrid_scale`、`eamalloy`、`eamfs`、`eampoly`、`lj612`、`zbl`、`sw`、`tersoff`、`pair_table`、`threebody_table`、`mindist`、`descriptor_linear`、`openkim`、`acelmp` 和 `shellfile`。

MTP 节点保存径向基定义、各基缩放、矩规则、明确的系数顺序和具名的头部/核字段。`mtp_m` 元数据区分 li/q2/fs/n1/n2，并记录网络维度和激活函数。数值数组不依赖 Eigen 或对象内存布局。现有 MTP 解析器读取这些字段在内存中生成的表达，不重新打开原模型。

PolyB 和 PolyBCos 径向基可包含预计算的正交化矩阵和归一化值。读取时会恢复这些数值，包括按有序元素对拆分的径向基；不含这些数值块的旧 GPM 文件仍可读取，并在本地初始化径向基。

其他 codec 保存有效样条表、函数参数、描述符配置、NN 归一化与权重或有序子模型。MTM 描述符内嵌依赖的 MTP；混合势保留训练标记和特征过渡参数，原有的计算与训练限制保持不变。

`EAMZhou2004::SaveParameterSet` / `LoadParameterSet` 保存独立的 `eam_zhou2004_parameters` 对象，包含活跃元素和具名参数。它是参数集文件，不是新增的可执行势工厂类型；`model_info` 和 `model_validate` 支持它。

KIM 节点声明模型标识并保存有效参数数组。ACE 内嵌已加载的基资源，仍需 PACE 运行库。Shell 内嵌捕获的脚本和显式列出的输入资源；用 `shellfile:dependencies` 声明程序依赖，用 `shellfile:resource:relative/name source/path` 声明输入。资源名不得逃出任务目录。解包后的脚本与所加载势对象的生命周期一致；显式计算 shell 模型时仍会运行这些脚本。

## 主动学习

`active_learning` 保存版本、方法、`target_model` 逻辑引用及唯一的 `binding_signature`。
纯推理文件不包含签名。签名覆盖有效模型配置、元素顺序、数值参数、builder 定义，
以及影响矩阵的 `weight`、`pow_scale`、`init_mode`、`init_value`。
它不覆盖历史/候选池的全部内容，也不包含块 ID、偏移或压缩字节。
没有顶层 `model_fingerprint`、AL 模型指纹或独立 builder 指纹。

该签名用于防止给训练后的筛选状态绑定其他模型，不是文件完整性校验。
模型修改后必须重建筛选状态；评分和保存仍检查运行期模型身份，批量操作在入口检查，
避免上下层重复校验。导出路径和设备设置仍是运行时选项。

MaxVol 保存 builder 设置、E/F/virial/site 权重、行缩放、初始化、活动行和来源、
选中构型、历史、进度及终止状态。矩阵工作区、分解、线程缓存和 mmap 文件在恢复时重建。
策略还保存阈值、间隔、计数器及候选池。MD 积分器、优化器和标注调度器检查点不属于此格式。
旧 `.sel` 输入仍可结合匹配模型重建并验证行；这与拒绝旧 GPM 主版本是两件事。


## 单文件筛选与训练输出

`maxvol_select` 对依赖模型的构建器默认保存 `<potential>.maxvol.gpm`，
`--state-output` 可指定其他 GPM 路径或旧式 `.sel` 路径。独立描述符仍默认输出 `.sel`。

```sh
imrmlp maxvol_select --potfile model.gpm --dtype pot_efs --pool train.cfg --state-output session.gpm
imrmlp maxvol_sample --state-file session.gpm --input candidates.xyz
imrmlp maxvol_sample --state-file session.gpm --dump-state true
imrmlp maxvol_sample --state-file session.gpm --extract-configs history.xyz
imrmlp maxvol_extend --potfile session.gpm --candidates candidates.xyz --state-output continued.gpm
```

恢复时使用嵌入模型及元素顺序，无需原模型或 `index.map`；显式提供的模型、元素顺序或筛选配置必须一致。
继续筛选保留会话策略、候选池及可选扩展。模型改变后必须重建筛选状态。

`trainmtp`、`trainnlpot` 和 `train_linear` 默认输出 `_Trained_Potential_.gpm`。
普通训练输出及默认检查点只含模型，去除输入模型中过期的 AL 状态。
使用 `--save-maxvol-state` 可在训练结束后用完整训练集重建 MaxVol，并嵌入最终 GPM；
构建器和选择器分别通过 `builder:`、`selector:` 设置。重建失败时保留原最终文件。
显式指定 `--output model.mtp` 和 `fit:save_to` 的原生格式路径仍受支持；
`--save-maxvol-state` 要求最终输出以 `.gpm` 结尾。

## Legacy 持久化模式

`trainmtp`、`trainnlpot`、`train_linear`、`maxvol_select`、`maxvol_sample`、
`maxvol_extend` 均支持 `--legacy`，恢复引入 GPM 工作流前的持久化默认行为。
配置文件可使用顶层键 `legacy true`；命令行 `--legacy=false` 优先于配置。
开关只作用于当前命令调用。

| 命令 | `--legacy` 下的行为 |
|---|---|
| `trainmtp` | 最终输出 `_Trained_Potential_.mtp`；默认 `fit:save_to potential.mtp` |
| `trainnlpot`、`train_linear` | 最终输出 `_Trained_Potential_`；默认 `fit:save_to potential_fitted` |
| 训练检查点 | 恢复各优化器原来的原生格式文件名和迭代编号约定 |
| `maxvol_select` | 默认写 `<potential>.sel`，沿用已有 v10 二进制读写器 |
| `maxvol_sample` | 默认只读 `<potential>.sel`，不自动选择 GPM sidecar 或内嵌 MaxVol 状态 |
| `maxvol_extend --state-file old.sel` | 用外部模型恢复状态，默认覆盖原 `.sel`；显式 `--state-output` 可另存 |
| `maxvol_extend --train train.cfg` | 恢复原来的三阶段流程，默认不额外保存状态；可显式指定 `--state-output old.sel` |

显式指定的原生输出和检查点路径保持有效。legacy 模式拒绝 GPM 状态路径以及
以 `.gpm` 结尾的训练输出、检查点路径，避免文件名与实际格式不一致。
`--save-maxvol-state` 表示 GPM 内嵌状态，与 `--legacy` 冲突；
需要外部 `.sel` 时，在训练后执行 `maxvol_select --legacy`。
扩展没有新构型时保留原结构输出文件，但显式续读的状态仍会更新。

```sh
imrmlp trainmtp --legacy --settings fit.ini --do-lbfgs
imrmlp maxvol_select --legacy --potfile _Trained_Potential_.mtp --pool train.cfg
imrmlp maxvol_sample --legacy --potfile _Trained_Potential_.mtp --input candidates.xyz
imrmlp maxvol_extend --legacy --potfile _Trained_Potential_.mtp --state-file _Trained_Potential_.mtp.sel --candidates candidates.xyz
```

此模式只切换持久化约定，不改变训练或选样算法。GPM 主版本 1 不受支持。

## 将 GPM 导出为原生模型文件

`model_convert --legacy` 将参数完整的 GPM 模型转换为原来的原生格式。
不需要训练数据，也不负责初始化空白 MTP 模板。不传 `--legacy` 时仍输出 GPM。

```sh
imrmlp model_convert --input session.gpm --legacy --output model.mtp
imrmlp model_convert --input session.gpm --legacy --output model.mtp --profile inference
```

默认 `preserve` 同时将内嵌 MaxVol 状态导出为 `model.mtp.sel`。
`resume` 要求存在兼容的 MaxVol 状态；`inference` 只输出模型，并删除同一目标位置
已有的 `model.mtp.sel`。其他 AL 方法无法表示为 `.sel`，除非选择 `inference`，
否则明确报错。AL 策略、候选队列及 GPM 扩展不在原生导出范围内。

GPM 元素表完整时额外生成 `model.mtp.index.map`；`model.mtp.ini` 记录原生模型
重载配置。使用该设置文件时，在导出目录运行命令。MTPNN1 在输出前缀后添加
`.mtp`、`.nn`、`.sc`；HMTP 添加 `_subN`。MTP-M 所需的 `extmodel` 记录在设置文件中。

`hybrid` 和 `hybrid_scale` 在指定输出路径写清单，将所有子势（包括固定子势）
保存为 `<output>.potN`。清单使用相对子势路径，保留训练标志与混合配置；移动时
需一起移动整套导出文件。基于 MTM 的 `descriptor_linear` 还会导出
`<output>.descriptor.mtp`，通过相对模型文件的路径引用。

原生导出支持 `mtp`、`mtp_m`、`mtp_les`、`mtpnn1`、
`hmtp`、`hybrid`、`hybrid_scale`、`eam_trainable`、`eampoly`、`lj612`、`zbl`、
`mindist`、`sw`、`tersoff`、`pair_table`、`threebody_table`、`descriptor_linear`。
没有完整可重载原生写入器的模型（包括 MEAM、EAM alloy/FS 表格、committee 和
外部运行时模型）明确报错，不进行近似或重新拟合。重载后的推理数据须与来源
一致，无法由原生格式精确表示的数据（例如自定义的预计算多项式矩阵）会被拒绝。
此校验不包含训练正则项和单次计算的累加标志。

对于 `pot_efs` MaxVol 状态，使用导出的原生模型和状态：

```sh
imrmlp maxvol_sample --potfile model.mtp --pottype mtp --dtype pot_efs \
  --state-file model.mtp.sel --index model.mtp.index.map --input candidates.xyz
```

配套设置文件也记录恢复的 builder、selector 和状态路径：

```sh
imrmlp maxvol_sample --settings model.mtp.ini --input candidates.xyz
imrmlp maxvol_extend --settings model.mtp.ini --candidates candidates.xyz
```

旧 CLI 的 builder 类型应与选样时一致；混合势还需指定对应 `--pottype`。
原生输出路径不能以 `.gpm` 结尾。所有输出先暂存并重载校验，再发布；发生可报告
的发布错误时回滚已有输出。多个文件的发布期间，并发读取方应等待转换完成。
