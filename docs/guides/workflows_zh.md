# 训练、评估与部署 {#guide_workflows}

[English](workflows.md) · [可运行的快速入门](../../examples/quickstart/README_zh.md)

@tableofcontents

本页面向需要训练模型的用户。已有合适模型、只想计算能量、弛豫或 NEB 时，
从[开始使用](getting-started_zh.md)选任务，不必先重做训练。

**第一次训练且没有数据？** 先完成[自包含训练入门](../../examples/quickstart/README_zh.md)，
直接使用它生成的 `fit.ini`、`train.cfg`、`valid.cfg` 和 `template.mtp`。
本页的三集合、EXTXYZ 配置用于自己的参考数据；入门示例只有训练/验证两份 CFG，
不是已经准备好的最终测试集；沿用该练习生成的配置即可。

需要完整的“基组生成 → 参考标注 → 训练/验证选型 → 最终测试 → 同一模型计算”时，
使用[自包含 MTP 连续练习](../../examples/mtp-workflow/README_zh.md)，其中提供全部模型与结构来源。

以下命令假定 `imrmlp` 位于 `PATH`，或已替换为对应构建版本的绝对路径。
每次计算使用独立工作目录。配置中的相对路径相对于进程工作目录解析，
不会自动相对于配置文件所在目录解析。

## 1. 确定元素映射与参考标签

`index.map` 按模型内部物种顺序，每行写一个元素符号。例如 Al/Ni 分别对应
物种索引 0、1：

```text
Al
Ni
```

预处理、训练和部署必须使用同一顺序。LAMMPS atom type 从 1 开始，
使用数字映射时模型 species 索引从 0 开始；新 LAMMPS 输入推荐按元素名映射。错误映射可能仍给出数值正常但
物理含义错误的结果。

逐个转换参考计算，然后拼接 CFG 记录：

```sh
imrmlp convstr OUTCAR reference.cfg --index index.map
imrmlp convstr vasprun.xml reference-xml.cfg --index index.map
cat reference.cfg reference-xml.cfg > dataset.cfg
```

以上是用户参考计算的文件名占位。XDATCAR 等仅含几何的轨迹不会因格式转换
自动获得能量、力或应力标签。应核查参考计算的电子收敛与标签单位，不能将
离子优化停止阈值当作电子/力计算精度。

位置和晶胞使用 Å，总能量为 eV，力为 eV/Å。原生广延应力张量使用 eV；
`calc_err` 还报告按体积归一化后的 GPa 应力。从其他代码导入时应遵循 IO
规定的符号、分量次序和归一化约定。

## 2. 划分训练、验证与最终测试集

尽量按轨迹、相、成分或生成计算划分，避免相邻 MD 帧泄漏到不同集合。
验证集用于选择模型和设置，最终测试集应在这些选择结束后评估。
简单演示可使用：

```sh
imrmlp split_set dataset.cfg split_ --nsplit 10 --shuffle yes
cat split_0.xyz split_1.xyz split_2.xyz split_3.xyz split_4.xyz \
    split_5.xyz split_6.xyz split_7.xyz > train.xyz
cp split_8.xyz valid.xyz
cp split_9.xyz test.xyz
```

`split_set` 写出 extxyz 文件；等分模式在可选打乱后按轮转方式分配。
也可在独立运行目录中直接生成约 80% / 10% / 10% 的子集：

```sh
imrmlp split_set dataset.cfg split_ --ratios=8,1,1 --shuffle yes
cp split_0.xyz train.xyz
cp split_1.xyz valid.xyz
cp split_2.xyz test.xyz
```

比例模式在可选打乱后按连续分段分配，整数子集大小采用最大余数法取整。
当前 CLI 没有随机种子参数，复现时需保存实际划分结果。不要将 extxyz 直接改名为 `.cfg`
后再依赖扩展名自动识别格式。

## 3. 选择匹配的模板

模板应包含预期的元素数与顺序、径向基、截断和矩基，并保留未经修改的副本。
[快速入门](../../examples/quickstart/README_zh.md)提供用于学习流程的小型合成模板。
用 `mtmbasis` 生成候选池、再用 `fastfit` 精简的过程见[基组指南](basis-selection_zh.md)；
生成器与径向基设置见[配置说明](../reference/configuration_zh.md)。生成器配置本身不是
完整可部署的模型。

`initmtpm` 用于 MTP-M 非线性头初始化，不是普通 MTP 训练必需的初始化步骤，
不要直接套用于普通模板。

## 4. 显式选择训练方式

保存 `train.ini`，键和值用空白分隔：

```text
index index.map
potfile template.mtp
trainset train.xyz
validset valid.xyz
fit:force_weight 0.05
fit:stress_weight 0.001
fit:weight_L2 1e-8
fit:save_to checkpoint.mtp
```

这些权重以默认能量权重 1 为基准，仅为示例；应结合标签质量、归一化和目标物理量选择。

```sh
# ALGLIB L-BFGS：联合优化径向与线性系数
imrmlp trainmtp train.ini --do-lbfgs
cp _Trained_Potential_.mtp fitted.mtp

# 另一种选择：固定径向函数，仅拟合线性系数
imrmlp trainmtp train.ini --linear-fit
cp _Trained_Potential_.mtp fitted-linear.mtp
```

**必须显式选择优化器或 `--linear-fit`。** 裸命令 `trainmtp train.ini` 不会
自动选择非线性优化方法。最终快照固定写入工作目录的 `_Trained_Potential_.mtp`；
`fit:save_to` 指定训练器检查点，不会重命名这个命令级最终输出。
使用独立运行目录可避免覆盖之前的模型。

非线性优化中可选的线性重拟合见[线性精修指南](../advanced/linear-refit_zh.md)。
`trainnlpot` 支持更多模型与设备路径，`train_linear` 则是另一套通用线性拟合入口。
这些命令接受的训练设置不同，复用配置前应查看所选命令的 `--help`。

## 5. 验证并保留复现信息

```sh
imrmlp calc_err --input valid.xyz --potfile fitted.mtp --index index.map \
  --output-summary validation.txt
imrmlp calc_err --input test.xyz --potfile fitted.mtp --index index.map \
  --output-summary test.txt
imrmlp calc_efs test.xyz predicted.xyz --potfile fitted.mtp --index index.map
imrmlp --version > version.txt
sha256sum fitted.mtp template.mtp train.xyz valid.xyz test.xyz > inputs.sha256
```

分别报告每原子能量、力和应力误差，检查异常构型、不同相/成分和目标性质。
在预期体积、温度和缺陷范围验证；静态 E/F 误差不足以证明长时间 MD 稳定性。
保留元素映射、配置、模型、划分后的数据、日志、构建模式和依赖版本。

### 小例子：怎样读误差报告？

在[自包含入门示例](../../examples/quickstart/README_zh.md)的 `errors.txt` 中，
能量统计应覆盖 4 个验证构型，力统计覆盖 12 个原子；误差接近数值精度，是因为
这个合成问题能被所选模型精确表示。真实材料不能以“也应全部为零”为目标。
先核对参与统计的数量和标签是否齐全，再读各类误差。

例如，`Energy per atom` 的 RMS 为 `0.004 eV`，就是 `4 meV/atom`；
`Forces` 的 RMS 为 `0.08 eV/Å`，就是 `80 meV/Å`。报告中的 `Forces` 按力差向量
统计，`Forces components` 按分量统计，比较不同模型时应使用同一行、同一验证集。
平均值较低但最大误差很大时，还需找出对应的相、缺陷或高温构型，决定是否补充
参考数据；降低训练误差不能替代这一步。

## 6. 用训练好的模型做计算

先完成[单点计算](getting-started_zh.md)，核对同一结构的 E/F/应力；再根据目标选择：

- [结构弛豫](relaxation_zh.md)：固定胞或变胞、轨迹、退出码和收敛判据。
- [NEB 路径](neb_zh.md)：端点准备、原子对应、路径初始化、势垒和续算。
- [晶体性质](properties_zh.md)：声子、弹性、状态方程及各自的数值收敛。

模型在验证集上的平均误差，不能替代目标计算条件下的稳定性和物性验证。

## 7. 部署同一模型

先[安装 LAMMPS 接口](../../LAMMPS/README_zh.md)，再使用：

```text
units metal
newton on
pair_style mlip/opt
pair_coeff * * fitted.mtp Al Ni
```

这里按 LAMMPS 类型顺序写 Al/Ni；单元素 Al 模型仅需 `Al`。旧模型无元素元数据时
改用已确认的数字 species 映射。该片段放在建盒/读入原子后，完整输入见接口指南。
先用 `run 0`，在同一几何、晶胞、映射和边界条件下与 CLI 比较 E/F。
样条模型使用 `mlip/spline`；LES/SOG 使用专用样式；热流计算需采用
transport pair style。

Python/ASE 使用[对应指南](../reference/python_zh.md)，显式指定模型顺序的 `elements`。
主动学习先读[选样循环引导](active-learning_zh.md)，原理见 [MaxVol](../theory/maxvol_zh.md)。
外推等级依赖具体模型与活动集，本身不是经过校准的误差估计。

### 过渡路径与变胞计算

基础路径流程见 [NEB 引导](neb_zh.md)；变胞与其他鞍点方法见[驻点专题](../advanced/saddle-workflows_zh.md)。
构建路径前，先在相同外压和约束下弛豫端点。应收敛镜像数量，并独立检查鞍点；
NEB 投影力很小本身不能证明找到了过渡态。

## 更多训练与选基方式

- [基组生成、拟合与选择](basis-selection_zh.md)
- [fastfit 成本选基](../advanced/fastfit-cost-selection_zh.md)
- [HVP 标签与拟合](../advanced/hvp-training_zh.md)
- [MTP 变量投影](../advanced/mtp-native-varpro_zh.md)
