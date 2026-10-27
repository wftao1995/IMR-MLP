# 自包含的 MTP 训练示例

[English](README.md) · [编译前提](../../docs/guides/installation_zh.md)

示例生成 20 个三原子构型，用一个小型合成 MTP 产生能量、力和应力标签：
16 个用于训练，4 个用于独立验证。元素名 `Al` 仅演示类型映射，参考势和拟合势
均不代表真实铝材料。

## 运行

完成[安装指南](../../docs/guides/installation_zh.md)的 CPU 构建和运行环境设置后，
在仓库根目录运行；如果安装页已成功生成 `build/quickstart/`，直接跳到下一节查看结果：

```sh
export PATH="$PWD/build/release:$PATH"
export LD_LIBRARY_PATH="$PWD/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 examples/quickstart/run.py
```

仅依赖 Python 标准库和已编译 CLI，不需要 Python 扩展、ASE、LAMMPS 或外部
数据。默认输出到本仓库的 `build/quickstart/`，若目录非空会拒绝覆盖。
可指定其他程序或输出目录：

```sh
python3 examples/quickstart/run.py \
  --imrmlp /absolute/path/to/imrmlp --work /tmp/imrmlp-demo
```

## 执行过程与输出

脚本生成 `reference.mtp`、`template.mtp`、`index.map`、构型及 `fit.ini`，
随后在**输出目录内**执行以下步骤。这里的 `imrmlp` 指上面指定的可执行程序：

```sh
imrmlp calc_efs train-input.cfg train.cfg --potfile reference.mtp --index index.map
imrmlp calc_efs valid-input.cfg valid.cfg --potfile reference.mtp --index index.map
imrmlp trainmtp fit.ini --linear-fit
cp _Trained_Potential_.mtp fitted.mtp
imrmlp calc_err --input valid.cfg --potfile fitted.mtp --index index.map --output-summary errors.txt
imrmlp calc_efs valid-input.cfg predicted.cfg --potfile fitted.mtp --index index.map
```

本例固定径向参数，只拟合线性系数；初始线性系数与参考势不同。
`trainmtp` 在工作目录写出最终 `_Trained_Potential_.mtp`，`fit:save_to`
用于训练器检查点，不会重命名这个最终文件。脚本显式复制为 `fitted.mtp`。

每个阶段会显示 `OK`。查看 `errors.txt` 和各阶段 `.log`；该精确可表示的
合成问题应得到接近数值精度的 E/F/应力误差，报告按六位小数显示。
真实材料的数据会有拟合残差，训练误差很低也不能证明模型具有可迁移性。

| 文件 | 从哪里来 / 用来做什么 |
| --- | --- |
| `train-input.cfg`、`valid-input.cfg` | 脚本生成的 16/4 帧几何；不带参考标签 |
| `train.cfg`、`valid.cfg` | 用合成 `reference.mtp` 标注的训练/验证数据，不能替换成无标签输入 |
| `template.mtp`、`fit.ini`、`index.map` | 初始模型、可直接使用的训练配置和元素映射 |
| `fitted.mtp` | 拟合后的模型，后续 CLI/Python/LAMMPS 练习从这里取得 |
| `predicted.cfg`、`errors.txt` | 四帧验证集的预测与误差报告；与参考 `valid.cfg` 对应 |
| 各阶段 `.log` | 脚本报错时，先查看对应阶段日志 |

这些文件都在 `--work` 目录内。手动执行上面的命令前先进入该目录；自定义 `--work`
时，其他页面中的 `build/quickstart/` 也应替换为实际路径。

## 非线性训练

在生成的目录内，可用同一模板尝试 ALGLIB L-BFGS：

```sh
imrmlp trainmtp fit.ini --do-lbfgs
```

这会覆盖默认最终快照；对比不同运行前应复制到独立文件。
生产数据准备、优化器选择和部署见[工作流程](../../docs/guides/workflows_zh.md)。
使用 Python/ASE 评估生成模型见[Python 文档](../../docs/reference/python_zh.md)，
LAMMPS 部署见[接口说明](../../LAMMPS/README_zh.md)。

想练习寻找值得标注的新构型，可继续运行自包含的
[MaxVol 选样练习](../active-learning/README_zh.md)，查看自动生成探索池的分数和候选。
想直接练习弛豫、NEB 和物性，转到[连续计算练习](../basic-calculations/README_zh.md)，
其中另附适用于稳定教学晶体的 LJ 势与全部输入。
需要把基组选择、训练/验证和后续计算串成同一条 MTP 路线时，使用
[MTP 连续练习](../mtp-workflow/README_zh.md)，其中训练的模型可直接完成教学晶体与空位任务。
