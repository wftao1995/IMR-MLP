# 示例配置文件

[English](README.md)

这里提供 `imrmlp` 子命令的配置模板和命令行参考。常用命令通常有精简的 `*-simple.ini` 和
较完整的 `*.ini`。键和值用空白分隔，不写等号；子系统使用 `fit:`、`relax:`、
`pot:`、`neb:` 或 `builder:` 等前缀。

| 用途 | 配置文件 |
| --- | --- |
| 能量、力、应力与误差 | `calc-efs*.ini`、`calc-err*.ini`、`testpot*.ini` |
| 训练 | `fit*.ini`、`fastfit*.ini`、`initmtpm*.ini`、`nnmtp*.ini` |
| 长程模型训练 | `fit-sog.ini`、`fit-sog-multichannel.ini` |
| 弛豫与驻点搜索 | `relax*.ini`、`neb*.ini`、`dimer*.ini`、`modefollowing*.ini`、`nroptimize.ini` |
| 主动学习 | `select*.ini`、`diverse-select*.ini`、`calc-fitness*.ini` |
| 结构操作 | `convstr*.ini`、`mk-*.ini`、`swap-axis*.ini`、`sameq*.ini`、`findint*.ini`、`trans-reduce*.ini`、`primi-reduce*.ini`、`split-set*.ini`、`remove-duplicate-*.ini`、`smooth-traj*.ini` |
| 势函数工具 | `adjustmtp*.ini`、`mtp2spline*.ini`、`mtpbasis*.ini`、`mtmbasis*.ini`、`pot2table*.ini` |
| 性质与声子 | `prop*.ini`、`phonopyfc*.ini`、`calc-trjavg*.ini`、`ev*.ini`、`calc-elastic*.ini`、`calc-numelastic*.ini`、`calc-fc*.ini`、`calc-ferr*.ini` |
| 混合势 | `zbl-mtp-hybrid-al/calc-efs.ini` |

这些是模板，不能在没有模型和数据时直接运行。先复制需要的配置到独立运行目录，
修改输入与模型路径，并核对物种映射。相对路径始终从启动 `imrmlp` 的工作目录
解析，即使配置文件在其他目录。从仓库根目录开始，例如：

```sh
mkdir -p build/my-efs-run
cp example-inputs/calc-efs-simple.ini build/my-efs-run/calc-efs.ini
```

修改副本，将结构、模型与 `index.map` 放到运行目录（也可使用绝对路径），然后执行：

```sh
cd build/my-efs-run
../release/imrmlp calc_efs --settings=calc-efs.ini --input POSCAR --output output-efs.cfg
```

精简 EFS 模板默认读取 `POSCAR`、`potential.mtp`、`index.map`，写入 `output-efs.cfg`。
需要附带输入、可直接运行的首个计算时，请使用[快速入门](../examples/quickstart/README_zh.md)。
弛豫、NEB 和物性需要完整输入时，使用[连续计算练习](../examples/basic-calculations/README_zh.md)，
其中会自动生成模型、结构与对应配置，并说明输出如何传给下一步。

标有 “CLI reference only” 的文件是命令行选项参考，不能作为配置加载；
包括许多结构工具、`adjustmtp`、`mtp2spline` 和 `nroptimize`。`phonopyfc` 使用
`--settings-file` 而非 `--settings`；`pot2table` 的文件只读取势函数配置，
导出选项需在命令行指定。

`nroptimize.ini` 是参数参考，不能直接作为配置加载。`dimer.ini` 是完整 Dimer 示例，
`dimer-simple.ini` 是最小起点。命令选项见[命令参考](../docs/reference/commands_zh.md)，
配置键见[配置说明](../docs/reference/configuration_zh.md)。
