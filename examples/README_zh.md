# 示例

[English](README.md)

| 示例 | 所需输入与用途 |
| --- | --- |
| [快速入门](quickstart/README_zh.md) | 自包含的合成 MTP 标注、拟合与验证，无需外部数据 |
| [连续计算练习](basic-calculations/README_zh.md) | 自动准备 LJ 势、结构和配置，逐步运行弛豫、端点准备、NEB、物性与可选 DOS |
| [MTP 连续练习](mtp-workflow/README_zh.md) | 生成基组、合成参考标注、训练与验证选型，再用同一个 MTP 做弛豫、NEB、物性与可选 DOS |
| [主动学习小练习](active-learning/README_zh.md) | 自包含的 MaxVol 几何选样、分数表与阈值练习 |
| [Al/Ni MTP GPU](mtp-gpu-alni/README_zh.md) | 已提供结构；需自行准备匹配的 Al/Ni 势与 GPU 构建，可替换为实际训练数据 |
| [LES 响应](les-response/README_zh.md) | 外场及响应分析的输入和适用范围；需要外部模型与参考数据 |

首次安装后先运行快速入门。它使用合成势，不能用于材料预测；主动学习小练习也
使用合成模型。材料研究示例面向特定
研究流程，历史集群路径和耗时不能作为通用默认值。仓库不附带构建产物、生产势
或私人参考数据，运行前请核对各示例的输入要求与许可说明。
