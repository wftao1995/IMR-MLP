# IMR-MLP

[English](README.md) · [文档导航](docs/README.md) · [LAMMPS 接口](LAMMPS/README_zh.md) · [Python](README-python_zh.md)

IMR-MLP 是用于训练和部署机器学习原子间势的 C++23 框架，提供 MTP 拟合、
能量/力/应力计算、主动学习、结构弛豫与分析、Phonopy 接入、Python 绑定和
MTP 的 LAMMPS 接口。

由中国科学院金属研究所材料人工智能研究部开发。

主程序为 `imrmlp`，共享库为 `libimrmlp.so`，Python 包为 `imrmlp`。

## 编译并运行第一个示例

默认 Linux 构建需要支持 C++23 的编译器、GNU Make 4.3+、Eigen、Intel
oneMKL、ALGLIB 和 FFTW。默认开启 OpenKIM 和 gzip；下方首次构建示例用 `USE_KIM=0` 关闭可选 KIM。
依赖路径、可选开关及 MPI/CUDA/HIP 模式见[安装指南](docs/guides/installation_zh.md)。

安装依赖后，在仓库根目录运行。第一行适用于 oneAPI 常见组件目录布局，
其他安装路径或统一布局按安装指南替换环境初始化脚本：

```sh
source /opt/intel/oneapi/setvars.sh
make -j2 imrmlp USE_KIM=0
export PATH="$PWD/build/release:$PATH"
export LD_LIBRARY_PATH="$PWD/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
imrmlp --version
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 examples/quickstart/run.py
```

示例自动生成小型合成数据、拟合 MTP，并在独立构型上评估误差。仅需要已编译
的 CLI 和 Python 标准库，结果及日志写入 `build/quickstart/`。再次运行时使用
新的 `--work` 目录。这是操作流程示范，不是可用于材料研究的势函数。
逐步说明见[快速入门](examples/quickstart/README_zh.md)。

## 按任务查阅

| 任务 | 文档 |
| --- | --- |
| 第一次使用：选任务、准备文件并做单点计算 | [开始使用](docs/guides/getting-started_zh.md) |
| 固定胞或变胞结构优化 | [弛豫引导](docs/guides/relaxation_zh.md) |
| 两个端点之间的迁移或反应势垒 | [NEB 引导](docs/guides/neb_zh.md) |
| 声子、弹性与状态方程 | [晶体性质引导](docs/guides/properties_zh.md) |
| 安装 CPU、MPI 或 GPU 版本 | [安装与构建](docs/guides/installation_zh.md) |
| 配置语法与物种映射 | [配置说明](docs/reference/configuration_zh.md) |
| 准备数据、训练并验证势函数 | [工作流程](docs/guides/workflows_zh.md) |
| 查询命令参数 | [命令参考](docs/reference/commands_zh.md)、`imrmlp COMMAND --help` |
| 在 LAMMPS 中使用 MTP | [接口安装与使用](LAMMPS/README_zh.md) |
| 使用 Python 或 ASE | [Python 与 ASE](README-python_zh.md) |
| 使用 LES/SOG 模型 | [LES](docs/reference/les-v2_zh.md)、[SOG](docs/reference/sog_zh.md) |
| 选样或在 LAMMPS MD 中在线采样 | [主动学习引导](docs/guides/active-learning_zh.md)、[LAMMPS MaxVol](LAMMPS/USER-MLIP/maxvol/README_zh.md) |
| 搜索驻点和过渡路径 | [NR、Dimer、MODECAR 与 NEB](docs/advanced/saddle-workflows_zh.md) |
| 查询命令、单位、势类型等“该用哪个”问题 | [FAQ](docs/guides/faq_zh.md) |
| 处理编译与运行错误 | [故障排查](docs/guides/troubleshooting_zh.md) |
| 运行测试、提交修改 | [开发与测试说明](CONTRIBUTING.md) |

配置文件通过空白分隔键和值，**不要写等号**：

```text
potfile model.mtp
fit:bfgs_maxiteration 1000
```

非线性 MTP 训练需要显式选择优化器：

```sh
build/release/imrmlp trainmtp train.ini --do-lbfgs
```

## 许可与数据

项目自身代码使用 [MIT](LICENSE)，版权为 2026 IMR-MLP contributors。
LAMMPS 派生接口保留 [GPLv2](LAMMPS/COPYING.GPL-2.0)，其他依赖与数据
保留原许可，详见[第三方许可说明](THIRD_PARTY_NOTICES.md)。ALGLIB、FFTW
的 GPL 版本会影响链接产物的再分发条件。

仓库提供合成测试和可运行示例，外部研究数据不随源码提供。
