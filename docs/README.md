# Documentation / 文档导航

Read the Markdown directly from this repository, or build the bilingual
Sphinx manual with navigation and search. Doxygen provides the C++ API reference.
English and Chinese user guides describe the same source
snapshot. Model-specific research notes may be available in only one language.

可以直接阅读 Markdown，也可以生成带导航、搜索和阅读主题的 Sphinx 双语手册；
C++ API 参考由 Doxygen 生成。
中英文用户指南面向同一份源码；部分模型专题目前仅有一种语言。

## Reading order / 阅读顺序

| Section / 分区 | English | 中文 |
|---|---|---|
| User guides / 使用指南 | [User guides](guides/index.md) | [使用指南](guides/index_zh.md) |
| Reference / 参考手册 | [Reference](reference/index.md) | [参考手册](reference/index_zh.md) |
| Advanced usage / 进阶使用 | [Advanced usage](advanced/index.md) | [进阶使用](advanced/index_zh.md) |
| Theory and algorithms / 理论与算法 | [Theory and algorithms](theory/index.md) | [理论与算法](theory/index_zh.md) |
| Development / 开发文档 | [Development](development/index.md) | [开发文档](development/index_zh.md) |

Start with [getting started](guides/getting-started.md) to choose between using
an existing model and training one. [Installation](guides/installation.md) and
the [self-contained quickstart](../examples/quickstart/README.md) verify the setup.
The [calculation walkthrough](../examples/basic-calculations/README.md) supplies
inputs and explicitly passes outputs between relaxation, NEB and property tasks.
The [MTP walkthrough](../examples/mtp-workflow/README.md) adds basis generation,
reference labeling, model selection and final testing before those calculations.
For calculations, follow [relaxation](guides/relaxation.md), [NEB](guides/neb.md),
[crystal properties](guides/properties.md) or [active learning](guides/active-learning.md).
Use the [FAQ](guides/faq.md) for choices and [troubleshooting](guides/troubleshooting.md) for symptoms.

先读[开始使用](guides/getting-started_zh.md)，区分已有模型做计算与自行训练两条路线。
[安装](guides/installation_zh.md)和[自包含示例](../examples/quickstart/README_zh.md)用于验证环境；
[连续计算练习](../examples/basic-calculations/README_zh.md)提供完整输入，明确衔接弛豫、NEB 与物性的输出。
[MTP 连续练习](../examples/mtp-workflow/README_zh.md)还串起基组生成、参考标注、训练选型与最终测试。
日常任务按[弛豫](guides/relaxation_zh.md)、[NEB](guides/neb_zh.md)、
[晶体性质](guides/properties_zh.md)或[主动学习](guides/active-learning_zh.md)的步骤运行。
“该用哪个”见 [FAQ](guides/faq_zh.md)，按症状排错见[故障排查](guides/troubleshooting_zh.md)。

## Build the HTML manual / 生成 HTML 手册

Install Python 3.12+, Node.js/npm, Doxygen and Graphviz, then run from the
repository root. The first command installs documentation tools under `build/`:

安装 Python 3.12+、Node.js/npm、Doxygen 和 Graphviz 后，在仓库根目录执行；
`doc-deps` 只需首次使用或更新依赖时运行：

> Python 3.12+ is required **only for building the manual**. The package and its
> bindings support Python 3.10+ (see the installation guide).
>
> Python 3.12+ **仅用于构建手册**；Python 包及其绑定支持 Python 3.10+（见安装指南）。

| Target / 目标 | Output / 产物 |
|---|---|
| `make doc-html` | Doxygen C++ API: `docs/build/html` (English) and `docs/build_zh/html` (Chinese) |
| `make doc-manual` | Bilingual Sphinx manual (includes the API): `docs/manual-html/{en,zh}` |
| `make doc-zh-html` | Chinese API Doxygen under `docs/build_zh/html` (depends on `doc-html`) |
| `make check-docs` | Catalog, link, command/option and bilingual checks |
| `make check-doc-html` | Generated HTML, search index and math checks |

`check-docs` runs `check_doc_commands.py` and `check_doc_locale.py` in advisory
mode by default. Run `make check-docs DOC_CHECK_STRICT=1` to fail on command,
option or alias differences and links that bypass an existing translation.
These checks cover entry points, examples and interface guides as well as the
manual. They check names and structure; they cannot verify a parameter's meaning
or guarantee that an example runs in your environment.

默认会报告命令与双语差异；使用 `make check-docs DOC_CHECK_STRICT=1` 可让命令、
选项、别名不一致及跳过已有译文的链接导致检查失败。检查覆盖文档入口、示例与
接口说明，但名称与结构检查不能替代参数语义核对和实际运行。

```sh
make doc-deps
make doc-manual
python3 -m http.server 8765 --bind 127.0.0.1 --directory docs
```

Open <http://127.0.0.1:8765/manual-html/zh/index.html> for Chinese, or
<http://127.0.0.1:8765/manual-html/en/index.html> for English. Press Ctrl+C to stop.
The manual reuses the Markdown and `catalog.json`; there is no second copy to edit.
Its MathJax assets are local. Pages without a translation show a notice and the original.
Use the language menu in the upper-right corner to switch to the same page in the other language.

中文入口：<http://127.0.0.1:8765/manual-html/zh/index.html>；
英文入口：<http://127.0.0.1:8765/manual-html/en/index.html>。按 Ctrl+C 停止服务。
正文继续修改原有 Markdown，分类与顺序由 `catalog.json` 管理；
公式使用本地 MathJax，缺少译文的页面会标明原文语言。
右上角的语言下拉菜单可切换到同一页面的另一语言版本。

`make doc-manual` also rebuilds Doxygen so API links and attachments are current.
To build only Doxygen, run:

```sh
make doc-html
```

This command builds both API languages. Open `docs/build/html/index.html` or `docs/build_zh/html/index.html`.
C++ API cross-references such as `@ref` are resolved in the generated manual;
ordinary Markdown links work directly in the repository. `make doc-pdf` also
requires a suitable TeX installation; verify the resulting PDF and build log,
since that legacy target does not propagate every LaTeX failure.

HTML 文档不需要先编译主程序。文档生成目录不应提交。
文档与示例检查见[开发说明](../CONTRIBUTING.md)。

## Maintaining the structure / 维护结构

`docs/catalog.json` defines the five sections and their reading order. Each
English/Chinese pair is registered once. Generated indexes and Doxygen input
lists stay in version control; build output remains under `docs/build`,
`docs/build_zh` and `docs/manual-html`. Homepage templates and styles live in
`docs/manual`. Build instructions in this section are for documentation
maintainers; reading the Markdown or running IMR-MLP does not require these tools.

```sh
python3 docs/tools/doc_catalog.py --write
make check-docs
make doc-manual
make check-doc-html
```

新增页面后先登记目录，再生成双语索引并检查链接。专题文档进入 `advanced`，
可运行示例保留在代码目录；手册描述当前行为，测试结果记录在提交或审查说明中。
