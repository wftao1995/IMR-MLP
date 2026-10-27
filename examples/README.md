# Examples

[中文](README_zh.md)

| Example | Inputs and purpose |
| --- | --- |
| [Quickstart / 快速入门](quickstart/README.md) · [中文](quickstart/README_zh.md) | Self-contained synthetic MTP labeling, fitting and validation; no external data needed |
| [Calculation walkthrough](basic-calculations/README.md) · [中文](basic-calculations/README_zh.md) | Supplied LJ potential, structures and settings for relaxation, endpoint preparation, NEB, properties and optional DOS |
| [MTP walkthrough](mtp-workflow/README.md) · [中文](mtp-workflow/README_zh.md) | Basis generation, synthetic reference labels, fitting and validation selection, then the same MTP for relaxation, NEB, properties and optional DOS |
| [Active-learning exercise](active-learning/README.md) · [中文](active-learning/README_zh.md) | Self-contained geometry-only MaxVol selection, score table and threshold exercise |
| [MTP GPU Al/Ni](mtp-gpu-alni/README.md) · [中文](mtp-gpu-alni/README_zh.md) | Supplied structures; user must provide a compatible Al/Ni potential, GPU build and optional training data |
| [LES response](les-response/README.md) · [中文](les-response/README_zh.md) | Response-analysis inputs and limitations; external model/reference data required |

Start with the quickstart to verify an installation. Its model is synthetic and
must not be used for material predictions. The active-learning exercise likewise
uses a synthetic model. Material research examples describe specific
research workflows; cluster paths and historical timings are not portable defaults.
Build artifacts, production potentials and private reference datasets are not
bundled. See each example's input requirements and licensing notes.
