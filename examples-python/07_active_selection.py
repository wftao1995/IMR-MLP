# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""07 — 使用描述符矩阵进行 MaxVol 和 CUR 选择。

每行代表一个候选结构，每列代表一个特征；此处使用合成矩阵演示接口。
结构级 MaxVol 及 FPS 的配置见 docs/reference/python_zh.md。
"""
import numpy as np
from imrmlp import MaxVol, cur_select

# 实际应用中用描述符矩阵替换此合成数据。
rng = np.random.default_rng(42)
features = rng.normal(size=(100, 16))

# MaxVol 选择与特征维数相等的行，并计算所有行的外推分数。
selector = MaxVol()
selected = selector.select_rows(features, k=features.shape[1])
scores = selector.compute_scores(features, selected)
print(f"MaxVol 索引: {selected}")
print(f"最大分数: {scores.max():.4f}")

# CUR 选择 20 行，计算相对重构误差。
result = cur_select(features, k=20, deterministic=True, compute_error=True)
print(f"CUR 索引: {result.selected_indices}")
print(f"CUR 重构误差: {result.reconstruction_error:.4f}")
