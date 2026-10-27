# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""
06 — 描述符线性模型训练（合成数据）

先构建 Python 扩展，再从仓库根目录运行：
PYTHONPATH=. python3 examples-python/06_train.py
普通 MTP 的固定径向基拟合请使用 imrmlp trainmtp train.ini --linear-fit。
"""
import argparse
from pathlib import Path

import numpy as np
import imrmlp
from imrmlp import LinearTrainer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--work", type=Path,
                    default=Path(__file__).resolve().parents[1] / "build/examples-python/06-train",
                    help="输出目录（同名模型文件会被覆盖）")
args = parser.parse_args()
args.work.mkdir(parents=True, exist_ok=True)

# 一个物种、两个径向特征；该模型支持通用线性拟合接口。
settings = {
    "pottype": "descriptor_linear",
    "descriptor_linear:species_count": "1",
    "descriptor_linear:descriptor:dtype": "behler",
    "descriptor_linear:descriptor:behler:cutoff": "5",
    "descriptor_linear:descriptor:behler:dispcount": "2",
    "descriptor_linear:descriptor:behler:dispselect": "1 2",
    "descriptor_linear:descriptor:behler:eta": "0.3",
}
pot = imrmlp.Potential.from_settings(settings)
assert pot.supports_linear_fit()

# 合成能量严格由相同特征的线性组合生成，不代表真实 Al 势。
# 不提供力/应力标签，所以本例仅拟合能量。
training_set = []
for distance in (1.5, 2.0, 2.8, 3.7):
    atoms = imrmlp.Atoms(
        numbers=[13, 13], elements=["Al"],
        positions=[[2, 2, 2], [2 + distance, 2, 2]], cell=np.eye(3) * 20,
    )
    cutoff = 0.5 * (np.cos(np.pi * distance / 5) + 1)
    atoms.energy = 2 * cutoff * (0.7 - 0.2 * np.exp(-0.3 * distance**2))
    training_set.append(atoms)
print(f"训练集: {len(training_set)} 个结构")

# ── 训练 ──────────────────────────────────────────────────────────────
trainer = LinearTrainer(
    energy_weight=1.0,
    force_weight=0,
    stress_weight=0,
    reg_L2=1e-12,
    save_to=str(args.work / "fitted.linear"),
)
result = trainer.train(pot, training_set)

print(f"特征数: {result.feature_count}")
print(f"E  RMSE: {result.train_erms:.6f} eV/atom")
print(f"模型已保存: {args.work / 'fitted.linear'}")

# ── 两步训练：独立模型，先组装再求解 ──────────────────────────────
pot2 = imrmlp.Potential.from_settings(settings)
trainer2 = LinearTrainer(force_weight=0, stress_weight=0, reg_L2=1e-12,
                         save_to=str(args.work / "fitted2.linear"))
trainer2.assemble(pot2, training_set)
result2 = trainer2.solve()
print(f"分步训练: E_RMSE={result2.train_erms:.6f} eV/atom")

# 重载时显式指定类型；两种训练方式应重现合成标签及有限的 E/F/S。
loaded = imrmlp.Potential(str(args.work / "fitted.linear"), "descriptor_linear")
for atoms in training_set:
    predictions = []
    for model in (pot, pot2, loaded):
        predicted = atoms.copy()
        model.calculate(predicted)
        predictions.append(predicted.energy)
        assert np.isfinite(predicted.forces).all()
        assert np.isfinite(predicted.stress).all()
    np.testing.assert_allclose(predictions, atoms.energy, atol=1e-8, rtol=0)
print("OK: 一步训练、分步训练及模型重载结果一致")
