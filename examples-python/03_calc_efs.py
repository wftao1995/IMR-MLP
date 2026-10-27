# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""
03 — 能量/力/应力计算

支持势类型：mtp, eamalloy, eamfs, eampoly,
            lj612, sw, tersoff, zbl, openkim 等
"""
import numpy as np
import imrmlp
from imrmlp import Potential

# ── 构造一个简单的 Al 二聚体 ─────────────────────────────────────────
atoms = imrmlp.Atoms(
    numbers=np.array([13, 13]),
    positions=np.array([[0.0, 0.0, 0.0], [2.86, 0.0, 0.0]]),
    cell=np.eye(3) * 10.0,
    elements=["Al"],
)

# ── 加载势，计算 EFS ──────────────────────────────────────────────────
pot = Potential("Al.mtp")
# pot = Potential("Al.eam.alloy", pottype="eamalloy")  # EAM 示例

pot.calculate(atoms)

print(f"势: {pot}")
print(f"能量: {atoms.energy:.6f} eV")
print(f"力:\n{atoms.forces}")
print(f"位力张量 (eV，virial = -V × 应力):\n{atoms.stress}")
print(f"site energies: {atoms.site_energies}")

# ── OpenMP 并行计算 (大数据集) ────────────────────────────────────────
imrmlp.set_num_threads(4)
# pot.calculate(atoms, use_omp=True)

# ── 只算能量 (快) ────────────────────────────────────────────────────
e = pot.calculate_energy(atoms)
print(f"能量(仅): {e:.6f} eV")
