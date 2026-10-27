# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""
02 — 手动构造 / 修改结构

演示 Atoms 对象的创建和属性访问。
"""
import numpy as np
from imrmlp import Atoms

# ── 构造 FCC-Al 4 原子晶胞 ───────────────────────────────────────────
a0 = 4.05  # Å
atoms = Atoms(
    numbers=np.array([13, 13, 13, 13]),          # Al = 原子序数 13
    positions=np.array([
        [0.0, 0.0, 0.0],
        [a0/2, a0/2, 0.0],
        [a0/2, 0.0, a0/2],
        [0.0, a0/2, a0/2],
    ]),
    cell=np.eye(3) * a0,
    elements=["Al"],
)
print(f"构造完成: {atoms}")
print(f"体积: {atoms.volume:.2f} Å³")
print(f"原子序数: {atoms.numbers}")

# ── 修改属性 ──────────────────────────────────────────────────────────
atoms.energy = -14.5
print(f"设能量: {atoms.energy:.4f} eV")

# ── 坐标转换 ──────────────────────────────────────────────────────────
atoms.to_fractional()
print("转为分数坐标:")
print(atoms.positions[:2])
atoms.to_cartesian()
print("转回笛卡尔坐标:")
print(atoms.positions[:2])

# ── ASE 互转 ──────────────────────────────────────────────────────────
try:
    ase_atoms = atoms.to_ase()
    print(f"→ ASE: {ase_atoms}")

    imrmlp_atoms = Atoms.from_ase(ase_atoms)
    print(f"← imrmlp: {imrmlp_atoms}")
except ImportError:
    print("ASE 未安装，跳过互转")
