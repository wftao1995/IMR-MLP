# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""
04 — 结构弛豫

支持优化器: lbfgs, fire, cg, gd
支持变胞、定体积、定形状等约束。
"""
import imrmlp
from imrmlp import Relaxation

# ── 准备 ──────────────────────────────────────────────────────────────
pot = imrmlp.Potential("Al.mtp")
source = imrmlp.read_one("POSCAR")
source.to_cartesian()
# read_one 不附带模型物种映射；在此明确将 Al 映射到模型物种 0。
# 对多元素模型，elements 必须按模型的物种顺序填写。
atoms = imrmlp.Atoms(numbers=source.numbers, positions=source.positions,
                     cell=source.cell, pbc=source.pbc, elements=["Al"])

# ── 原子位置弛豫 (FIRE) ───────────────────────────────────────────────
relax = Relaxation(pot, driver="fire", gtol=1e-2, max_iter=500)
result = relax.run(atoms)
print(f"FIRE 弛豫: converged={result.converged}, "
      f"E: {result.init_energy:.4f} → {result.final_energy:.4f}, "
      f"iters={result.n_iter}")

# ── 变胞弛豫 (L-BFGS) ─────────────────────────────────────────────────
# relax_vc = Relaxation(pot, driver="lbfgs", gtol=1e-3,
#                       relax_cell=True, pressure=0.0)
# result = relax_vc.run(atoms)
# print(f"变胞弛豫: converged={result.converged}, "
#       f"E: {result.init_energy:.4f} → {result.final_energy:.4f}")

# ── 固定体积、只变形状 ────────────────────────────────────────────────
# relax_shape = Relaxation(pot, relax_cell=True, fix_vol=True)
# result = relax_shape.run(atoms)

# ── 保存弛豫轨迹 ──────────────────────────────────────────────────────
# relax_traj = Relaxation(pot, trajfile="relax_traj.extxyz")
# result = relax_traj.run(atoms)
