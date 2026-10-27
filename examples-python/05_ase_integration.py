# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""
05 — ASE 集成

IMRMLPCalculator 把 imrmlp 势函数包装成 ASE Calculator，
可以直接用 ASE 的优化器、MD、NEB、振动分析等全套生态。
"""
import numpy as np
from ase.build import bulk
from ase.optimize import BFGS
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import ase.units as units

from imrmlp.calculator import IMRMLPCalculator

# ── 构建结构 ──────────────────────────────────────────────────────────
atoms = bulk("Al", "fcc", a=4.05) * (3, 3, 3)
atoms.calc = IMRMLPCalculator(filename="Al.mtp", elements=["Al"])

print(f"初始: natoms={len(atoms)}, 晶格矢量长度={atoms.get_cell().lengths()}")

# ── ASE 优化器弛豫 ────────────────────────────────────────────────────
opt = BFGS(atoms)
opt.run(fmax=0.01)
print(f"弛豫后能量: {atoms.get_potential_energy():.4f} eV")
print(f"弛豫后力最大值: {np.abs(atoms.get_forces()).max():.4f} eV/Å")

# ── 短 MD ─────────────────────────────────────────────────────────────
atoms.set_velocities(np.zeros((len(atoms), 3)))
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)
dyn.run(100)
print(f"MD 后温度: {atoms.get_temperature():.0f} K")
