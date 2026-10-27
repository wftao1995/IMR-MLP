# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""
01 — 结构读写 (Structure I/O)

支持格式：poscar, outcar, extxyz, mlpcfg, cif, lmptrj, lmpdat,
           vasprun, xsf, vtk, xdatcar, lmpyaml 等
"""
import imrmlp

# ── 读取单个结构 ──────────────────────────────────────────────────────
atoms = imrmlp.read_one("POSCAR")
print(f"单结构: {atoms}")

# ── 批量读取 ──────────────────────────────────────────────────────────
frames = imrmlp.read("trajectory.xyz")
print(f"共 {len(frames)} 帧")
for i, f in enumerate(frames[:3]):
    e = f.energy if f.energy is not None else float('nan')
    print(f"  帧{i}: natoms={f.natoms}, E={e:.4f}")

# ── 格式自动检测 / 显式指定 ──────────────────────────────────────────
# atoms = imrmlp.read_one("CONTCAR", format="poscar")
# atoms = imrmlp.read("dump.lammpstrj", format="lmptrj")

# ── 写入文件 ──────────────────────────────────────────────────────────
imrmlp.write("output.cfg", atoms)                    # 单结构
imrmlp.write("output.extxyz", frames[:10])           # 多帧

# ── 格式探测 ──────────────────────────────────────────────────────────
fmt = imrmlp.io.guess_format("structure.POSCAR")
print(f"POSCAR 文件 -> 格式: {fmt}")
