# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""
08 — 并行控制 & 工具函数

设置 OpenMP 线程数、检查 MPI 等。
"""
import imrmlp

# ── 线程控制 ──────────────────────────────────────────────────────────
print(f"默认线程数: {imrmlp.get_num_threads()}")
imrmlp.set_num_threads(8)
print(f"设置后: {imrmlp.get_num_threads()}")

# ── MPI ────────────────────────────────────────────────────────────────
if imrmlp.has_mpi:
    print("MPI 已启用")
    # from mpi4py import MPI
    # rank = MPI.COMM_WORLD.Get_rank()
else:
    print("MPI 未启用（编译时未开启 IMRMLP_MPI）")

# ── 版本 ──────────────────────────────────────────────────────────────
print(f"imrmlp 版本: {imrmlp.__version__}")
