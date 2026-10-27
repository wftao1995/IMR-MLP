#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: GPL-2.0-only
# Adapted for the IMR-MLP public source layout, 2026-09-22.


from __future__ import annotations

import argparse
from pathlib import Path

from verify_lammps import close, read_thermo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    thermo = read_thermo(args.log)
    for component in range(1, 7):
        conventional = thermo[f"c_conventional[{component}]"]
        transport = thermo[f"c_transport[{component}]"]
        if not close(conventional, transport, rtol=3.0e-10, atol=2.0e-9):
            raise AssertionError(
                f"heat-current component {component} differs: "
                f"{conventional} vs {transport}")
    print("MTP heat current matches LAMMPS compute heat/flux in the pairwise limit")


if __name__ == "__main__":
    main()
