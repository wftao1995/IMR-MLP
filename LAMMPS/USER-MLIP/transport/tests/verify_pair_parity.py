#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: GPL-2.0-only
# Adapted for the IMR-MLP public source layout, 2026-09-22.


from __future__ import annotations

import argparse
from pathlib import Path

from verify_lammps import close, read_dump, read_thermo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_log", type=Path)
    parser.add_argument("baseline_dump", type=Path)
    parser.add_argument("transport_log", type=Path)
    parser.add_argument("transport_dump", type=Path)
    args = parser.parse_args()

    baseline_thermo = read_thermo(args.baseline_log)
    transport_thermo = read_thermo(args.transport_log)
    baseline_atoms = read_dump(args.baseline_dump)
    transport_atoms = read_dump(args.transport_dump)
    if baseline_atoms.keys() != transport_atoms.keys():
        raise AssertionError("atom IDs differ")

    for field in ("PotEng", "Press", "Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"):
        if not close(baseline_thermo[field], transport_thermo[field],
                     rtol=2.0e-11, atol=2.0e-9):
            raise AssertionError(f"thermo {field} differs")
    for atom_id in baseline_atoms:
        for field in ("fx", "fy", "fz", "c_eatom"):
            if not close(baseline_atoms[atom_id][field],
                         transport_atoms[atom_id][field],
                         rtol=2.0e-11, atol=2.0e-9):
                raise AssertionError(f"atom {atom_id} field {field} differs")
    print("mlip/transport matches baseline energy, forces, and global virial")


if __name__ == "__main__":
    main()
