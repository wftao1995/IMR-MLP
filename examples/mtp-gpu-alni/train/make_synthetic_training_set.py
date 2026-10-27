#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Scale one structure's E/F/S labels so the GPU-fit example has nonzero loss."""

from __future__ import annotations

import argparse
from pathlib import Path

LABEL_SCALE = 1.02


def perturb_mlpcfg(source: Path, destination: Path) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    atom_count = None
    atom_rows_left = 0
    scaled_atom_columns: list[int] | None = None
    perturb_energy = False
    perturb_stress = False

    output: list[str] = []
    for line_number, line in enumerate(lines):
        stripped = line.strip()

        if line_number > 0 and lines[line_number - 1].strip() == "Size":
            atom_count = int(stripped)

        if stripped.startswith("AtomData:"):
            fields = stripped.split()[1:]
            if "fx" not in fields:
                raise ValueError("input mlpcfg does not contain force columns")
            scaled_atom_columns = [
                fields.index(name)
                for name in ("site_en", "fx", "fy", "fz")
                if name in fields
            ]
            if atom_count is None:
                raise ValueError("AtomData appears before Size")
            atom_rows_left = atom_count
            output.append(line)
            continue

        if atom_rows_left > 0:
            values = stripped.split()
            assert scaled_atom_columns is not None
            for column in scaled_atom_columns:
                values[column] = f"{float(values[column]) * LABEL_SCALE:.16g}"
            output.append(" ".join(values))
            atom_rows_left -= 1
            continue

        if perturb_energy:
            output.append(f"{float(stripped) * LABEL_SCALE:.16g}")
            perturb_energy = False
            continue

        if perturb_stress:
            values = stripped.split()
            values = [f"{float(value) * LABEL_SCALE:.16g}" for value in values]
            output.append(" ".join(values))
            perturb_stress = False
            continue

        output.append(line)
        if stripped == "Energy":
            perturb_energy = True
        elif stripped.startswith("PlusStress:") or stripped.startswith("Stress:"):
            perturb_stress = True

    if atom_count is None or scaled_atom_columns is None:
        raise ValueError("input is not a supported text mlpcfg file")
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    perturb_mlpcfg(args.source, args.destination)


if __name__ == "__main__":
    main()
