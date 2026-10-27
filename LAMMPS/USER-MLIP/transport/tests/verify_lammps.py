#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: GPL-2.0-only
# Adapted for the IMR-MLP public source layout, 2026-09-22.

"""Validate one run and compare serial, OpenMP, and optional MPI results."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def close(lhs: float, rhs: float, *, rtol: float = 2.0e-10,
          atol: float = 2.0e-10) -> bool:
    return abs(lhs - rhs) <= atol + rtol * max(abs(lhs), abs(rhs))


def read_thermo(path: Path) -> dict[str, float]:
    columns = None
    result = None
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "Step":
            columns = fields
            continue
        if columns and len(fields) == len(columns):
            try:
                values = [float(value) for value in fields]
            except ValueError:
                continue
            result = dict(zip(columns, values))
    if result is None:
        raise RuntimeError(f"no thermo table in {path}")
    return result


def read_dump(path: Path) -> dict[int, dict[str, float]]:
    lines = path.read_text().splitlines()
    header = max(i for i, line in enumerate(lines)
                 if line.startswith("ITEM: ATOMS "))
    columns = lines[header].split()[2:]
    atoms = {}
    for line in lines[header + 1:]:
        if line.startswith("ITEM:"):
            break
        fields = line.split()
        if len(fields) == len(columns):
            row = dict(zip(columns, map(float, fields)))
            atoms[int(row["id"])] = row
    return atoms


def validate(log: Path, dump: Path) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
    thermo = read_thermo(log)
    atoms = read_dump(dump)
    if int(thermo["Atoms"]) != len(atoms):
        raise AssertionError("thermo/dump atom counts differ")
    if not all(math.isfinite(value) for value in thermo.values()):
        raise AssertionError("non-finite thermo value")

    atom_energy = sum(atom["c_eatom"] for atom in atoms.values())
    if not close(atom_energy, thermo["PotEng"]):
        raise AssertionError(f"site energies do not sum to PE: {atom_energy} vs {thermo['PotEng']}")

    for component in range(1, 7):
        stress_sum = thermo[f"c_ssum[{component}]"]
        pressure_volume = -thermo[f"c_pressure[{component}]"] * thermo["Volume"]
        if not close(stress_sum, pressure_volume, rtol=3.0e-10, atol=2.0e-7):
            raise AssertionError(
                f"per-atom virial mismatch in component {component}: "
                f"{stress_sum} vs {pressure_volume}")

    if not close(thermo["c_qlr"], -thermo["c_qrl"], atol=2.0e-9):
        raise AssertionError("interface current is not antisymmetric")

    q_from_dump = 0.0
    for atom in atoms.values():
        for component in range(1, 4):
            if not close(atom[f"c_qlr[{component}]"],
                         -atom[f"c_qrl[{component}]"], atol=2.0e-9):
                raise AssertionError("per-atom interface force is not antisymmetric")
        q_from_dump += sum(atom[f"c_qlr[{component}]"] * atom[velocity]
                           for component, velocity in enumerate(("vx", "vy", "vz"), 1))
    if not close(q_from_dump, thermo["c_qlr"], atol=2.0e-9):
        raise AssertionError(
            "scalar/per-atom interface current mismatch (reverse communication may have repeated)")

    for component in ("fx", "fy", "fz"):
        if not close(sum(atom[component] for atom in atoms.values()), 0.0,
                     atol=2.0e-9):
            raise AssertionError(f"net {component} is nonzero")
    return thermo, atoms


def compare(reference, candidate, name: str) -> None:
    ref_thermo, ref_atoms = reference
    test_thermo, test_atoms = candidate
    if ref_atoms.keys() != test_atoms.keys():
        raise AssertionError(f"{name}: atom IDs differ")
    fields = [field for field in ref_thermo if field not in {"Step"}]
    for field in fields:
        if not close(ref_thermo[field], test_thermo[field], rtol=3.0e-9,
                     atol=3.0e-8):
            raise AssertionError(
                f"{name}: thermo {field} differs: {ref_thermo[field]} vs {test_thermo[field]}")
    atom_fields = [field for field in next(iter(ref_atoms.values()))
                   if field not in {"id"}]
    for atom_id in ref_atoms:
        for field in atom_fields:
            if not close(ref_atoms[atom_id][field], test_atoms[atom_id][field],
                         rtol=3.0e-9, atol=3.0e-8):
                raise AssertionError(f"{name}: atom {atom_id} field {field} differs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path,
                        help="run prefixes, each with .log and .dump")
    args = parser.parse_args()
    results = [validate(prefix.with_suffix(".log"), prefix.with_suffix(".dump"))
               for prefix in args.runs]
    for prefix, result in zip(args.runs[1:], results[1:]):
        compare(results[0], result, prefix.name)
    print(f"validated {len(results)} transport run(s): " +
          ", ".join(prefix.name for prefix in args.runs))


if __name__ == "__main__":
    main()
