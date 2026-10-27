#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Export validated DFT_BEC reference cells on the DFT host (NumPy and ASE)."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from ase.io import read


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(root, phase, output):
    directory = root/phase/'high_cutoff/dfpt'
    response = json.loads((directory/'response.json').read_text())
    if response['gap']['indirect_gap_eV'] <= 0.01:
        raise ValueError(f'{phase}: insulating reference required')
    if 'General timing and accounting' not in (directory/'OUTCAR').read_text():
        raise ValueError(f'{phase}: VASP did not finish normally')
    structure = read(directory/'POSCAR')
    tree = ET.parse(directory/'vasprun.xml')
    arrays = tree.findall('.//array[@name="born_charges"]')
    if not arrays:
        raise ValueError('Missing XML Born charges')
    z = np.array([[list(map(float, row.text.split())) for row in s.findall('v')]
                  for s in arrays[-1].findall('set')])
    epsilon_nodes = tree.findall('.//varray[@name="epsilon"]')
    epsilon = np.array([list(map(float, row.text.split())) for row in epsilon_nodes[-1].findall('v')])
    if z.shape != (len(structure), 3, 3) or not np.isfinite(z).all() or not np.isfinite(epsilon).all():
        raise ValueError('Invalid response shape or value')
    z_error = float(np.max(np.abs(z-np.array(response['BEC_e']))))
    eps_error = float(np.max(np.abs(epsilon-np.array(response['epsilon_infinity']))))
    if z_error > 1e-5 or eps_error > 1e-5:
        raise ValueError('XML and parsed OUTCAR disagree')
    if np.max(np.abs(z.sum(axis=0))) > 1e-6:
        raise ValueError('Unexpected XML acoustic-sum residual')
    result = dict(schema='fixed-bec-reference-v1', phase=phase,
                  cell_A=structure.cell.array.tolist(), positions_A=structure.positions.tolist(),
                  species=structure.get_chemical_symbols(), masses_amu=structure.get_masses().tolist(),
                  bec_e=z.tolist(), bec_axes='polarization,displacement',
                  epsilon_infinity=epsilon.tolist(), gap_eV=response['gap']['indirect_gap_eV'],
                  source_directory=str(directory),
                  source_sha256={name:digest(directory/name) for name in ['POSCAR','INCAR','KPOINTS','OUTCAR','vasprun.xml','response.json']},
                  validation=dict(xml_outcar_bec_difference_e=z_error, xml_outcar_epsilon_difference=eps_error,
                                  asr_max_e=float(np.max(np.abs(z.sum(axis=0))))))
    (output/(phase+'.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(phase, result['validation'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phases', nargs='+', default=['alpha_2H','beta_2H','alpha_3R'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for phase in args.phases:
        export(args.source.resolve(), phase, args.output)
