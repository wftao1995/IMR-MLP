#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Prepare constant-field, fixed-cell LAMMPS input with external fixed BECs.

This is a local linear-response approximation around an insulating reference,
independent of the zero-field potential. It does not interpret SOG latents as
charges or replace them. Requires NumPy; no LAMMPS source modification.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def prepare(reference, field, replicate=(1, 1, 1), project_asr=False):
    if reference.get('schema') != 'fixed-bec-reference-v1' or reference.get('bec_axes') != 'polarization,displacement':
        raise ValueError('Explicit fixed BEC schema and tensor axes required')
    cell = np.asarray(reference['cell_A'], dtype=float)
    positions = np.asarray(reference['positions_A'], dtype=float)
    z = np.asarray(reference['bec_e'], dtype=float)
    species = list(reference['species'])
    masses = np.asarray(reference['masses_amu'], dtype=float)
    n = len(species)
    field = np.asarray(field, dtype=float)
    if (n == 0 or cell.shape != (3, 3) or positions.shape != (n, 3)
        or z.shape != (n, 3, 3) or masses.shape != (n,) or field.shape != (3,)):
        raise ValueError('Inconsistent reference arrays')
    if not all(np.isfinite(x).all() for x in [cell, positions, z, masses, field]):
        raise ValueError('Nonfinite reference or field')
    if np.linalg.det(cell) <= 1e-8 or np.min(masses) <= 0:
        raise ValueError('Right-handed nonsingular cell and positive masses required')
    if len(replicate) != 3 or any(int(v) != v or v < 1 for v in replicate):
        raise ValueError('Positive integer replication factors required')
    residual = z.sum(axis=0)
    if project_asr:
        z = z-residual/n
    elif np.max(np.abs(residual)) > 1e-6:
        raise ValueError('BEC acoustic sum exceeds tolerance; review data or explicitly request --project-asr')
    # A proper Cartesian rotation, not a tensor rescaling by cell dimensions.
    ex = cell[0]/np.linalg.norm(cell[0])
    ez = np.cross(cell[0], cell[1]); ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex)
    rotation = np.column_stack([ex, ey, ez])
    shifts = list(itertools.product(*(range(int(v)) for v in replicate)))
    positions = np.concatenate([positions+np.asarray(s)@cell for s in shifts])@rotation
    cell = np.diag(replicate)@cell@rotation
    cell[np.triu_indices(3, 1)] = 0.  # roundoff in the orthonormal construction
    z = np.tile(rotation.T@z@rotation, (len(shifts), 1, 1))
    field_rotated = rotation.T@field
    forces = np.einsum('iab,a->ib', z, field_rotated)
    # Establish one branch at initialization. MD subsequently uses image flags.
    fractional = positions@np.linalg.inv(cell)
    fractional -= np.floor(fractional)
    positions = fractional@cell
    return dict(cell_A=cell.tolist(), positions_A=positions.tolist(), bec_e=z.tolist(),
                species=species*len(shifts), masses_amu=np.tile(masses, len(shifts)).tolist(),
                field_source_V_A=field.tolist(), field_V_A=field_rotated.tolist(),
                source_to_lammps_rotation=rotation.tolist(), forces_eV_A=forces.tolist(),
                original_asr_sum_e=residual.tolist(), projected_asr=project_asr,
                asr_sum_e=z.sum(axis=0).tolist(), replica_count=len(shifts))


def quote(path):
    value = str(path)
    if any(c in value for c in ['"', '$', '\n', '\r']):
        raise ValueError('Unsupported characters in LAMMPS path')
    return '"'+value+'"'


def write_case(state, work, model=None, type_order=None):
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    cell = np.array(state['cell_A'])
    positions = np.array(state['positions_A'])
    forces = np.array(state['forces_eV_A'])
    species = state['species']
    order = type_order or list(dict.fromkeys(species))
    if len(set(order)) != len(order) or set(order) != set(species):
        raise ValueError('Explicit type order must match the reference elements')
    if model is not None and type_order is None:
        raise ValueError('Model deployment requires explicit --type-order matching its training map')
    n = len(species)
    lines = ['Fixed BEC reference; units metal; Cartesian tensor rotation recorded', '',
             f'{n} atoms', f'{len(order)} atom types', '',
             f'0 {cell[0,0]:.17g} xlo xhi', f'0 {cell[1,1]:.17g} ylo yhi', f'0 {cell[2,2]:.17g} zlo zhi',
             f'{cell[1,0]:.17g} {cell[2,0]:.17g} {cell[2,1]:.17g} xy xz yz', '', 'Masses', '']
    for i, name in enumerate(order):
        selected = [m for s,m in zip(species,state['masses_amu']) if s == name]
        if np.max(selected)-np.min(selected) > 1e-8:
            raise ValueError('One mass per species required')
        lines.append(f'{i+1} {selected[0]:.17g} # {name}')
    lines += ['', 'Atoms # atomic', '']
    for i, (name, p) in enumerate(zip(species, positions)):
        lines.append(f'{i+1} {order.index(name)+1} '+' '.join(f'{v:.17g}' for v in p)+' 0 0 0')
    (work/'reference.data').write_text('\n'.join(lines)+'\n')
    include = ['# Constant field, fixed BEC and fixed cell; units metal.',
               '# Preserve atom IDs, atom count, image flags and this reference across a restart.',
               '# No change_box, fix npt/deform, atom insertion/deletion or replication after loading.',
               f'if "$(count(all)) != {n}" then "quit 1"']
    for d, axis in enumerate('xyz'):
        for name, values in [('f'+axis, forces[:, d]), ('r'+axis, positions[:, d])]:
            path = work/(name+'.atom')
            path.write_text(str(n)+'\n'+''.join(f'{i+1} {v:.17g}\n' for i,v in enumerate(values)))
            include.append(f'variable bec_{name} atomfile {quote(path)}')
    include += ['compute bec_unwrapped all property/atom xu yu zu',
                'variable bec_u atom "-(v_bec_fx*(c_bec_unwrapped[1]-v_bec_rx)+v_bec_fy*(c_bec_unwrapped[2]-v_bec_ry)+v_bec_fz*(c_bec_unwrapped[3]-v_bec_rz))"',
                'fix bec_field all addforce v_bec_fx v_bec_fy v_bec_fz energy v_bec_u',
                'fix_modify bec_field energy yes virial no']
    box = dict(lx=cell[0,0], ly=cell[1,1], lz=cell[2,2], xy=cell[1,0], xz=cell[2,0], yz=cell[2,1])
    expression = '+'.join(f'({name}-({value:.17g}))^2' for name,value in box.items())
    include += [f'variable bec_cell_error equal "{expression}"',
                'if "$(v_bec_cell_error) > 1.0e-16" then "quit 1"',
                'fix bec_fixed_cell all halt 1 v_bec_cell_error > 1.0e-16 error hard']
    (work/'field.inc').write_text('\n'.join(include)+'\n')
    init = ['units metal', 'atom_style atomic', 'atom_modify map array', 'boundary p p p',
            f'read_data {quote(work / "reference.data")}', 'neighbor 1.0 bin', 'neigh_modify every 1 delay 0 check yes']
    if model is None:
        init += ['pair_style zero 4.0', 'pair_coeff * *']
    else:
        model = model.resolve()
        if not model.is_file():
            raise ValueError('Missing zero-field model')
        state['model'] = str(model)
        state['model_sha256'] = hashlib.sha256(model.read_bytes()).hexdigest()
        init += [f'pair_style mlip/opt-les {quote(model)} '+' '.join(str(i) for i in range(len(order))),
                 'pair_coeff * *']
    (work/'init.inc').write_text('\n'.join(init)+'\n')
    # The generated smoke input only evaluates a reference frame. A user may add
    # fix nvt/nve at fixed cell; this utility does not invent a production protocol.
    (work/'in.field').write_text(f'include {quote(work / "init.inc")}\ninclude {quote(work / "field.inc")}\n'
                               'thermo 1\nthermo_style custom step pe f_bec_field\nrun 0\n'
                               'write_dump all custom field.dump id type xu yu zu fx fy fz modify sort id format float %.17g\n')
    state['type_order'] = order
    (work/'manifest.json').write_text(json.dumps(state, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--field', type=float, nargs=3, required=True, metavar=('EX','EY','EZ'))
    parser.add_argument('--replicate', type=int, nargs=3, default=[1,1,1])
    parser.add_argument('--project-asr', action='store_true')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--type-order', nargs='+')
    args = parser.parse_args()
    state = prepare(json.loads(args.reference.read_text()), args.field, args.replicate, args.project_asr)
    state['reference'] = str(args.reference.resolve())
    state['reference_sha256'] = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    write_case(state, args.work, args.model, args.type_order)
