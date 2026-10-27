#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT
"""Prepare synthetic LJ inputs for the step-by-step calculation exercise."""

import argparse
import itertools
from pathlib import Path


def poscar(points, cell, title='Synthetic LJ exercise, not an Al model'):
    lines = [title, '1.0',
             f'{cell} 0 0', f'0 {cell} 0', f'0 0 {cell}', 'Al',
             str(len(points)), 'Cartesian']
    return '\n'.join(lines + [' '.join(f'{v:.12f}' for v in p) for p in points]) + '\n'


def relaxation(input_file, output_file, trajectory, variable_cell=False,
               pottype='lj612', potfile='lj.ini'):
    settings = f'''input {input_file}
output {output_file}
index index.map
pot:potfile {potfile}
relax:gtol 0.0001
relax:maxiteration 500
relax:stepmax 0.05
relax:write_traj true
'''
    if trajectory != 'relax-trajectory.xyz':
        settings += f'relax:trajfile {trajectory}\n'
    if pottype != 'mtp':
        settings += f'pot:pottype {pottype}\n'
    if variable_cell:
        settings += 'relax:relax_cell true\n'
    else:
        settings += 'relax:cartesian true\n'
    return settings


def prepare_inputs(work, pottype='lj612', potfile='lj.ini',
                   parameters='ntypes 1\n0-0 0.1 2.5\ncut:0-0 6.0\n',
                   title='Synthetic LJ exercise, not an Al model'):
    """Write shared calculation inputs; parameters=None leaves model deployment to the caller."""
    if work.exists() and (not work.is_dir() or any(work.iterdir())):
        raise ValueError('--work must be an empty directory; choose a new path')
    work.mkdir(parents=True, exist_ok=True)

    for task in ('relax', 'neb', 'fc', 'elastic', 'eos', 'phonopy'):
        directory = work/task
        directory.mkdir()
        (directory/'index.map').write_text('Al\n')
        if parameters is not None:
            (directory/potfile).write_text(parameters)

    def relax_settings(input_file, output_file, trajectory, variable_cell=False):
        return relaxation(input_file, output_file, trajectory, variable_cell, pottype, potfile)

    a = 3.9
    basis = ((0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5))
    bulk = [[a*v for v in p] for p in basis]
    bulk[0] = [.03, -.02, .01]
    directory = work/'relax'
    (directory/'POSCAR').write_text(poscar(bulk, a, title))
    (directory/'relax.ini').write_text(relax_settings('POSCAR', 'relaxed.xyz', 'relax-trajectory.xyz'))
    (directory/'bulk.ini').write_text(relax_settings('relaxed.xyz', 'bulk.xyz', 'bulk-trajectory.xyz', True))

    sites = [[a*(x+dx), a*(y+dy), a*(z+dz)]
             for x, y, z in itertools.product(range(2), repeat=3)
             for dx, dy, dz in basis]
    initial = sites[1:]  # Vacancy at the origin; 31 atoms in a 2x2x2 FCC cell.
    final = [p.copy() for p in initial]
    final[0] = [0., 0., 0.]  # The same row-0 atom hops into the vacancy.
    directory = work/'neb'
    for name, points in (('initial', initial), ('final', final)):
        (directory/f'{name}-input.vasp').write_text(poscar(points, 2*a, title))
        (directory/f'{name}.ini').write_text(relax_settings(
            f'{name}-input.vasp', f'{name}.xyz', f'{name}-trajectory.xyz'))
    (directory/'neb.ini').write_text(f'''index index.map
pot:potfile {potfile}
nimgs 7
image:0 initial.vasp
image:6 final.vasp
driver fire
maxiteration 500
gtol 0.02
max_move 0.05
num_threads 1
neb:solver_policy stable-v1
neb:init_method idpp
neb:spring 1
neb:climb true
''' + (f'pot:pottype {pottype}\n' if pottype != 'mtp' else ''))
    (work/'elastic/elastic.ini').write_text('''relax:cartesian true
relax:gtol 0.0001
relax:maxiteration 500
''')
    (work/'eos/eos.ini').write_text(f'''pot:potfile {potfile}
relax:gtol 0.0001
relax:maxiteration 500
''' + (f'pot:pottype {pottype}\n' if pottype != 'mtp' else ''))
    (work/'phonopy/settings.ini').write_text(f'pot:pottype {pottype}\n')


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=root/'build/basic-calculations')
    work = parser.parse_args().work.resolve()
    try:
        prepare_inputs(work)
    except ValueError as error:
        parser.error(str(error))
    print(f'Inputs prepared in: {work}')
    print('Next: follow examples/basic-calculations/README.md (or README_zh.md).')
    print('Run relaxation first; copy its bulk.vasp to each property directory as relaxed.vasp.')
    print('The Al label is illustrative; the LJ parameters are not a material potential.')


if __name__ == '__main__':
    main()
