#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT
"""Prepare a synthetic MTP basis/fit/calculation workflow, or complete its generated templates."""

import argparse
import importlib.util
import itertools
import json
from pathlib import Path
import random
import re


CANDIDATES = (8, 16)


def configuration(points, cell):
    lines = ['BEGIN_CFG', ' Size', f' {len(points)}', ' Supercell',
             f' {cell:.12f} 0 0', f' 0 {cell:.12f} 0', f' 0 0 {cell:.12f}',
             ' AtomData: id type cartes_x cartes_y cartes_z']
    lines += [f' {i+1} 0 ' + ' '.join(f'{x:.12f}' for x in point)
              for i, point in enumerate(points)]
    return '\n'.join(lines + ['END_CFG', '']) + '\n'


def prepare(root, work):
    if work.exists() and (not work.is_dir() or any(work.iterdir())):
        raise ValueError('--work must be an empty directory; choose a new path')
    work.mkdir(parents=True, exist_ok=True)
    (work/'index.map').write_text('Al\n')
    reference = work/'reference'
    reference.mkdir()
    epsilon, sigma, cutoff = .1, 2.5, 6.0

    def lj(r):
        return 4*epsilon*((sigma/r)**12 - (sigma/r)**6)

    derivative_at_cutoff = 24*epsilon/cutoff*((sigma/cutoff)**6 - 2*(sigma/cutoff)**12)
    # Native pair_table format stores energy columns; its spline supplies derivatives.
    rows = ['Al', 'N 4201']
    for i in range(4201):
        r = 1.8 + .001*i
        energy = lj(r) - lj(cutoff) - (r-cutoff)*derivative_at_cutoff
        rows.append(f'{i} {r:.12f} {energy:.16g}')
    (reference/'force-shifted-lj.table').write_text('\n'.join(rows) + '\n')

    basis = ((0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5))
    splits = {}
    for split, seed, counts in [('train', 1701, (60, 24, 16)),
                                ('valid', 1702, (16, 8, 4)),
                                ('test', 1704, (16, 8, 4))]:
        rng = random.Random(seed)
        frames = []
        for i in range(counts[0]):
            # Cover both ends of the declared interval; holdouts sample random distances.
            distance = (2.1 + (5.95-2.1)*i/(counts[0]-1) if split == 'train'
                        else rng.uniform(2.1, 5.95))
            frames.append(configuration([[7, 7, 7], [7+distance, 7, 7]], 20))
        for _ in range(counts[1]):
            a = rng.uniform(3.72, 4.08)
            points = [[a*x+rng.uniform(-.04, .04), a*y+rng.uniform(-.04, .04),
                       a*z+rng.uniform(-.04, .04)] for x, y, z in basis]
            frames.append(configuration(points, a))
        for _ in range(counts[2]):
            a = rng.uniform(3.84, 3.96)
            points = [[a*(x+dx), a*(y+dy), a*(z+dz)]
                      for x, y, z in itertools.product(range(2), repeat=3)
                      for dx, dy, dz in basis][1:]
            fraction = rng.random()
            points[0] = [a*.5*(1-fraction), a*.5*(1-fraction), 0]
            for point in points:
                for j in range(3):
                    point[j] += rng.uniform(-.02, .02)
            frames.append(configuration(points, 2*a))
        (work/f'{split}-input.cfg').write_text(''.join(frames))
        splits[split] = dict(seed=seed, dimers=counts[0], bulk=counts[1], vacancy_path=counts[2],
                             dimer_sampling='uniform grid' if split == 'train' else 'random uniform')

    for count in CANDIDATES:
        directory = work/f'basis-{count}'
        directory.mkdir()
        (directory/'basis.ini').write_text(f'''max_lev {2*count}
max_dim 1
max_mu {count-1}
max_nv 0
mode 3
seed 17
maxiter 100
''')
        (directory/'fit.ini').write_text('''trainset ../train.cfg
validset ../valid.cfg
index ../index.map
potfile template.mtp
fit:skip_preinit true
fit:do_orthogonalization false
fit:force_weight 0.1
fit:stress_weight 0.001
fit:weight_L2 1e-12
fit:bfgs_m 2
fit:bfgs_maxiteration 100
fit:save_to checkpoint.mtp
''')

    # Reuse the LJ exercise's geometry/settings writer, with MTP model deployment deferred.
    module_path = root/'examples/basic-calculations/prepare.py'
    spec = importlib.util.spec_from_file_location('calculation_inputs', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.prepare_inputs(work/'calculations', pottype='mtp', potfile='model.mtp',
                          parameters=None, title='Synthetic MTP workflow, not an Al material model')
    provenance = dict(reference=dict(type='pair_table', epsilon_eV=epsilon, sigma_A=sigma,
                                    cutoff_A=cutoff, grid_min_A=1.8, grid_step_A=.001,
                                    grid_points=4201, modification='force-shifted LJ'),
                      splits=splits, candidates=list(CANDIDATES),
                      basis='ordinary MTP, nu=0 and max_dim=1; fixed identity radial coefficients',
                      units=dict(length='angstrom', energy='eV', force='eV/angstrom'))
    (work/'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Prepared inputs: {work}')
    print('Next: reference-label the three splits and run mtmbasis in basis-8 and basis-16.')
    print('No trained model has been deployed to calculations/ yet.')


def complete_templates(work):
    provenance = json.loads((work/'provenance.json').read_text())
    if provenance['candidates'] != list(CANDIDATES):
        raise ValueError('candidate settings differ from this exercise')
    templates = []
    for count in CANDIDATES:
        directory = work/f'basis-{count}'
        output = directory/'template.mtp'
        if output.exists():
            raise ValueError(f'{output} already exists; reuse it or start a new --work')
        text = (directory/'potential.mtp_0').read_text()
        for key in ('radial_funcs_count', 'alpha_scalar_moments'):
            match = re.search(rf'^\s*{key}\s*=\s*(\d+)', text, re.M)
            if not match or int(match[1]) != count:
                raise ValueError(f'{directory}: unexpected {key}; use the supplied basis.ini')
        text = re.sub(r'radial_basis_size = \d+', f'radial_basis_size = {count}', text)
        text = text.replace('max_dist = 5', 'max_dist = 6')
        text += 'radial_coeffs\n0-0\n'
        text += ''.join('{' + ', '.join('1' if i == j else '0' for j in range(count)) + '}\n'
                        for i in range(count))
        text += 'species_coeffs = {0}\nmoment_coeffs = {' + ', '.join('0' for _ in range(count)) + '}\n'
        templates.append((output, text))
    for output, text in templates:
        output.write_text(text)
        print(f'Completed untrained template: {output}')


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=root/'build/mtp-workflow')
    parser.add_argument('--complete-templates', action='store_true',
                        help='complete templates after both mtmbasis commands; do not regenerate inputs')
    args = parser.parse_args()
    work = args.work.resolve()
    try:
        if args.complete_templates:
            complete_templates(work)
        else:
            prepare(root, work)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
