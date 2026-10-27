#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT
"""Run a self-contained synthetic MTP fitting example (not a material model)."""

import argparse
import math
from pathlib import Path
import subprocess


def configuration(i):
    distance = 2.1 + 0.085 * i
    positions = [(7.0, 7.0, 7.0), (7.0 + distance, 7.0, 7.0),
                 (7.0 + 0.35 * distance, 8.8 + 0.04 * i, 7.0 + 0.2 * math.sin(i))]
    rows = ['BEGIN_CFG', ' Size', ' 3', ' Supercell',
            ' 20 0 0', ' 0 20 0', ' 0 0 20',
            ' AtomData: id type cartes_x cartes_y cartes_z']
    rows += [f' {j + 1} 0 {x:.12f} {y:.12f} {z:.12f}'
             for j, (x, y, z) in enumerate(positions)]
    return '\n'.join(rows + ['END_CFG', '']) + '\n'


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imrmlp', type=Path, default=root/'build/release/imrmlp')
    parser.add_argument('--work', type=Path, default=root/'build/quickstart')
    args = parser.parse_args()
    executable, work = args.imrmlp.resolve(), args.work.resolve()
    if not executable.is_file():
        parser.error('build the CLI with make imrmlp, or supply --imrmlp')
    if work.exists() and any(work.iterdir()):
        parser.error('--work must be empty; use a new directory for another run')
    work.mkdir(parents=True, exist_ok=True)

    model = (root/'LAMMPS/USER-MLIP/transport/tests/simple_analytic.mtp').read_text()
    (work/'reference.mtp').write_text(model)
    (work/'template.mtp').write_text(model.replace('species_coeffs = {0.125}',
                                                 'species_coeffs = {0.0}')
                                        .replace('moment_coeffs = {0.75}',
                                                 'moment_coeffs = {0.1}'))
    (work/'index.map').write_text('Al\n')
    (work/'train-input.cfg').write_text(''.join(configuration(i) for i in range(20) if i % 5))
    (work/'valid-input.cfg').write_text(''.join(configuration(i) for i in range(20) if not i % 5))
    (work/'fit.ini').write_text('''trainset train.cfg
validset valid.cfg
potfile template.mtp
index index.map
fit:skip_preinit true
fit:force_weight 0.1
fit:stress_weight 0.001
fit:weight_L2 1e-12
fit:save_to checkpoint.mtp
fit:bfgs_m 2
fit:bfgs_maxiteration 30
''')

    def run(label, *command):
        with (work/f'{label}.log').open('w') as log:
            subprocess.run([str(executable), *command], cwd=work,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'{label}: OK', flush=True)

    for split in ('train', 'valid'):
        run(f'label-{split}', 'calc_efs', f'{split}-input.cfg', f'{split}.cfg',
            '--potfile', 'reference.mtp', '--pottype', 'mtp', '--index', 'index.map')
    run('linear-fit', 'trainmtp', 'fit.ini', '--linear-fit')
    # trainmtp writes this final snapshot in its current working directory.
    (work/'fitted.mtp').write_bytes((work/'_Trained_Potential_.mtp').read_bytes())
    run('validation', 'calc_err', '--input', 'valid.cfg', '--potfile', 'fitted.mtp',
        '--pottype', 'mtp', '--index', 'index.map', '--output-summary', 'errors.txt')
    run('predict', 'calc_efs', 'valid-input.cfg', 'predicted.cfg', '--potfile',
        'fitted.mtp', '--pottype', 'mtp', '--index', 'index.map')
    print((work/'errors.txt').read_text())
    print(f'Artifacts: {work}')
    print('Synthetic data only: this model is not a fitted Al material potential.')


if __name__ == '__main__':
    main()
