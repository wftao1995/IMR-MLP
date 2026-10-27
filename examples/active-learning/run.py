#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT
"""Run a synthetic, geometry-only MaxVol selection example (not a material model)."""

import argparse
from pathlib import Path
import re
import shutil
import subprocess


def configuration(distance):
    return f"""BEGIN_CFG
 Size
 2
 Supercell
 20 0 0
 0 20 0
 0 0 20
 AtomData: id type cartes_x cartes_y cartes_z
 1 0 7 7 7
 2 0 {7 + distance:.8f} 7 7
END_CFG
"""


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imrmlp', type=Path, default=root/'build/release/imrmlp')
    parser.add_argument('--work', type=Path, default=root/'build/active-learning')
    args = parser.parse_args()
    executable, work = args.imrmlp.resolve(), args.work.resolve()
    if not executable.is_file():
        parser.error('build the CLI with make imrmlp, or supply --imrmlp')
    if work.exists() and (not work.is_dir() or any(work.iterdir())):
        parser.error('--work must be an empty directory; choose a new path')
    work.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(root/'LAMMPS/USER-MLIP/transport/tests/simple_analytic.mtp',
                    work/'model.mtp')
    (work/'index.map').write_text('Al\n')
    # This exercise needs geometry only, not E/F/stress labels.
    (work/'coverage.cfg').write_text(''.join(configuration(d) for d in (3.8, 3.9, 4.0)))
    distances = (3.9, 3.5, 3.0, 2.5)
    (work/'pool.cfg').write_text(''.join(configuration(d) for d in distances))
    (work/'select.ini').write_text("builder:potfile model.mtp\n")

    def run(label, *command):
        log_path = work/f'{label}.log'
        with log_path.open('w') as log:
            result = subprocess.run([str(executable), *command], cwd=work,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f'{label} failed (exit {result.returncode}); see {log_path}')
        print(f'{label}: OK', flush=True)

    run('select', 'maxvol_select', 'select.ini', '--pool', 'coverage.cfg',
        '--index', 'index.map', '--output', 'selected.xyz',
        '--score-file', 'coverage-grades.txt')
    # The first pass saves all positive-grade frames for an inspectable score table.
    common = ('maxvol_sample', 'select.ini', '--input', 'pool.cfg',
              '--index', 'index.map', '--state-file', 'model.mtp.sel',
              '--threshold-break', '1000000000')
    run('grade-pool', *common, '--output', 'graded-pool.xyz', '--threshold-save', '0')
    threshold = 2.1
    run('sample', *common, '--output', 'candidates.xyz', '--threshold-save', str(threshold))

    grades = [float(value) for value in re.findall(
        r'MV_grade="([^"]+)"', (work/'graded-pool.xyz').read_text())]
    if len(grades) != len(distances):
        raise SystemExit(f'Expected four graded frames; inspect {work}/grade-pool.log')
    rows = ['frame\tdistance_A\tgrade\tselected']
    rows += [f'{i}\t{d:.2f}\t{g:.6f}\t{int(g > threshold)}'
             for i, (d, g) in enumerate(zip(distances, grades))]
    (work/'grades.tsv').write_text('\n'.join(rows) + '\n')
    print('\n'.join(rows))
    print(f'Selected {sum(g > threshold for g in grades)} / {len(grades)} candidates')
    print(f'Artifacts: {work}')
    print('Synthetic selection only: no reference labels, retraining or material accuracy claim.')


if __name__ == '__main__':
    main()
