#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT
"""Compare CPU MTP styles, including valid models with no moment products."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

from verify_lammps import close, read_dump, read_thermo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lmp', type=Path)
    parser.add_argument('--potential', type=Path,
                        help='Optional additional analytic MTP with a valid species 0')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    models = [(base / 'simple_analytic.mtp', False), (base / 'simple_spline.mtp', True)]
    if args.potential:
        models.append((args.potential.resolve(), False))

    with tempfile.TemporaryDirectory(prefix='imrmlp-style-parity-') as directory:
        work = Path(directory)
        for model, spline in models:
            styles = (['mlip/transport/spline', 'mlip/spline'] if spline else
                      ['mlip/transport', 'mlip', 'mlip/opt', 'mlip/opt/omp'])
            reference = None
            for index, style in enumerate(styles):
                log, dump = work / f'{index}.log', work / f'{index}.dump'
                subprocess.run([
                    str(args.lmp.resolve()), '-pk', 'omp', '2',
                    '-log', str(log), '-screen', 'none', '-var', 'style', style,
                    '-var', 'potential', str(model), '-var', 'dump_file', str(dump),
                    '-in', str(base / 'in.pair_parity')],
                    env={**os.environ, 'OMP_NUM_THREADS': '2'}, check=True)
                result = read_thermo(log), read_dump(dump)
                if reference is None:
                    reference = result
                    continue
                for key in ('PotEng', 'Press', 'Pxx', 'Pyy', 'Pzz', 'Pxy', 'Pxz', 'Pyz'):
                    if not close(reference[0][key], result[0][key], rtol=2e-11, atol=2e-9):
                        raise AssertionError((model.name, style, key))
                if reference[1].keys() != result[1].keys():
                    raise AssertionError('Atom IDs differ')
                for atom in reference[1]:
                    for key in ('fx', 'fy', 'fz', 'c_eatom'):
                        if not close(reference[1][atom][key], result[1][atom][key],
                                     rtol=2e-11, atol=2e-9):
                            raise AssertionError((model.name, style, atom, key))
                print(f'{model.name}: {style} matches {styles[0]}', flush=True)


if __name__ == '__main__':
    main()
