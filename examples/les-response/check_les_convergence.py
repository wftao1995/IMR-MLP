#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Compare an EFS-trained Coulomb LES to existing DFPT anchors on ptliu2."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def structure(path, reference, displacement=None):
    cell = np.asarray(reference['cell_A'])
    pos = np.asarray(reference['positions_A'])
    if displacement is not None:
        pos = pos + displacement
    text = str(len(pos)) + '\n'
    text += '\n'.join(' '.join(f'{v:.17g}' for v in row) for row in cell) + '\n'
    for symbol, row in zip(reference['species'], pos):
        text += str(['In', 'Se'].index(symbol)) + ' ' + ' '.join(f'{v:.17g}' for v in row) + '\n'
    path.write_text(text)


def statistics(predicted, reference):
    delta = predicted-reference
    diagonal = np.eye(3, dtype=bool)
    return dict(rmse_e=float(np.sqrt(np.mean(delta**2))),
                relative_rmse=float(np.linalg.norm(delta)/np.linalg.norm(reference)),
                diagonal_rmse_e=float(np.sqrt(np.mean(delta[:, diagonal]**2))),
                offdiagonal_rmse_e=float(np.sqrt(np.mean(delta[:, ~diagonal]**2))),
                maximum_e=float(np.max(np.abs(delta))))


def run(args):
    args.work.mkdir(parents=True, exist_ok=True)
    phases = ['alpha_2H', 'beta_2H', 'alpha_3R']
    factors = [1, 2, 3, 4, 6, 8, 12]
    manifest = dict(model=str(args.model), model_sha256=digest(args.model),
                    executable_sha256=digest(args.executable), factors=factors,
                    library_sha256=digest(args.executable.parent/'libimrmlp.so'),
                    reference_hashes={p: digest(args.references/(p+'.json')) for p in phases},
                    parity_anchor='alpha_2H at replication 12; one global sign only',
                    epsilon='trace(DFPT epsilon_infinity)/3; scalar approximation',
                    perturbation_seed=20260915, perturbation_std_A=.01,
                    provisional_screen=dict(relative_BEC_RMSE=.20, diagonal_BEC_RMSE_e=.30,
                                            last_step_max_tensor_difference_e=.02))
    provenance = args.work/'provenance.json'
    if provenance.exists() and json.loads(provenance.read_text()) != manifest:
        raise RuntimeError('Work directory belongs to different inputs')
    provenance.write_text(json.dumps(manifest, indent=2))
    rng = np.random.default_rng(manifest['perturbation_seed'])
    all_results, references = {}, {}
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    # Use the exact library whose hash is recorded, including runtime snapshots.
    env['LD_LIBRARY_PATH'] = str(args.executable.parent) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    def call(phase, factor, epsilon, variant, flags):
        path = args.work/f'{phase}-{factor}-{variant}.json'
        inp = args.work/f'{phase}{"-displaced" if variant == "displaced" else ""}.in'
        if not path.exists():
            command = [str(args.executable), str(args.model), str(inp),
                       *([str(factor)]*3), str(epsilon), str(path), *flags]
            with path.with_suffix('.log').open('w') as out:
                subprocess.run(command, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)
            path.with_suffix('.command.json').write_text(json.dumps(command, indent=2))
        value = json.loads(path.read_text())
        if not np.isfinite(value['bec_e']).all():
            raise RuntimeError('Nonfinite response')
        if value['replica_spread_e'] > 1.e-8:
            raise RuntimeError('Equivalent copies have inconsistent BEC')
        if value.get('fd_max_column_error_e', 0.) > 2.e-6:
            raise RuntimeError('Analytic BEC disagrees with finite differences')
        return value
    for phase in phases:
        reference = json.loads((args.references/(phase+'.json')).read_text())
        if reference['bec_axes'] != 'polarization,displacement':
            raise ValueError('Unexpected BEC axis convention')
        references[phase] = reference
        structure(args.work/(phase+'.in'), reference)
        disp = rng.normal(0., .01, (len(reference['species']), 3))
        disp -= disp.mean(axis=0)
        structure(args.work/(phase+'-displaced.in'), reference, disp)
        epsilon = float(np.trace(reference['epsilon_infinity'])/3)
        rows = []
        for factor in factors:
            base = call(phase, factor, epsilon, 'base', ['fd'] if factor <= 2 else [])
            rebase = call(phase, factor, epsilon, 'rebased', ['rebase'])
            value = dict(factor=factor, base=base,
                         rebase_max_tensor_difference_e=float(np.linalg.norm(
                             np.array(base['bec_e'])-rebase['bec_e'], axis=(1, 2)).max()))
            if factor in [1, 4, 12]:
                value['displaced_diagnostics'] = call(phase, factor, epsilon, 'displaced', ['diagnostics'])
            rows.append(value)
            print(phase, factor, 'atoms', base['atoms'], 'rebase', value['rebase_max_tensor_difference_e'], flush=True)
        all_results[phase] = rows
    anchor = np.array(all_results['alpha_2H'][-1]['base']['bec_e'])
    parity = 1 if np.sum(anchor*np.array(references['alpha_2H']['bec_e'])) >= 0 else -1
    summary = dict(parity=parity, provisional_screen=manifest['provisional_screen'], phases={})
    for phase, rows in all_results.items():
        ref = np.asarray(references[phase]['bec_e'])
        for index, row in enumerate(rows):
            predicted = parity*np.array(row['base']['bec_e'])
            row['accuracy'] = statistics(predicted, ref)
            row['asr_projected_accuracy'] = statistics(predicted-predicted.mean(axis=0), ref)
            if index:
                row['previous_size_max_tensor_difference_e'] = float(np.linalg.norm(
                    predicted-parity*np.array(rows[index-1]['base']['bec_e']), axis=(1, 2)).max())
        largest = rows[-1]
        passed = (largest['accuracy']['relative_rmse'] < .2
                  and largest['accuracy']['diagonal_rmse_e'] < .3
                  and largest['previous_size_max_tensor_difference_e'] < .02)
        summary['phases'][phase] = dict(reference_species=references[phase]['species'],
                                       reference_bec_e=ref.tolist(), rows=rows,
                                       preliminary_BEC_screen_passed=passed)
    summary['all_phases_passed'] = all(p['preliminary_BEC_screen_passed'] for p in summary['phases'].values())
    (args.work/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({p: r['rows'][-1]['accuracy'] for p, r in summary['phases'].items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'executable', 'references', 'work']:
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    for key, value in vars(args).items():
        setattr(args, key, value.resolve())
    run(args)
