#!/usr/bin/env python3
"""Check standalone CPU/GPU/MPI inference against generate_models GPM fixtures."""
import argparse
import csv
import json
import os
from pathlib import Path
from run_regression import close, observables, read_dump, run, setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True,
                        help='Output of IMRMLP_NATIVE_GPM=1 generate_native_test_models')
    parser.add_argument('--lmp', type=Path, required=True)
    parser.add_argument('--plugin', type=Path, help='Omit for styles embedded in standard lmp')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'hip'], default='cpu')
    parser.add_argument('--gpu-id', default='auto')
    parser.add_argument('--ranks', nargs='+', type=int, default=[1, 2, 4])
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1')
    if args.device != 'cpu':
        env['IMRMLP_REQUIRE_GPU'] = '1'
    with (args.models / 'cases.tsv').open() as handle:
        cases = list(csv.DictReader(handle, delimiter='\t'))
    supported = {case['name'] for case in cases if case['name'] in {'mtp', 'mtp_split'} or
                 (case['name'].startswith('mtp_m_') and not case['name'].endswith(('_les', '_sog')))}
    count = 0
    for case in cases:
        if case['name'] not in supported:
            continue
        if Path(case['model']).suffix != '.gpm':
            raise AssertionError('Generate fixtures with IMRMLP_NATIVE_GPM=1')
        reference = Path(case['reference']).read_text().splitlines()
        expected_global = list(map(float, reference[0].split()))
        expected_atoms = [list(map(float, line.split())) for line in reference[1:]]
        for ranks in args.ranks:
            directory = args.work / f"{case['name']}-{case['geometry']}-{ranks}"
            directory.mkdir(exist_ok=True)
            dump, metrics = directory / 'atoms.dump', directory / 'global.txt'
            style = 'mlip/opt' if args.device == 'cpu' else f'mlip/gpudirect device {args.device} precision fp64 gpu_id {args.gpu_id}'
            commands = ((f'plugin load {args.plugin.resolve()}\n' if args.plugin else '') + setup(case) +
                        f'pair_style {style}\npair_coeff * * {case["model"]} 0 1\n' +
                        observables(case, dump.resolve(), metrics.resolve(), 1.6021765e6))
            script = directory / 'in.gpm'
            script.write_text(commands)
            output = run(['mpiexec', '-n', str(ranks), str(args.lmp.resolve()),
                          '-log', 'none', '-in', str(script.resolve())], env)
            (directory / 'run.log').write_text(output)
            close(expected_global, list(map(float, metrics.read_text().split())), str(directory))
            close(expected_atoms, read_dump(dump, 1.6021765e6), str(directory))
            count += 1
    rejected = 0
    for case in cases:
        if case['name'] in supported or case['geometry'] != 'orthogonal':
            continue
        script = args.work / f'in.reject-{case["name"]}'
        script.write_text((f'plugin load {args.plugin.resolve()}\n' if args.plugin else '') + setup(case) +
                          f'pair_style mlip/opt\npair_coeff * * {case["model"]} 0 1\n')
        run(['mpiexec', '-n', str(args.ranks[-1]), str(args.lmp.resolve()),
             '-log', 'none', '-in', str(script.resolve())], env, 'does not support')
        rejected += 1
    report = dict(device=args.device, comparisons=count, unsupported_rejections=rejected)
    (args.work / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
