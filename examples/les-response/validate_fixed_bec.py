#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Validate fixed-BEC energy/forces, PBC crossings and MPI on the compute host."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from fixed_bec_field import prepare, write_case, quote


def run(lmp, directory, text, ranks, label):
    path = directory/(label+'.in')
    path.write_text(text)
    command = (['mpirun', '-np', str(ranks)] if ranks > 1 else [])+[str(lmp), '-in', str(path), '-log', label+'.log']
    env = os.environ.copy()
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
        env[key] = '1'
    with (directory/(label+'.stdout')).open('w') as log:
        result = subprocess.run(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    (directory/(label+'.command.json')).write_text(json.dumps(dict(command=command, returncode=result.returncode), indent=2))
    return result.returncode


def dump(path):
    lines = path.read_text().splitlines()
    header = next(i for i,line in enumerate(lines) if line.startswith('ITEM: ATOMS'))
    return np.array([[float(v) for v in line.split()] for line in lines[header+1:]])


def main(args):
    args.work.mkdir(parents=True, exist_ok=False)
    z = np.array([[1.,.4,-.2],[.1,2.,.3],[.2,-.5,1.5]])
    synthetic = dict(schema='fixed-bec-reference-v1', bec_axes='polarization,displacement',
                     cell_A=[[8.,0.,0.],[1.2,7.,0.],[-.4,.7,9.]],
                     positions_A=[[7.995,.2,.3],[3.,3.,3.]],
                     bec_e=[z.tolist(),(-z).tolist()],species=['In','Se'],masses_amu=[114.818,78.971])
    field = np.array([.03,-.02,.01])
    state = prepare(synthetic, field)
    # Nonsymmetric Z catches an incorrect Ze versus Z^T e contraction.
    assert np.max(np.abs(np.array(state['forces_eV_A'])[0]-z.T@field)) < 1e-14
    assert np.linalg.norm(z.T@field-z@field) > 1e-3
    # Cartesian rotation must preserve scalar work and vector force covariance.
    angle = .47
    rotation = np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    rotated = copy.deepcopy(synthetic)
    rotated['cell_A'] = (np.array(synthetic['cell_A'])@rotation.T).tolist()
    rotated['positions_A'] = (np.array(synthetic['positions_A'])@rotation.T).tolist()
    rotated['bec_e'] = (rotation@np.array(synthetic['bec_e'])@rotation.T).tolist()
    rotated_state = prepare(rotated, rotation@field)
    assert np.max(np.abs(np.array(rotated_state['forces_eV_A'])-state['forces_eV_A'])) < 1e-14
    assert np.max(np.abs(np.array(rotated_state['cell_A'])-state['cell_A'])) < 1e-13
    replicated_state = prepare(synthetic, field, replicate=(2,3,1))
    assert len(replicated_state['species']) == 12
    assert np.max(np.abs(np.array(replicated_state['forces_eV_A'])-np.tile(state['forces_eV_A'],(6,1)))) < 1e-14
    bad = copy.deepcopy(synthetic);bad['bec_e'][0][0][0] += .1
    try:
        prepare(bad, field)
    except ValueError:
        pass
    else:
        raise AssertionError('Nonneutral BEC accepted silently')
    assert np.max(np.abs(prepare(bad, field, project_asr=True)['asr_sum_e'])) < 1e-14
    cases = [('synthetic', synthetic, None)]
    paths = sorted(args.references.glob('*.json'))
    if not paths:
        raise ValueError('No DFT reference files found')
    for path in paths:
        cases.append((path.stem, json.loads(path.read_text()), args.model))
    results = []
    dump_command = 'write_dump all custom {name}.dump id type xu yu zu fx fy fz modify sort id format float %.17g\n'
    for name, reference, model in cases:
        state = prepare(reference, field)
        case = args.work/name
        write_case(state, case, model, ['In','Se'])
        baseline = None
        for ranks in [1,2,4]:
            directory = case/str(ranks);directory.mkdir()
            text = f'include {quote(case / "init.inc")}\n'
            text += 'group probe id 1\ndisplace_atoms probe move 0.03 -0.02 0.015 units box\nthermo_style custom step pe\nrun 0\n'
            text += dump_command.format(name='before')
            text += 'print "BEFORE $(pe:%.17g)" file energy.txt\n'
            text += f'include {quote(case / "field.inc")}\nrun 0\n'
            text += dump_command.format(name='field')
            text += 'print "AFTER $(pe:%.17g) $(f_bec_field:%.17g)" append energy.txt\n'
            assert run(args.lmp, directory, text, ranks, 'comparison') == 0, directory
            before, after = dump(directory/'before.dump'), dump(directory/'field.dump')
            assert np.array_equal(before[:,:2],after[:,:2])
            expected_force = np.array(state['forces_eV_A'])
            force_error = float(np.max(np.abs(after[:,5:8]-before[:,5:8]-expected_force)))
            expected_energy = -np.sum(expected_force*(after[:,2:5]-state['positions_A']))
            energies = (directory/'energy.txt').read_text().splitlines()
            base_energy = float(energies[0].split()[1])
            total_energy, field_energy = map(float, energies[1].split()[1:])
            energy_error = max(abs(total_energy-base_energy-expected_energy),abs(field_energy-expected_energy))
            assert force_error < 1e-8 and energy_error < 1e-9
            if baseline is None:
                baseline = after
            mpi_error = float(np.max(np.abs(after-baseline)))
            assert mpi_error < 1e-8
            results.append(dict(case=name, ranks=ranks, force_error_eV_A=force_error,
                                energy_error_eV=energy_error, mpi_max_difference=mpi_error))
    # Constant force has an exact Verlet trajectory. Cross PBC and check energy,
    # then resume from a restart with the same reference branch and atom IDs.
    case = args.work/'synthetic'
    directory = case/'dynamics';directory.mkdir()
    text = f'include {quote(case / "init.inc")}\ninclude {quote(case / "field.inc")}\n'
    text += 'group probe id 1\nvelocity all set 0 0 0\nvelocity probe set 5 0 0 units box\n'
    text += 'fix integrate all nve\ntimestep 0.001\nthermo 100\nthermo_style custom step pe ke etotal\nrun 0\n'
    text += 'print "START $(etotal:%.17g)" file energy.txt\nrun 100\n'
    text += 'print "END $(etotal:%.17g)" append energy.txt\nwrite_restart state.restart\n'
    text += dump_command.format(name='end')
    assert run(args.lmp, directory, text, 4, 'dynamics') == 0
    energy = [float(line.split()[1]) for line in (directory/'energy.txt').read_text().splitlines()]
    drift = abs(energy[1]-energy[0])
    assert drift < 1e-10
    assert dump(directory/'end.dump')[0,2] > 8.  # actually crossed x boundary
    restart = 'units metal\natom_style atomic\natom_modify map array\nread_restart state.restart\n'
    restart += 'pair_style zero 4.0\npair_coeff * *\n'
    restart += f'include {quote(case / "field.inc")}\nfix integrate all nve\n'
    restart += 'thermo_style custom step pe ke etotal\nrun 0\nprint "RESTART $(etotal:%.17g)" file restart-energy.txt\n'
    restart += 'run 100\nprint "END $(etotal:%.17g)" append restart-energy.txt\n'
    assert run(args.lmp, directory, restart, 2, 'restart') == 0
    restart_energy = [float(line.split()[1]) for line in (directory/'restart-energy.txt').read_text().splitlines()]
    restart_error = max(abs(x-energy[0]) for x in restart_energy)
    assert restart_error < 1e-10
    # Reject a deformed cell instead of reporting a physically incomplete NPT.
    guard = f'include {quote(case / "init.inc")}\ninclude {quote(case / "field.inc")}\n'
    guard += 'change_box all x scale 1.001 remap\nfix integrate all nve\nrun 1\n'
    assert run(args.lmp, directory, guard, 2, 'expected-cell-error') != 0
    assert 'Fix halt' in (directory/'expected-cell-error.stdout').read_text()
    report = dict(passed=True, comparisons=results, synthetic_energy_drift_eV=drift,
                  restart_energy_error_eV=restart_error, rejects_deformed_cell=True,
                  binary=str(args.lmp), binary_sha256=hashlib.sha256(args.lmp.read_bytes()).hexdigest())
    (args.work/'checks.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


def dynamics(args):
    """Additional short NVE check with the real SOG + DFT BEC combinations."""
    if not json.loads((args.work/'checks.json').read_text())['passed']:
        raise ValueError('Complete the static/PBC tests first')
    results = []
    for path in sorted(args.references.glob('*.json')):
        case = args.work/path.stem
        state = json.loads((case/'manifest.json').read_text())
        directory = case/'nve';directory.mkdir()
        text = f'include {quote(case / "init.inc")}\ninclude {quote(case / "field.inc")}\n'
        text += 'velocity all create 100.0 731027 mom yes rot no dist gaussian\n'
        text += 'fix integrate all nve\ntimestep 0.0001\nthermo 10\nthermo_style custom step pe ke etotal f_bec_field\n'
        text += 'run 0\nprint "START $(etotal:%.17g)" file energy.txt\nrun 100\n'
        text += 'print "END $(etotal:%.17g)" append energy.txt\n'
        text += 'write_dump all custom end.dump id type xu yu zu fx fy fz modify sort id format float %.17g\n'
        assert run(args.lmp, directory, text, 4, 'nve') == 0
        energy = [float(line.split()[1]) for line in (directory/'energy.txt').read_text().splitlines()]
        positions = dump(directory/'end.dump')[:,2:5]
        drift = abs(energy[1]-energy[0])/len(positions)
        displacement = float(np.max(np.linalg.norm(positions-state['positions_A'],axis=1)))
        assert np.isfinite(positions).all() and drift < 1e-6 and displacement < .3
        results.append(dict(phase=path.stem, atoms=len(positions), ranks=4, steps=100,
                            timestep_ps=.0001, initial_temperature_K=100.,
                            energy_drift_eV_atom=drift, maximum_displacement_A=displacement))
    report = dict(passed=True, cases=results, purpose='interface smoke test, not equilibrium or switching validation')
    (args.work/'sog-dynamics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmp', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dynamics-only', action='store_true', help='Add short SOG NVE runs to a completed validation directory')
    args = parser.parse_args()
    for name in ['lmp','work','references','model']:
        setattr(args,name,getattr(args,name).resolve())
    (dynamics if args.dynamics_only else main)(args)
