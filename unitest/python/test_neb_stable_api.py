"""Public CLI/Python equivalence, policy provenance, and restart contracts."""
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
from imrmlp import Atoms, NEB, Potential

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT/'build'/os.environ.get('BUILD_MODE', 'release')


def command(directory, *arguments):
    if not (BUILD/'imrmlp').exists():
        pytest.skip('Build imrmlp first')
    env = os.environ.copy()
    env.update(LD_LIBRARY_PATH=str(BUILD)+':'+env.get('LD_LIBRARY_PATH', ''),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return subprocess.run([str(BUILD/'imrmlp'), 'neb', 'neb.ini', *arguments],
                          cwd=directory, env=env, capture_output=True, text=True, timeout=30)


def summary(directory):
    return dict(line[2:].split(maxsplit=1) for line in (directory/'neb.dat').read_text().splitlines()
                if line.startswith('# ') and len(line.split()) == 3)


def inputs(directory, vcneb, driver, policy='stable-v1'):
    initial = Atoms(numbers=[1, 1], elements=['H'], positions=[[0, 0, 0], [1.3, 0, 0]],
                    cell=np.eye(3)*20, pbc=vcneb)
    final = Atoms(numbers=[1, 1], elements=['H'], positions=[[0, 0, 0], [.8, 1.2, .1]],
                  cell=np.diag([21, 20, 20]) if vcneb else np.eye(3)*20, pbc=vcneb)
    (directory/'index.map').write_text('H\n')
    for i, atoms in [(0, initial), (2, final)]:
        positions=''.join('H '+' '.join(map(str, p))+'\n' for p in atoms.positions)
        cell=' '.join(map(str, atoms.cell.flat))
        pbc='T T T' if vcneb else 'F F F'
        (directory/f'image{i}.xyz').write_text(
            f'2\nLattice="{cell}" pbc="{pbc}" Properties=species:S:1:pos:R:3\n'+positions)
    (directory/'neb.ini').write_text('\n'.join([
        'pot:pottype lj612', 'pot:ntypes 1', 'pot:0-0 1 1', 'pot:cut:0-0 3',
        'nimgs 3', 'input-format extxyz', f'driver {driver}', 'maxiteration 10',
        'max_total_iter 10', 'max_move .03', 'neb:gtol 1e-6',
        f'neb:solver_policy {policy}', f'neb:vcneb {str(vcneb).lower()}',
        'neb:pressure .2', 'neb:max_cell_strain .002', 'neb:max_force_evaluations 3',
        'image:0 image0.xyz', 'image:2 image2.xyz', 'neb:climb false',
    ])+'\n')
    return initial, final


@pytest.mark.parametrize('vcneb', [False, True])
@pytest.mark.parametrize('driver', ['fire', 'lbfgs', 'cg', 'sgd'])
def test_neb_stable_cli_and_python_share_controller(tmp_path, vcneb, driver):
    initial, final = inputs(tmp_path, vcneb, driver)
    potential = Potential.from_settings({'pottype': 'lj612', 'ntypes': 1, '0-0': '1 1', 'cut:0-0': 3})
    result = NEB(potential, n_images=3, optimizer=driver, vcneb=vcneb, pressure=.2,
                 max_iter=10, max_move=.03, neb_ftol=1e-6, max_force_evaluations=3,
                 solver_policy='stable-v1').run(initial, final)
    process = command(tmp_path)
    assert process.returncode == (0 if result.converged else 2), process.stdout+process.stderr
    meta = summary(tmp_path)
    assert meta['solver_policy'] == result.solver_policy == 'stable-v1'
    assert meta['climb_delay_units'] == result.climb_delay_units == 'accepted_steps'
    assert meta['solver_phase'] == result.solver_phase
    assert meta['optimizer_iteration_units'] == result.optimizer_iteration_units
    assert meta['force_evaluation_units'] == result.force_evaluation_units
    assert meta['status'] == result.stop_reason
    assert int(meta['optimizer_iterations']) == result.n_iter
    assert int(meta['force_evaluations']) == result.n_force_evals <= 3
    assert int(meta['accepted_steps']) == result.accepted_steps
    assert int(meta['rejected_steps']) == result.rejected_steps
    assert int(meta['image_force_evaluations']) == result.n_image_force_evals
    assert float(meta['max_neb_force']) == pytest.approx(result.max_neb_force, abs=2e-10)
    assert float(meta['last_atomic_move_A']) <= .03*(1+1e-10)
    assert float(meta['last_cell_change']) <= .002*(1+1e-10)
    assert (tmp_path/'neb.dat.cells.csv').exists()
    assert (tmp_path/'neb.dat.iterations.csv').exists()


@pytest.mark.parametrize('stored', ['stable-v1', 'legacy', None])
def test_neb_restart_inherits_recorded_policy_or_legacy(tmp_path, stored):
    inputs(tmp_path, False, 'fire')
    assert command(tmp_path).returncode in (0, 2)
    path = tmp_path/'neb.ini'
    path.write_text('\n'.join(line for line in path.read_text().splitlines()
                              if not line.startswith('neb:solver_policy'))+'\n')
    path = tmp_path/'neb.dat'
    lines=[line for line in path.read_text().splitlines() if not line.startswith('# solver_policy')]
    if stored is not None:
        lines.insert(0, '# solver_policy '+stored)
    path.write_text('\n'.join(lines)+'\n')
    process=command(tmp_path, '--next_step')
    assert process.returncode in (0, 2), process.stdout+process.stderr
    assert summary(tmp_path)['solver_policy'] == (stored or 'legacy')


def test_neb_restart_preserves_accepted_phase_clock(tmp_path):
    inputs(tmp_path, False, 'fire')
    assert command(tmp_path).returncode in (0, 2)
    count=int(summary(tmp_path)['accepted_steps'])
    assert count>0
    process=command(tmp_path, '--next_step')
    assert process.returncode in (0, 2), process.stdout+process.stderr
    meta=summary(tmp_path)
    assert int(meta['initial_accepted_steps'])==count
    assert int(meta['accepted_steps'])>=count


def test_neb_stable_vc_restart_requires_reference_cell(tmp_path):
    inputs(tmp_path, True, 'fire')
    assert command(tmp_path).returncode in (0, 2)
    (tmp_path/'neb_reference_cell.data').unlink()
    process=command(tmp_path, '--next_step')
    assert process.returncode not in (0, 2)
    assert 'reference_cell' in process.stdout+process.stderr


@pytest.mark.parametrize('keyword,value', [('solver_policy', 'unknown'),
                                           ('max_cell_strain', 0),
                                           ('max_force_evaluations', -1)])
def test_neb_invalid_solver_configuration_is_rejected(keyword, value):
    potential=Potential.from_settings({'pottype': 'lj612', 'ntypes': 1, '0-0': '1 1', 'cut:0-0': 3})
    a=Atoms(numbers=[1], elements=['H'], positions=[[0, 0, 0]], cell=np.eye(3)*20, pbc=False)
    b=a.copy();b.positions=[[1, 0, 0]]
    with pytest.raises(ValueError):
        NEB(potential, n_images=3, **{keyword: value}).run(a, b)


@pytest.mark.parametrize('cap', [160, 320, 640])
def test_extended_krylov_is_an_explicit_labeled_diagnostic(tmp_path, cap):
    inputs(tmp_path, False, 'fire')
    with (tmp_path/'neb.ini').open('a') as stream:
        stream.write(f'neb:diagnostic_gmres_max_directions {cap}\n')
    process=command(tmp_path)
    assert process.returncode in (0, 2), process.stdout+process.stderr
    meta=summary(tmp_path)
    assert int(meta['max_gmres_directions'])==cap
    assert meta['numerical_variant']==('extended_krylov_diagnostic' if cap>160 else 'standard')
    assert int(meta['force_evaluations'])<=3


@pytest.mark.parametrize('cap', [0, 641])
def test_invalid_krylov_diagnostic_bounds_are_rejected(tmp_path, cap):
    inputs(tmp_path, False, 'fire')
    with (tmp_path/'neb.ini').open('a') as stream:
        stream.write(f'neb:diagnostic_gmres_max_directions {cap}\n')
    process=command(tmp_path)
    assert process.returncode not in (0, 2)


@pytest.mark.parametrize('variant', ['none', 'secant', 'anderson', 'combined', 'multisecant','guarded','adaptive','cell-block','graph','colored','retained'])
def test_experimental_acceleration_is_labeled_and_budgeted(tmp_path, variant):
    inputs(tmp_path, True, 'lbfgs')
    with (tmp_path/'neb.ini').open('a') as stream:
        stream.write(f'neb:experimental_acceleration {variant}\n')
    process=command(tmp_path)
    assert process.returncode in (0, 2), process.stdout+process.stderr
    meta=summary(tmp_path)
    assert meta['experimental_acceleration']==variant
    assert int(meta['force_evaluations'])<=3
    assert float(meta['last_atomic_move_A'])<=.03*(1+1e-10)
    assert float(meta['last_cell_change'])<=.002*(1+1e-10)


@pytest.mark.parametrize('policy,variant', [('stable-v1','unknown'),('legacy','anderson')])
def test_experimental_acceleration_rejects_invalid_configuration(tmp_path, policy, variant):
    inputs(tmp_path, False, 'fire', policy=policy)
    with (tmp_path/'neb.ini').open('a') as stream:
        stream.write(f'neb:experimental_acceleration {variant}\n')
    process=command(tmp_path)
    assert process.returncode not in (0, 2)
