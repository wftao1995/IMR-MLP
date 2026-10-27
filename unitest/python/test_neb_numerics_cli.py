"""NEB optimizer regression with a real LJ potential and distorted image chain."""
import math

import pytest

from test_cli_error_paths import run_command


@pytest.mark.parametrize('policy', ['stable-v1', 'legacy'])
def test_improved_tangent_fire_writes_final_step_and_residuals(tmp_path, policy):
    (tmp_path / 'index.map').write_text('H\n')
    settings = [
        'pot:pottype lj612', 'pot:ntypes 1', 'pot:0-0 1 1', 'pot:cut:0-0 4',
        'nimgs 3', 'input-format extxyz', 'driver fire',
        'max_total_iter 1', 'maxiteration 1', 'max_move 0.1', 'ts_init 2',
        'neb:gtol 1e-6', 'neb:spring 1', 'neb:ttype 1', 'neb:stype 1',
        'neb:climb false', 'neb:check_mic false', f'neb:solver_policy {policy}',
    ]
    for i, x in enumerate((0, 0.3, 1)):
        (tmp_path / f'image{i}.xyz').write_text(
            '1\nLattice="10 0 0 0 10 0 0 0 10" pbc="F F F" '
            'Properties=species:S:1:pos:R:3\n'
            f'H {x:.16g} 0 0\n'
        )
        settings.append(f'image:{i} image{i}.xyz')
    (tmp_path / 'neb.ini').write_text('\n'.join(settings) + '\n')
    result = run_command('imrmlp', ['neb', '--settings', 'neb.ini', '--num_threads', '1'], tmp_path)
    assert result.returncode == 2, result.stdout[-3000:] + result.stderr

    # With no physical neighbors, the initial spring force is 0.7 - 0.3.
    # FIRE's capped single step moves the center to 0.4, leaving force 0.2.
    frames = (tmp_path / 'mep.xyz').read_text().splitlines()
    assert len(frames) == 9
    for i, x in enumerate((0, 0.4, 1)):
        position = [float(v) for v in frames[3*i+2].split()[1:4]]
        assert position == pytest.approx([x, 0, 0], abs=1e-12)

    summary = (tmp_path / 'neb.dat').read_text()
    metadata = dict(line[2:].split(maxsplit=1) for line in summary.splitlines() if line.startswith('# '))
    assert metadata['status'] == 'max_iterations'
    assert int(metadata['optimizer_iterations']) == 1
    assert float(metadata['max_neb_force']) == pytest.approx(0.2, abs=1e-12)
    assert float(metadata['max_spring_force']) == pytest.approx(0.2, abs=1e-12)
    assert float(metadata['max_perp_force']) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize('driver', ['fire', 'cg', 'lbfgs', 'sgd'])
@pytest.mark.parametrize('budget', [1, pytest.param(1500, marks=pytest.mark.extended)])
@pytest.mark.parametrize('tangent', [1, 2, 3])
@pytest.mark.parametrize('policy', ['stable-v1', 'legacy'])
def test_tangents_relax_distorted_lj_path(tmp_path, driver, budget, tangent, policy):
    (tmp_path / 'index.map').write_text('H\n')
    equilibrium = 2 ** (1 / 6)
    settings = [
        'pot:pottype lj612', 'pot:ntypes 1', 'pot:0-0 1 1', 'pot:cut:0-0 4',
        'nimgs 7', 'input-format extxyz', f'driver {driver}',
        'max_total_iter 1500', f'maxiteration {budget}', 'max_move 0.03', 'neb:gtol 1e-6',
        f'neb:ttype {tangent}', 'neb:climb false', 'neb:check_mic false', f'neb:solver_policy {policy}',
    ]
    # The legacy fixed SGD step of 0.1 oscillates on this stiff LJ path.
    # Compare valid legacy paths using its existing explicit step parameter.
    if policy == 'legacy' and driver == 'sgd':
        settings.append('max_step_init 0.01')
    for i in range(7):
        t = i / 6
        bend = math.sin(math.pi * t)
        left = 8 + 2 * t
        (tmp_path / f'image{i}.xyz').write_text(
            '2\nLattice="20 0 0 0 20 0 0 0 20" Properties=species:S:1:pos:R:3\n'
            f'H {left:.16g} 10 10\n'
            f'H {left + equilibrium + 0.02*bend:.16g} {10+0.03*bend:.16g} 10\n'
        )
        settings.append(f'image:{i} image{i}.xyz')
    (tmp_path / 'neb.ini').write_text('\n'.join(settings) + '\n')
    result = run_command('imrmlp', ['neb', '--settings', 'neb.ini', '--num_threads', '1'], tmp_path)
    assert result.returncode == (2 if budget == 1 else 0), result.stdout[-3000:] + result.stderr
    summary = (tmp_path / 'neb.dat').read_text()
    metadata = dict(line[2:].split(maxsplit=1) for line in summary.splitlines() if line.startswith('# '))
    assert int(metadata['optimizer_iterations']) <= budget
    if budget == 1:
        assert metadata['status'] == 'max_iterations'
        assert float(metadata['max_neb_force']) > 1e-6
        return
    assert metadata['status'] == 'converged'
    assert float(metadata['max_neb_force']) < 1e-6
    frames = (tmp_path / 'mep.xyz').read_text().splitlines()
    assert len(frames) == 28
    for i in range(7):
        positions = [[float(v) for v in frames[4*i+j].split()[1:4]] for j in (2, 3)]
        separation = math.dist(*positions)
        assert separation == pytest.approx(equilibrium, abs=2e-7)
