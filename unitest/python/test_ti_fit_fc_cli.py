"""Exercise harmonic fitting/export through the public CLI and file formats."""
import csv

import numpy as np
import pytest

from test_calc_fc_cli import MODEL, read_full_fc
from test_cli_error_paths import run_command


def prepare(tmp_path, energy=True):
    positions = np.array([[7., 7., 7.], [9.5, 7., 7.], [7.8, 9., 7.1]])
    hessian = np.kron(3 * np.eye(3) - np.ones((3, 3)), np.eye(3))
    (tmp_path / 'index.map').write_text('H 0.4 1.0\n')

    def frame(pos, forces=None, cell=20.):
        props = 'species:S:1:pos:R:3' + (':forces:R:3' if forces is not None else '')
        header = f'3\nLattice="{cell} 0 0 0 {cell} 0 0 0 {cell}" Properties={props}'
        header += ' energy=-1\n' if energy else '\n'
        return header + ''.join('H ' + ' '.join(map(str, row)) + '\n'
                                for row in (pos if forces is None else np.hstack((pos, forces))))

    (tmp_path / 'reference.xyz').write_text(frame(positions))
    lines = ['3 3']
    for i in range(3):
        for j in range(3):
            lines += [f'{i+1} {j+1}']
            lines += [' '.join(map(str, row)) for row in hessian[3*i:3*i+3, 3*j:3*j+3]]
    (tmp_path / 'initial.fc').write_text('\n'.join(lines) + '\n')
    displacements = np.random.default_rng(42).normal(scale=.005, size=(10, 3, 3))
    displacements -= displacements.mean(axis=1, keepdims=True)
    displacements = np.concatenate((displacements, -displacements))
    trajectory = ''.join(frame(positions + u, (-hessian @ u.ravel()).reshape(3, 3))
                         for u in displacements)
    (tmp_path / 'trajectory.xyz').write_text(trajectory)
    args = ['ti_fit_fc', '--input', 'reference.xyz', '--input-format', 'extxyz',
            '--initial-fc', 'initial.fc', '--index', 'index.map',
            '--output-prefix', 'fit', '--write-validation-ensemble=false', '--num_threads', '1']
    return args, hessian, trajectory


def outputs(tmp_path):
    for name in ('FORCE_CONSTANTS', 'REFERENCE.xyz', 'REFERENCE.lmp', 'REPORT.dat'):
        assert (tmp_path / ('fit_' + name)).is_file(), name
    text = (tmp_path / 'fit_LAMMPS_REFERENCE').read_text()
    assert text.startswith('TI_HARMONIC_REFERENCE 1\nunits metal\nnatoms 3\n')
    assert '# lammps_mode eigenvalue frequency_thz' in (tmp_path / 'fit_REPORT.dat').read_text()
    return text


def test_export_needs_no_potential_and_preserves_force_constants(tmp_path):
    args, hessian, _ = prepare(tmp_path)
    result = run_command('imrmlp', args + ['--workflow', 'export', '--potfile', 'missing.mtp'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    text = outputs(tmp_path)
    assert 'energy0 -1\n' in text
    np.testing.assert_allclose(read_full_fc(tmp_path / 'fit_FORCE_CONSTANTS'), hessian, atol=1e-12)
    report = dict(line.split(maxsplit=1) for line in (tmp_path / 'fit_REPORT.dat').read_text().splitlines()
                  if line and not line.startswith('#'))
    assert float(report['lammps_asr_error']) < 1e-12


def test_md_fits_saved_trajectory_forces(tmp_path):
    args, hessian, _ = prepare(tmp_path)
    # Omit --input to also test the fixed-cell centroid workflow.
    del args[1:5]
    result = run_command('imrmlp', args + ['--workflow', 'md', '--potfile', MODEL,
                         '--traj', 'trajectory.xyz', '--traj-format', 'extxyz',
                         '--force-source', 'trajectory'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    outputs(tmp_path)
    np.testing.assert_allclose(read_full_fc(tmp_path / 'fit_FORCE_CONSTANTS'), hessian, atol=1e-8)


@pytest.mark.parametrize('workflow', ['ensemble', 'md-ensemble'])
def test_self_consistent_workflows_write_iteration_diagnostics(tmp_path, workflow):
    args, _, _ = prepare(tmp_path)
    args += ['--workflow', workflow, '--potfile', MODEL, '--temperature', '1',
             '--ensemble-size', '8', '--warmup-iterations', '0',
             '--min-iterations', '1', '--max-iterations', '2', '--consecutive-converged', '1']
    if workflow == 'md-ensemble':
        args += ['--traj', 'trajectory.xyz', '--traj-format', 'extxyz', '--force-source', 'trajectory']
    result = run_command('imrmlp', args, tmp_path)
    # Two iterations deliberately exhaust the budget: retain diagnostics and signal nonconvergence.
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'converged: no' in result.stdout
    outputs(tmp_path)
    with (tmp_path / 'fit_ITERATIONS.csv').open() as stream:
        records = list(csv.DictReader(stream))
    assert 1 <= len(records) <= 2
    assert all(float(row['sampling_attempts']) >= 8 for row in records)


def test_md_rejects_changing_cell(tmp_path):
    args, _, trajectory = prepare(tmp_path)
    trajectory = trajectory.replace('20.0 0 0 0 20.0 0 0 0 20.0',
                                    '21.0 0 0 0 21.0 0 0 0 21.0', 1)
    (tmp_path / 'trajectory.xyz').write_text(trajectory)
    result = run_command('imrmlp', args + ['--workflow', 'md', '--traj', 'trajectory.xyz',
                         '--traj-format', 'extxyz', '--potfile', MODEL], tmp_path)
    assert result.returncode != 0
    assert 'cell' in (result.stdout + result.stderr).lower()
    assert not (tmp_path / 'fit_LAMMPS_REFERENCE').exists()


def test_sparse_export_reconstructs_onsite_blocks(tmp_path):
    args, _, _ = prepare(tmp_path)
    result = run_command('imrmlp', args + ['--workflow', 'export',
                         '--lammps-fc-source', 'initial', '--lammps-fc-cutoff', '2.6',
                         '--lammps-fc-min-frequency-thz', '.5'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    text = outputs(tmp_path)
    assert 'blocks 5\n' in text  # Remove the longest edge (2,3), leaving a connected graph.
    block_tokens = text.split('blocks 5\n')[1].split('end')[0].split()
    matrix = np.zeros((9, 9))
    for offset in range(0, len(block_tokens), 11):
        i, j, *values = block_tokens[offset:offset+11]
        i, j = int(i)-1, int(j)-1
        block = np.array(values, dtype=float).reshape(3, 3)
        matrix[3*i:3*i+3, 3*j:3*j+3] = block
        matrix[3*j:3*j+3, 3*i:3*i+3] = block.T
    np.testing.assert_allclose(matrix.reshape(3, 3, 3, 3).sum(axis=2), 0, atol=1e-12)
    assert np.linalg.eigvalsh(matrix)[3] > 0
