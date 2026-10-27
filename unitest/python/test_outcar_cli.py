"""OUTCAR conversion consumes complete steps within the selected input range."""
import gzip

import numpy as np
import pytest

from test_cli_error_paths import run_command
from test_streaming_python_contracts import bounded_outcar, OUTCAR_ITER
import imrmlp


def outcar_with_stress(*, bad_second=False):
    text = bounded_outcar(second=OUTCAR_ITER + ' 1(1)\n' if bad_second else None)
    stress = ('  FORCE on cell =-STRESS in cart. coord. units (eV):\n'
              '  Direction XX YY ZZ XY YZ ZX\n  in kB 1 2 3 4 5 6\n')
    return text.replace(' VOLUME and BASIS-vectors', stress + ' VOLUME and BASIS-vectors')


def convert_outcar(tmp_path, *, entry='imrmlp', loadrate=1, outrate=1,
                   compressed=False, bad_second=False, options=(), explicit=True):
    (tmp_path / 'index.map').write_text('H 0\n')
    source = tmp_path / ('OUTCAR.gz' if compressed else 'OUTCAR')
    text = outcar_with_stress(bad_second=bad_second)
    source.write_bytes(gzip.compress(text.encode()) if compressed else text.encode())
    args = ['convstr'] if entry == 'imrmlp' else []
    if explicit:
        args += ['--input-format', 'outcar']
    args += ['--output-format', 'extxyz', '--index', 'index.map',
             '--loadrate', str(loadrate), '--outrate', str(outrate),
             '--lossless', 'true', *options, '--inputfiles', source.name,
             '--outputfile', 'output.xyz']
    return run_command(entry, args, tmp_path)


def check_outcar_frames(path, expected_steps):
    frames = imrmlp.read(path)
    assert len(frames) == len(expected_steps)
    tensor = np.array([[1., 4., 6.], [4., 2., 5.], [6., 5., 3.]])
    for atom, step in zip(frames, expected_steps):
        side = step + 2
        assert atom.natoms == 2
        np.testing.assert_array_equal(atom.numbers, [1, 1])
        np.testing.assert_allclose(atom.cell, np.eye(3) * side)
        np.testing.assert_allclose(atom.positions, [[step + 1, 2, 3], [step + 4, 5, 6]])
        np.testing.assert_allclose(atom.forces, [[.1, .2, .3], [-.1, -.2, -.3]])
        assert atom.energy == pytest.approx(-10 - step)
        np.testing.assert_allclose(atom.stress, tensor * side**3 * .000624150912588,
                                   rtol=1e-12, atol=1e-14)
    return frames


@pytest.mark.parametrize('entry,loadrate,outrate,compressed,explicit,steps', [
    ('imrmlp', 0, 0, False, True, [0, 1]),
    ('imrmlp', 1, 1, False, False, [0, 1]),
    ('convstr', 1, 0, False, True, [0, 1]),
    ('convstr', 1, 1, True, False, [0, 1]),
    ('imrmlp', 2, 1, False, True, [0]),
    ('imrmlp', 1, 2, False, True, [0]),
])
def test_outcar_conversion_preserves_physical_fields_and_sampling(
        tmp_path, entry, loadrate, outrate, compressed, explicit, steps):
    result = convert_outcar(tmp_path, entry=entry, loadrate=loadrate, outrate=outrate,
                            compressed=compressed, explicit=explicit)
    assert result.returncode == 0, result.stdout + result.stderr
    check_outcar_frames(tmp_path / 'output.xyz', steps)


@pytest.mark.parametrize('outrate', [0, 1])
@pytest.mark.parametrize('compressed', [False, True])
def test_outcar_sequential_end_does_not_parse_unrequested_bad_step(tmp_path, outrate, compressed):
    result = convert_outcar(tmp_path, outrate=outrate, compressed=compressed,
                            bad_second=True, options=('--end', '0'))
    assert result.returncode == 0, result.stdout + result.stderr
    check_outcar_frames(tmp_path / 'output.xyz', [0])


@pytest.mark.parametrize('mode', [(0, 0), (1, 0), (1, 1)])
def test_outcar_in_range_bad_step_fails_conversion(tmp_path, mode):
    result = convert_outcar(tmp_path, loadrate=mode[0], outrate=mode[1], bad_second=True)
    assert result.returncode != 0, result.stdout + result.stderr
    assert 'OUTCAR' in result.stdout + result.stderr or 'ERROR WHEN READING FILE' in result.stdout + result.stderr
    if mode == (1, 1):
        check_outcar_frames(tmp_path / 'output.xyz', [0])


def test_outcar_missing_input_is_a_failure_even_after_valid_input(tmp_path):
    (tmp_path / 'OUTCAR').write_text(outcar_with_stress())
    (tmp_path / 'index.map').write_text('H 0\n')
    output = tmp_path / 'output.xyz'
    output.write_text('existing output\n')
    result = run_command('convstr', [
        '--input-format', 'outcar', '--output-format', 'extxyz', '--index', 'index.map',
        '--loadrate', '1', '--outrate', '1', '--overwrite',
        '--inputfiles', 'OUTCAR', 'missing.OUTCAR', '--outputfile', output.name,
    ], tmp_path)
    assert result.returncode != 0
    assert 'cannot open' in result.stdout + result.stderr
    assert output.read_text() == 'existing output\n'
