"""Known-count AXSF output has standard framing and independent unit oracles."""
import gzip

import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
import imrmlp
from test_cli_error_paths import run_command


def frames(kind):
    pbc = {'fixed': [True]*3, 'variable': [True, True, False],
           'molecule': [False]*3, 'polymer': [True, False, False]}[kind]
    cell = np.diag([4., 5., 6.])
    a = imrmlp.Atoms(numbers=[1, 8], elements=['O', 'C', 'H'],
                   positions=[[1, 2, 3], [2, 3, 4]], cell=cell, pbc=pbc)
    a.forces = [[54.422793282616, -27.211396641308, 0], [0, 0, 0]]
    a.energy = -1
    b = a.copy(); b.positions = [[2, 3, 4], [3, 4, 5]]
    if kind in ('variable', 'molecule'): b.cell = cell * 2
    if kind != 'molecule': a.to_fractional()
    return [a, b]


@pytest.mark.parametrize('kind', ['fixed', 'variable', 'molecule', 'polymer'])
@pytest.mark.parametrize('suffix', ['.axsf', '.axsf.gz'])
def test_axsf_known_list_writes_standard_cells_units_and_preserves_sources(tmp_path, kind, suffix):
    source = frames(kind)
    saved = [a.copy() for a in source]
    path = tmp_path / ('trajectory' + suffix)
    imrmlp.write(path, source) # Extension selects known-count animation output.
    text = gzip.decompress(path.read_bytes()).decode() if suffix.endswith('.gz') else path.read_text()
    assert text.startswith('ANIMSTEPS 2\n')
    rows = [line.split() for line in text.splitlines() if line.startswith('H ')]
    np.testing.assert_allclose([float(x) for x in rows[0][1:]], [1, 2, 3, 2, -1, 0], atol=1e-13)
    if kind == 'variable':
        assert 'PRIMVEC 1\n' in text and 'PRIMVEC 2\n' in text
    elif kind == 'molecule':
        assert 'PRIMVEC' not in text and 'ATOMS 1\n' in text and 'ATOMS 2\n' in text
    else:
        assert text.count('PRIMVEC\n') == 1 and 'PRIMVEC 1' not in text
    result = list(imrmlp.iread(path))
    assert len(result) == 2
    for i, a in enumerate(result):
        np.testing.assert_array_equal(a.numbers, [1, 8])
        np.testing.assert_array_equal(a.pbc, saved[i].pbc)
        np.testing.assert_allclose(a.positions, [[1+i, 2+i, 3+i], [2+i, 3+i, 4+i]], atol=1e-13)
        np.testing.assert_allclose(a.forces, saved[i].forces, atol=1e-13)
        assert a.energy is None # Standard XSF stores geometry and optional forces.
        np.testing.assert_allclose(a.cell, np.zeros((3, 3)) if kind == 'molecule' else saved[i].cell, atol=1e-13)
        np.testing.assert_array_equal(source[i].positions, saved[i].positions)
        assert source[i]._impl.cartesian == saved[i]._impl.cartesian




@pytest.mark.parametrize('invalid', ['count', 'pbc', 'position'])
def test_axsf_bad_later_frames_leave_existing_target_unchanged(tmp_path, invalid):
    source = frames('fixed')
    if invalid == 'count':
        source[1] = imrmlp.Atoms(numbers=[1], elements=['O', 'C', 'H'], positions=[[0, 0, 0]], cell=np.eye(3)*4)
    elif invalid == 'pbc': source[1].pbc = [False]*3
    else: source[1].positions = [[np.nan, 0, 0], [1, 2, 3]]
    path = tmp_path / 'keep.axsf'; path.write_bytes(b'keep original\n')
    with pytest.raises(RuntimeError): imrmlp.write(path, source, format='axsf')
    assert path.read_bytes() == b'keep original\n'


def test_axsf_empty_list_and_unsupported_unknown_count_writer_do_not_touch_target(tmp_path):
    path = tmp_path / 'keep.axsf'; path.write_bytes(b'keep original\n')
    imrmlp.write(path, [], format='axsf')
    assert path.read_bytes() == b'keep original\n'
    consumed = []
    def unknown():
        consumed.append(True)
        yield frames('fixed')[0]
    with pytest.raises(ValueError, match='write_iter supports only'):
        imrmlp.write_iter(path, unknown(), format='axsf')
    assert consumed == [] and path.read_bytes() == b'keep original\n'


def test_axsf_cli_uses_known_vector_mode_and_rejects_unknown_count_before_open(tmp_path):
    (tmp_path / 'input.xyz').write_text(
        '1\nLattice="4 0 0 0 5 0 0 0 6" Properties=species:S:1:pos:R:3\nH 1 2 3\n'
        '1\nLattice="4 0 0 0 5 0 0 0 6" Properties=species:S:1:pos:R:3\nH 2 3 4\n')
    args = ['convstr', '--input-format=extxyz', '--output-format=axsf', '--inputfiles', 'input.xyz']
    result = run_command('imrmlp', args + ['--outputfile=output.axsf'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'output.axsf').read_text().startswith('ANIMSTEPS 2\n')
    result = run_command('imrmlp', args + ['--loadrate=1', '--outrate=1', '--outputfile=unopened.axsf'], tmp_path)
    assert result.returncode != 0 and 'known-count' in result.stdout + result.stderr
    assert not (tmp_path / 'unopened.axsf').exists()
