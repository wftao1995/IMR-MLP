"""Ordinary convstr bulk and sequential modes honor the same discard options."""
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
import imrmlp
from test_cli_error_paths import run_command
from test_streaming_python_contracts import bounded_outcar


def source_file(tmp_path, format):
    source = tmp_path / 'input'
    if format == 'outcar':
        source.write_text(bounded_outcar())
    elif format == 'n2p2':
        source.write_text('begin\nlattice 6 0 0\nlattice 0 6 0\nlattice 0 0 6\n'
                          'atom 1 2 3 H .25 -1 .1 .2 .3\nenergy -1\ncharge .25\nend\n'
                          'begin\nlattice 6 0 0\nlattice 0 6 0\nlattice 0 0 6\n'
                          'atom 2 3 4 H -.25 -2 .2 .3 .4\nenergy -2\ncharge -.25\nend\n')
    else:
        frames = []
        for i in range(2):
            atom = imrmlp.Atoms(numbers=[1], positions=[[1. + i, 2., 3.]], cell=np.eye(3) * 3)
            atom.energy = -9. - i
            atom.free_energy = -10. - i
            atom.forces = [[.1 + i, .2 + i, .3 + i]]
            atom.stress = np.eye(3) * (i + 1.)
            atom._impl.site_energies = [-9. - i]
            frames.append(atom)
        imrmlp.write(source, frames, format=format)
    return source


@pytest.mark.parametrize('format', ['extxyz', 'mlpcfg', 'bin', 'outcar', 'n2p2'])
@pytest.mark.parametrize('mode', [(0, 0), (1, 0), (1, 1)])
@pytest.mark.parametrize('discard', [[], ['free_energy'], ['energy', 'fenergy', 'forces', 'virial', 'site_en']])
def test_convstr_ordinary_modes_preserve_only_unselected_labels(tmp_path, format, mode, discard):
    source = source_file(tmp_path, format)
    original_bytes = source.read_bytes()
    before = imrmlp.read(source, format=format)
    args = ['--inputfiles', source, '--input-format', format,
            '--outputfile', 'filtered.xyz', '--output-format', 'extxyz',
            '--loadrate', str(mode[0]), '--outrate', str(mode[1])]
    if discard:
        args += ['--discard', *discard]
    result = run_command('convstr', args, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert source.read_bytes() == original_bytes
    after = imrmlp.read(tmp_path / 'filtered.xyz')
    assert len(after) == len(before) == 2
    for actual, prior in zip(after, before):
        assert actual.energy == (None if 'energy' in discard else prior.energy)
        assert actual.free_energy == (None if {'fenergy', 'free_energy'}.intersection(discard) else prior.free_energy)
        np.testing.assert_array_equal(actual.positions, prior.positions)
        np.testing.assert_allclose(actual.cell, prior.cell, atol=1e-13, rtol=0)
        np.testing.assert_array_equal(actual.numbers, prior.numbers)
        for property, label in [('forces', 'forces'), ('stress', 'virial'), ('site_energies', 'site_en')]:
            value = getattr(actual, property)
            expected = None if label in discard else getattr(prior, property)
            if expected is None:
                assert value is None
            else:
                np.testing.assert_allclose(value, expected, atol=1e-13, rtol=0)
        if format == 'n2p2':
            np.testing.assert_array_equal(actual.charges, prior.charges)


@pytest.mark.parametrize('mode', [(1, 0), (1, 1)])
def test_imrmlp_sequential_discard_removes_only_requested_arbitrary_metadata(tmp_path, mode):
    source = tmp_path / 'input.xyz'
    source.write_text('1\nLattice="3 0 0 0 3 0 0 0 3" Properties=species:S:1:pos:R:3 '
                      'energy=-9 fenergy=-10 temperature=300 keep=7\nH 1 2 3\n' * 2)
    result = run_command('imrmlp', ['convstr', '--inputfiles', source, '--input-format', 'extxyz',
                                   '--outputfile', 'filtered.xyz', '--output-format', 'extxyz',
                                   '--loadrate', str(mode[0]), '--outrate', str(mode[1]),
                                   '--discard', 'temperature'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    text = (tmp_path / 'filtered.xyz').read_text()
    assert 'temperature=' not in text
    assert text.count('keep=7') == 2
    for atom in imrmlp.read(tmp_path / 'filtered.xyz'):
        assert atom.energy == -9
        assert atom.free_energy == -10
