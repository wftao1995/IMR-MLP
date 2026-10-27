"""Standard lattice-free N2P2 molecules stay nonperiodic through bounded IO."""
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
import imrmlp

MOLECULE = ('begin\natom 1 -2 0 H .25 2 1 -2 0\nenergy -3\ncharge -.5\nend\n')
PERIODIC = ('begin\nlattice 2 0 0\nlattice 0 2 0\nlattice 0 0 2\n'
            'atom 0 0 0 H 0 0 0 0 0\nend\n')


@pytest.mark.parametrize('suffix', ['.n2p2', '.n2p2.gz'])
def test_molecular_and_periodic_n2p2_blocks_share_bounded_io(tmp_path, suffix):
    source = tmp_path / 'authored.n2p2'
    source.write_text(MOLECULE + 'begin\nend\n' + PERIODIC + MOLECULE)
    first, empty, periodic, last = imrmlp.read(source)
    for a in (first, empty, last):
        np.testing.assert_array_equal(a.pbc, [False]*3)
        np.testing.assert_array_equal(a.cell, np.zeros((3, 3)))
    assert len(empty) == 0 and empty.energy is None
    np.testing.assert_array_equal(periodic.pbc, [True]*3)
    np.testing.assert_allclose(first.positions, [[.529177210903, -1.058354421806, 0]], atol=1e-14)
    np.testing.assert_allclose(first.forces, [[51.422087120633655, -102.84417424126731, 0]], atol=1e-12)
    assert first.energy == pytest.approx(-81.634189923924)
    assert first.total_charge == -.5
    path = tmp_path / ('mixed' + suffix)
    imrmlp.write_iter(path, imrmlp.iread(source), format='n2p2')
    result = list(imrmlp.iread(path, format='n2p2'))
    assert [len(a) for a in result] == [1, 0, 1, 1]
    for a, b in zip(result, (first, empty, periodic, last)):
        np.testing.assert_array_equal(a.pbc, b.pbc)
        np.testing.assert_allclose(a.cell, b.cell, atol=1e-14)
        np.testing.assert_allclose(a.positions, b.positions, atol=1e-14)
        assert a.energy == b.energy
    first.cell = np.eye(3) * 10
    imrmlp.write_one(tmp_path / 'padded.n2p2', first)
    assert 'lattice' not in (tmp_path / 'padded.n2p2').read_text()
    np.testing.assert_array_equal(first.cell, np.eye(3) * 10)


@pytest.mark.parametrize('lattice', [
    'lattice 1 0 0\n', 'lattice 1 0 0\nlattice 0 1 0\n'])
def test_n2p2_partial_lattice_remains_malformed(tmp_path, lattice):
    path = tmp_path / 'bad.n2p2'
    path.write_text('begin\n' + lattice + 'atom 0 0 0 H 0 0 0 0 0\nend\n')
    with pytest.raises(RuntimeError):
        list(imrmlp.iread(path))
