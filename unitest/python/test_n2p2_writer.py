"""N2P2 output uses standard training blocks and bounded iterable state."""
import gzip

import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
import imrmlp


def training_frame(count=1):
    a = imrmlp.Atoms(numbers=[8] * count, positions=[[.529177210903, 0, 0]] * count,
                   cell=np.eye(3) * 1.058354421806)
    a.forces = [[51.422087120633655, 0, 0]] * count
    a.energy = -27.211396641308
    a._impl.charges = [.25] * count
    a._impl.site_energies = [54.422793282616] * count
    a.total_charge = -.5
    return a


@pytest.mark.parametrize('format', ['', 'n2p2', 'n2p2dat'])
@pytest.mark.parametrize('single', [False, True])
def test_n2p2_public_writer_text_units_and_label_roundtrip(tmp_path, format, single):
    a = training_frame()
    path = tmp_path / 'training.n2p2'
    imrmlp.write(path, a if single else [a, a.copy()], format=format)
    row = next(line.split() for line in path.read_text().splitlines() if line.startswith('atom '))
    assert row[4] == 'O'
    np.testing.assert_allclose([float(x) for x in row[1:4]], [1, 0, 0], atol=1e-14)
    np.testing.assert_allclose([float(x) for x in row[5:]], [.25, 2, 1, 0, 0], atol=1e-14)
    frames = imrmlp.read(path, format='n2p2')
    assert len(frames) == (1 if single else 2)
    np.testing.assert_allclose(frames[0].forces, a.forces)
    np.testing.assert_allclose(frames[0].site_energies, a.site_energies)
    assert frames[0].total_charge == -.5
    assert frames[0].energy == a.energy


@pytest.mark.parametrize('suffix', ['.n2p2', '.n2p2.gz'])
def test_n2p2_write_iter_one_pass_variable_counts_fractional_and_compression(tmp_path, suffix):
    frames = [training_frame(1), training_frame(2)]
    frames[0].to_fractional()
    before = frames[0].positions.copy()
    visited = []
    def once():
        for i, a in enumerate(frames):
            visited.append(i)
            yield a
    path = tmp_path / ('frames' + suffix)
    imrmlp.write_iter(path, once(), format='n2p2dat')
    assert visited == [0, 1]
    assert not frames[0]._impl.cartesian
    np.testing.assert_array_equal(frames[0].positions, before)
    result = list(imrmlp.iread(path, format='n2p2'))
    assert [len(a) for a in result] == [1, 2]
    np.testing.assert_allclose(result[0].positions, [[.529177210903, 0, 0]], atol=1e-14)
    if suffix.endswith('.gz'):
        assert gzip.decompress(path.read_bytes()).count(b'begin\n') == 2


def test_n2p2_write_iter_empty_keeps_target_and_late_error_has_complete_prefix(tmp_path):
    path = tmp_path / 'frames.n2p2'
    path.write_bytes(b'keep\n')
    imrmlp.write_iter(path, iter(()), format='n2p2')
    assert path.read_bytes() == b'keep\n'
    invalid = imrmlp.Atoms(numbers=[8], positions=[[0, 0, 0]], cell=np.eye(3))
    with pytest.raises(RuntimeError, match='Failed to write'):
        imrmlp.write_iter(path, iter([training_frame(), invalid]), format='n2p2')
    assert path.read_text().count('begin\n') == 1
    assert len(imrmlp.read(path, format='n2p2')) == 1
