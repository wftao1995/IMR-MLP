"""Bounded Python I/O owns frames and preserves single-pass writer contracts."""
from contextlib import closing
import gzip
import shlex

import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
import imrmlp


FIRST = ('2\nLattice="6 0 0 .2 7 0 .1 .3 8" pbc="T T F" '
         'Properties=species:S:1:pos:R:3:forces:R:3:charges:R:1 '
         'energy=-.5 virial="1 0 0 0 2 0 0 0 3" '
         'label="first frame" config_type="bulk"\n'
         'H .1 .2 .3 .4 .5 .6 .25\nHe 1.6 .7 .8 -.4 -.5 -.6 -.25\n')


SECOND = ('1\nLattice="9 .2 .1 0 10 .3 0 0 11" pbc="F F F" '
          'Properties=species:S:1:pos:R:3 energy=.25 label="second frame"\n'
          'He 2 3 4\n')


THIRD = ('3\nLattice="12 0 0 0 13 0 0 0 14" pbc="T T T" '
         'Properties=species:S:1:pos:R:3:forces:R:3 energy=1.5 '
         'label="third frame"\n'
         'He .2 .3 .4 .1 .2 .3\nH 2.2 .5 .6 -.1 -.2 -.3\nH 1.1 2.1 .8 0 0 0\n')


EMPTY_FRAME = ('0\nLattice="4 0 0 0 5 0 0 0 6" pbc="F F F" '
               'Properties=species:S:1:pos:R:3 label="empty frame"\n')


def assert_same(actual, expected):
    np.testing.assert_array_equal(actual.numbers, expected.numbers)
    np.testing.assert_array_equal(actual.pbc, expected.pbc)
    np.testing.assert_allclose(actual.positions, expected.positions, atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual.cell, expected.cell, atol=1e-12, rtol=0)
    assert actual.energy == expected.energy
    for field in ('forces', 'stress'):
        reference = getattr(expected, field)
        if reference is None:
            assert getattr(actual, field) is None
        else:
            np.testing.assert_allclose(getattr(actual, field), reference, atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual._impl.charges, expected._impl.charges, atol=1e-12, rtol=0)


@pytest.mark.parametrize('format', ['', 'extxyz', 'xyz'])
def test_streamed_mixed_frames_match_bulk_and_keep_their_own_labels(tmp_path, format):
    path = tmp_path / 'frames.extxyz'
    path.write_text(FIRST + SECOND + THIRD)
    expected = imrmlp.read(path)
    actual = list(imrmlp.iread(path, format=format))
    assert [len(frame) for frame in actual] == [2, 1, 3]
    for i, (frame, reference) in enumerate(zip(actual, expected)):
        assert_same(frame, reference)
        output = tmp_path / f'frame-{i}.xyz'
        imrmlp.write_one(output, frame)
        fields = shlex.split(output.read_text().splitlines()[1])
        assert f'label={"first second third".split()[i]} frame' in fields


def test_next_and_close_do_not_reuse_or_invalidate_previous_frames(tmp_path):
    path = tmp_path / 'frames.xyz'
    path.write_text(FIRST + SECOND)
    frames = imrmlp.iread(path)
    first = next(frames)
    saved = first.copy()
    second = next(frames)
    assert_same(first, saved)
    second.positions = [[9., 8., 7.]]
    frames.close()
    frames.close()
    with pytest.raises(StopIteration):
        next(frames)
    assert_same(first, saved)


def test_closing_supports_early_exit_without_consuming_bad_suffix(tmp_path):
    path = tmp_path / 'early.xyz'
    path.write_text(FIRST + 'invalid later frame\n')
    with closing(imrmlp.iread(path)) as frames:
        first = next(frames)
    assert len(first) == 2
    np.testing.assert_allclose(first.positions[0], [.1, .2, .3])
    with pytest.raises(StopIteration):
        next(frames)


def test_empty_input_and_complete_zero_atom_frame_are_distinct(tmp_path):
    path = tmp_path / 'empty.xyz'
    path.write_text(' \n\t\n')
    assert list(imrmlp.iread(path)) == []
    path.write_text(EMPTY_FRAME + SECOND)
    frames = list(imrmlp.iread(path))
    assert [len(frame) for frame in frames] == [0, 1]
    assert frames[0].positions.shape == (0, 3)
    np.testing.assert_array_equal(frames[0].pbc, [False, False, False])


def test_bad_second_frame_is_reported_when_consumed_and_closes_iteration(tmp_path):
    path = tmp_path / 'bad-tail.xyz'
    path.write_text(FIRST + '1\nProperties=species:S:1:pos:R:3\n')
    frames = imrmlp.iread(path)
    first = next(frames)
    saved = first.copy()
    with pytest.raises(RuntimeError, match='EXTXYZ frame 1'):
        next(frames)
    with pytest.raises(StopIteration):
        next(frames)
    assert_same(first, saved)


def test_gzip_decode_is_lazy_and_bad_trailer_does_not_become_eof(tmp_path):
    path = tmp_path / 'bad-tail.xyz.gz'
    # More than two native output buffers separates the first frame from the
    # damaged member, so normal decoder read-ahead cannot reach it on next().
    first_member = gzip.compress((FIRST + '\n' * 150000).encode())
    damaged_member = gzip.compress(SECOND.encode())[:-1]
    path.write_bytes(first_member + damaged_member)
    frames = imrmlp.iread(path)
    first = next(frames)
    saved = first.copy()
    with pytest.raises(RuntimeError, match='EXTXYZ frame'):
        list(frames)
    with pytest.raises(StopIteration):
        next(frames)
    assert_same(first, saved)


def test_file_is_opened_on_first_iteration_and_missing_file_fails(tmp_path):
    path = tmp_path / 'created-later.xyz'
    frames = imrmlp.iread(path)
    path.write_text(SECOND)
    assert len(next(frames)) == 1
    frames.close()
    missing = imrmlp.iread(tmp_path / 'missing.xyz')
    with pytest.raises(RuntimeError, match='streaming input'):
        next(missing)


CFG_FIRST = ('BEGIN_CFG\nSize\n3\nSupercell\n10 0 0\n.2 11 0\n.1 .3 12\n'
             'AtomData: id type cartes_x cartes_y cartes_z fx fy fz\n'
             '3 2 3.1 3.2 3.3 .7 .8 .9\n'
             '1 2 .1 .2 .3 .1 .2 .3\n'
             '2 0 1.1 1.2 1.3 -.1 -.2 -.3\n'
             'Energy\n-4.5\nPlusStress: xx yy zz yz xz xy\n1 2 3 .4 .5 .6\n'
             'Feature label train1\nEND_CFG\n')


CFG_SECOND = ('BEGIN_CFG\nSize\n1\nSupercell\n20 0 0\n0 18 0\n0 0 17\n'
              'AtomData: id type cartes_x cartes_y cartes_z\n'
              '1 2 2 1 .5\nEnergy\n1\nFeature label train2\nEND_CFG\n')


@pytest.fixture
def training_paths(tmp_path):
    cfg = tmp_path / 'training.cfg'
    cfg.write_text(CFG_FIRST + CFG_SECOND)

    # These map to internal model types [2, 0, 2], not atomic numbers.
    first = imrmlp.Atoms(numbers=[13, 8, 13], elements=['O', 'N', 'Al'],
                       positions=[[.1, .2, .3], [1.1, .5, .8], [2.4, 1.3, .7]],
                       cell=[[8., .2, .1], [.1, 9., .3], [0., .2, 10.]],
                       pbc=[True, False, True])
    first.energy = -4.5
    first.forces = [[.1, .2, .3], [-.1, -.2, -.3], [.7, .8, .9]]
    first.stress = [[1., .6, .5], [.6, 2., .4], [.5, .4, 3.]]
    first._impl.site_energies = [-1., -2., -1.5]
    first._impl.charges = [.25, -.1, -.15]
    directions = np.arange(18, dtype=float).reshape(9, 2) / 17
    first.set_hvp(directions, 1.5 * directions)
    second = imrmlp.Atoms(numbers=[8], elements=['O', 'N', 'Al'],
                        positions=[[2., 1., .5]], cell=np.diag([20., 18., 17.]),
                        pbc=False)
    second.energy = 1.
    binary = tmp_path / 'training.bin'
    imrmlp.write(binary, [first, second], format='bin')
    return {'cfg': cfg, 'bin': binary}


def test_cfg_stream_keeps_id_slots_raw_types_and_optional_labels(training_paths, tmp_path):
    path = training_paths['cfg']
    expected = imrmlp.read(path)
    actual = list(imrmlp.iread(path))
    assert [len(frame) for frame in actual] == [3, 1]
    for frame, reference in zip(actual, expected):
        assert_same(frame, reference)
    assert actual[0].chemmap == [2, 0, 2]
    np.testing.assert_array_equal(actual[0].numbers, [2, 0, 2])
    np.testing.assert_allclose(actual[0].positions,
                               [[.1, .2, .3], [1.1, 1.2, 1.3], [3.1, 3.2, 3.3]])
    np.testing.assert_allclose(actual[0].forces,
                               [[.1, .2, .3], [-.1, -.2, -.3], [.7, .8, .9]])
    np.testing.assert_allclose(actual[0].stress,
                               [[1., .6, .5], [.6, 2., .4], [.5, .4, 3.]])
    assert actual[1].chemmap == [2]
    assert actual[1].forces is None and actual[1].stress is None
    np.testing.assert_array_equal(actual[0].pbc, [True, True, True])
    for i, frame in enumerate(actual):
        output = tmp_path / f'cfg-frame-{i}.cfg'
        imrmlp.write_one(output, frame)
        assert f'Feature label train{i + 1}' in output.read_text()


def test_binary_stream_keeps_partial_pbc_hvp_and_independent_frames(training_paths):
    path = training_paths['bin']
    expected = imrmlp.read(path)
    frames = imrmlp.iread(path)
    first = next(frames)
    saved = first.copy()
    second = next(frames)
    with pytest.raises(StopIteration):
        next(frames)
    assert_same(first, saved)
    assert_same(first, expected[0])
    assert_same(second, expected[1])
    assert first.chemmap == [2, 0, 2]
    assert second.chemmap == [0]
    np.testing.assert_array_equal(first.pbc, [True, False, True])
    np.testing.assert_array_equal(second.pbc, [False, False, False])
    np.testing.assert_allclose(first.site_energies, [-1., -2., -1.5])
    np.testing.assert_allclose(first.charges, [.25, -.1, -.15])
    directions = np.arange(18, dtype=float).reshape(9, 2) / 17
    np.testing.assert_allclose(first.hvp_directions, directions, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(first.hvp_targets, 1.5 * directions, rtol=1e-14, atol=1e-14)
    assert second.site_energies is None and second.charges is None
    assert second.hvp_directions.size == second.hvp_targets.size == 0


@pytest.mark.parametrize('codec,error', [('cfg', 'MLPCFG frame 1'), ('bin', 'MLIPBIN frame 1')])
def test_started_training_frame_truncation_is_not_eof(training_paths, codec, error):
    path = training_paths[codec]
    if codec == 'cfg':
        # A BEGIN_CFG frame has started; unrelated text outside markers retains
        # the native CFG scanning contract and is not used as a corrupt frame.
        path.write_text(CFG_FIRST + CFG_SECOND.removesuffix('END_CFG\n'))
    else:
        path.write_bytes(path.read_bytes()[:-1])
    frames = imrmlp.iread(path)
    first = next(frames)
    saved = first.copy()
    with pytest.raises(RuntimeError, match=error):
        next(frames)
    with pytest.raises(StopIteration):
        next(frames)
    assert_same(first, saved)


N2P2_BOHR = .529177210903


N2P2_HARTREE = 27.211396641308


N2P2_LATTICE = 'lattice 6 0 0\nlattice .2 7 0\nlattice .1 .3 8\n'


N2P2_FIRST = ('begin\ncomment label=ignored config_type=ignored\n' + N2P2_LATTICE +
              'atom 1 2 3 H .25 -.5 .1 -.2 .3\n'
              'atom 2 3 4 O -.1 .125 -.4 .5 -.6\n'
              'energy -.25\ncharge .8\nend\n')


N2P2_EMPTY = 'begin\n' + N2P2_LATTICE + 'end\n'


N2P2_SECOND = ('begin\nlattice 8 0 0\nlattice 0 9 0\nlattice 0 0 10\n'
               'atom 3 1 2 O -.2 .125 0 0 .3\nend\n')


def assert_n2p2_first(frame):
    np.testing.assert_array_equal(frame.numbers, [1, 8])
    assert frame.chemmap == [1, 8]
    np.testing.assert_array_equal(frame.pbc, [True, True, True])
    np.testing.assert_allclose(frame.positions,
                               np.array([[1., 2., 3.], [2., 3., 4.]]) * N2P2_BOHR,
                               rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(frame.cell,
                               np.array([[6., 0., 0.], [.2, 7., 0.], [.1, .3, 8.]]) * N2P2_BOHR,
                               rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(frame.forces,
                               np.array([[.1, -.2, .3], [-.4, .5, -.6]]) *
                               N2P2_HARTREE / N2P2_BOHR, rtol=1e-13, atol=1e-13)
    assert frame.energy == pytest.approx(-.25 * N2P2_HARTREE, rel=1e-13, abs=1e-13)
    np.testing.assert_allclose(frame.site_energies, np.array([-.5, .125]) * N2P2_HARTREE,
                               rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(frame.charges, [.25, -.1], rtol=0, atol=1e-14)
    assert frame.stress is None


@pytest.mark.parametrize('format,filename', [
    ('', 'frames.n2p2'), ('n2p2', 'input.data'), ('n2p2dat', 'input.data'),
])
def test_n2p2_stream_units_elements_and_independent_charge_labels(tmp_path, format, filename):
    path = tmp_path / filename
    path.write_text(N2P2_FIRST + N2P2_EMPTY + N2P2_SECOND)
    expected = imrmlp.read(path, format=format)
    with closing(imrmlp.iread(path, format=format)) as frames:
        first = next(frames)
        saved = first.copy()
        empty = next(frames)
        second = next(frames)
        with pytest.raises(StopIteration):
            next(frames)
    actual = [first, empty, second]
    assert [len(frame) for frame in actual] == [2, 0, 1]
    assert len(expected) == len(actual)
    for frame, reference in zip(actual, expected):
        assert_same(frame, reference)
    assert_same(first, saved)
    assert_n2p2_first(first)
    assert empty.energy is None and empty.charges is None and empty.site_energies is None
    assert second.chemmap == [8]
    assert second.energy is None
    np.testing.assert_allclose(second.charges, [-.2])
    np.testing.assert_allclose(second.site_energies, [.125 * N2P2_HARTREE])
    assert second.cell[0, 0] == pytest.approx(8 * N2P2_BOHR)

    for i, frame in enumerate(actual):
        output = tmp_path / f'n2p2-frame-{i}.xyz'
        imrmlp.write_one(output, frame)
        fields = dict(token.split('=', 1) for token in shlex.split(
            output.read_text().splitlines()[1]) if '=' in token)
        if i == 0:
            assert float(fields['total_charge']) == pytest.approx(.8)
            assert first.charges.sum() == pytest.approx(.15)
        else:
            assert 'total_charge' not in fields
        assert 'label' not in fields and 'config_type' not in fields


XDAT_HEADER = 'first cell\n1\n4 0 0\n.5 6 0\n0 .2 8\nH He\n1 1\n'


XDAT_HEADER2 = 'second cell\n1\n5 .2 0\n0 7 .1\n.3 0 9\nHe H\n1 1\n'


XDAT_FIRST = 'Direct configuration= 7\n.1 .2 .3\n.4 .5 .6\n'


XDAT_SECOND = 'Direct configuration= 41\n.2 .3 .4\n.5 .6 .7\n'


def xdat_trajectory(variable, *, bad_second=False):
    second = XDAT_SECOND.replace('.5 .6 .7', '.5 .6') if bad_second else XDAT_SECOND
    return XDAT_HEADER + XDAT_FIRST + (XDAT_HEADER2 if variable else '') + second


@pytest.mark.parametrize('format,filename,variable', [
    ('', 'XDATCAR', False), ('xdatcar', 'input.data', False),
    ('', 'xdatcar6', True), ('XDATCAR6', 'input.data', True),
])
def test_xdatcar_stream_keeps_first_frame_cells_species_and_cartesian_coordinates(
        tmp_path, format, filename, variable):
    path = tmp_path / filename
    path.write_text(xdat_trajectory(variable).rstrip('\n'))
    bulk = imrmlp.read(path, format=format)
    with closing(imrmlp.iread(path, format=format)) as frames:
        first = next(frames)
        saved = first.copy()
        second = next(frames)
        with pytest.raises(StopIteration):
            next(frames)
    for actual, expected in zip([first, second], bulk):
        assert_same(actual, expected)
    cells = [np.array([[4., 0., 0.], [.5, 6., 0.], [0., .2, 8.]])]
    cells.append(np.array([[5., .2, 0.], [0., 7., .1], [.3, 0., 9.]])
                 if variable else cells[0])
    fractions = [np.array([[.1, .2, .3], [.4, .5, .6]]),
                 np.array([[.2, .3, .4], [.5, .6, .7]])]
    for index, atom in enumerate([first, second]):
        np.testing.assert_allclose(atom.cell, cells[index], atol=1e-13, rtol=0)
        np.testing.assert_allclose(atom.positions, fractions[index] @ cells[index], atol=1e-13, rtol=0)
        np.testing.assert_array_equal(atom.numbers, [2, 1] if variable and index else [1, 2])
        np.testing.assert_array_equal(atom.pbc, [True, True, True])
        assert atom.energy is None and atom.forces is None and atom.stress is None
    second.positions = np.zeros((2, 3))
    assert_same(first, saved)


@pytest.mark.parametrize('format', ['xdatcar', 'xdatcar6'])
def test_xdatcar_stream_reports_a_started_bad_second_frame_and_can_close_early(tmp_path, format):
    path = tmp_path / f'bad.{format}'
    path.write_text(xdat_trajectory(format == 'xdatcar6', bad_second=True))
    with closing(imrmlp.iread(path, format=format)) as early:
        assert len(next(early)) == 2
    frames = imrmlp.iread(path, format=format)
    first = next(frames)
    saved = first.copy()
    with pytest.raises(RuntimeError, match=format.upper() + ' frame 1'):
        next(frames)
    with pytest.raises(StopIteration):
        next(frames)
    assert_same(first, saved)


@pytest.mark.parametrize('format,suffix', [('', 'extxyz'), ('cfg', 'cfg'), ('mlpbin', 'bin')])
def test_write_iter_consumes_frames_once_and_matches_bulk(tmp_path, format, suffix):
    source = tmp_path / 'writer-source.xyz'
    source.write_text(FIRST + EMPTY_FRAME + SECOND)
    frames = imrmlp.read(source)
    for frame in frames:
        frame.pbc = True  # CFG retains its native full-PBC convention.
    first = frames[0]
    first._impl.velocities = np.array([[.2, .3, .4], [-.1, .5, .6]])
    directions = np.arange(1., 13.).reshape(6, 2) / 10
    first.set_hvp(directions, 1.7 * directions)
    expected = [frame.copy() for frame in frames]
    first.to_fractional()
    fractional = first.positions.copy()
    output = tmp_path / ('streamed.' + suffix)
    consumed = []

    def single_pass():
        for i, frame in enumerate(frames):
            if i:
                # A list-first implementation would consume this before opening.
                assert output.exists()
            consumed.append(i)
            yield frame

    assert imrmlp.write_iter(output, single_pass(), format=format) is None
    assert consumed == [0, 1, 2]
    assert not first._impl.cartesian
    np.testing.assert_array_equal(first.positions, fractional)
    np.testing.assert_array_equal(first.hvp_targets, 1.7 * directions)
    actual = imrmlp.read(output)
    assert [len(frame) for frame in actual] == [2, 0, 1]
    bulk = tmp_path / ('bulk.' + suffix)
    imrmlp.write(bulk, expected, format=format)
    for frame, reference, bulk_frame in zip(actual, expected, imrmlp.read(bulk)):
        assert_same(frame, reference)
        assert_same(frame, bulk_frame)
    np.testing.assert_allclose(actual[0].velocities, [[.2, .3, .4], [-.1, .5, .6]])
    np.testing.assert_allclose(actual[0].hvp_directions, directions)
    np.testing.assert_allclose(actual[0].hvp_targets, 1.7 * directions)
    assert actual[2].velocities is None and actual[2].charges is None


def test_write_iter_composes_with_iread_and_finalizes_gzip(tmp_path):
    source = tmp_path / 'input.xyz'
    source.write_text(FIRST + SECOND)
    output = tmp_path / 'output.extxyz.gz'
    with closing(imrmlp.iread(source)) as frames:
        imrmlp.write_iter(output, frames)
    assert len(gzip.decompress(output.read_bytes()).splitlines()) == 7
    for frame, reference in zip(imrmlp.read(output), imrmlp.read(source)):
        assert_same(frame, reference)


def test_write_iter_uses_the_declared_element_order_for_extxyz(tmp_path):
    frames = [imrmlp.Atoms(numbers=[1, 2], elements=['He', 'H'],
                         positions=[[0, 0, 0], [2, 0, 0]], cell=np.eye(3)*10),
              imrmlp.Atoms(numbers=[1], elements=['He', 'H'],
                         positions=[[1, 0, 0]], cell=np.eye(3)*10)]
    output = tmp_path / 'mapped.xyz'
    imrmlp.write_iter(output, iter(frames), format='xyz')
    actual = imrmlp.read(output)
    assert [a.numbers.tolist() for a in actual] == [[1, 2], [1]]
    assert [a.chemmap for a in frames] == [[1, 0], [1]]


def test_write_iter_preserves_an_iterator_error_and_the_written_prefix(tmp_path):
    atom = imrmlp.Atoms(numbers=[1], positions=[[1, 2, 3]], cell=np.eye(3)*10)
    error = ValueError('source generation failed')
    output = tmp_path / 'prefix.cfg'

    def interrupted():
        yield atom
        raise error

    with pytest.raises(ValueError, match='source generation failed') as caught:
        imrmlp.write_iter(output, interrupted())
    assert caught.value is error
    actual = imrmlp.read(output)
    assert len(actual) == 1
    np.testing.assert_allclose(actual[0].positions, [[1, 2, 3]])


def test_write_iter_snapshots_the_mapping_and_rejects_later_changes(tmp_path):
    elements = ['H', 'He']
    atom = imrmlp.Atoms(numbers=[1], elements=elements, positions=[[0, 0, 0]],
                      cell=np.eye(3)*10)
    output = tmp_path / 'mapping.xyz'

    def changed_mapping():
        yield atom
        elements.reverse()
        # Atoms snapshots the caller list: the original remains compatible.
        yield atom
        # A later Atoms with a genuinely different mapping must be rejected.
        yield imrmlp.Atoms(numbers=[1], elements=elements, positions=[[1, 0, 0]],
                         cell=np.eye(3)*10)

    with pytest.raises(ValueError, match='same element mapping'):
        imrmlp.write_iter(output, changed_mapping())
    actual = imrmlp.read(output)
    assert len(actual) == 2
    assert [frame.numbers.tolist() for frame in actual] == [[1], [1]]
    assert atom.chemmap == [0]
    np.testing.assert_array_equal(atom.numbers, [1])


@pytest.mark.parametrize('existing', [False, True])
def test_write_iter_empty_iterator_never_opens_the_target(tmp_path, existing):
    output = tmp_path / 'empty.bin'
    if existing:
        output.write_bytes(b'keep existing data')
    imrmlp.write_iter(output, iter(()))
    assert output.exists() == existing
    if existing:
        assert output.read_bytes() == b'keep existing data'


@pytest.mark.parametrize('format', ['poscar', 'CFG'])
def test_write_iter_rejects_other_formats_before_consuming_or_opening(tmp_path, format):
    output = tmp_path / 'existing'
    output.write_text('keep existing data')
    consumed = []

    def untouched():
        consumed.append(True)
        yield imrmlp.Atoms()

    with pytest.raises(ValueError, match='write_iter supports only'):
        imrmlp.write_iter(output, untouched(), format=format)
    assert consumed == []
    assert output.read_text() == 'keep existing data'


@pytest.mark.parametrize('compressed', [False, True])
def test_write_iter_reports_output_close_failures(tmp_path, compressed):
    from pathlib import Path
    if not Path('/dev/full').exists():
        pytest.skip('/dev/full is unavailable')
    output = tmp_path / ('full.xyz.gz' if compressed else 'full.xyz')
    output.symlink_to('/dev/full')
    atom = imrmlp.Atoms(numbers=[1], positions=[[0, 0, 0]], cell=np.eye(3)*10)
    with pytest.raises(RuntimeError, match='write|clos'):
        imrmlp.write_iter(output, iter([atom]))


def test_write_iter_normalizes_numeric_element_symbols_without_changing_sources(tmp_path):
    elements = ['28', 'Al']
    frames = [imrmlp.Atoms(numbers=[13, 28, 13], elements=elements,
                         positions=[[1, 2, 3], [3, 2, 1], [2, 3, 1]],
                         cell=np.eye(3) * 10),
              imrmlp.Atoms(numbers=[28], elements=elements,
                         positions=[[2, 1, 3]], cell=np.eye(3) * 10)]
    frames[0].to_fractional()
    positions = frames[0].positions.copy()
    output = tmp_path / 'numeric-map.xyz'
    imrmlp.write_iter(output, iter(frames))
    with closing(imrmlp.iread(output)) as stream:
        actual = list(stream)
    assert [a.numbers.tolist() for a in actual] == [[13, 28, 13], [28]]
    assert [a.chemmap for a in frames] == [[1, 0, 1], [0]]
    assert elements == ['28', 'Al']
    assert frames[0]._elements == elements
    assert not frames[0]._impl.cartesian
    np.testing.assert_array_equal(frames[0].positions, positions)
    np.testing.assert_allclose(actual[0].positions,
                               [[1, 2, 3], [3, 2, 1], [2, 3, 1]])
    rows = output.read_text().splitlines()
    assert [row.split()[0] for row in rows[2:5]] == ['Al', 'Ni', 'Al']
    assert rows[-1].split()[0] == 'Ni'


OUTCAR_ITER = '--------------------------------------- Iteration'
OUTCAR_ABORT = '------------------------ aborting loop because EDIFF is reached'
OUTCAR_POSITION = ' POSITION                                       TOTAL-FORCE (eV/Angst)'
OUTCAR_TOTEN = '  free  energy   TOTEN  ='


def bounded_outcar_step(step, *, mlff=False):
    electronic = '' if mlff else f'{OUTCAR_ITER} 1(1)\n{OUTCAR_ABORT}\n'
    side = step + 2
    return (electronic + ' VOLUME and BASIS-vectors are now :\n'
            ' ---\n energy-cutoff : 400\n volume of cell : 8\n'
            ' direct lattice vectors reciprocal lattice vectors\n'
            f' {side} 0 0 0 0 0\n 0 {side} 0 0 0 0\n 0 0 {side} 0 0 0\n'
            f'{OUTCAR_POSITION}\n ---\n'
            f' {step + 1} 2 3 .1 .2 .3\n {step + 4} 5 6 -.1 -.2 -.3\n'
            f'{OUTCAR_TOTEN} {-10 - step} eV\n')


def bounded_outcar(version=5, *, mlff=False, second=None):
    header = (f' vasp.{version}.4.4\n INCAR: SYSTEM = bounded\n'
              + (' MLFF enabled\n' if mlff else '')
              + ' POTCAR: PAW_PBE H 2000\n VRHFIN =H: s1\n'
              ' ions per type = 2\n NELM = 60\n EDIFF = .00001\n'
              ' EDIFFG = -.1\n NSW = 2\n IBRION = 0\n ISIF = 2\n')
    return (header + bounded_outcar_step(0, mlff=mlff)
            + (bounded_outcar_step(1, mlff=mlff) if second is None else second))


@pytest.mark.parametrize('version,mlff', [(5, False), (6, False), (6, True)])
@pytest.mark.parametrize('compressed', [False, True])
def test_outcar_iread_owns_complete_steps_and_accepts_statistics_tail(tmp_path, version, mlff, compressed):
    path = tmp_path / ('OUTCAR.gz' if compressed else 'OUTCAR')
    text = bounded_outcar(version, mlff=mlff) + ' General timing and accounting informations'
    data = text.encode()
    path.write_bytes(gzip.compress(data) if compressed else data)
    with closing(imrmlp.iread(path)) as frames:
        first = next(frames)
        saved = first.copy()
        second = next(frames)
        with pytest.raises(StopIteration):
            next(frames)
    assert_same(first, saved)
    for step, atom in enumerate([first, second]):
        assert atom.natoms == 2
        np.testing.assert_allclose(atom.cell, np.eye(3) * (step + 2))
        np.testing.assert_allclose(atom.positions, [[step + 1, 2, 3], [step + 4, 5, 6]])
        np.testing.assert_allclose(atom.forces, [[.1, .2, .3], [-.1, -.2, -.3]])
        assert atom.energy == pytest.approx(-10 - step)
    assert_same(first, saved)


@pytest.mark.parametrize('broken', [
    OUTCAR_ITER + ' 1(1)\n',
    bounded_outcar_step(1).split(OUTCAR_POSITION)[0],
    bounded_outcar_step(1).replace(' 2 2 3 .1 .2 .3', ' 2 2 3 .1 .2'),
    bounded_outcar_step(1).replace(f'{OUTCAR_TOTEN} -11', f'{OUTCAR_TOTEN} invalid'),
])
def test_outcar_iread_reports_started_incomplete_step_after_valid_prefix(tmp_path, broken):
    path = tmp_path / 'OUTCAR'
    path.write_text(bounded_outcar(second=broken))
    with closing(imrmlp.iread(path)) as frames:
        assert next(frames).energy == pytest.approx(-10)
        with pytest.raises(RuntimeError, match='OUTCAR frame 1'):
            next(frames)
    with pytest.raises(RuntimeError):
        imrmlp.read(path)
    assert imrmlp.read_one(path).energy == pytest.approx(-10)


def test_outcar_iread_is_lazy_and_early_close_leaves_bad_suffix_unread(tmp_path):
    path = tmp_path / 'OUTCAR'
    frames = imrmlp.iread(path)
    path.write_text(bounded_outcar(second=OUTCAR_ITER + ' 1(1)\n'))
    assert next(frames).energy == pytest.approx(-10)
    frames.close()


def test_outcar_iread_validates_gzip_trailer_when_fully_consumed(tmp_path):
    path = tmp_path / 'OUTCAR.gz'
    data = bytearray(gzip.compress(bounded_outcar().encode()))
    data[-8] ^= 1
    path.write_bytes(data)
    with pytest.raises(RuntimeError):
        list(imrmlp.iread(path))
