"""Owned parser/stream state for current Python structure I/O."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import numpy as np
import pytest
import imrmlp
from imrmlp._imrmlp import _GPIO


def frames(path, n=12, symbol="H"):
    path.write_text("".join(f'1\nProperties=species:S:1:pos:R:3 energy={i}\n{symbol} {i} 0 0\n' for i in range(n)))
    return path


def test_independent_readers_snapshot_ranges_and_lifetimes(tmp_path):
    source = frames(tmp_path / "frames.xyz")
    io = _GPIO()
    first = io.open_reader(str(source), "extxyz")
    io.read_startfrom = 1
    io.read_endto = 7
    io.read_irate = 2
    second = io.open_reader(str(source), "extxyz")
    io.read_startfrom = 10
    assert first.read_next().energy == 0
    assert second.read_next().energy == 1
    first.close()
    assert [second.read_next().energy for _ in range(3)] == [3, 5, 7]
    assert second.read_next() is None
    second.close()
    assert first.read_next() is None


def test_one_reader_serializes_concurrent_next(tmp_path):
    source = frames(tmp_path / "frames.xyz", n=48)
    reader = _GPIO().open_reader(str(source), "extxyz")

    def consume():
        result = []
        while (frame := reader.read_next()) is not None:
            result.append(frame.energy)
        return result

    with ThreadPoolExecutor(max_workers=4) as workers:
        got = sum(workers.map(lambda _: consume(), range(4)), [])
    assert sorted(got) == list(range(48))
    reader.close()


def test_concurrent_bulk_reads_share_only_configuration(tmp_path):
    paths = [frames(tmp_path / f"{i}.xyz", n=i + 2, symbol="He" if i % 2 else "H")
             for i in range(8)]
    io = _GPIO()
    with ThreadPoolExecutor(max_workers=4) as workers:
        got = list(workers.map(lambda p: io.read(str(p), "extxyz"), paths))
    assert [len(group) for group in got] == list(range(2, 10))
    for i, group in enumerate(got):
        assert group[0].chemmap == [2 if i % 2 else 1]
        assert group[-1].energy == i + 1


def test_writer_map_snapshot_and_compressed_finalization(tmp_path):
    io = _GPIO()
    io.set_elements(["H"])
    first_path = tmp_path / "first.xyz.gz"
    first = io.open_writer(str(first_path), "extxyz")
    io.set_elements(["He"])
    second_path = tmp_path / "second.xyz.gz"
    second = io.open_writer(str(second_path), "extxyz")
    atom = imrmlp.Atoms(numbers=[1], positions=[[.2, .3, .4]], cell=np.eye(3), elements=["H"])
    atom._impl.to_fractional()
    before = atom.positions.copy()
    with ThreadPoolExecutor(max_workers=2) as workers:
        list(workers.map(lambda writer: writer.write(atom._impl), [first, second]))
    first.close()
    second.close()
    assert gzip.open(first_path, "rt").read().endswith("\n")
    assert imrmlp.read_one(first_path).numbers.tolist() == [1]
    assert imrmlp.read_one(second_path).numbers.tolist() == [2]
    assert not atom._impl.cartesian
    np.testing.assert_array_equal(atom.positions, before)
    with pytest.raises(RuntimeError, match="closed"):
        first.write(atom._impl)


def test_bad_reader_does_not_poison_other_session(tmp_path):
    good = frames(tmp_path / "good.xyz", n=3)
    bad = tmp_path / "bad.xyz"
    bad.write_text("1\nProperties=species:S:1:pos:R:3 energy=0\nH 0 0 0\n2\nbroken\nH 0 0 0\n")
    io = _GPIO()
    a = io.open_reader(str(good), "extxyz")
    b = io.open_reader(str(bad), "extxyz")
    assert b.read_next().energy == 0
    with pytest.raises(RuntimeError):
        b.read_next()
    assert b.read_next() is None
    assert [a.read_next().energy for _ in range(3)] == [0, 1, 2]


def test_end_selection_stops_before_malformed_tail(tmp_path):
    source = frames(tmp_path / "selected.xyz", n=4)
    with source.open("a") as output:
        output.write("2\nbroken\nH 0 0 0\n")
    io = _GPIO()
    io.read_endto = 3
    io.read_irate = 2
    reader = io.open_reader(str(source), "extxyz")
    assert [reader.read_next().energy, reader.read_next().energy] == [0, 2]
    assert reader.read_next() is None


def test_public_contexts_and_empty_writer(tmp_path):
    unused = tmp_path / "unused.xyz"
    with imrmlp.open_writer(unused):
        pass
    assert not unused.exists()
    target = tmp_path / "frames.xyz"
    atom = imrmlp.Atoms(numbers=[8], positions=[[1, 2, 3]], elements=["O"])
    with imrmlp.open_writer(target, elements=["O"]) as writer:
        writer.write(atom)
        atom.positions = [[2, 3, 4]]
        writer.write(atom)
    with imrmlp.open_reader(target) as reader:
        assert [a.numbers.tolist() for a in reader] == [[8], [8]]
    with imrmlp.open_writer(tmp_path / "wrong.xyz") as writer:
        with pytest.raises(ValueError, match="mapping"):
            writer.write(atom)


def test_highlevel_unit_copy_preserves_explicit_mapping():
    atom = imrmlp.Atoms(numbers=[8], positions=[[1, 2, 3]], elements=["O"])
    converted = imrmlp.canonicalize_atoms(atom, length="nm")
    assert converted._elements == ["O"]
    assert converted.numbers.tolist() == [8]
    np.testing.assert_allclose(converted.positions, [[10, 20, 30]])
    np.testing.assert_array_equal(atom.positions, [[1, 2, 3]])
    np.testing.assert_allclose(imrmlp.convert_units([1, 2], "length", "nm", "A"), [10, 20])


@pytest.mark.parametrize("reader", [imrmlp.read, imrmlp.read_one, imrmlp.iread, imrmlp.open_reader])
def test_extxyz_declared_units_stress_uses_source_volume(tmp_path, reader):
    source = tmp_path / "source.xyz"
    source.write_text('1\nLattice="2 0 0 0 2 0 0 0 2" pbc="T T T" '
                      'energy=4 free_energy=3 stress="2 0 0 0 2 0 0 0 2" '
                      'Properties=species:S:1:pos:R:3:forces:R:3\nH 1 0 0 2 0 0\n')
    result = reader(source, input_units={"length": "nm", "energy": "Hartree"})
    if reader is imrmlp.read:
        result = result[0]
    elif reader is not imrmlp.read_one:
        iterator = result
        try:
            result = next(iterator)
        finally:
            iterator.close()
    energy = imrmlp.unit_conversion_factor("energy", "Hartree", "eV")
    np.testing.assert_allclose(result.cell, np.eye(3) * 20.)
    np.testing.assert_allclose(result.positions, [[10, 0, 0]])
    np.testing.assert_allclose(result.forces, [[.2 * energy, 0, 0]])
    np.testing.assert_allclose(result.stress, np.eye(3) * -16. * energy)
    assert result.energy == pytest.approx(4 * energy)
    assert result.free_energy == pytest.approx(3 * energy)


@pytest.mark.parametrize("reader", [imrmlp.read, imrmlp.read_one, imrmlp.iread, imrmlp.open_reader])
def test_units_reject_fixed_format_before_read(tmp_path, reader):
    path = tmp_path / "missing.cfg"
    with pytest.raises(ValueError, match="only for EXTXYZ"):
        result = reader(path, input_units="atomic")
        if reader is imrmlp.iread:
            next(result)
