"""MSI atom filtering preserves valid bond endpoints through public readers."""

import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
from imrmlp._imrmlp import _GPIO
import imrmlp


RECORDS = [(41, '6 C'), (7, '1 H'), (29, '8 O'), (13, '7 N')]


def model(records, bonds, with_charges=True):
    lines = ['(1 Model']
    for index, (atom_id, acl) in enumerate(records):
        lines += [f'  ({atom_id} Atom', f'    (A C ACL "{acl}")',
                  f'    (A D XYZ ({index} 0 0))']
        if with_charges:
            lines.append(f'    (A D Charge {index + .25})')
        lines.append('  )')
    for index, (a, b) in enumerate(bonds):
        lines += [f'  ({index + 1} Bond', f'    (A I Atom1 {a})',
                  f'    (A I Atom2 {b})', '  )']
    return '\n'.join(lines + [')', ''])


def check_topology(raw, expected):
    assert raw.bonds == expected
    assert all(0 <= endpoint < len(raw) for bond in raw.bonds for endpoint in bond)


@pytest.mark.parametrize('skipped', range(4))
@pytest.mark.parametrize('reader', [imrmlp.read, imrmlp.read_one])
@pytest.mark.parametrize('fmt', ['', 'msi'])
def test_filtered_atom_at_every_position(tmp_path, capfd, skipped, reader, fmt):
    records = list(RECORDS)
    records[skipped] = (records[skipped][0], '0 Zz')
    bonds = [(a, b) for a, _ in RECORDS for b, _ in RECORDS]
    bonds += [(999, 41), (41, 999), (999, 998)]
    path = tmp_path / 'filtered.msi'
    path.write_text(model(records, bonds))
    result = reader(path, format=fmt)
    atom = result[0] if reader is imrmlp.read else result
    assert atom.natoms == 3
    check_topology(atom._impl, [[a, b] for a in range(3) for b in range(3)])
    np.testing.assert_array_equal(atom.numbers, np.delete([6, 1, 8, 7], skipped))
    np.testing.assert_array_equal(atom.positions[:, 0], np.delete(range(4), skipped))
    assert atom._impl.charges == []
    errors = capfd.readouterr().err
    assert 'unknown element "Zz", skipping atom' in errors
    assert 'references a missing atom, skipping bond' in errors
    assert 'references a skipped atom, skipping bond' in errors


@pytest.mark.parametrize('reader_name', ['read', 'read_one'])
def test_custom_map_removes_unknown_elements_and_their_bonds(tmp_path, reader_name):
    path = tmp_path / 'custom.msi'
    path.write_text(model(RECORDS, [(41, 29), (29, 41), (7, 29),
                                   (13, 41), (7, 13), (41, 999)]))
    io = _GPIO()
    io.set_elements(['O', 'C'])
    result = getattr(io, reader_name)(str(path), 'msi')
    atom = result[0] if reader_name == 'read' else result
    assert atom.chemmap == [1, 0]
    np.testing.assert_array_equal(atom.positions[:, 0], [0, 2])
    check_topology(atom, [[0, 1], [1, 0]])
    assert atom.charges == []


def test_filtered_and_complete_models_do_not_share_indices_or_charges(tmp_path):
    filtered = model([(9, '0 Zz'), (4, '6 C'), (2, '1 H')], [(4, 2), (9, 2)])
    complete = model([(9, '8 O'), (4, '1 H'), (2, '6 C')], [(2, 9), (9, 4)])
    path = tmp_path / 'models.msi'
    path.write_text(filtered + complete + filtered)
    frames = imrmlp.read(path)
    assert [frame.natoms for frame in frames] == [2, 3, 2]
    for index in (0, 2):
        check_topology(frames[index]._impl, [[0, 1]])
        assert frames[index]._impl.charges == []
    check_topology(frames[1]._impl, [[2, 0], [0, 1]])
    assert frames[1]._impl.charges == [.25, 1.25, 2.25]


@pytest.mark.parametrize('reader', [imrmlp.read, imrmlp.read_one])
def test_all_bonds_can_be_dropped_with_a_retained_atom(tmp_path, reader):
    path = tmp_path / 'dangling.msi'
    path.write_text(model([(41, '0 Zz'), (7, '6 C')],
                          [(41, 7), (7, 41), (999, 7)], with_charges=False))
    result = reader(path)
    atom = result[0] if reader is imrmlp.read else result
    assert atom.natoms == 1
    check_topology(atom._impl, [])
    assert atom._impl.charges == []
