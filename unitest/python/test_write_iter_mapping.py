"""Bounded output snapshots owned element mappings rather than caller lists."""
import numpy as np
import pytest

pytest.importorskip("imrmlp._imrmlp")
import imrmlp


def test_write_iter_ignores_changes_to_the_constructor_elements_list(tmp_path):
    elements = ['H', 'He']
    atom = imrmlp.Atoms(numbers=[1], elements=elements, positions=[[0, 0, 0]],
                      cell=np.eye(3)*10)
    output = tmp_path / 'mapping.xyz'

    def changed_input_list():
        yield atom
        elements.reverse()
        yield atom

    imrmlp.write_iter(output, changed_input_list())
    actual = imrmlp.read(output)
    assert len(actual) == 2
    assert all(frame.numbers.tolist() == [1] for frame in actual)
    assert atom.numbers.tolist() == [1]


def test_write_iter_snapshots_the_mapping_and_rejects_later_changes(tmp_path):
    elements = ['H', 'He']
    atom = imrmlp.Atoms(numbers=[1], elements=elements, positions=[[0, 0, 0]],
                      cell=np.eye(3)*10)
    output = tmp_path / 'mapping.xyz'

    def changed_mapping():
        yield atom
        # Mutate the frame's owned mapping, not the copied constructor input.
        atom._elements.reverse()
        yield atom

    with pytest.raises(ValueError, match='same element mapping'):
        imrmlp.write_iter(output, changed_mapping())
    actual = imrmlp.read(output)
    assert len(actual) == 1 and actual[0].numbers.tolist() == [1]
