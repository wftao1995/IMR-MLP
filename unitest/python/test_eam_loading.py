# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""EAM factory loading, automatic species maps, and malformed tables."""
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
from imrmlp import Atoms, Potential


def write_table(path, kind):
    lines = ['test', 'test', 'test', '3 H He Li', '5 1 5 1 4']
    for element in range(3):
        lines += [f'{element+1} 1 1 fcc', '0 0 0 0 0']
        lines += ['0 0 0 0 0'] * (3 if kind == 'eamfs' else 1)
    for k in range(1, 7):
        lines.append(' '.join(str(k*r*r) for r in range(5)))
    path.write_text('\n'.join(lines) + '\n')


@pytest.mark.parametrize('kind', ['eamalloy', 'eamfs'])
@pytest.mark.parametrize('mapping', ['auto', 'reverse', 'subset'])
@pytest.mark.parametrize('omp', [False, True])
def test_eam_species_maps_preserve_pair_energy(tmp_path, monkeypatch, kind, mapping, omp):
    monkeypatch.chdir(tmp_path)
    write_table(tmp_path / 'potential.eam', kind)
    elements = {'auto': ['H', 'He', 'Li'], 'reverse': ['Li', 'He', 'H'], 'subset': ['Li', 'H']}[mapping]
    if mapping != 'auto':
        (tmp_path / 'index.map').write_text('\n'.join(elements) + '\n')
    potential = Potential('potential.eam', pottype=kind)
    atoms = Atoms(numbers=[1, 3], elements=elements,
                  positions=[[0, 0, 0], [2, 0, 0]], cell=np.eye(3)*20, pbc=False)
    potential.calculate(atoms, use_omp=omp)
    assert potential.species_count == len(elements)
    assert atoms.energy == pytest.approx(8)
    np.testing.assert_allclose(atoms.forces, [[4, 0, 0], [-4, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(atoms.site_energies, [4, 4], atol=1e-12)


@pytest.mark.parametrize('kind', ['eamalloy', 'eamfs'])
@pytest.mark.parametrize('contents', [None, '', 'a\nb\nc\n1 H\n5 1 5 1 4\n1 1 1 fcc\n0 0 0 0 0\n'])
def test_eam_missing_and_truncated_files_fail_on_load(tmp_path, monkeypatch, kind, contents):
    monkeypatch.chdir(tmp_path)
    if contents is not None:
        (tmp_path / 'broken.eam').write_text(contents)
    with pytest.raises(RuntimeError, match='broken.eam'):
        Potential('broken.eam', pottype=kind)
