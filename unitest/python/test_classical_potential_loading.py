# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Species-map line endings and classical potential availability/settings."""
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
from imrmlp import Atoms, Potential


def test_removed_meam_is_not_available(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not hasattr(Potential, 'MEAM')
    with pytest.raises(RuntimeError, match='Unknown format'):
        Potential('unused.pot', pottype='meam')


@pytest.mark.parametrize('kind', ['sw', 'tersoff'])
@pytest.mark.parametrize('newline', ['', '\n'])
def test_species_map_keeps_last_element(tmp_path, monkeypatch, kind, newline):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'index.map').write_text('H\nHe' + newline)
    contents = ('He He 2body 1 1 2 0.5 4 0 3 3\n' if kind == 'sw' else
                'He He A=10 B=3 lambda1=2 lambda2=1 R=3 D=0.2 beta=1 n=1 c=1 d=1 h=-0.3\n')
    (tmp_path / 'potential').write_text(contents)
    potential = Potential('potential', pottype=kind)
    assert potential.species_count == 2
    atoms = Atoms(numbers=[2, 2], elements=['H', 'He'], positions=[[0, 0, 0], [1.3, 0, 0]],
                  cell=np.eye(3)*20, pbc=False)
    potential.calculate(atoms)
    assert np.isfinite(atoms.energy)
    assert abs(atoms.energy) > 0.01


@pytest.mark.parametrize('mode_line', ['mode=brenner', 'mode = brenner', 'mode= brenner'])
def test_brenner_mode_is_parsed_as_text(tmp_path, monkeypatch, mode_line):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'index.map').write_text('H\n')
    (tmp_path / 'potential').write_text(mode_line + '\nH H D0=1 S=2 r0=1 beta=1 '
                                      'R=3 D=0.2 c=1 d=1 h=-0.3 gamma=1\n')
    potential = Potential('potential', pottype='tersoff')
    atoms = Atoms(numbers=[1, 1], elements=['H'], positions=[[0, 0, 0], [1.3, 0, 0]],
                  cell=np.eye(3)*20, pbc=False)
    potential.calculate(atoms)
    assert atoms.energy == pytest.approx(np.exp(-0.6) - 2*np.exp(-0.3))


def test_unknown_brenner_mode_reports_the_setting(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'potential').write_text('mode=unknown\nH H R=3 D=0.2\n')
    with pytest.raises(RuntimeError, match='unknown mode unknown'):
        Potential('potential', pottype='tersoff')


@pytest.mark.parametrize('kind,contents,cutoff', [
    ('lj612', 'pot:ntypes 1\npot:0-0 1 1\npot:cut:0-0 3\npot:ele:0 0.7\n', 3),
    ('mindist', 'pot:power 4\npot:factor 2\npot:cutoff 3\n', 3),
    ('zbl', 'pot:z 1\npot:r0 0.8\npot:r1 2\n', 2),
])
def test_settings_potentials_load_parameters_from_file(tmp_path, monkeypatch, kind, contents, cutoff):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'index.map').write_text('H\n')
    (tmp_path / 'potential').write_text(contents)
    potential = Potential('potential', pottype=kind)
    assert potential.cutoff == pytest.approx(cutoff)
    assert potential.species_count == (0 if kind == 'mindist' else 1)
    atoms = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [1.3, 0, 0]],
                  elements=['H'], cell=np.eye(3)*20, pbc=False)
    potential.calculate(atoms)
    assert np.isfinite(atoms.energy)
    assert abs(atoms.energy) > 0.01
    assert np.sum(atoms.site_energies) == pytest.approx(atoms.energy)
    if kind == 'lj612':
        assert atoms.energy == pytest.approx(4*((1/1.3)**12 - (1/1.3)**6) + 1.4)
    elif kind == 'mindist':
        assert atoms.energy == pytest.approx(2*(1.3 - 3)**4)
    with pytest.raises(RuntimeError, match='Cannot open potential file'):
        Potential('missing.pot', pottype=kind)
