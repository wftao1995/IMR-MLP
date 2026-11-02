# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Model-declared element order replaces index.map for structure reading."""
import re

import pytest

from test_cli_error_paths import ROOT, run_command

XYZ = """2
Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3 pbc="T T T"
Al 0 0 0
Ni 2 0 0
"""

POSCAR = """Al Ni
1.0
10 0 0
0 10 0
0 0 10
Al Ni
1 1
Direct
0 0 0
0.2 0 0
"""


def tagged_model(directory):
    result = run_command(
        'imrmlp',
        ['adjustmtp', '--input', str(ROOT / 'unitest/test-data/potential.mtp'),
         '--output', 'tagged.mtp', '--elements', 'Al', 'Ni'],
        directory)
    assert result.returncode == 0, result.stdout + result.stderr
    return 'tagged.mtp'


def legacy_model(directory, tagged):
    text = (directory / tagged).read_text()
    text = re.sub(r'^potential_tag = .*$', 'potential_tag = ', text, flags=re.M)
    (directory / 'legacy.mtp').write_text(text)
    return 'legacy.mtp'


def calc_efs(directory, model, input_name, output_name, extra=()):
    return run_command(
        'imrmlp',
        ['calc_efs', '--input', input_name, '--output', output_name,
         '--potfile', model, '--pottype', 'mtp', *extra],
        directory)


def read_result(path):
    lines = path.read_text().splitlines()
    energy = float(re.search(r'\benergy=([^\s]+)', lines[1]).group(1))
    rows = [line.split() for line in lines[2:] if line.strip()]
    species = [row[0] for row in rows]
    forces = [tuple(float(value) for value in row[4:7]) for row in rows]
    return energy, species, forces


@pytest.fixture
def workdir(tmp_path):
    (tmp_path / 'input.xyz').write_text(XYZ)
    (tmp_path / 'POSCAR').write_text(POSCAR)
    (tmp_path / 'index.map').write_text('Al\nNi\n')
    return tmp_path


def test_extxyz_without_index_map_matches_explicit_map(workdir):
    model = tagged_model(workdir)
    with_map = calc_efs(workdir, model, 'input.xyz', 'with-map.xyz')
    assert with_map.returncode == 0, with_map.stdout + with_map.stderr
    (workdir / 'index.map').unlink()
    without_map = calc_efs(workdir, model, 'input.xyz', 'without-map.xyz')
    assert without_map.returncode == 0, without_map.stdout + without_map.stderr
    assert 'index file' not in without_map.stderr.lower()

    reference = read_result(workdir / 'with-map.xyz')
    predicted = read_result(workdir / 'without-map.xyz')
    assert predicted[0] == pytest.approx(reference[0], abs=1e-12)
    assert predicted[1] == ['Al', 'Ni']
    for got, expected in zip(predicted[2], reference[2]):
        assert got == pytest.approx(expected, abs=1e-12)


def test_poscar_without_index_map_matches_explicit_map(workdir):
    model = tagged_model(workdir)
    assert calc_efs(workdir, model, 'POSCAR', 'with-map.xyz').returncode == 0
    (workdir / 'index.map').unlink()
    result = calc_efs(workdir, model, 'POSCAR', 'without-map.xyz')
    assert result.returncode == 0, result.stdout + result.stderr
    reference = read_result(workdir / 'with-map.xyz')
    predicted = read_result(workdir / 'without-map.xyz')
    assert predicted[0] == pytest.approx(reference[0], abs=1e-12)
    assert predicted[1] == ['Al', 'Ni']


def test_gpm_model_without_index_map(workdir):
    model = tagged_model(workdir)
    assert calc_efs(workdir, model, 'input.xyz', 'reference.xyz').returncode == 0
    reference = read_result(workdir / 'reference.xyz')
    convert = run_command(
        'imrmlp',
        ['model_convert', '--input', model, '--output', 'model.gpm'],
        workdir)
    assert convert.returncode == 0, convert.stdout + convert.stderr
    (workdir / 'index.map').unlink()
    result = calc_efs(workdir, 'model.gpm', 'input.xyz', 'out.xyz')
    assert result.returncode == 0, result.stdout + result.stderr
    energy, species, forces = read_result(workdir / 'out.xyz')
    assert species == ['Al', 'Ni']
    assert energy == pytest.approx(reference[0], abs=1e-12)
    for got, expected in zip(forces, reference[2]):
        assert got == pytest.approx(expected, abs=1e-12)


def test_numeric_cfg_without_index_map_keeps_element_names(workdir):
    model = tagged_model(workdir)
    convert = run_command(
        'imrmlp',
        ['convstr', 'input.xyz', 'input.cfg', '--index', 'index.map'],
        workdir)
    assert convert.returncode == 0, convert.stdout + convert.stderr
    (workdir / 'index.map').unlink()
    result = calc_efs(workdir, model, 'input.cfg', 'out.xyz')
    assert result.returncode == 0, result.stdout + result.stderr
    _, species, _ = read_result(workdir / 'out.xyz')
    assert species == ['Al', 'Ni']


def test_explicit_missing_map_uses_model_or_fails_without_metadata(workdir):
    model = tagged_model(workdir)
    (workdir / 'index.map').unlink()
    result = calc_efs(workdir, model, 'input.xyz', 'out.xyz',
                      ['--index', 'missing.map'])
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'declared by the model' in result.stderr

    legacy = legacy_model(workdir, model)
    result = calc_efs(workdir, legacy, 'input.xyz', 'legacy.xyz',
                      ['--index', 'missing.map'])
    assert result.returncode != 0
    assert 'does not declare an element order' in result.stderr


def test_poscar_unknown_element_is_rejected(workdir):
    model = tagged_model(workdir)
    (workdir / 'POSCAR-Cu').write_text(POSCAR.replace('Al', 'Cu'))
    (workdir / 'index.map').unlink()
    result = calc_efs(workdir, model, 'POSCAR-Cu', 'out.xyz')
    assert result.returncode != 0
    assert 'not in the species map' in result.stderr


def test_relax_trajectory_without_index_map(workdir):
    model = tagged_model(workdir)
    (workdir / 'relax.ini').write_text(
        f'input input.xyz\noutput relaxed.xyz\n'
        f'pot:potfile {model}\npot:pottype mtp\n'
        'relax:gtol 1e20\nrelax:maxiteration 1\nrelax:write_traj true\n')
    (workdir / 'index.map').unlink()
    result = run_command('imrmlp', ['relax', 'relax.ini'], workdir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'index file' not in result.stderr.lower()
