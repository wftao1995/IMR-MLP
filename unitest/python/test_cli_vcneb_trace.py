# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT
"""Real CLI pressure/enthalpy trace and six-component cell-metric regression."""
import math
import os
from pathlib import Path
import re
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / 'build' / os.environ.get('BUILD_MODE', 'release')


def command(args, directory):
    executable = BUILD / 'imrmlp'
    if not executable.exists():
        pytest.skip('Build imrmlp before the bounded CLI trace test')
    env = os.environ.copy()
    env.update(LD_LIBRARY_PATH=str(BUILD)+':'+env.get('LD_LIBRARY_PATH', ''),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return subprocess.run([str(executable), *args], cwd=directory, env=env,
                          capture_output=True, text=True, timeout=30)


def cfg(length, separation):
    return ('BEGIN_CFG\nSize\n2\nSupercell\n'
            f'{length} 0 0\n0 {length} 0\n0 0 {length}\n'
            'AtomData: id type cartes_x cartes_y cartes_z\n'
            f'1 0 {length/2-separation/2} {length/2} {length/2}\n'
            f'2 0 {length/2+separation/2} {length/2} {length/2}\nEND_CFG\n')


@pytest.mark.parametrize('pressure', [-1.0, 1.0])
@pytest.mark.parametrize('old_weight', ['none', 'settings', 'cli'])
def test_vcneb_exposes_actual_metric_and_independent_e_h_pv(tmp_path, pressure, old_weight):
    # Technical LJ inputs; intentionally not same-pressure relaxed endpoints.
    # The portal must reject these at its endpoint gate. Here the standalone
    # kernel is exercised for just one step to test its numerical trace.
    r = 2**(1/6)*2.5
    (tmp_path/'potential.lj').write_text('ntypes 1\nele:0 0\n0-0 0.1 2.5\ncut:0-0 5\n')
    (tmp_path/'index.map').write_text('Cu\n')
    for i, (length, distance) in enumerate([(10,r),(12,6),(11,r)]):
        (tmp_path/f'image-{i}.cfg').write_text(cfg(length,distance))
    settings = 'pot:pottype lj612\npot:potfile potential.lj\npot:typemap index.map\n'
    (tmp_path/'tool.ini').write_text(settings)
    (tmp_path/'vcneb.ini').write_text(settings +
        'input-format mlpcfg\nindex index.map\nnimgs 3\nmepout mep.cfg\n'
        'mepformat mlpcfg\nsummary vcneb.dat\ndriver fire\nmaxiteration 1\nmax_total_iter 1\n'
        'neb:vcneb true\nneb:symm_strain true\nneb:recast_atomic_force true\n'
        'neb:apply_scal false\nneb:iref 0\nneb:cellweight 4\n'
        f'neb:pressure {pressure}\nneb:climb true\nneb:climb_delay 5\nneb:check_mic false\n' +
        ''.join(f'image:{i} image-{i}.cfg\n' for i in range(3)))
    if old_weight == 'settings':
        with (tmp_path/'vcneb.ini').open('a') as f:
            f.write('cellweight 9\n')
    args = ['neb', 'vcneb.ini']
    if old_weight == 'cli':
        args += ['--cellweight', '9']
    result = command(args,tmp_path)
    assert ('cellweight is deprecated' in result.stderr) == (old_weight != 'none')
    assert result.returncode == 2, result.stdout+result.stderr  # one-step budget
    text = (tmp_path/'vcneb.dat').read_text()
    metadata = dict(re.findall(r'^# (\w+) (\S+)$',text,re.M))
    assert metadata['joint_metric'] == 'vcneb-frobenius-v1'
    assert float(metadata['pressure_GPa']) == pressure
    assert float(metadata['cell_weight']) == (4 if old_weight == 'none' else 9)
    weight_scale = 2 if old_weight == 'none' else 3
    scales = [float(v) for _,v in re.findall(r'^# cell_scale (\d+) (\S+)$',text,re.M)]
    assert scales == pytest.approx([weight_scale,weight_scale*math.sqrt(2),weight_scale*math.sqrt(2),weight_scale,weight_scale*math.sqrt(2),weight_scale])
    trace = [[float(v) for v in row] for row in re.findall(
        r'^# image_energy (\d+) (\S+) (\S+) (\S+) (\S+)$',text,re.M)]
    assert len(trace) == 3
    raw = command(['calc_efs','mep.cfg','independent.cfg','--input-format','mlpcfg',
                   '--output-format','mlpcfg','--settings','tool.ini','--index','index.map'],tmp_path)
    assert raw.returncode == 0, raw.stdout+raw.stderr
    energies = [float(v) for v in re.findall(r'^Energy\s*\n\s*(\S+)',(tmp_path/'independent.cfg').read_text(),re.M)]
    assert len(energies) == 3
    for i, energy, enthalpy, pv, volume in trace:
        assert energy == pytest.approx(energies[int(i)], abs=2e-10)
        assert pv == pytest.approx(pressure*volume/160.21766208, abs=2e-10)
        assert enthalpy == pytest.approx(energy+pv, abs=2e-10)
    reference = [float(v) for v in (tmp_path/'neb_reference_cell.data').read_text().split()]
    assert reference == [10,0,0,0,10,0,0,0,10]


@pytest.mark.parametrize('option', ['--cur-o=1', '--reset-score'])
def test_removed_fastfit_options_are_rejected(tmp_path, option):
    result = command(['fastfit', option], tmp_path)
    assert result.returncode != 0
    assert option.split('=')[0].lstrip('-') in result.stderr + result.stdout
