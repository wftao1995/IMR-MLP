"""CLI NEB execution modes; real MPI cases opt in with IMRMLP_RUN_MPI_TESTS=1."""
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def binary(mpi=False):
    path = ROOT / ('build/mpi/imrmlp-mpi' if mpi else 'build/release/imrmlp')
    if not path.exists():
        pytest.skip(f'Build {path} first')
    if mpi and (os.environ.get('IMRMLP_RUN_MPI_TESTS') != '1' or not shutil.which('mpiexec')):
        pytest.skip('Set IMRMLP_RUN_MPI_TESTS=1 with MPI available')
    return path


def inputs(work, mode='images', vc=False, variable=False, potential='lj612'):
    work.mkdir()
    (work/'index.map').write_text('H\n')
    n = 5 if variable else 3
    for i, position in [(0, (1.3, 0, 0)), (n-1, (.8, 1.2, .1))]:
        cell = '21 0 0 0 20 0 0 0 20' if vc and i else '20 0 0 0 20 0 0 0 20'
        (work/f'{i}.xyz').write_text(f'2\nLattice="{cell}" pbc="T T T" Properties=species:S:1:pos:R:3\n'
                                    'H 0 0 0\nH '+' '.join(map(str, position))+'\n')
    lines = [f'pot:pottype {potential}', 'pot:ntypes 1', 'pot:0-0 1 1', 'pot:cut:0-0 3',
             f'nimgs {n}', 'input-format extxyz', 'driver fire', 'maxiteration 6',
             'max_total_iter 6', 'max_move .03', 'neb:solver_policy stable-v1',
             f'neb:omp_mode {mode}', f'neb:vcneb {str(vc).lower()}',
             'neb:gtol 1e-8', 'neb:climb true', 'neb:climb_delay 2',
             'image:0 0.xyz', f'image:{n-1} {n-1}.xyz',
             'neb:write_traj true', 'neb:nblocks 1', 'mepformat extxyz']
    if variable:
        # A deliberately crowded supplied image forces adaptive insertion.
        (work/'1.xyz').write_text((work/'0.xyz').read_text().replace('1.3 0 0', '1.29 .01 0'))
        lines += ['image:1 1.xyz', 'neb:variableimage true', 'neb:img_adjust_interval 1',
                  'neb:img_insert_thresh 1.05', 'neb:max_images 7']
    if potential == 'mtp':
        lines += [f'pot:potfile {ROOT / "unitest/test-data/portable-mtp/plain.mtp"}']
        (work/'index.map').write_text('Al\nNi\n')
        for path in work.glob('*.xyz'):
            path.write_text(path.read_text().replace('H 0 0 0', 'Al 0 0 0').replace('\nH ', '\nNi '))
    (work/'neb.ini').write_text('\n'.join(lines)+'\n')


def run(work, ranks=1, threads=1, mpi=False, args=('neb', 'neb.ini')):
    exe = binary(mpi)
    env = os.environ.copy()
    env.update(LD_LIBRARY_PATH=str(exe.parent)+':'+env.get('LD_LIBRARY_PATH', ''),
               OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               OMPI_MCA_rmaps_base_oversubscribe='1')
    cmd = [str(exe), *args, '--num_threads', str(threads)]
    if mpi:
        cmd = ['mpiexec', '-n', str(ranks), *cmd]
    return subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=90)


def compare(a, b):
    def meta(work):
        return dict(line[2:].split(maxsplit=1) for line in (work/'neb.dat').read_text().splitlines()
                    if line.startswith('# ') and len(line.split()) == 3)
    x, y = meta(a), meta(b)
    for key in ['status', 'accepted_steps', 'force_evaluations', 'climbing_image']:
        assert x[key] == y[key], (key, x, y)
    for key in ['max_neb_force', 'max_perp_force', 'max_spring_force']:
        assert float(x[key]) == pytest.approx(float(y[key]), rel=2e-10, abs=1e-11)
    np.testing.assert_allclose(np.loadtxt(a/'neb.dat'), np.loadtxt(b/'neb.dat'), rtol=1e-9, atol=1e-10)
    first, second = sorted(a.glob('nebimage_*_final.vasp')), sorted(b.glob('nebimage_*_final.vasp'))
    assert len(first) == len(second)
    for fa, fb in zip(first, second):
        np.testing.assert_allclose(np.loadtxt(fa, skiprows=8), np.loadtxt(fb, skiprows=8), rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize('mode', ['images', 'atoms'])
@pytest.mark.parametrize('potential', ['lj612', 'mtp'])
def test_omp_modes_match_serial(tmp_path, mode, potential):
    for name, level, threads in [('serial', 'images', 1), ('parallel', mode, 3)]:
        work=tmp_path/name; inputs(work, level, potential=potential)
        result=run(work, threads=threads)
        assert result.returncode in (0,2), result.stdout+result.stderr
    compare(tmp_path/'serial', tmp_path/'parallel')


@pytest.mark.parametrize('mode,ranks,threads,vc,variable', [
    ('images', 2, 1, False, False), ('images', 3, 2, True, False),
    ('atoms', 4, 2, False, False), ('atoms', 2, 2, True, True)])
@pytest.mark.parametrize('potential', ['lj612', 'mtp'])
def test_mpi_modes_match_serial(tmp_path, mode, ranks, threads, vc, variable, potential):
    binary(True)
    for name in ['serial', 'parallel']:
        work=tmp_path/name; inputs(work, mode, vc, variable, potential)
        result=run(work, ranks, threads, mpi=name=='parallel')
        assert result.returncode in (0,2), result.stdout+result.stderr
        assert result.stdout.count('NEB stopped after') == 1
    compare(tmp_path/'serial', tmp_path/'parallel')


@pytest.mark.parametrize('failure', ['input', 'output', 'worker'])
def test_mpi_failure_exits_without_hanging(tmp_path, failure):
    binary(True)
    work=tmp_path/'job'; inputs(work)
    if failure=='input':
        (work/'0.xyz').unlink()
    elif failure=='output':
        with (work/'neb.ini').open('a') as stream: stream.write('summary missing/neb.dat\n')
    else:
        # Calculator failure on a worker must reach the controller.
        script=work/'fail.sh'
        script.write_text('case "$1" in *mpi-rank-1) exit 7;; esac\n'
                          'cat > "$1/result.xyz" <<EOF\n2\nLattice="20 0 0 0 20 0 0 0 20" '
                          'Properties=species:S:1:pos:R:3:forces:R:3 energy=0 virial="0 0 0 0 0 0 0 0 0"\n'
                          'H 0 0 0 0 0 0\nH 1 0 0 0 0 0\nEOF\n')
        path=work/'neb.ini'
        path.write_text(path.read_text().replace('pot:pottype lj612', 'pot:pottype shellfile')+
                        f'pot:shellfile:runscript {script}\npot:shellfile:index index.map\n'
                        'pot:shellfile:read_outfile result.xyz\npot:shellfile:read_outformat extxyz\n')
    result=run(work, 3, 2, mpi=True)
    assert result.returncode not in (0,2), result.stdout+result.stderr
    if failure=='worker': assert 'NEB failure on rank 1' in result.stdout+result.stderr


def test_mpi_help_is_printed_once(tmp_path):
    result=run(tmp_path, 3, 2, mpi=True, args=('neb', '--help'))
    assert result.returncode == 0, result.stdout+result.stderr
    assert result.stdout.count('Nudged elastic band calculation') == 1
    assert 'neb:omp_mode' in result.stdout
    assert 'neb:keep_symm' in result.stdout and 'neb:perturb_amplitude' in result.stdout
    assert 'Always set to true' not in result.stdout


def test_mpi_evaluates_each_image_once(tmp_path):
    binary(True)
    work=tmp_path/'job';inputs(work)
    script=work/'calculate.sh'
    script.write_text('echo evaluated >> "$1/calls"\n'
                      'cat > "$1/result.xyz" <<EOF\n2\nLattice="20 0 0 0 20 0 0 0 20" '
                      'Properties=species:S:1:pos:R:3:forces:R:3 energy=0 virial="0 0 0 0 0 0 0 0 0"\n'
                      'H 0 0 0 0 0 0\nH 1 0 0 0 0 0\nEOF\n')
    path=work/'neb.ini'
    path.write_text(path.read_text().replace('pot:pottype lj612','pot:pottype shellfile')+
                    f'pot:shellfile:runscript {script}\npot:shellfile:index index.map\n'
                    'pot:shellfile:read_outfile result.xyz\npot:shellfile:read_outformat extxyz\n')
    result=run(work,4,2,mpi=True)
    assert result.returncode in (0,2),result.stdout+result.stderr
    calls=sum(len(path.read_text().splitlines()) for path in work.glob('run-job-*/calls'))
    counts=[int(line.split()[-1]) for line in (work/'neb.dat').read_text().splitlines()
            if line.startswith('# image_force_evaluations ')]
    assert counts==[calls]


@pytest.mark.parametrize('feature', ['constraints','perturbation'])
def test_mpi_symmetry_features_match_serial(tmp_path,feature):
    binary(True)
    for name in ['serial','parallel']:
        work=tmp_path/name;inputs(work)
        for index,dx,y in [(0,1.38,1.19),(2,1.68,1.47)]:
            (work/f'{index}.xyz').write_text('2\nLattice="6 0 0 0 7 0 0 0 8" '
                                           'Properties=species:S:1:pos:R:3\n'
                                           f'H {dx} {y} 1.52\nH {6-dx} {y} 1.52\n')
        with (work/'neb.ini').open('a') as stream:
            stream.write('neb:symmetry_diagnostics symmetry.dat\n')
            stream.write('neb:keep_symm true\n' if feature=='constraints' else
                         'neb:perturb_amplitude .02\nneb:perturb_seed 19\n')
        result=run(work,3,2,mpi=name=='parallel')
        assert result.returncode in (0,2),result.stdout+result.stderr
        report=(work/'symmetry.dat').read_text()
        assert report.count('# stage final')==1
        assert '# current_common_operations' in report
    compare(tmp_path/'serial',tmp_path/'parallel')
