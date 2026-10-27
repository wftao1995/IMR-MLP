"""Opt-in real MPI inference and coordinated failure checks for calc_err."""
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('IMRMLP_RUN_MPI_TESTS') != '1',
                                reason='requires matching MPI and serial executables')
ROOT = Path(os.environ.get('IMRMLP_TEST_REPO', Path(__file__).resolve().parents[2]))
OUTPUTS = {'energy': '--output-energy-diff', 'energy_at': '--output-energy-at-diff',
           'force': '--output-force-diff', 'stress': '--output-stress-diff',
           'virial': '--output-virial-diff', 'summary': '--output-summary'}


def binaries():
    serial, mpi = ROOT/'build/release/imrmlp', ROOT/'build/mpi/imrmlp-mpi'
    if not serial.exists() or not mpi.exists() or not shutil.which('mpiexec'):
        pytest.skip('matching imrmlp, imrmlp-mpi and mpiexec required')
    return serial, mpi


def dataset(path, count, invalid=False):
    frames = []
    for frame in range(count):
        natoms = 2 + frame % 3
        cell = '0 0 0 0 0 0 0 0 0' if invalid and frame == count-1 else '16 0 0 .3 17 0 .2 .4 18'
        props = 'species:S:1:pos:R:3' + (':forces:R:3' if frame % 3 != 1 else '')
        labels = (f' energy={frame*.7}' if frame % 3 != 2 else '')
        if frame % 3 != 0:
            labels += ' virial="1 .2 .3 .4 2 .5 .6 .7 3"'
        lines = [str(natoms), f'Lattice="{cell}" Properties={props}{labels}']
        for atom in range(natoms):
            line = f'{"Al" if atom%2 == 0 else "Ni"} {2+atom*2.1+frame*.03} {2+atom*.4} {2+atom*.3}'
            if frame % 3 != 1:
                line += f' {atom*.1} {frame*.2} -.3'
            lines.append(line)
        frames.append('\n'.join(lines)+'\n')
    path.write_text(''.join(frames))


def model_settings(path, kind):
    fixtures = ROOT/'unitest/test-data/portable-mtp'
    plain = fixtures/'plain.mtp'
    if not plain.exists():
        pytest.skip('portable native MTP fixtures required')
    settings = {'pottype': kind}
    if kind == 'lj612':
        settings.update(ntypes='2', **{'0-0': '.2 1.1', '0-1': '.3 1.2', '1-1': '.25 1.15',
                                     'cut:0-0': '5', 'cut:0-1': '5', 'cut:1-1': '5'})
    elif kind == 'sog':
        original = (fixtures/'les.mtp').read_text()
        count = int(re.search(r'alpha_scalar_moments\s*=\s*(\d+)', original).group(1))
        original = re.split(r'(?m)^lr_sigma\s*=', original, maxsplit=1)[0]
        charges = 'les_species_coeffs = {.23,-.17}\nles_moment_coeffs = {' + ','.join(['.001']*count) + '}\n'
        footer = ('les_version = 3\nles_kernel = sog\nsog_channels = 1\nsog_terms = 2\n'
                  'sog_log_bandwidth_min = -4\nsog_log_bandwidth_max = 1\n'
                  'sog_zero_mode = include\nsog_self_mode = onsite\nles_dl = 2\n'
                  'les_cutoff_mode = smooth\nles_taper_fraction = .1\n'
                  'sog_weights = {.3,-.1}\nsog_log_bandwidths = {-1.5,-.7}\n')
        model = path.with_suffix('.mtp'); model.write_text(original+charges+footer)
        settings.update(pottype='mtp_les', potfile=str(model))
    else:
        settings['potfile'] = str(fixtures/('les.mtp' if kind == 'mtp_les' else 'plain.mtp'))
    path.write_text(''.join(f'pot:{key} {value}\n' for key, value in settings.items()))


def invoke(binary, directory, prefix, ranks=1, threads=1, mode='frames', extra=()):
    args = [str(binary), 'calc_err', '--num_threads', str(threads), '--mpi-mode', mode, '--input', str(directory/'data.xyz'),
            '--input-format', 'extxyz', '--index', str(directory/'index.map'),
            '--settings', str(directory/'model.ini'), '--percentile', '.75']
    for name, option in OUTPUTS.items():
        if option not in extra:
            args += [option, str(directory/f'{prefix}-{name}')]
    args += list(extra)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env['LD_LIBRARY_PATH'] = str(binary.parent)+':'+env.get('LD_LIBRARY_PATH', '')
    if ranks > 1:
        args = ['mpiexec', '-n', str(ranks)] + args
    return subprocess.run(args, env=env, text=True, capture_output=True, timeout=60)


@pytest.mark.parametrize('kind', ['lj612', 'mtp', 'mtp_les', 'sog'])
@pytest.mark.parametrize('ranks,count,threads', [(2,5,1), (3,5,2), (3,2,2)])
def test_frames_matches_serial_outputs(tmp_path, kind, ranks, count, threads):
    serial, mpi = binaries()
    (tmp_path/'index.map').write_text('Al 0\nNi 1\n')
    dataset(tmp_path/'data.xyz', count)
    model_settings(tmp_path/'model.ini', kind)
    baseline = invoke(serial, tmp_path, 'serial')
    assert baseline.returncode == 0, baseline.stdout+baseline.stderr
    parallel = invoke(mpi, tmp_path, 'mpi', ranks, threads)
    assert parallel.returncode == 0, parallel.stdout+parallel.stderr
    assert f'calc_err frame evaluations: {count}' in parallel.stdout
    for name in list(OUTPUTS)+['forcecmpt']:
        assert (tmp_path/f'mpi-{name}').read_bytes() == (tmp_path/f'serial-{name}').read_bytes(), name


@pytest.mark.parametrize('failure', ['nonroot-geometry', 'root-output'])
def test_frames_failure_is_collective(tmp_path, failure):
    _, mpi = binaries()
    (tmp_path/'index.map').write_text('Al 0\nNi 1\n')
    dataset(tmp_path/'data.xyz', 5, invalid=failure == 'nonroot-geometry')
    model_settings(tmp_path/'model.ini', 'lj612')
    extra = ['--output-summary', str(tmp_path/'missing'/'summary')] if failure == 'root-output' else []
    result = invoke(mpi, tmp_path, 'failed', 3, 2, extra=extra)
    assert result.returncode != 0
    output = result.stdout+result.stderr
    assert 'calc_err failure on rank' in output
    if failure == 'nonroot-geometry':
        assert 'rank 2: evaluation: frame 4:' in output
    else:
        assert 'rank 0: output:' in output


def test_gather_crosses_bounded_chunk_with_many_small_frames(tmp_path):
    test_frames_matches_serial_outputs(tmp_path, 'lj612', 2, 3000, 2)


def test_gather_splits_a_large_frame_and_prediction_header(tmp_path):
    serial, mpi = binaries()
    (tmp_path/'index.map').write_text('Al 0\nNi 1\n')
    model_settings(tmp_path/'model.ini', 'lj612')
    with (tmp_path/'data.xyz').open('w') as stream:
        # 13 + 3*21840 = 65533: the next record's flags finish the
        # first 65536-double chunk, while its energy starts the second.
        for count in [21840, 21844]:
            stream.write(f'{count}\nLattice="{6*(count+2)} 0 0 0 20 0 0 0 20" '
                         'Properties=species:S:1:pos:R:3:forces:R:3 energy=0 '
                         'virial="1 .2 .3 .4 2 .5 .6 .7 3"\n')
            for atom in range(count):
                x = 1.7 if atom == 1 else 6*atom
                stream.write(f'{"Al" if atom%2 == 0 else "Ni"} {x} 2 2 0 0 0\n')
    baseline = invoke(serial, tmp_path, 'serial')
    assert baseline.returncode == 0, baseline.stdout+baseline.stderr
    parallel = invoke(mpi, tmp_path, 'mpi', 3, 2)
    assert parallel.returncode == 0, parallel.stdout+parallel.stderr
    assert 'calc_err frame evaluations: 2' in parallel.stdout
    for name in list(OUTPUTS)+['forcecmpt']:
        assert (tmp_path/f'mpi-{name}').read_bytes() == (tmp_path/f'serial-{name}').read_bytes(), name


def test_replicated_progress_still_completes(tmp_path):
    serial, _ = binaries()
    (tmp_path/'index.map').write_text('Al 0\nNi 1\n')
    dataset(tmp_path/'data.xyz', 2)
    model_settings(tmp_path/'model.ini', 'lj612')
    result = invoke(serial, tmp_path, 'serial', mode='replicated', extra=['--show-progress'])
    assert result.returncode == 0, result.stdout+result.stderr
    assert '100%' in result.stdout
    assert 'Calculating EFS' in result.stdout


def test_frames_nonperiodic_zero_cell_molecule(tmp_path):
    serial, mpi = binaries()
    (tmp_path/'index.map').write_text('Al 0\nNi 1\n')
    model_settings(tmp_path/'model.ini', 'lj612')
    (tmp_path/'data.xyz').write_text(''.join(
        f'2\nLattice="0 0 0 0 0 0 0 0 0" pbc="F F F" '
        f'Properties=species:S:1:pos:R:3:forces:R:3 energy={frame}\n'
        f'Al 0 0 0 0 0 0\nNi {1.5+frame*.1} 0 0 0 0 0\n'
        for frame in range(5)))
    baseline = invoke(serial, tmp_path, 'serial')
    assert baseline.returncode == 0, baseline.stdout+baseline.stderr
    parallel = invoke(mpi, tmp_path, 'mpi', 3, 2)
    assert parallel.returncode == 0, parallel.stdout+parallel.stderr
    assert 'calc_err frame evaluations: 5' in parallel.stdout
    for name in list(OUTPUTS)+['forcecmpt']:
        assert (tmp_path/f'mpi-{name}').read_bytes() == (tmp_path/f'serial-{name}').read_bytes(), name
