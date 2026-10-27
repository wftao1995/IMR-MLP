"""Reduction commands preserve periodicity and reject unsupported inputs before output."""
import re
import struct

import pytest

from test_cli_error_paths import run_command


COMMANDS = [('trans_reduce', ()), ('primi_reduce', ('--two-stage', 'true')),
            ('primi_reduce', ('--two-stage', 'false'))]
CELL = (4., 0., 0., 0., 5., 0., 0., 0., 6.)
SENTINEL = b'existing reduction output\n'


def frame(fmt, count=1, cell=CELL, positions=None, pbc='T T T', metadata=''):
    positions = positions if positions is not None else [(0.5, 1.25, 2.25)] * count
    if fmt == 'bin':
        # MLIPBIN uses column-major lattice entries; the shared fixture is diagonal.
        return (b'GPMLBIN2' + struct.pack('=Ii', 3, count)
                + b'l' + struct.pack('=9d', *cell)
                + b'p' + struct.pack(f'={3 * count}d', *(x for p in positions for x in p))
                + b't' + struct.pack(f'={count}i', *([1] * count)) + b'#')
    return (f'{count}\nLattice="{" ".join(map(str, cell))}" '
            f'Properties=species:S:1:pos:R:3 pbc="{pbc}" {metadata}\n'
            + ''.join('H ' + ' '.join(map(str, pos)) + '\n' for pos in positions)).encode()


def run_reduction(tmp_path, command, fmt, *, options=()):
    name, arguments = command
    return run_command('imrmlp', [name, *arguments, '--input', f'input.{fmt}',
                                '--input-format', fmt, '--output', 'output.xyz',
                                '--output-format', 'extxyz', '--overwrite',
                                '--num_threads', '1', *options], tmp_path)


@pytest.mark.parametrize('command', COMMANDS)
@pytest.mark.parametrize('fmt', ['extxyz', 'bin'])
@pytest.mark.parametrize('prefix', [False, True], ids=['empty-first', 'empty-second'])
@pytest.mark.parametrize('existing', [False, True])
def test_complete_zero_atom_frame_is_rejected_without_output(tmp_path, command, fmt, prefix, existing):
    (tmp_path / f'input.{fmt}').write_bytes((frame(fmt) if prefix else b'') + frame(fmt, 0))
    output = tmp_path / 'output.xyz'
    if existing:
        output.write_bytes(SENTINEL)
    result = run_reduction(tmp_path, command, fmt)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'Cell reduction requires a nonempty structure' in result.stderr
    if existing:
        assert output.read_bytes() == SENTINEL
    else:
        assert not output.exists()


@pytest.mark.parametrize('command', COMMANDS)
@pytest.mark.parametrize('tol', ['0', '-1', 'nan', 'inf', '-inf', '1e308'])
def test_invalid_tolerance_preserves_existing_output(tmp_path, command, tol):
    (tmp_path / 'input.extxyz').write_bytes(frame('extxyz'))
    output = tmp_path / 'output.xyz'
    output.write_bytes(SENTINEL)
    result = run_reduction(tmp_path, command, 'extxyz', options=['--tol', tol])
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stderr
    assert output.read_bytes() == SENTINEL


@pytest.mark.parametrize('command', COMMANDS)
@pytest.mark.parametrize('fmt', ['extxyz', 'bin'])
@pytest.mark.parametrize('prefix', [False, True])
def test_singular_cell_preserves_existing_output(tmp_path, command, fmt, prefix):
    singular = (4., 0., 0., 0., 5., 0., 0., 0., 0.)
    payload = (frame(fmt) if prefix else b'') + frame(fmt, cell=singular)
    (tmp_path / f'input.{fmt}').write_bytes(payload)
    output = tmp_path / 'output.xyz'
    output.write_bytes(SENTINEL)
    result = run_reduction(tmp_path, command, fmt)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'Cell reduction requires a finite, nonsingular cell' in result.stderr
    assert output.read_bytes() == SENTINEL


@pytest.mark.parametrize('command', COMMANDS)
@pytest.mark.parametrize('failure', ['nonfinite-position', 'nonfinite-cell', 'truncated-frame'])
def test_invalid_frame_never_reports_success_or_replaces_output(tmp_path, command, failure):
    payload = frame('extxyz')
    if failure == 'nonfinite-position':
        payload = frame('extxyz', positions=[(float('nan'), 1., 2.)])
    elif failure == 'nonfinite-cell':
        payload = frame('extxyz', cell=(float('inf'), *CELL[1:]))
    else:
        payload = frame('extxyz') + b'1\n'
    (tmp_path / 'input.extxyz').write_bytes(payload)
    output = tmp_path / 'output.xyz'
    output.write_bytes(SENTINEL)
    result = run_reduction(tmp_path, command, 'extxyz')
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stderr
    assert output.read_bytes() == SENTINEL


@pytest.mark.parametrize('command', COMMANDS)
@pytest.mark.parametrize('copies', [1, 2, 3])
@pytest.mark.parametrize('tilted', [False, True])
@pytest.mark.parametrize('left_handed', [False, True])
def test_valid_single_atom_and_repeated_cells_reduce(tmp_path, command, copies, tilted, left_handed):
    ax = -4. if left_handed else 4.
    bx, cx, cy = (.5, .25, .75) if tilted else (0., 0., 0.)
    cell = (ax * copies, 0., 0., bx, 5., 0., cx, cy, 6.)
    origin = (ax * .125 + bx * .25 + cx * .375, 5 * .25 + cy * .375, 6 * .375)
    positions = [(origin[0] + copy * ax, origin[1], origin[2]) for copy in range(copies)]
    (tmp_path / 'input.extxyz').write_bytes(frame('extxyz', copies, cell, positions))
    result = run_reduction(tmp_path, command, 'extxyz')
    assert result.returncode == 0, result.stdout + result.stderr
    assert f'Cell size reduced by {copies} times' in result.stdout
    output = (tmp_path / 'output.xyz').read_text()
    assert output.splitlines()[0] == '1'
    assert 'pbc="T T T"' in output
    lattice = [float(x) for x in re.search(r'Lattice="([^"]+)"', output).group(1).split()]
    determinant = (lattice[0] * (lattice[4] * lattice[8] - lattice[5] * lattice[7])
                   - lattice[1] * (lattice[3] * lattice[8] - lattice[5] * lattice[6])
                   + lattice[2] * (lattice[3] * lattice[7] - lattice[4] * lattice[6]))
    assert determinant == pytest.approx(ax * 5 * 6)


@pytest.fixture
def labeled_repeated_frames(tmp_path):
    labels = [('bulk sweep 0007', 'bulk'), ('bulk sweep 0008', 'strained_bulk')]
    cell = (8., 0., 0., 0., 10., 0., 0., 0., 6.)
    positions = [(.5, 1., 1.), (4.5, 1., 1.), (.5, 6., 1.), (4.5, 6., 1.)]
    payload = b''.join(frame('extxyz', 4, cell, positions,
                            metadata=f'label="{label}" config_type="{config_type}"')
                       for label, config_type in labels)
    (tmp_path / 'input.extxyz').write_bytes(payload)
    return labels


@pytest.mark.parametrize('command', COMMANDS)
def test_reduced_frames_keep_their_text_labels(tmp_path, command, labeled_repeated_frames):
    result = run_reduction(tmp_path, command, 'extxyz')
    assert result.returncode == 0, result.stdout + result.stderr
    lines = (tmp_path / 'output.xyz').read_text().splitlines()
    assert len(lines) == 6
    for index, (label, config_type) in enumerate(labeled_repeated_frames):
        assert lines[3 * index] == '1'
        header = lines[3 * index + 1]
        assert re.search(r'label="([^"]*)"', header).group(1) == label
        assert re.search(r'config_type="([^"]*)"', header).group(1) == config_type


@pytest.mark.parametrize('command', COMMANDS)
def test_reduced_and_primitive_frames_preserve_extensive_labels(tmp_path, command):
    cell = (8., 0., 0., 0., 10., 0., 0., 0., 6.)
    positions = [(.5, 1.25, 2.25), (4.5, 1.25, 2.25),
                 (.5, 6.25, 2.25), (4.5, 6.25, 2.25)]
    virial = [4., 1., 2., 1., 8., 3., 2., 3., 12.]
    reduced_virial = [value / 4 for value in virial]
    repeated = frame('extxyz', 4, cell, positions,
                     metadata='energy=-20 fenergy=-18 virial="4 1 2 1 8 3 2 3 12" '
                              'total_charge=3.2 label="repeated bulk"')
    unchanged = frame('extxyz', metadata='energy=-5 fenergy=-4.5 '
                                       'virial="1 .25 .5 .25 2 .75 .5 .75 3" '
                                       'total_charge=.8 label="already primitive"')
    (tmp_path / 'input.extxyz').write_bytes(repeated + unchanged)
    result = run_reduction(tmp_path, command, 'extxyz')
    assert result.returncode == 0, result.stdout + result.stderr
    assert re.findall(r'Cell size reduced by (\d+) times', result.stdout) == ['4', '1']
    lines = (tmp_path / 'output.xyz').read_text().splitlines()
    assert len(lines) == 6
    for i, label in enumerate(('repeated bulk', 'already primitive')):
        assert lines[3 * i] == '1'
        header = lines[3 * i + 1]
        assert float(re.search(r'\benergy=([^\s]+)', header).group(1)) == pytest.approx(-5.)
        assert float(re.search(r'\bfenergy=([^\s]+)', header).group(1)) == pytest.approx(-4.5)
        assert float(re.search(r'\btotal_charge=([^\s]+)', header).group(1)) == pytest.approx(.8)
        actual_virial = [float(value) for value in re.search(r'\bvirial="([^"]+)"', header).group(1).split()]
        assert actual_virial == pytest.approx(reduced_virial)
        lattice = [float(value) for value in re.search(r'Lattice="([^"]+)"', header).group(1).split()]
        volume = (lattice[0] * (lattice[4] * lattice[8] - lattice[5] * lattice[7])
                  - lattice[1] * (lattice[3] * lattice[8] - lattice[5] * lattice[6])
                  + lattice[2] * (lattice[3] * lattice[7] - lattice[4] * lattice[6]))
        assert volume == pytest.approx(120.)
        assert [value / volume for value in actual_virial] == pytest.approx([value / 480 for value in virial])
        assert re.search(r'label="([^"]*)"', header).group(1) == label


@pytest.mark.parametrize('command', COMMANDS)
def test_explicit_total_charge_survives_supercell_reduction_roundtrip(tmp_path, command):
    (tmp_path / 'source.extxyz').write_text(
        '1\nLattice="4 0 0 0 5 0 0 0 6" pbc="T T T" '
        'Properties=species:S:1:pos:R:3:charges:R:1 total_charge=.8\n'
        'H .5 1.25 2.25 .15\n')
    expanded = run_command('imrmlp', [
        'mk_super', '--input', 'source.extxyz', '--input-format', 'extxyz',
        '--output', 'input.extxyz', '--output-format', 'extxyz',
        '--supercell', '2', '1', '1', '--num_threads', '1'], tmp_path)
    assert expanded.returncode == 0, expanded.stdout + expanded.stderr
    lines = (tmp_path / 'input.extxyz').read_text().splitlines()
    assert lines[0] == '2'
    assert float(re.search(r'\btotal_charge=([^\s]+)', lines[1]).group(1)) == pytest.approx(1.6)
    assert sum(float(row.split()[-1]) for row in lines[2:]) == pytest.approx(.3)

    result = run_reduction(tmp_path, command, 'extxyz')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Cell size reduced by 2 times' in result.stdout
    lines = (tmp_path / 'output.xyz').read_text().splitlines()
    assert lines[0] == '1'
    assert float(re.search(r'\btotal_charge=([^\s]+)', lines[1]).group(1)) == pytest.approx(.8)


def test_reduction_keeps_xdatcar_frame_numbers(tmp_path, labeled_repeated_frames):
    result = run_command('imrmlp', [
        'trans_reduce', '--input', 'input.extxyz', '--input-format', 'extxyz',
        '--output', 'output.XDATCAR', '--output-format', 'xdatcar',
        '--overwrite', '--num_threads', '1'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    output = (tmp_path / 'output.XDATCAR').read_text()
    assert re.findall(r'Direct configuration=\s*(-?\d+)', output) == ['0', '1']


@pytest.mark.parametrize('command', COMMANDS)
def test_reduction_preserves_finite_slab_layers(tmp_path, command):
    cell = (8., 0., 0., 0., 5., 0., 0., 0., 20.)
    positions = [(.5, 1., 0.), (4.5, 1., 0.), (.5, 1., 10.), (4.5, 1., 10.)]
    (tmp_path / 'input.extxyz').write_bytes(frame('extxyz', 4, cell, positions, pbc='T T F'))
    result = run_reduction(tmp_path, command, 'extxyz')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Cell size reduced by 2 times' in result.stdout
    output = (tmp_path / 'output.xyz').read_text()
    lines = output.splitlines()
    assert lines[0] == '2'
    assert 'pbc="T T F"' in lines[1]
    lattice = [float(x) for x in re.search(r'Lattice="([^"]+)"', output).group(1).split()]
    assert lattice == pytest.approx((4., 0., 0., 0., 5., 0., 0., 0., 20.))
    assert sorted(float(line.split()[3]) for line in lines[2:]) == pytest.approx([0., 10.])
