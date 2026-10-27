"""Composition and scalar-coordinate lower envelopes use fixed reference sets."""
import re
import struct

import pytest
from test_cli_error_paths import run_command


DIAGNOSTICS = ('hull_energy_per_atom', 'energy_relative_to_hull', 'energy_above_hull')


def xyz_frame(species, energy, *, metadata='', force=None, virial=None):
    tags = ('Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3'
            + (':forces:R:3' if force is not None else ''))
    if energy is not None:
        tags += f' energy={energy}'
    if virial is not None:
        tags += f' virial="{virial} 0 0 0 0 0 0 0 0"'
    tags += ' ' + metadata
    positions = ''.join(f'{symbol} {j} 0 0' + (f' {force} 0 0' if force is not None else '')
                        + '\n' for j, symbol in enumerate(species))
    return f'{len(species)}\n{tags}\n{positions}'


def binary_frame(*, energy=0., doubles=(), integers=(), features=()):
    # Native MLIPBIN fixtures preserve the distinction between typed scalars
    # and Feature strings, which EXTXYZ cannot represent for integer globals.
    def string(value):
        encoded = value.encode()
        return struct.pack('=i', len(encoded)) + encoded
    payload = (b'GPMLBIN2' + struct.pack('=Ii', 3, 1)
               + b'l' + struct.pack('=9d', 10, 0, 0, 0, 10, 0, 0, 0, 10)
               + b'p' + struct.pack('=3d', 0, 0, 0) + b't' + struct.pack('=i', 0))
    if energy is not None:
        payload += b'e' + struct.pack('=d', energy)
    for key, value in doubles:
        payload += b'C' + string(key) + struct.pack('=d', value)
    for key, value in integers:
        payload += b'E' + string(key) + struct.pack('=i', value)
    for key, value in features:
        payload += b'F' + string(key) + string(value)
    return payload + b'#'


@pytest.fixture
def hull_input(tmp_path):
    (tmp_path / 'index.map').write_text('H 99\nHe -31\nLi 18\n')
    return ['--input-format', 'extxyz', '--trainset', 'input.xyz', '--index', 'index.map',
            '--output', 'output.cfg', '--energy-mode', 'composition-hull']


def values(tmp_path, key):
    return [float(value) for value in re.findall(r'^Feature\s+' + key + r'\s+(\S+)',
                                                (tmp_path / 'output.cfg').read_text(), re.M)]


def succeeds(tmp_path, args):
    result = run_command('calc_fitness', args, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def fails_preserving_output(tmp_path, args, message):
    (tmp_path / 'output.cfg').write_text('original output\n')
    result = run_command('calc_fitness', args, tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert message in result.stderr
    assert (tmp_path / 'output.cfg').read_text() == 'original output\n'


def test_binary_self_hull_supercells_and_map_order(tmp_path, hull_input):
    frames = [xyz_frame(['H'], 0), xyz_frame(['He'], 0),
              xyz_frame(['H', 'He'], -2), xyz_frame(['He', 'H', 'H', 'He'], -2.8),
              xyz_frame(['H', 'H', 'H', 'He'], -1.2)]
    (tmp_path / 'input.xyz').write_text(''.join(frames))
    succeeds(tmp_path, hull_input)
    assert values(tmp_path, 'hull_energy_per_atom') == pytest.approx([0, 0, -1, -1, -.5])
    assert values(tmp_path, 'energy_above_hull') == pytest.approx([0, 0, 0, .3, .2], abs=1e-8)
    assert values(tmp_path, 'Fitness') == pytest.approx([1, 1, 1, .25, 1/3], abs=1e-8)
    assert [float(x) for x in re.findall(r'\nEnergy\n\s*(\S+)',
                                       (tmp_path / 'output.cfg').read_text())] == [0, 0, -2, -2.8, -1.2]
    # The index references are intentionally nonzero and unrelated to the hull.
    (tmp_path / 'index.map').write_text('Li -100\nHe 700\nH -900\n')
    (tmp_path / 'input.xyz').write_text(''.join(reversed(frames)))
    succeeds(tmp_path, hull_input)
    assert values(tmp_path, 'energy_above_hull') == pytest.approx([.2, .3, 0, 0, 0], abs=1e-8)


def test_ternary_hull_and_fixed_composition(tmp_path, hull_input):
    (tmp_path / 'input.xyz').write_text(''.join([
        xyz_frame(['H'], 0), xyz_frame(['He'], 0), xyz_frame(['Li'], 0),
        xyz_frame(['H', 'He', 'Li'], -1.8), xyz_frame(['Li', 'He', 'H'], -1.2)]))
    succeeds(tmp_path, hull_input)
    assert values(tmp_path, 'energy_above_hull') == pytest.approx([0, 0, 0, 0, .2], abs=1e-8)
    (tmp_path / 'input.xyz').write_text(xyz_frame(['H', 'He'], -2)
                                         + xyz_frame(['H', 'He', 'H', 'He'], -2))
    succeeds(tmp_path, hull_input)
    assert values(tmp_path, 'energy_above_hull') == pytest.approx([0, .5], abs=1e-8)


def test_reference_hull_is_fixed_and_allows_below_query(tmp_path, hull_input):
    # Reference .cfg must be guessed independently of query --input-format extxyz.
    (tmp_path / 'ref.cfg').write_text('BEGIN_CFG\nSize\n1\nSupercell\n10 0 0\n0 10 0\n0 0 10\n'
                                        'AtomData: id type cartes_x cartes_y cartes_z\n1 0 0 0 0\nEnergy\n0\nEND_CFG\n')
    (tmp_path / 'input.xyz').write_text(xyz_frame(['H'], -1) + xyz_frame(['H'], .2))
    succeeds(tmp_path, hull_input + ['--hull-reference', 'ref.cfg'])
    assert values(tmp_path, 'hull_energy_per_atom') == pytest.approx([0, 0])
    assert values(tmp_path, 'energy_relative_to_hull') == pytest.approx([-1, .2])
    assert values(tmp_path, 'energy_above_hull') == pytest.approx([0, .2])
    assert values(tmp_path, 'Fitness') == pytest.approx([1, 1/3])


@pytest.mark.parametrize('coords', ['c=0.2 d=0.3 z=2', 'c=0.8 d=0.8 z=1'])
def test_property_reference_outside_affine_or_convex_domain(tmp_path, hull_input, coords):
    (tmp_path / 'ref.xyz').write_text(''.join(xyz_frame(['H'], e, metadata=m) for e, m in [
        (0, 'c=0 d=0 z=1'), (1, 'c=1 d=0 z=1'), (2, 'c=0 d=1 z=1')]))
    (tmp_path / 'input.xyz').write_text(xyz_frame(['H'], 0, metadata=coords))
    fails_preserving_output(tmp_path, hull_input + ['--energy-mode', 'property-hull',
        '--hull-reference', 'ref.xyz', '--hull-coordinates', 'c', 'd', 'z'], 'outside')


def test_composition_outside_reference_domain(tmp_path, hull_input):
    (tmp_path / 'ref.xyz').write_text(xyz_frame(['H'], 0))
    (tmp_path / 'input.xyz').write_text(xyz_frame(['He'], 0))
    fails_preserving_output(tmp_path, hull_input + ['--hull-reference', 'ref.xyz'], 'outside')


def test_property_integer_double_and_redundant_coordinates(tmp_path, hull_input):
    (tmp_path / 'input.bin').write_bytes(b''.join([
        binary_frame(energy=0, integers=[('i', 0)], doubles=[('d', 10)]),
        binary_frame(energy=-1, integers=[('i', 1)], doubles=[('d', 20)]),
        binary_frame(energy=0, integers=[('i', 2)], doubles=[('d', 30)]),
        binary_frame(energy=-.6, integers=[('i', 1)], doubles=[('d', 20)])]))
    succeeds(tmp_path, hull_input + ['--input-format', 'bin', '--trainset', 'input.bin',
        '--energy-mode', 'property-hull', '--hull-coordinates', 'i', 'd'])
    assert values(tmp_path, 'energy_above_hull') == pytest.approx([0, 0, 0, .4], abs=1e-8)
    assert values(tmp_path, 'Fitness') == pytest.approx([1, 1, 1, .2], abs=1e-8)


@pytest.mark.parametrize('payload,diagnostic', [
    (binary_frame(features=[('c', '0')]), 'existing global'),
    (binary_frame(doubles=[('c', float('nan'))]), 'finite'),
    (binary_frame(doubles=[('c', float('inf'))]), 'finite'),
    (binary_frame(doubles=[('c', 0)], integers=[('c', 0)]), 'ambiguous'),
], ids=['feature-string', 'nan', 'inf', 'ambiguous'])
def test_property_coordinate_contract(tmp_path, hull_input, payload, diagnostic):
    (tmp_path / 'input.bin').write_bytes(payload)
    fails_preserving_output(tmp_path, hull_input + ['--input-format', 'bin', '--trainset', 'input.bin',
        '--energy-mode', 'property-hull', '--hull-coordinates', 'c'], diagnostic)


@pytest.mark.parametrize('energy', [None, float('nan'), float('inf')])
def test_explicit_reference_requires_energy_in_every_frame(tmp_path, hull_input, energy):
    (tmp_path / 'ref.bin').write_bytes(binary_frame(energy=0) + binary_frame(energy=energy))
    (tmp_path / 'input.xyz').write_text(xyz_frame(['H'], 0))
    fails_preserving_output(tmp_path, hull_input + ['--hull-reference', 'ref.bin'], 'reference configuration 2')


def test_missing_query_energy_is_neutral_and_clears_old_diagnostics(tmp_path, hull_input):
    stale = [(key, 99) for key in DIAGNOSTICS]
    (tmp_path / 'input.bin').write_bytes(binary_frame(energy=None, doubles=stale,
        features=[(key, '99') for key in DIAGNOSTICS]))
    succeeds(tmp_path, hull_input + ['--input-format', 'bin', '--trainset', 'input.bin',
                                    '--energy-mode', 'property-hull', '--hull-coordinates', 'absent'])
    assert values(tmp_path, 'Fitness') == [1]
    for key in DIAGNOSTICS:
        assert values(tmp_path, key) == []


def test_hull_multiplies_force_virial_and_preserves_free_energy(tmp_path, hull_input):
    (tmp_path / 'ref.xyz').write_text(xyz_frame(['H'], 0))
    (tmp_path / 'input.xyz').write_text(xyz_frame(['H'], .1, metadata='fenergy=-3', force=2, virial=2))
    succeeds(tmp_path, hull_input + ['--hull-reference', 'ref.xyz',
        '--select_by_force', 'true', '--select_by_stress', 'true'])
    assert values(tmp_path, 'Fitness') == pytest.approx([.5 * .2 * 100/102])
    assert values(tmp_path, 'fenergy') == [-3]


@pytest.mark.parametrize('option,value', [('hullevl', '0'), ('hullevl', '-1'), ('hullevl', 'nan'),
    ('hullref', 'inf'), ('hullpow', 'nan'), ('divide_by_size', 'false')])
def test_enabled_hull_parameter_validation(tmp_path, hull_input, option, value):
    (tmp_path / 'input.xyz').write_text(xyz_frame(['H'], 0))
    fails_preserving_output(tmp_path, hull_input + ['--' + option, value],
                            'divide_by_size' if option == 'divide_by_size' else 'Hull fitness requires')


def test_only_enabled_energy_parameters_are_validated(tmp_path, hull_input):
    (tmp_path / 'input.xyz').write_text(xyz_frame(['H'], 0))
    succeeds(tmp_path, hull_input + ['--engevl', 'nan', '--engref', 'inf', '--engpow', 'nan'])
    succeeds(tmp_path, hull_input + ['--energy-mode', 'formation', '--hullevl', 'nan', '--hullref', 'inf'])
    succeeds(tmp_path, hull_input + ['--select_by_energy', 'false', '--hullevl', 'nan',
        '--divide_by_size', 'false', '--hull-reference', 'absent', '--energy-mode', 'property-hull'])


def test_hull_reference_built_before_threshold_and_coordinates_keys_unique(tmp_path, hull_input):
    (tmp_path / 'input.xyz').write_text(xyz_frame(['H'], -1) + xyz_frame(['H'], -.9))
    succeeds(tmp_path, hull_input + ['--threshold', '.9'])
    assert values(tmp_path, 'hull_energy_per_atom') == [-1]
    fails_preserving_output(tmp_path, hull_input + ['--energy-mode', 'property-hull',
        '--hull-coordinates', 'c', 'c'], 'keys must be unique')
    fails_preserving_output(tmp_path, hull_input + ['--energy-mode', 'property-hull'], 'requires hull-coordinates')


@pytest.mark.parametrize('mode,enabled', [('formation', 'true'), ('property-hull', 'false')])
def test_inactive_hull_preserves_existing_metadata(tmp_path, hull_input, mode, enabled):
    (tmp_path / 'input.xyz').write_text(xyz_frame(['H'], 0,
        metadata=' '.join(key + '=99' for key in DIAGNOSTICS)))
    # EXTXYZ scalars become typed globals; use Feature strings as well to
    # verify the command does not erase unrelated input metadata in old modes.
    (tmp_path / 'input.bin').write_bytes(binary_frame(doubles=[(key, 99) for key in DIAGNOSTICS],
        features=[(key, '99') for key in DIAGNOSTICS]))
    succeeds(tmp_path, hull_input + ['--input-format', 'bin', '--trainset', 'input.bin',
        '--energy-mode', mode, '--select_by_energy', enabled])
    for key in DIAGNOSTICS:
        assert values(tmp_path, key) == [99]


def test_separate_hull_reference_scale_power_and_query_snapshot(tmp_path, hull_input):
    (tmp_path / 'ref.xyz').write_text(xyz_frame(['H'], 0, metadata='fitness=0')
                                    + xyz_frame(['H'], 0, metadata='fitness=1'))
    (tmp_path / 'input.xyz').write_text(xyz_frame(['H'], .5, metadata='fitness=0.5'))
    succeeds(tmp_path, hull_input + ['--energy-mode', 'property-hull', '--hull-coordinates', 'fitness',
        '--hull-reference', 'ref.xyz', '--hull-reference-format', 'extxyz',
        '--hullref', '.1', '--hullevl', '.2', '--hullpow', '2'])
    assert values(tmp_path, 'energy_above_hull') == pytest.approx([.5])
    assert values(tmp_path, 'Fitness') == pytest.approx([.2/(.2+.4**2)])


@pytest.mark.parametrize('power', ['0', '.01'])
@pytest.mark.parametrize('external', [False, True])
def test_on_hull_reference_endpoints_have_exact_zero_excess(tmp_path, hull_input, power, external):
    frames = xyz_frame(['H'], -.6, metadata='c=0') + xyz_frame(['H'], .3, metadata='c=1')
    (tmp_path / 'input.xyz').write_text(frames)
    args = hull_input + ['--energy-mode', 'property-hull', '--hull-coordinates', 'c', '--hullpow', power]
    if external:
        (tmp_path / 'ref.xyz').write_text(frames)
        args += ['--hull-reference', 'ref.xyz']
    succeeds(tmp_path, args)
    assert values(tmp_path, 'hull_energy_per_atom') == [-.6, .3]
    assert values(tmp_path, 'energy_relative_to_hull') == [0, 0]
    assert values(tmp_path, 'energy_above_hull') == [0, 0]
    assert values(tmp_path, 'Fitness') == [1, 1]


def test_external_hull_preserves_small_real_signed_differences(tmp_path, hull_input):
    (tmp_path / 'ref.xyz').write_text(xyz_frame(['H'], -.6, metadata='c=0')
                                    + xyz_frame(['H'], .3, metadata='c=1'))
    energies = [.3 - 1e-12, .3 + 1e-12]
    (tmp_path / 'input.xyz').write_text(''.join(xyz_frame(['H'], e, metadata='c=1') for e in energies))
    succeeds(tmp_path, hull_input + ['--energy-mode', 'property-hull', '--hull-coordinates', 'c',
        '--hull-reference', 'ref.xyz', '--hullpow', '0'])
    assert values(tmp_path, 'energy_relative_to_hull') == [e - .3 for e in energies]
    assert values(tmp_path, 'energy_above_hull') == [0, energies[1] - .3]
    assert values(tmp_path, 'Fitness') == pytest.approx([1, 1/11])
