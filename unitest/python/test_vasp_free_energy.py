"""Explicit VASP free energies remain available to ASE and EXTXYZ consumers."""
import re
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
import imrmlp
from test_cli_error_paths import run_command
from test_streaming_python_contracts import bounded_outcar
from test_vasprun_consumers import xml_trajectory


def source_text(format):
    return bounded_outcar() if format == 'outcar' else xml_trajectory()


@pytest.mark.parametrize('format,stream', [('outcar', False), ('outcar', True), ('vasprun', False)])
def test_public_vasp_explicit_free_energy_reaches_ase_and_extxyz_without_source_changes(tmp_path, format, stream):
    source = tmp_path / ('OUTCAR' if format == 'outcar' else 'vasprun.xml')
    source.write_text(source_text(format))
    raw = source.read_bytes()
    frames = list(imrmlp.iread(source, format=format)) if stream else imrmlp.read(source, format=format)
    assert len(frames) == 2
    snapshots = [atom.copy() for atom in frames]
    for i, atom in enumerate(frames):
        assert atom.energy == -10 - i
        assert atom.free_energy == -10 - i
        ase_atom = atom.to_ase()
        assert ase_atom.get_potential_energy(force_consistent=True) == -10 - i
        assert ase_atom.calc.results['energy'] == -10 - i
    output = tmp_path / 'roundtrip.xyz'
    imrmlp.write(output, frames, format='extxyz')
    restored = imrmlp.read(output)
    assert len(restored) == 2
    for i, (atom, before, after) in enumerate(zip(frames, snapshots, restored)):
        assert atom.energy == before.energy == after.energy
        assert atom.free_energy == before.free_energy == after.free_energy == -10 - i
        for name in ('positions', 'cell', 'forces'):
            np.testing.assert_allclose(getattr(after, name), getattr(before, name), atol=1e-13, rtol=0)
            np.testing.assert_array_equal(getattr(atom, name), getattr(before, name))
        if before.stress is not None:
            np.testing.assert_allclose(after.stress, before.stress, atol=1e-13, rtol=0)
    assert source.read_bytes() == raw


@pytest.mark.parametrize('format', ['outcar', 'vasprun', 'extxyz'])
@pytest.mark.parametrize('discard', ['fenergy', 'free_energy', 'energy'])
def test_current_cli_free_energy_discard_aliases_keep_independent_fields_and_labels(tmp_path, format, discard):
    source = tmp_path / 'input'
    if format == 'extxyz':
        source.write_text('1\nLattice="3 0 0 0 3 0 0 0 3" energy=-9 fenergy=-10 '
                          'virial="1 0 0 0 2 0 0 0 3" Properties=species:S:1:pos:R:3:forces:R:3\nH 1 2 3 .1 .2 .3\n')
    else:
        source.write_text(source_text(format))
    before = imrmlp.read(source, format=format)
    result = run_command('convstr', ['--inputfiles', source, '--input-format', format,
                                    '--outputfile', 'filtered.xyz', '--output-format', 'extxyz',
                                    '--loadrate', '0', '--outrate', '0', '--discard', discard], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / 'filtered.xyz'
    after = imrmlp.read(output)
    assert len(after) == len(before)
    headers = [line for line in output.read_text().splitlines() if line.startswith('Lattice=')]
    for atom, prior, header in zip(after, before, headers):
        # Match complete fields so energy never also matches fenergy.
        assert bool(re.search(r'(?:^|\s)energy=', header)) == (discard != 'energy')
        assert bool(re.search(r'(?:^|\s)fenergy=', header)) == (discard == 'energy')
        assert atom.energy == (None if discard == 'energy' else prior.energy)
        assert atom.free_energy == (prior.free_energy if discard == 'energy' else None)
        np.testing.assert_array_equal(atom.positions, prior.positions)
        np.testing.assert_array_equal(atom.forces, prior.forces)
        if prior.stress is not None:
            np.testing.assert_allclose(atom.stress, prior.stress, atol=1e-13, rtol=0)
