# Exercise the public filename reader through real convstr consumers.
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
import imrmlp
from test_cli_error_paths import run_command


def xml_trajectory():
    header = ('<modeling><generator><i name="program">vasp</i>'
              '<i name="version">6.3.2</i></generator>'
              '<atominfo><atoms>1</atoms><types>1</types><array name="atomtypes">'
              '<set><rc><c>1</c><c>H</c></rc></set></array></atominfo>')
    steps = []
    for i in range(2):
        steps.append(f'''<calculation><scstep/><structure><crystal><varray name="basis">
<v>3 0 0</v><v>0 3 0</v><v>0 0 3</v></varray><i name="volume">27</i></crystal>
<varray name="positions"><v>{.1 + .2 * i} .2 .3</v></varray></structure>
<varray name="forces"><v>1 2 3</v></varray><varray name="stress">
<v>1 0 0</v><v>0 2 0</v><v>0 0 3</v></varray><energy>
<i name="e_fr_energy">{-10-i}</i><i name="e_wo_entrp">-9</i>
<i name="e_0_energy">-9.5</i></energy></calculation>''')
    return header + ''.join(steps) + '</modeling>'


@pytest.mark.parametrize('format', ['xml', 'vasprun', 'vasprun.xml'])
def test_vasprun_convstr_split_keeps_both_frame_files_and_values(tmp_path, format):
    source = tmp_path / 'vasprun.xml'
    source.write_text(xml_trajectory())
    result = run_command('convstr', ['--inputfiles', source, '--input-format', format,
                                    '--outputfile', 'frame', '--output-format', 'xsf'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(p.name for p in tmp_path.glob('frame*')) == ['frame0', 'frame1']
    for i in range(2):
        atom = imrmlp.read_one(tmp_path / f'frame{i}', format='xsf')
        np.testing.assert_allclose(atom.positions, [[.3 + .6 * i, .6, .9]], atol=1e-14)
        np.testing.assert_array_equal(atom.cell, np.eye(3) * 3)
        np.testing.assert_array_equal(atom.numbers, [1])
        np.testing.assert_allclose(atom.forces, [[1., 2., 3.]], atol=1e-13)


@pytest.mark.parametrize('format', ['xml', 'vasprun', 'vasprun.xml'])
@pytest.mark.parametrize('discard', [[], ['energy'], ['forces', 'virial'], ['energy', 'forces', 'virial']])
def test_vasprun_convstr_discards_only_requested_labels(tmp_path, format, discard):
    source = tmp_path / 'vasprun.xml'
    source.write_text(xml_trajectory())
    options = ['--discard', *discard] if discard else []
    result = run_command('convstr', ['--inputfiles', source, '--input-format', format,
                                    '--outputfile', 'frames.xyz', '--output-format', 'extxyz', *options], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    frames = imrmlp.read(tmp_path / 'frames.xyz')
    assert len(frames) == 2
    for i, atom in enumerate(frames):
        assert (atom.energy is None) == ('energy' in discard)
        assert (atom.forces is None) == ('forces' in discard)
        assert (atom.stress is None) == ('virial' in discard)
        if atom.energy is not None:
            assert atom.energy == -10 - i
        if atom.forces is not None:
            np.testing.assert_array_equal(atom.forces, [[1., 2., 3.]])
        if atom.stress is not None:
            np.testing.assert_allclose(atom.stress, np.diag([27., 54., 81.]) / 1602.1766208)
        np.testing.assert_allclose(atom.positions, [[.3 + .6 * i, .6, .9]], atol=1e-14)
