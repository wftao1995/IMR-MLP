"""Primitive XSF geometry is independent of the optional conventional cell."""
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
import imrmlp
from test_cli_error_paths import run_command

# XCrySDen's own ZnS example:
# https://web.mit.edu/xcrysden_v1.5.60/www/XCRYSDEN/doc/XSF.html
PRIMITIVE = np.array([[0., 2.71, 2.71], [2.71, 0., 2.71], [2.71, 2.71, 0.]])
POSITIONS = np.array([[0., 0., 0.], [1.355, -1.355, -1.355]])


def zns_text(conventional):
    extra = 'CONVVEC\n5.42 0 0\n0 5.42 0\n0 0 5.42\n' if conventional else ''
    return ('CRYSTAL\nPRIMVEC\n0 2.71 2.71\n2.71 0 2.71\n2.71 2.71 0\n' + extra +
            'PRIMCOORD\n2 1\n16 0 0 0\n30 1.355 -1.355 -1.355\n')


def check_zns(atom):
    np.testing.assert_array_equal(atom.cell, PRIMITIVE)
    np.testing.assert_array_equal(atom.positions, POSITIONS)
    np.testing.assert_array_equal(atom.numbers, [16, 30])
    np.testing.assert_array_equal(atom.pbc, [True, True, True])
    assert atom.forces is None


@pytest.mark.parametrize('conventional', [False, True])
def test_xsf_standard_zns_keeps_primitive_cell_and_cartesian_positions(tmp_path, conventional):
    source = tmp_path / 'ZnS.xsf'
    source.write_text(zns_text(conventional))
    check_zns(imrmlp.read_one(source))
    frames = imrmlp.read(source)
    assert len(frames) == 1
    check_zns(frames[0])
    result = run_command('convstr', ['--inputfiles', source, '--input-format', 'xsf',
                                    '--outputfile', 'ZnS.xyz', '--output-format', 'extxyz'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    check_zns(imrmlp.read_one(tmp_path / 'ZnS.xyz'))
