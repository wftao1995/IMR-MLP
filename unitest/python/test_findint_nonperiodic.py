"""Finite-cluster cavities use host geometry, not periodic images of a box."""
import itertools
import numpy as np
import pytest
from test_cli_error_paths import run_command


@pytest.mark.parametrize('zero_cell',[False,True])
@pytest.mark.parametrize('local',[False,True])
def test_findint_fff_finds_only_cube_center_in_cartesian_coordinates(tmp_path,zero_cell,local):
    center=np.array([4.,5.,6.])
    points=[center+2*np.array(signs) for signs in itertools.product([-1,1],repeat=3)]
    cell='0 0 0 0 0 0 0 0 0' if zero_cell else '12 0 0 0 12 0 0 0 12'
    source='8\nLattice="'+cell+'" pbc="F F F" Properties=species:S:1:pos:R:3 energy=9\n'
    source+=''.join('Si '+' '.join(map(str,p))+'\n' for p in points)
    (tmp_path/'host.xyz').write_text(source)
    args=['findint','--input=host.xyz','--input-format=extxyz',
          '--output=sites.xyz','--output-format=extxyz','--sites-only','--insert=H',
          '--cluster-tol=.01','--volprec=.1']
    if local: args+=['--near-pos-car=4,5,6','--near-radius=4']
    result=run_command('imrmlp',args,tmp_path)
    assert result.returncode==0,result.stdout+result.stderr
    rows=(tmp_path/'sites.xyz').read_text().splitlines()
    assert rows[0]=='1'
    assert 'pbc="F F F"' in rows[1]
    assert 'energy=' not in rows[1]
    assert rows[2].split()[0]=='H'
    np.testing.assert_allclose(list(map(float,rows[2].split()[1:4])),center,atol=1e-10)
    assert 'Position (Cartesian Angstrom)' in result.stdout
    assert (tmp_path/'host.xyz').read_text()==source


def test_findint_fff_planar_input_fails_without_replacing_output(tmp_path):
    (tmp_path/'host.xyz').write_text('4\nLattice="8 0 0 0 8 0 0 0 8" pbc="F F F" '
        'Properties=species:S:1:pos:R:3\nSi 0 0 0\nSi 2 0 0\nSi 0 2 0\nSi 2 2 0\n')
    (tmp_path/'sites.xyz').write_text('keep output\n')
    result=run_command('imrmlp',['findint','--input=host.xyz','--input-format=extxyz',
        '--output=sites.xyz','--output-format=extxyz'],tmp_path)
    assert result.returncode!=0
    assert 'three-dimensional atomic convex hull' in result.stdout+result.stderr
    assert (tmp_path/'sites.xyz').read_text()=='keep output\n'
