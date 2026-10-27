"""The production primitive-reduction CLI finds non-axis-aligned slab repeats."""
import re
import numpy as np
import pytest
from test_cli_error_paths import run_command


@pytest.mark.parametrize('singular', [False, True])
@pytest.mark.parametrize('two_stage', [False, True])
def test_primitive_reduction_finds_plane_group_and_preserves_finite_layers(tmp_path,singular,two_stage):
    (tmp_path/'index.map').write_text('H 1\nHe 1\n')
    height=0 if singular else 12
    (tmp_path/'slab.xyz').write_text(
        f'4\nLattice="2 3 0 -2 3 0 0 0 {height}" pbc="T T F" '
        'Properties=species:S:1:pos:R:3 energy=8 fenergy=6\n'
        'H .2 .3 .4\nHe .7 .8 1.4\nH 2.2 .3 .4\nHe 2.7 .8 1.4\n')
    args=['primi_reduce','--input','slab.xyz','--input-format','extxyz',
          '--output','primitive.xyz','--output-format','extxyz','--num_threads','1']
    if two_stage: args+=['--two-stage','true']
    result=run_command('imrmlp',args,tmp_path)
    assert result.returncode==0,result.stdout+result.stderr
    rows=(tmp_path/'primitive.xyz').read_text().splitlines()
    assert rows[0]=='2' and len(rows)==4
    cell=np.array(list(map(float,re.search(r'Lattice="([^"]+)"',rows[1]).group(1).split()))).reshape(3,3)
    assert np.linalg.norm(np.cross(cell[0],cell[1]))==pytest.approx(6.)
    np.testing.assert_array_equal(cell[2],[0,0,height])
    assert 'pbc="T T F"' in rows[1]
    assert float(re.search(r'\benergy=([^ ]+)',rows[1]).group(1))==4
    assert float(re.search(r'\bfenergy=([^ ]+)',rows[1]).group(1))==3
    positions={row.split()[0]:np.array(list(map(float,row.split()[1:4]))) for row in rows[2:]}
    for symbol,old in [('H',np.array([.2,.3,.4])),('He',np.array([.7,.8,1.4]))]:
        delta=old-positions[symbol]
        assert delta[2]==pytest.approx(0,abs=1e-12)
        coeff=np.linalg.solve(cell[:2,:2].T,delta[:2])
        np.testing.assert_allclose(coeff,np.rint(coeff),atol=1e-12)
