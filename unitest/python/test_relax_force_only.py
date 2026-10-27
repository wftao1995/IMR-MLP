import numpy as np
import pytest
from test_cli_error_paths import run_command


def pair():
    import imrmlp
    potential=imrmlp.Potential.from_settings({'pottype':'lj612','ntypes':1,'0-0':'1 1','cut:0-0':4})
    atoms=imrmlp.Atoms(numbers=[1,1],elements=['H'],positions=[[0,0,0],[1.5,0,0]],cell=np.zeros((3,3)),pbc=False)
    return potential,atoms


def test_force_only_public_python_converges_a_real_pair():
    import imrmlp
    potential,atoms=pair()
    result=imrmlp.Relaxation(potential,driver='lbfgspp',force_only=True,max_iter=100,gtol=1e-8,max_move=.05).run(atoms)
    assert result.converged
    assert np.linalg.norm(atoms.positions[1]-atoms.positions[0])==pytest.approx(2**(1/6),abs=1e-8)
    assert result.final_energy==pytest.approx(-1.,abs=1e-12)


def test_force_only_requires_lbfgspp():
    import imrmlp
    potential,_=pair()
    with pytest.raises(ValueError,match='lbfgspp'):
        imrmlp.Relaxation(potential,driver='fire',force_only=True)


def test_force_only_actual_cli_runs_existing_mode_four(tmp_path):
    (tmp_path/'index.map').write_text('H 0\n')
    (tmp_path/'pair.xyz').write_text('2\npbc="F F F" Properties=species:S:1:pos:R:3\nH 0 0 0\nH 1.5 0 0\n')
    (tmp_path/'input.ini').write_text('input pair.xyz\ninput-format extxyz\noutput relaxed.xyz\noutput-format extxyz\nisif 2\npot:pottype lj612\npot:ntypes 1\npot:0-0 1 1\npot:cut:0-0 4\nrelax:driver lbfgspp\nrelax:lbfgspp:linesearch 4\nrelax:gtol 1e-8\nrelax:stepmax .05\nrelax:maxiteration 100\nrelax:write_traj false\n')
    result=run_command('imrmlp',['relax','input.ini','--num_threads','1'],tmp_path)
    assert result.returncode==0,result.stdout+result.stderr
    import imrmlp
    final=imrmlp.read_one(tmp_path/'relaxed.xyz')
    assert np.linalg.norm(final.positions[1]-final.positions[0])==pytest.approx(2**(1/6),abs=1e-8)
