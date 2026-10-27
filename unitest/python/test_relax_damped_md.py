import numpy as np
import pytest
from test_cli_error_paths import run_command

@pytest.mark.parametrize('driver,factor',[('mdmin',.5),('quickmin',1.)])
def test_damped_md_public_python_first_step_and_current_state(driver,factor):
    import imrmlp
    pot=imrmlp.Potential.from_settings({'pottype':'lj612','ntypes':1,'0-0':'1 1','cut:0-0':4})
    a=imrmlp.Atoms(numbers=[1,1],elements=['H'],positions=[[0,0,0],[1.5,0,0]],cell=np.zeros((3,3)),pbc=False)
    dt=.1
    result=imrmlp.Relaxation(pot,driver=driver,timestep=dt,max_iter=1,gtol=0,max_move=10).run(a)
    derivative=24*(1/1.5**7-2/1.5**13)
    displacement=factor*dt**2*derivative
    np.testing.assert_allclose(a.positions[:,0],[displacement,1.5-displacement],atol=1e-13)
    assert result.n_iter==1 and result.stop_reason=='iteration_limit' and not result.converged
    r=1.5-2*displacement
    assert result.final_energy==pytest.approx(4*(r**-12-r**-6))

@pytest.mark.parametrize('driver',['mdmin','quickmin'])
def test_damped_md_actual_relax_cli_converges_a_pair(tmp_path,driver):
    import imrmlp
    (tmp_path/'index.map').write_text('H 0\n')
    (tmp_path/'input.xyz').write_text('2\npbc="F F F" Lattice="0 0 0 0 0 0 0 0 0" Properties=species:S:1:pos:R:3\nH 0 0 0\nH 1.5 0 0\n')
    (tmp_path/'settings.ini').write_text('input input.xyz\ninput-format extxyz\noutput output.xyz\noutput-format extxyz\nisif 2\npot:pottype lj612\npot:ntypes 1\npot:0-0 1 1\npot:cut:0-0 4\nrelax:driver '+driver+'\nrelax:'+driver+':dt .03\nrelax:maxiteration 1000\nrelax:gtol 1e-7\nrelax:stepmax .05\nrelax:write_traj false\n')
    result=run_command('imrmlp',['relax','settings.ini','--num_threads','1'],tmp_path)
    assert result.returncode==0,result.stdout+result.stderr
    a=imrmlp.read_one(tmp_path/'output.xyz')
    assert np.linalg.norm(a.positions[1]-a.positions[0])==pytest.approx(2**(1/6),abs=1e-7)
    assert a.energy==pytest.approx(-1.,abs=1e-12)

def test_damped_md_timestep_does_not_silently_apply_to_other_optimizers():
    import imrmlp
    pot=imrmlp.Potential.from_settings({'pottype':'lj612','ntypes':1,'0-0':'1 1','cut:0-0':4})
    with pytest.raises(ValueError,match='timestep'):
        imrmlp.Relaxation(pot,driver='fire',timestep=.2)
