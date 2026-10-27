"""Explicit opt-in checks a matching MPI executable without racing shared output."""
import os
from pathlib import Path
import shutil
import subprocess
import pytest


@pytest.mark.skipif(os.environ.get('IMRMLP_RUN_MPI_TESTS') != '1', reason='requires matching MPI build; set IMRMLP_RUN_MPI_TESTS=1')
def test_calc_err_nonroot_does_not_open_report_paths(tmp_path):
    root=Path(__file__).resolve().parents[2]
    # Tests may also be executed from an un-applied temp candidate.
    root=Path(os.environ.get('IMRMLP_TEST_REPO',root))
    binary=root/'build/mpi/imrmlp-mpi'
    if not binary.exists() or not shutil.which('mpiexec'):
        pytest.skip('build/mpi/imrmlp-mpi and mpiexec required')
    (tmp_path/'index.map').write_text('H 0\n')
    for name,energy in [('ref',1),('pred',3)]:
        (tmp_path/f'{name}.xyz').write_text(f'1\nLattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3 energy={energy}\nH 1 1 1\n')
    for rank in (0,1): (tmp_path/f'rank-{rank}').mkdir()
    env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    env['LD_LIBRARY_PATH']=str(root/'build/mpi')+':'+env.get('LD_LIBRARY_PATH','')
    script='exec "$1" calc_err --input "$2/ref.xyz" --input-predict "$2/pred.xyz" --input-format extxyz --index "$2/index.map" --output-summary "$2/rank-$OMPI_COMM_WORLD_RANK/summary" --output-energy-diff "$2/rank-$OMPI_COMM_WORLD_RANK/energy"'
    result=subprocess.run(['mpiexec','-n','2','/bin/sh','-c',script,'calc-err-test',str(binary),str(tmp_path)],env=env,text=True,capture_output=True,timeout=40)
    assert result.returncode==0,result.stdout+result.stderr
    assert (tmp_path/'rank-0/summary').exists()
    rows=(tmp_path/'rank-0/energy').read_text().splitlines()
    assert list(map(float,rows[1].split())) == [1.,3.]
    assert list((tmp_path/'rank-1').iterdir()) == []
