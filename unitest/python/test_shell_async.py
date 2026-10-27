"""Explicit Shell jobs retain their inputs and outlive their calculator."""
import gc
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
from imrmlp import Atoms, Potential


def calculator(path, script):
    (path / 'index.map').write_text('H\n')
    (path / 'run.sh').write_text(script)
    return Potential.from_settings({
        'pottype': 'shellfile', 'shellfile:index': 'index.map',
        'shellfile:runscript': 'run.sh', 'shellfile:write_posfile': 'input.xyz',
        'shellfile:write_posformat': 'extxyz', 'shellfile:read_outfile': 'output.xyz',
        'shellfile:read_outformat': 'extxyz',
    })


def test_jobs_return_owned_results_after_parent_release(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    potential = calculator(tmp_path, 'cp "$1/input.xyz" "$1/output.xyz"\n')
    atom = Atoms(numbers=[1], elements=['H'], positions=[[1, 2, 3]], cell=np.eye(3) * 7)
    jobs = [potential.submit(atom) for _ in range(3)]
    del potential
    gc.collect()
    atom.positions = [[9, 9, 9]]
    assert len({job.run_directory for job in jobs}) == 3
    for job in jobs:
        first = job.result()
        assert job.ready()
        np.testing.assert_array_equal(first.positions, [[1, 2, 3]])
        first.positions = [[5, 5, 5]]
        np.testing.assert_array_equal(job.result().positions, [[1, 2, 3]])
        assert Path(job.run_directory).is_absolute()


def test_job_repeats_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    potential = calculator(tmp_path, 'exit 17\n')
    atom = Atoms(numbers=[1], elements=['H'], positions=[[1, 2, 3]], cell=np.eye(3) * 7)
    job = potential.submit(atom)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='run'):
            job.result()
    assert job.ready()
