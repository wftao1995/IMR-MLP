# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Native objects release owned resources and preserve borrowed dependencies."""
import ctypes
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip('imrmlp._imrmlp')
from imrmlp._imrmlp import _GPIO


def test_read_map_failure_is_catchable(tmp_path):
    io = _GPIO()
    missing = tmp_path / 'missing.map'
    with pytest.raises(RuntimeError, match='missing.map'):
        io.read_map(str(missing))
    empty = tmp_path / 'empty.map'
    empty.write_text('')
    with pytest.raises(RuntimeError, match='empty.map'):
        io.read_map(str(empty))
    valid = tmp_path / 'valid.map'
    valid.write_text('H\t-2.0\r\nHe\t-3.0\r\n')
    io.read_map(str(valid))


def run_python(code, tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root) + ':' + env.get('PYTHONPATH', '')
    env['OMP_NUM_THREADS'] = '1'
    result = subprocess.run([sys.executable, '-c', code, str(root)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_relaxation_cleanup_reinitialization_and_interpreter_exit(tmp_path):
    run_python('''
import gc, sys, weakref
import numpy as np
from imrmlp import Atoms, Potential
from imrmlp._imrmlp import _Relaxation
potential = Potential(sys.argv[1] + '/unitest/test-data/potential.mtp')
atoms = Atoms(positions=[[1., 1., 1.], [3.5, 1., 1.]], cell=np.eye(3)*12)
atoms.chemmap = [0, 0]
for i in range(20):
    driver = _Relaxation()
    driver.selective_dynamics = True
    driver.max_iter = 1
    for repeat in range(2):
        driver.init(potential.inner, atoms._impl)
        driver.run()
        assert np.isfinite(driver.energy)
        driver.cleanup()
        driver.cleanup()
    del driver
    gc.collect()
# The low-level driver must retain the borrowed potential and its wrapper.
driver = _Relaxation()
driver.init(potential.inner, atoms._impl)
driver.max_iter = 1
wrapper = weakref.ref(potential._wrapper)
del potential
gc.collect()
assert wrapper() is not None
driver.run()
del driver
gc.collect()
assert wrapper() is None
# Leave an initialized object for interpreter shutdown as well.
potential = Potential(sys.argv[1] + '/unitest/test-data/potential.mtp')
driver = _Relaxation()
driver.init(potential.inner, atoms._impl)
''', tmp_path)


@pytest.mark.skipif(not Path('/proc/self/status').exists() or
                    not hasattr(ctypes.CDLL(None), 'malloc_trim'),
                    reason='RSS regression requires Linux and glibc')
def test_discarded_relaxation_objects_do_not_accumulate(tmp_path):
    run_python('''
import ctypes, gc
from pathlib import Path
from imrmlp._imrmlp import _Relaxation
libc = ctypes.CDLL(None)
def rss():
    gc.collect()
    libc.malloc_trim(0)
    return int(next(line for line in Path('/proc/self/status').read_text().splitlines()
                    if line.startswith('VmRSS:')).split()[1])
for _ in range(30):
    driver = _Relaxation()
    del driver
before = rss()
for _ in range(600):
    driver = _Relaxation()
    driver.cleanup()
    del driver
after = rss()
# The leaking holder accumulated about 50 MiB over these 600 iterations.
# Allow allocator/runtime noise while detecting that sustained growth.
assert after - before < 16 * 1024, (before, after)
''', tmp_path)


def test_bmatrix_results_survive_cleanup_and_failed_calls(tmp_path):
    run_python('''
import gc, sys
import numpy as np
from imrmlp import Atoms
from imrmlp.al import build_bmatrix
from imrmlp._imrmlp import _build_bmatrix
path = sys.argv[1] + '/unitest/test-data/potential.mtp'
atoms = Atoms(positions=[[1., 1., 1.], [3.5, 1., 1.]], cell=np.eye(3)*12)
atoms.chemmap = [0, 0]
for dtype in ('mtm', 'pot_efs'):
    expected = build_bmatrix([atoms], path, dtype=dtype).copy()
    assert expected.size > 0 and np.isfinite(expected).all()
    for _ in range(3):
        result = build_bmatrix([atoms], path, dtype=dtype)
        gc.collect()
        np.testing.assert_allclose(result, expected)
        try:
            _build_bmatrix([atoms._impl, None], path, dtype=dtype)
        except TypeError:
            pass
        else:
            raise AssertionError('invalid structure must raise')
        np.testing.assert_allclose(result, expected)
# Keep the copied array alive through interpreter shutdown.
assert result.flags.owndata
''', tmp_path)


@pytest.mark.skipif(not Path('/proc/self/status').exists() or
                    not hasattr(ctypes.CDLL(None), 'malloc_trim'),
                    reason='RSS regression requires Linux and glibc')
def test_bmatrix_temporary_storage_does_not_accumulate(tmp_path):
    run_python('''
import ctypes, gc, sys
from pathlib import Path
import numpy as np
from imrmlp import Atoms
from imrmlp.al import build_bmatrix
path = sys.argv[1] + '/unitest/test-data/potential.mtp'
atoms = Atoms(positions=[[1., 1., 1.], [3.5, 1., 1.]], cell=np.eye(3)*12)
atoms.chemmap = [0, 0]
libc = ctypes.CDLL(None)
def rss():
    gc.collect()
    libc.malloc_trim(0)
    return int(next(line for line in Path('/proc/self/status').read_text().splitlines()
                    if line.startswith('VmRSS:')).split()[1])
for _ in range(3):
    build_bmatrix([atoms]*30, path)
before = rss()
for _ in range(20):
    build_bmatrix([atoms]*30, path)
after = rss()
# The leaked matrix and builder accumulated over 20 MiB in this workload.
assert after - before < 8 * 1024, (before, after)
''', tmp_path)


@pytest.mark.skipif(not Path('/proc/self/status').exists() or
                    not hasattr(ctypes.CDLL(None), 'malloc_trim'),
                    reason='RSS regression requires Linux and glibc')
@pytest.mark.parametrize('kind', ['potential', 'selector'])
def test_owned_potentials_and_selectors_do_not_accumulate(tmp_path, kind):
    run_python('''
import ctypes, gc, sys
from pathlib import Path
from imrmlp._imrmlp import _PotentialWrapper, _ActiveSelector
path = sys.argv[1] + '/unitest/test-data/potential.mtp'
factory = ''' + ('_PotentialWrapper' if kind == 'potential' else '_ActiveSelector') + '''
libc = ctypes.CDLL(None)
def rss():
    gc.collect()
    libc.malloc_trim(0)
    return int(next(line for line in Path('/proc/self/status').read_text().splitlines()
                    if line.startswith('VmRSS:')).split()[1])
for _ in range(20):
    value = factory(path)
    del value
before = rss()
for _ in range(150):
    value = factory(path)
    del value
after = rss()
# Previously each 100 objects leaked roughly 24 MiB (selector) or 40 MiB (potential).
assert after - before < 12 * 1024, (before, after)
''', tmp_path)
