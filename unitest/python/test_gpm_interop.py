# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Cross-library GPM interoperability; set both *_GPM_INTEROP_TEST variables.

Each variable names the matching build's ut_gpm.test. The test launches separate
processes so each loads its own shared library. Run after building both tests.
"""

import os
from pathlib import Path
import subprocess

import pytest


def _executable(variable):
    value = os.environ.get(variable)
    if not value:
        pytest.skip('Set GPML_GPM_INTEROP_TEST and IMRMLP_GPM_INTEROP_TEST')
    path = Path(value).resolve()
    assert path.is_file(), path
    return path


def _worker(executable, tmp_path, mode, model, result, basis, output=None):
    env = os.environ.copy()
    # A test executable lives in build/<mode>/unitest, next to its library's directory.
    env['LD_LIBRARY_PATH'] = str(executable.parent.parent) + ':' + env.get('LD_LIBRARY_PATH', '')
    env.update(
        GPM_INTEROP_MODE=mode,
        GPM_INTEROP_MODEL=str(model),
        GPM_INTEROP_RESULT=str(result),
        GPM_INTEROP_BASIS=basis,
    )
    if output is not None:
        env['GPM_INTEROP_OUTPUT'] = str(output)
    else:
        env.pop('GPM_INTEROP_OUTPUT', None)
    proc = subprocess.run(
        [str(executable), 'GPM interoperability worker', '--reporter', 'compact'],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return [float(x) for x in result.read_text().splitlines()]


@pytest.mark.parametrize('basis', ['RBPolyB', 'RBPolyBCos'])
@pytest.mark.parametrize('writer_name', ['GPML', 'IMRMLP'])
@pytest.mark.parametrize('write_mode', ['write', 'legacy'])
def test_gpm_polynomial_cross_library_roundtrip(tmp_path, basis, writer_name, write_mode):
    gpml = _executable('GPML_GPM_INTEROP_TEST')
    imrmlp = _executable('IMRMLP_GPM_INTEROP_TEST')
    writer, reader = (gpml, imrmlp) if writer_name == 'GPML' else (imrmlp, gpml)
    source, copied = tmp_path / 'source.gpm', tmp_path / 'copied.gpm'
    expected = _worker(writer, tmp_path, write_mode, source, tmp_path / 'original.txt', basis)
    received = _worker(reader, tmp_path, 'read', source, tmp_path / 'received.txt', basis, copied)
    returned = _worker(writer, tmp_path, 'read', copied, tmp_path / 'returned.txt', basis)
    # Energy, all force components and all nine virial components.
    assert len(expected) == 19
    assert received == pytest.approx(expected, rel=1e-10, abs=1e-10)
    assert returned == pytest.approx(expected, rel=1e-10, abs=1e-10)
