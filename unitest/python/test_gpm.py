import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import struct

import numpy as np
import pytest
from imrmlp import Atoms, ModelBundle, Potential


def settings():
    return {"pottype": "lj612", "typemap": "", "ntypes": "1",
            "0-0": "0.13 1.7", "cut:0-0": "5", "ele:0": "0.07"}


def atoms():
    return Atoms(positions=[[0, 0, 0], [2.1, 0.1, 0.2]],
                 cell=np.eye(3) * 20, numbers=[0, 0], pbc=[False] * 3)


def test_gpm_fresh_process_and_explicit_type(tmp_path):
    bundle = ModelBundle.from_settings(settings())
    path = tmp_path / "independent.gpm"
    bundle.save(path)
    ModelBundle.validate(path)
    raw = path.read_bytes()
    magic, major, minor, flags, size, offset, digest = struct.unpack('<8sHHIQQ32s', raw[:64])
    assert (magic, major, minor, flags) == (b'GPMLMODL', 1, 0, 0)
    assert offset == (64 + size + 63) // 64 * 64
    assert hashlib.sha256(raw[64:64 + size]).digest() == digest
    for block in json.loads(raw[64:64 + size])['blocks']:
        assert all(isinstance(block[key], str) for key in ('offset', 'stored_size', 'raw_size'))
        assert int(block['offset']) % 64 == 0
        start = offset + int(block['offset'])
        assert block['codec'] == 'none'
        assert hashlib.sha256(raw[start:start + int(block['raw_size'])]).hexdigest() == block['sha256']
    expected = atoms()
    bundle.potential.calculate(expected)
    import imrmlp
    root = Path(imrmlp.__file__).resolve().parent.parent
    from imrmlp import _imrmlp
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), str(Path(_imrmlp.__file__).parent)]))
    code = """
import json, sys
import numpy as np
from imrmlp import Potential, Atoms
p=Potential(sys.argv[1])
a=Atoms(positions=[[0,0,0],[2.1,.1,.2]],cell=np.eye(3)*20,numbers=[0,0],pbc=[False]*3)
p.calculate(a)
print('GPM_RESULT='+json.dumps([a.energy, a.forces.tolist()]))
"""
    run = subprocess.run([sys.executable, "-c", code, str(path)], cwd=tmp_path,
                         env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    actual = json.loads(next(line.split("=", 1)[1] for line in run.stdout.splitlines()
                             if line.startswith("GPM_RESULT=")))
    assert actual[0] == pytest.approx(expected.energy, abs=1e-12)
    np.testing.assert_allclose(actual[1], expected.forces, atol=1e-12)
    with pytest.raises(RuntimeError, match="type conflicts"):
        Potential(path, "mtp")
    with pytest.raises(RuntimeError, match="configuration conflicts"):
        Potential.from_settings({"potfile": str(path), "ntypes": "2"})
    assert "active_learning" not in ModelBundle.inspect(path)
    with pytest.raises(RuntimeError):
        ModelBundle.load(path).create_al()


def test_gpm_default_save_preserves_and_corruption_is_detected(tmp_path):
    path = tmp_path / "model.gpm"
    Potential.from_settings(settings()).save(path)
    p = Potential(path)
    copy = tmp_path / "copy.gpm"
    p.save(copy)
    ModelBundle.validate(copy)
    data = bytearray(copy.read_bytes())
    data[-1] ^= 1
    copy.write_bytes(data)
    with pytest.raises(RuntimeError, match="checksum"):
        ModelBundle.validate(copy)
    with pytest.raises(RuntimeError, match="requires AL"):
        ModelBundle.load(path).save(copy, profile="resume")


def test_gpm_explicit_al_resume_and_selection(tmp_path):
    from imrmlp.al import ActiveSelector
    source = Path(__file__).resolve().parents[2] / 'unitest/test-data/potential.mtp'
    selector = ActiveSelector(potfile=source, dtype='pot_efs', select_mode='full')
    a = atoms()
    selector.add_candidates([a])
    selector.select()
    path = tmp_path / 'resume.gpm'
    selector.save(path)
    bundle = ModelBundle.load(path)
    session = bundle.create_al(threshold=-1, interval=2)
    expected = selector.grade(a)
    assert session.evaluate(a)['uncertainty'] == pytest.approx(expected)
    bundle.potential.calculate(a)
    assert session.pool == []
    session.check(a)
    session.save(tmp_path / 'continued.gpm')
    resumed = ModelBundle.load(tmp_path / 'continued.gpm').create_al()
    session.check(a)
    resumed.check(a)
    assert len(session.pool) == len(resumed.pool) == 1
    assert resumed.evaluate(a)['uncertainty'] == pytest.approx(expected)
    assert ActiveSelector.load(path).grade(a) == pytest.approx(expected)
    resumed.selector.add_candidates([a])
    resumed.selector.select()
    resumed.save(tmp_path / 'reselected.gpm')
    ModelBundle.validate(tmp_path / 'reselected.gpm')
