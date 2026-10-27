# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Neural MTP binary loading through the public potential factory."""
from pathlib import Path
import shutil
import struct

import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
from imrmlp import Atoms, Potential

FIXTURE = Path(__file__).resolve().parents[2] / 'tests/test-nnmtp/_Trained_Potential_'


def write_model(tmp_path, layers, inputs=None):
    shutil.copyfile(str(FIXTURE) + '.mtp', tmp_path / 'model.mtp')
    (tmp_path / 'model.sc').write_text('0.5 -0.75\n')
    if inputs is None:
        _, inputs = struct.unpack_from('=ii', Path(str(FIXTURE) + '.nn').read_bytes())
    hidden1, hidden2 = 2, 3
    header = struct.pack('=4i', -0x4e4e, 1, layers, 2)  # version 1, sigmoid
    if layers == 1:
        dimensions = struct.pack('=2i', hidden1, inputs)
        values = [0.] * (inputs * hidden1) + [1.] * hidden1 + [0.] * hidden1
    else:
        dimensions = struct.pack('=3i', hidden1, hidden2, inputs)
        values = [0.] * (inputs * hidden1 + hidden1 + hidden1 * hidden2 + hidden2)
        values += [1.] * hidden2
    values += [0.] * inputs + [1.] * inputs
    (tmp_path / 'model.nn').write_bytes(header + dimensions + struct.pack(f'={len(values)}d', *values))


@pytest.mark.parametrize('layers', [1, 2])
@pytest.mark.parametrize('omp', [False, True])
def test_factory_reads_network_architecture_and_activation(tmp_path, monkeypatch, layers, omp):
    monkeypatch.chdir(tmp_path)
    write_model(tmp_path, layers)
    pot = Potential('model', pottype='mtpnn1')
    atom = Atoms(numbers=[1, 2], elements=['H', 'He'], positions=[[0, 0, 0], [2, 0, 0]],
                 cell=np.eye(3)*20, pbc=False)
    pot.calculate(atom, use_omp=omp)
    assert atom.energy == pytest.approx(1.75 if layers == 1 else 2.75)
    np.testing.assert_allclose(atom.forces, 0, atol=1e-12)
    atom.forces = np.ones((2, 3))
    atom.stress = np.ones((3, 3))
    pot.calculate(atom, properties=['energy'])
    np.testing.assert_array_equal(atom.forces, np.ones((2, 3)))
    assert atom.stress is None


@pytest.mark.parametrize('layers', [1, 2])
@pytest.mark.parametrize('problem', ['truncated', 'mismatch'])
def test_factory_rejects_incomplete_or_incompatible_networks(tmp_path, monkeypatch, layers, problem):
    monkeypatch.chdir(tmp_path)
    write_model(tmp_path, layers, inputs=1 if problem == 'mismatch' else None)
    if problem == 'truncated':
        path = tmp_path / 'model.nn'
        path.write_bytes(path.read_bytes()[:28])
    with pytest.raises(RuntimeError, match=r'model\.nn'):
        Potential('model', pottype='mtpnn1')
