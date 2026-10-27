# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

pytest.importorskip("imrmlp._imrmlp")
import imrmlp


def structure():
    atoms = imrmlp.Atoms(positions=[[1., 2., 3.]], cell=np.eye(3) * 10)
    atoms.chemmap = [0]
    return atoms


@pytest.mark.parametrize("format", ["", "nc", "netcdf"])
@pytest.mark.parametrize("single", [False, True])
def test_removed_format_does_not_overwrite(tmp_path, format, single):
    target = tmp_path / "target.nc"
    target.write_bytes(b"preserve")
    atoms = structure()
    with pytest.raises(RuntimeError):
        imrmlp.write(target, atoms if single else [atoms], format=format)
    assert target.read_bytes() == b"preserve"


@pytest.mark.parametrize("format", ["", "nc", "netcdf"])
def test_removed_format_read_is_rejected(tmp_path, format):
    target = tmp_path / "target.nc"
    target.write_bytes(b"unsupported input")
    with pytest.raises(RuntimeError):
        imrmlp.read(target, format=format)
    with pytest.raises(RuntimeError):
        imrmlp.read_one(target, format=format)


@pytest.mark.parametrize("suffix", ["cfg", "xyz"])
def test_supported_trajectory_roundtrip(tmp_path, suffix):
    target = tmp_path / f"trajectory.{suffix}"
    atoms = structure()
    imrmlp.write(target, [atoms, atoms])
    frames = imrmlp.read(target)
    assert len(frames) == 2
    for frame in frames:
        np.testing.assert_allclose(frame.positions, atoms.positions)
        np.testing.assert_allclose(frame.cell, atoms.cell)


@pytest.mark.parametrize("single", [False, True])
def test_unsupported_compression_does_not_overwrite(tmp_path, single):
    target = tmp_path / "target.xyz.zst"
    target.write_bytes(b"preserve")
    atoms = structure()
    with pytest.raises(RuntimeError):
        imrmlp.write(target, atoms if single else [atoms])
    assert target.read_bytes() == b"preserve"


@pytest.mark.parametrize("suffix", ["xyz.zst", "xyz"])
def test_unsupported_compression_read_is_rejected(tmp_path, suffix):
    target = tmp_path / f"target.{suffix}"
    target.write_bytes(b"\x28\xb5\x2f\xfd")
    with pytest.raises(RuntimeError):
        imrmlp.read(target)
    with pytest.raises(RuntimeError):
        imrmlp.read_one(target)


def test_gzip_trajectory_roundtrip(tmp_path):
    target = tmp_path / "trajectory.xyz.gz"
    atoms = structure()
    imrmlp.write(target, [atoms, atoms])
    assert target.read_bytes().startswith(b"\x1f\x8b")
    frames = imrmlp.read(target)
    assert len(frames) == 2
    for frame in frames:
        np.testing.assert_allclose(frame.positions, atoms.positions)
        np.testing.assert_allclose(frame.cell, atoms.cell)
