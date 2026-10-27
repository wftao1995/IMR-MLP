"""CUR CLI must reject zero features before indexing per-structure scores."""
import numpy as np
import pytest

from test_cli_error_paths import run_command


@pytest.fixture
def pool(tmp_path):
    (tmp_path / "index.map").write_text("H 0\n")
    frames = []
    for distance in (1.5, 2., 2.8):
        frames.append(
            '2\nLattice="20 0 0 0 20 0 0 0 20" pbc="F F F" '
            'Properties=species:S:1:pos:R:3\n'
            f'H 1 1 1\nH {1 + distance} 1 1\n')
    (tmp_path / "pool.xyz").write_text("".join(frames))
    return ["diverse_select", "--method", "cur", "--pool", "pool.xyz",
            "--format", "extxyz", "--index", "index.map", "--output", "selected.xyz",
            "--score-file", "scores.dat", "--num_threads", "1"]


def test_cur_cli_zero_parameter_model_exits_with_a_diagnostic(tmp_path, pool):
    # This valid potential has no trainable parameters, hence a zero-column
    # pot_efs descriptor. The positive row request used to reach empty scores.
    (tmp_path / "penalty.pot").write_text("pot:pottype mindist\npot:cutoff 5\n")
    result = run_command("imrmlp", pool + ["--n-select", "1", "--dtype", "pot_efs",
                         "--pottype", "mindist", "--potfile", "penalty.pot"], tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "CUR requires at least one descriptor feature (got 0)" in result.stdout + result.stderr
    assert not (tmp_path / "scores.dat").exists()
    assert not (tmp_path / "selected.xyz").exists()


def test_cur_cli_accepts_more_rows_than_descriptor_features(tmp_path, pool):
    (tmp_path / "cur.ini").write_text(
        "behler:dispcount 1\nbehler:dispselect 1\nbehler:cutoff 5\n")
    result = run_command("imrmlp", pool + ["--n-select", "2", "--dtype", "behler",
                         "--settings", "cur.ini", "--cur-deterministic", "true",
                         "--compute-error", "true"], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    scores = np.loadtxt(tmp_path / "scores.dat")
    assert scores.shape == (3, 3)
    np.testing.assert_array_equal(scores[:, 0], np.arange(3))
    assert np.isfinite(scores).all()
    assert scores[:, 1].sum() == pytest.approx(1., abs=2e-6)
    assert scores[:, 2].sum() == 2
    assert (tmp_path / "selected.xyz").read_text().count("Properties=") == 2
    assert "Relative reconstruction error: 0.000000" in result.stdout
