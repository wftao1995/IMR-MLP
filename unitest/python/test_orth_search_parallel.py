"""The parallel early-stop search must return one valid integer supercell."""
import re

import numpy as np
import pytest

from test_cli_error_paths import run_command


@pytest.mark.parametrize("threads", [1, 2, 4])
@pytest.mark.parametrize("repeat", range(3))
def test_orth_search_one_parallel_result_is_a_valid_supercell(tmp_path, threads, repeat):
    cell = np.array([[2., 0., 0.], [1., 2., 0.], [0., 0., 3.]])
    positions = {"H": np.array([.5, .6, .7]), "He": np.array([1.2, 1.3, 1.4])}
    (tmp_path / "index.map").write_text("H 1\nHe 1\n")
    (tmp_path / "input.xyz").write_text(
        '2\nLattice="2 0 0 1 2 0 0 0 3" pbc="T T T" '
        'Properties=species:S:1:pos:R:3\nH .5 .6 .7\nHe 1.2 1.3 1.4\n')
    result = run_command("imrmlp", [
        "mk_orth_search", "--input", "input.xyz", "--input-format", "extxyz",
        "--output", "output.xyz", "--output-format", "extxyz", "--one", "true",
        "--max-expand", "2", "--tol", "1e-8", "--num_threads", str(threads),
    ], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = (tmp_path / "output.xyz").read_text().splitlines()
    count = int(lines[0])
    assert len(lines) == count + 2  # Exactly one complete output frame.
    output_cell = np.fromstring(re.search(r'Lattice="([^"]+)"', lines[1]).group(1),
                                sep=" ").reshape(3, 3)
    transform = output_cell @ np.linalg.inv(cell)
    np.testing.assert_allclose(transform, np.rint(transform), atol=1e-10, rtol=0)
    copies = round(np.linalg.det(transform))
    assert 0 < copies <= 2
    assert count == copies * len(positions)
    gram = output_cell @ output_cell.T
    np.testing.assert_allclose(gram - np.diag(np.diag(gram)), 0, atol=1e-10, rtol=0)
    for species, source in positions.items():
        selected = [list(map(float, row.split()[1:4])) for row in lines[2:]
                    if row.split()[0] == species]
        assert len(selected) == copies
        translations = (np.array(selected) - source) @ np.linalg.inv(cell)
        np.testing.assert_allclose(translations, np.rint(translations), atol=1e-10, rtol=0)
        # Distinct source images must remain distinct in the new cell.
        fractional = np.mod(np.array(selected) @ np.linalg.inv(output_cell), 1.)
        assert len(np.unique(np.round(fractional, 10), axis=0)) == copies
