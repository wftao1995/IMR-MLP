"""RDF rejects undefined normalization volumes without restricting other analyses."""
import math
import pytest
from test_cli_error_paths import run_command


def frame(cell):
    return ('2\nLattice="' + cell + '" pbc="F F F" '
            'Properties=species:S:1:pos:R:3\nH 0 0 0\nH 1 0 0\n')


def run_rdf(tmp_path, text, average=False):
    (tmp_path / 'index.map').write_text('H 0\nHe 1\n')
    (tmp_path / 'input.xyz').write_text(text)
    args = ['calc_prop', '--input', 'input.xyz', '--input-format', 'extxyz',
            '--rdf', '--rdf-kd', '--rdf-no-pbc', '--ref-rlimit', '2', '--cutoff', '2']
    if average:
        args += ['--rdf-rho-all']
    return run_command('imrmlp', args, tmp_path)


@pytest.mark.parametrize('cell', ['0 0 0 0 0 0 0 0 0', '-2 0 0 0 2 0 0 0 2'])
@pytest.mark.parametrize('average', [False, True])
def test_rdf_rejects_invalid_normalization_volume(tmp_path, cell, average):
    result = run_rdf(tmp_path, frame(cell), average)
    assert result.returncode > 0, result.stdout + result.stderr
    assert 'RDF normalization requires a finite positive cell volume' in result.stdout + result.stderr
    assert not (tmp_path / 'gr_total.txt').exists()


def test_rdf_checks_only_the_volume_selected_by_density_policy(tmp_path):
    text = frame('2 0 0 0 2 0 0 0 2') + frame('0 0 0 0 0 0 0 0 0')
    result = run_rdf(tmp_path, text)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [list(map(float, row.split())) for row in
            (tmp_path / 'gr_total.txt').read_text().splitlines()[1:]]
    peak = next(row for row in rows if row[0] == 1.)
    expected = 1. / (.25 * (4. * math.pi / 3.) * (1.05**3 - .95**3))
    assert peak[1] == pytest.approx(expected, rel=1e-5)
    assert peak[2] == 1.
    (tmp_path / 'gr_total.txt').unlink()
    result = run_rdf(tmp_path, text, average=True)
    assert result.returncode > 0, result.stdout + result.stderr
    assert 'finite positive cell volume' in result.stdout + result.stderr
    assert not (tmp_path / 'gr_total.txt').exists()


def test_nonrdf_output_accepts_cellfree_structures(tmp_path):
    (tmp_path / 'index.map').write_text('H 0\n')
    (tmp_path / 'input.xyz').write_text(frame('0 0 0 0 0 0 0 0 0'))
    result = run_command('imrmlp', ['calc_prop', '--input', 'input.xyz',
                         '--input-format', 'extxyz', '--output-xyz', 'output.xyz'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'output.xyz').exists()
