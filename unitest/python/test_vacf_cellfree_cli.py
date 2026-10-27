"""CP2KXYZ geometry-only trajectories supply finite derived VACF velocities."""
import math
import pytest
from test_cli_error_paths import run_command


@pytest.mark.parametrize('algorithm', [1, 2])
def test_vacf_cp2kxyz_zero_cell_ballistic_signal(tmp_path, algorithm):
    (tmp_path / 'index.map').write_text('H 0\n')
    text = ''.join(f'1\n i = {f}, time = {f}, E = 0\n'
                   f'H {1 + .2*f:.17g} {2 + .3*f:.17g} {3 + .4*f:.17g}\n' for f in range(8))
    (tmp_path / 'input.xyz').write_text(text)
    (tmp_path / 'settings.ini').write_text(
        f'vacf:algo {algorithm}\nvacf:timestep 1\nvacf:total true\n'
        'vacf:vdos false\nvacf:normalize false\nvacf:types false\n')
    result = run_command('imrmlp', ['calc_prop', '--input', 'input.xyz', '--input-format',
                         'cp2kxyz', '--vacf', '--settings', 'settings.ini', '--num_threads', '1'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [list(map(float, line.split())) for line in (tmp_path / 'VACF_total.dat').read_text().splitlines()
            if line.strip() and not line.startswith('#')]
    assert len(rows) == 6
    for lag, row in enumerate(rows):
        assert all(math.isfinite(v) for v in row)
        assert row == pytest.approx([lag, .04, .09, .16, .29 / 3], abs=1e-11)
    assert (tmp_path / 'input.xyz').read_text() == text
