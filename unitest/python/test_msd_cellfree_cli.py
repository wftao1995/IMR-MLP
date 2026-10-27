"""Ordinary CP2KXYZ ballistic molecules need no periodic cell for MSD."""
import math
import pytest
from test_cli_error_paths import run_command


@pytest.mark.parametrize('fft', [False, True])
def test_msd_cp2kxyz_zero_cell_ballistic_signal(tmp_path, fft):
    (tmp_path / 'index.map').write_text('H 0\n')
    text = ''.join(f'1\n i = {f}, time = {f}, E = 0\n'
                   f'H {1 + .2*f:.17g} {2 + .3*f:.17g} {3 + .4*f:.17g}\n' for f in range(8))
    (tmp_path / 'input.xyz').write_text(text)
    (tmp_path / 'settings.ini').write_text(
        f'msd:fft {str(fft).lower()}\nmsd:timestep 1\nmsd:total true\n'
        'msd:drift_cm false\nmsd:der false\nmsd:do_lfit false\n')
    result = run_command('imrmlp', ['calc_prop', '--input', 'input.xyz', '--input-format',
                         'cp2kxyz', '--msd', '--settings', 'settings.ini', '--num_threads', '1'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [list(map(float, line.split())) for line in (tmp_path / 'MSD_TOTAL.dat').read_text().splitlines()
            if line.strip() and not line.startswith('#')]
    assert len(rows) == 8
    for lag, row in enumerate(rows):
        assert all(math.isfinite(v) for v in row)
        assert row == pytest.approx([lag, .29 * lag * lag], abs=1e-10)
    assert (tmp_path / 'input.xyz').read_text() == text
