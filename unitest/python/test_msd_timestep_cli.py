"""calc_prop must retain the precision of the selected MSD time spacing."""
import math
import pytest
from test_cli_error_paths import run_command


def run_msd(tmp_path, timestep, skip, fft):
    (tmp_path / 'index.map').write_text('H 0\n')
    text = ''.join(f'1\n i = {frame}, time = {frame}, E = 0\n'
                   f'H {1 + .2*frame:.17g} {2 + .3*frame:.17g} {3 + .4*frame:.17g}\n'
                   for frame in range(8))
    (tmp_path / 'input.xyz').write_text(text)
    (tmp_path / 'settings.ini').write_text(
        f'msd:timestep {timestep}\nmsd:fft {str(fft).lower()}\n'
        'msd:total true\nmsd:drift_cm false\nmsd:der false\nmsd:do_lfit false\n')
    result = run_command('imrmlp', ['calc_prop', '--input', 'input.xyz',
                         '--input-format', 'cp2kxyz', '--msd', '--settings', 'settings.ini',
                         '--skip-every', str(skip), '--num_threads', '1'], tmp_path)
    assert (tmp_path / 'input.xyz').read_text() == text
    return result


@pytest.mark.parametrize('timestep', [1e-7, 1.23456789])
@pytest.mark.parametrize('skip', [1, 2])
@pytest.mark.parametrize('fft', [False, True])
def test_msd_keeps_raw_timestep_precision_after_frame_selection(tmp_path, timestep, skip, fft):
    result = run_msd(tmp_path, timestep, skip, fft)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [list(map(float, line.split())) for line in
            (tmp_path / 'MSD_TOTAL.dat').read_text().splitlines()
            if line.strip() and not line.startswith('#')]
    assert len(rows) == 8 // skip
    for lag, row in enumerate(rows):
        assert all(math.isfinite(value) for value in row)
        assert row[0] == pytest.approx(lag * timestep * skip, rel=2e-14, abs=1e-20)
        # Independent ballistic result for the retained frame indices.
        assert row[1] == pytest.approx(.29 * (lag * skip)**2, rel=1e-11, abs=1e-11)


@pytest.mark.parametrize('timestep', ['0', '-1', 'nan', 'inf', '-inf'])
@pytest.mark.parametrize('fft', [False, True])
def test_msd_invalid_timestep_still_fails_before_result_output(tmp_path, timestep, fft):
    result = run_msd(tmp_path, timestep, 2, fft)
    assert result.returncode > 0, result.stdout + result.stderr
    assert 'MSD timestep must be finite and positive' in result.stdout + result.stderr
    assert not (tmp_path / 'MSD_TOTAL.dat').exists()
