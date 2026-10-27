"""The existing 3D MSD finite-size correction requires three periodic axes."""
import math
import re
import pytest
from test_cli_error_paths import run_command


@pytest.mark.parametrize('mode', [0, 1, 2])
@pytest.mark.parametrize('boundary', ['TTT', 'FFF', 'TTF', 'later_TTF'])
def test_msd_finite_size_correction_periodicity_and_full_pbc_control(tmp_path, mode, boundary):
    (tmp_path / 'index.map').write_text('H 0\n')
    records = []
    for f in range(4):
        pbc = 'TTF' if boundary == 'later_TTF' and f == 3 else (
            'TTT' if boundary == 'later_TTF' else boundary)
        records.append('1\nLattice="10 0 0 0 10 0 0 0 10" '
                       f'pbc="{" ".join(pbc)}" Properties=species:S:1:pos:R:3\n'
                       f'H {math.sqrt(6 * f):.17g} 0 0\n')
    (tmp_path / 'input.xyz').write_text(''.join(records))
    (tmp_path / 'settings.ini').write_text(
        'msd:fft false\nmsd:tacavg false\nmsd:drift_cm false\n'
        'msd:do_lfit true\nmsd:timestep 1\nmsd:fit_start 1\nmsd:fit_end 3\n'
        f'msd:fsc_mode {mode}\nmsd:temperature 300\nmsd:viscosity 1\n')
    result = run_command('imrmlp', ['calc_prop', '--input', 'input.xyz', '--input-format',
                         'extxyz', '--msd', '--settings', 'settings.ini', '--num_threads', '1'], tmp_path)
    if mode and boundary != 'TTT':
        assert result.returncode != 0, result.stdout + result.stderr
        assert 'fully periodic boundaries in every frame' in result.stdout + result.stderr
        assert 'MSD FSC:' not in result.stdout + result.stderr
        assert 'Complete!' not in result.stdout
        assert not (tmp_path / 'MSD_TOTAL.dat').exists()
        return
    assert result.returncode == 0, result.stdout + result.stderr
    text = (tmp_path / 'MSD_TOTAL.dat').read_text()
    correction = (8.617333262e-5 * 300 * 2.837298 /
                  (6 * math.pi * 6.241509074e-3 * 10)) if mode else 0.
    for key, expected in [('slope', 6.), ('D_PBC', 1.),
                          ('D_correction', correction), ('D_infinite', 1. + correction)]:
        value = float(re.search(rf'\b{key} (\S+)', text).group(1))
        assert value == pytest.approx(expected, abs=1e-10)
