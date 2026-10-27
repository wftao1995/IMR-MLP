"""calc_ferr accepts prediction datasets outside the working-directory default."""
import math
from pathlib import Path
import pytest
from test_cli_error_paths import run_command


def write_dataset(path, energies):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for energy in energies:
        label = '' if energy is None else f' energy={energy:.17g}'
        frames.append('2\nLattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3' + label +
                      '\nH 1 1 1\nH 3 1 1\n')
    path.write_text(''.join(frames))


def expected_per_atom(reference, prediction, temperature):
    differences = [a - b for a, b in zip(reference, prediction)]
    minimum = min(differences)
    kt = temperature * 8.61733326e-5
    return (minimum - kt * math.log(sum(math.exp(-(delta - minimum) / kt)
                                       for delta in differences) / len(differences))) / 2.


@pytest.mark.parametrize('absolute,inline', [(False, False), (True, True)])
def test_calc_ferr_selects_explicit_prediction_path_with_independent_energy_oracle(tmp_path, absolute, inline):
    (tmp_path / 'index.map').write_text('H 0\n')
    reference, prediction = [2.1, 3.5, 6.3], [1.8, 3.4, 5.9]
    source = tmp_path / 'reference' / 'data.xyz'
    chosen = tmp_path / 'predictions' / 'model outputs.xyz'
    write_dataset(source, reference)
    write_dataset(chosen, prediction)
    # A different healthy default proves the new option actually selects a file.
    write_dataset(tmp_path / 'mlp.cfg', [0., 0., 0.])
    prediction_path = chosen if absolute else chosen.relative_to(tmp_path)
    option = [f'--prediction={prediction_path}'] if inline else ['--prediction', prediction_path]
    result = run_command('imrmlp', ['calc_ferr', '--input', source, '--input-format', 'extxyz',
                                   '--temperature', '425', *option], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    value = float(result.stdout.splitlines()[-1])
    assert value == pytest.approx(expected_per_atom(reference, prediction, 425.), abs=1e-12)
    assert value != pytest.approx(expected_per_atom(reference, [0., 0., 0.], 425.), abs=1e-6)


def test_calc_ferr_omitted_prediction_keeps_mlp_cfg_default(tmp_path):
    (tmp_path / 'index.map').write_text('H 0\n')
    reference, prediction = [2., 5.], [1., 4.]
    write_dataset(tmp_path / 'reference.xyz', reference)
    write_dataset(tmp_path / 'mlp.cfg', prediction)
    result = run_command('imrmlp', ['calc_ferr', '--input', 'reference.xyz',
                                   '--input-format', 'extxyz'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert float(result.stdout.splitlines()[-1]) == pytest.approx(.5, abs=1e-12)


@pytest.mark.parametrize('missing_energy', ['reference', 'prediction'])
def test_calc_ferr_explicit_prediction_keeps_required_energy_checks(tmp_path, missing_energy):
    (tmp_path / 'index.map').write_text('H 0\n')
    reference = [2., None] if missing_energy == 'reference' else [2., 5.]
    prediction = [1., None] if missing_energy == 'prediction' else [1., 4.]
    write_dataset(tmp_path / 'reference.xyz', reference)
    write_dataset(tmp_path / 'predictions' / 'model.xyz', prediction)
    result = run_command('imrmlp', ['calc_ferr', '--input', 'reference.xyz',
                                   '--prediction', 'predictions/model.xyz', '--input-format', 'extxyz'], tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'energy is required on both sides at frame 2' in result.stderr


def test_calc_ferr_explicit_missing_prediction_fails(tmp_path):
    (tmp_path / 'index.map').write_text('H 0\n')
    write_dataset(tmp_path / 'reference.xyz', [2., 5.])
    result = run_command('imrmlp', ['calc_ferr', '--input', 'reference.xyz',
                                   '--prediction', 'missing/model.xyz', '--input-format', 'extxyz'], tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'prediction dataset must match reference frame count' in result.stderr


def test_calc_ferr_help_documents_prediction_and_default_without_opening_data(tmp_path):
    result = run_command('imrmlp', ['calc_ferr', '--help'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '--prediction' in result.stdout
    assert 'default: mlp.cfg' in result.stdout
    assert 'Input format for both datasets' in result.stdout
