"""Keep chemistry tags aligned with reordered or expanded model coefficients."""
import json

import pytest

from test_cli_error_paths import ROOT, run_command


@pytest.fixture
def tagged_model(tmp_path):
    data = (ROOT / 'unitest/test-data/potential.mtp').read_bytes()
    end = data.index(b'\n', data.index(b'moment_coeffs ='))
    text = data[:end + 1].decode()
    fields = dict(schema='gpml.mtp.metadata', version=1, type='MTP', model='linear',
                  elements=['Al', 'Ni'], custom='keep-me')
    text = '\n'.join('potential_tag = ' + json.dumps(fields) if line.startswith('potential_tag')
                     else line for line in text.splitlines()) + '\n'
    (tmp_path / 'input.mtp').write_text(text)
    return ['adjustmtp', '--input', 'input.mtp', '--output', 'output.mtp']


def metadata(tmp_path):
    lines = (tmp_path / 'output.mtp').read_text().splitlines()
    return json.loads(next(line.split('=', 1)[1] for line in lines if line.startswith('potential_tag')))


def test_species_permutation_reorders_element_tag(tmp_path, tagged_model):
    result = run_command('gpmain', tagged_model + ['--neworder', '1', '0'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    fields = metadata(tmp_path)
    assert fields['elements'] == ['Ni', 'Al']
    assert fields['custom'] == 'keep-me'


def test_tagged_expansion_accepts_explicit_output_elements(tmp_path, tagged_model):
    result = run_command('gpmain', tagged_model + ['--neworder', '1', '2', '0',
                         '--elements', 'Ni', 'Cu', 'Al'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert metadata(tmp_path)['elements'] == ['Ni', 'Cu', 'Al']


@pytest.mark.parametrize('extra,message', [
    ([], 'requires --elements'),
    (['--elements', 'Ni', 'Cu'], 'requires --elements'),
    (['--elements', 'Al', 'Cu', 'Ni'], 'conflicts with existing species identities'),
    (['--elements', 'Ni', 'Ni', 'Al'], 'duplicate element'),
])
def test_tagged_expansion_rejects_missing_or_conflicting_elements(tmp_path, tagged_model, extra, message):
    result = run_command('gpmain', tagged_model + ['--neworder', '1', '2', '0'] + extra, tmp_path)
    assert result.returncode != 0
    assert message in result.stdout + result.stderr
    assert not (tmp_path / 'output.mtp').exists()
