"""Malformed XML reaches the normal reader failure path through convstr."""
import pytest
from test_cli_error_paths import run_command
from test_vasprun_consumers import xml_trajectory


@pytest.mark.parametrize('damage', ['root', 'basis', 'last_forces'])
def test_convstr_rejects_incomplete_xml_without_publishing_output(tmp_path, damage):
    text = xml_trajectory()
    if damage == 'root':
        text = '<modeling/>'
    elif damage == 'basis':
        text = text.replace('<v>3 0 0</v>', '<v>3 0</v>', 1)
    else:
        left, right = text.rsplit('<v>1 2 3</v>', 1)
        text = left + '<v>1 2</v>' + right
    (tmp_path / 'index.map').write_text('H 0\n')
    (tmp_path / 'vasprun.xml').write_text(text)
    output = tmp_path / 'frames.xyz'
    output.write_text('previous successful output')
    result = run_command('convstr', ['--inputfiles', 'vasprun.xml', '--input-format', 'xml',
                                    '--outputfile', output, '--output-format', 'extxyz', '--overwrite'], tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'Error reading vasprun.xml:' in result.stdout + result.stderr
    assert output.read_text() == 'previous successful output'
