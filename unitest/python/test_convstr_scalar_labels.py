"""Scalar metadata expressions exercise the actual standalone convstr process."""
import re

import pytest
from test_cli_error_paths import run_command
from test_convstr_import_options import BOHR, HARTREE, frame, invoke, read_output


def input_frame(energy=2, x=3):
    return frame(energy).replace(' pbc=', f' x={x} y="old text" pbc=')


def convert_to(tmp_path, output_format, outfile, extra):
    (tmp_path/'index.map').write_text('H 0\n')
    return run_command('convstr', [
        '--inputfiles', 'input.xyz', '--input-format', 'extxyz',
        '--outputfile', outfile, '--output-format', output_format,
        '--index', 'index.map', *extra], tmp_path)


@pytest.mark.parametrize('mode', [(), ('--loadrate', '1'), ('--outrate', '1'),
                                  ('--loadrate', '1', '--outrate', '1')])
def test_scalar_assignments_use_original_snapshot_in_every_path(tmp_path, mode):
    (tmp_path/'input.xyz').write_text(input_frame(2, 3) + input_frame(8, 5))
    result = invoke(tmp_path, ['input.xyz'], [
        '--input-format', 'extxyz', '--label-expr',
        'x=x+1', 'y=x+energy/size', 'density=energy/volume',
        '--label', 'note', '--value', 'energy/size', *mode])
    assert result.returncode == 0, result.stdout + result.stderr
    output = read_output(tmp_path/'output.xyz')
    assert [float(t['x']) for t, _ in output] == [4., 6.]
    assert [float(t['y']) for t, _ in output] == [5., 13.]
    assert [float(t['density']) for t, _ in output] == [.25, 1.]
    assert [t['note'] for t, _ in output] == ['energy/size'] * 2
    for header in (tmp_path/'output.xyz').read_text().splitlines()[1::3]:
        assert len(re.findall(r'\bx=', header)) == 1
        assert len(re.findall(r'\by=', header)) == 1
        assert not re.search(r'\b[xy]="', header)  # Numeric outputs, no duplicate Feature.


@pytest.mark.parametrize('mode', [(), ('--loadrate', '1'), ('--outrate', '1')])
def test_scalar_units_precede_assignment_and_derived_sort(tmp_path, mode):
    (tmp_path/'input.xyz').write_text(input_frame(2) + input_frame(8))
    result = invoke(tmp_path, ['input.xyz'], [
        '--input-format', 'extxyz', '--input-unit-system', 'atomic',
        '--label-expr', 'rank=-energy/size', 'density=energy/volume',
        '--sort-by', 'rank', *mode])
    assert result.returncode == 0, result.stdout + result.stderr
    output = read_output(tmp_path/'output.xyz')
    assert [float(t['rank']) for t, _ in output] == pytest.approx([-8*HARTREE, -2*HARTREE])
    assert [float(t['density']) for t, _ in output] == pytest.approx([
        HARTREE / BOHR**3, HARTREE / (4*BOHR**3)])


def test_scalar_bin_and_extxyz_roundtrip_remain_numeric(tmp_path):
    (tmp_path/'input.xyz').write_text(input_frame())
    for output_format, outfile in [('bin', 'derived.bin'), ('extxyz', 'derived.xyz')]:
        result = convert_to(tmp_path, output_format, outfile, [
            '--label-expr', 'derived=energy/size'])
        assert result.returncode == 0, result.stdout + result.stderr
        result = invoke(tmp_path, [outfile], [
            '--input-format', output_format, '--label-expr', 'check=derived+1', '--overwrite'])
        assert result.returncode == 0, result.stdout + result.stderr
        tags, _ = read_output(tmp_path/'output.xyz')[0]
        assert float(tags['derived']) == 2.
        assert float(tags['check']) == 3.


@pytest.mark.parametrize('mode', [(), ('--loadrate', '1', '--outrate', '1')])
def test_scalar_cfg_export_uses_existing_feature_text_bridge(tmp_path, mode):
    (tmp_path/'input.xyz').write_text(input_frame())
    result = convert_to(tmp_path, 'mlpcfg', 'derived.cfg', [
        '--label-expr', 'derived=energy/3', *mode])
    assert result.returncode == 0, result.stdout + result.stderr
    text = (tmp_path/'derived.cfg').read_text()
    entries = re.findall(r'^Feature derived (.+)$', text, re.MULTILINE)
    assert len(entries) == 1
    assert float(entries[0]) == pytest.approx(2/3)
    # Reading legacy CFG Feature text does not turn arbitrary labels into scalars.
    result = invoke(tmp_path, ['derived.cfg'], [
        '--input-format', 'mlpcfg', '--label-expr', 'check=derived+1'])
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'nonnumeric scalar' in result.stdout + result.stderr


@pytest.mark.parametrize('expression', [
    'energy=1', 'free_energy=1', 'size=2', 'bad=2^energy', 'bad=sin(1,2)', 'bad=1;', 'bad='])
def test_invalid_scalar_syntax_is_preflighted_before_output(tmp_path, expression):
    (tmp_path/'input.xyz').write_text(input_frame())
    (tmp_path/'output.xyz').write_text('original output')
    result = invoke(tmp_path, ['input.xyz'], [
        '--input-format', 'extxyz', '--label-expr', expression,
        '--loadrate', '1', '--outrate', '1', '--overwrite'])
    assert result.returncode == 1, result.stdout + result.stderr
    assert expression in result.stdout + result.stderr
    assert (tmp_path/'output.xyz').read_text() == 'original output'


@pytest.mark.parametrize('expressions', [('a=1', 'a=2'), ('a=1',)])
def test_conflicting_scalar_targets_are_preflighted(tmp_path, expressions):
    (tmp_path/'input.xyz').write_text(input_frame())
    (tmp_path/'output.xyz').write_text('original output')
    extra = ['--label', 'a', '--value', 'text'] if len(expressions) == 1 else []
    result = invoke(tmp_path, ['input.xyz'], [
        '--input-format', 'extxyz', '--label-expr', *expressions, *extra,
        '--loadrate', '1', '--outrate', '1', '--overwrite'])
    assert result.returncode == 1, result.stdout + result.stderr
    assert (tmp_path/'output.xyz').read_text() == 'original output'


@pytest.mark.parametrize('expression', ['bad=missing', 'bad=y', 'bad=1/0',
                                        'bad=sqrt(-1)', 'bad=log(0)', 'bad=exp(1000)'])
def test_scalar_runtime_errors_name_frame_and_expression(tmp_path, expression):
    (tmp_path/'input.xyz').write_text(input_frame())
    result = invoke(tmp_path, ['input.xyz'], [
        '--input-format', 'extxyz', '--label-expr', expression])
    assert result.returncode == 1, result.stdout + result.stderr
    diagnostic = result.stdout + result.stderr
    assert 'frame 0' in diagnostic
    assert expression in diagnostic


def test_scalar_stream_error_keeps_completed_prefix(tmp_path):
    (tmp_path/'input.xyz').write_text(input_frame(2) + input_frame(0))
    result = invoke(tmp_path, ['input.xyz'], [
        '--input-format', 'extxyz', '--label-expr', 'inverse=1/energy',
        '--loadrate', '1', '--outrate', '1'])
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'frame 1' in result.stdout + result.stderr
    output = read_output(tmp_path/'output.xyz')
    assert len(output) == 1
    assert float(output[0][0]['inverse']) == .5


def test_old_text_label_default_is_unchanged(tmp_path):
    (tmp_path/'input.xyz').write_text(input_frame())
    result = invoke(tmp_path, ['input.xyz'], [
        '--input-format', 'extxyz', '--label', 'note', '--value', '1+2'])
    assert result.returncode == 0, result.stdout + result.stderr
    assert read_output(tmp_path/'output.xyz')[0][0]['note'] == '1+2'
    assert 'note="1+2"' in (tmp_path/'output.xyz').read_text()
