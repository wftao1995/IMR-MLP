"""Fitness validates only enabled contributions before publishing output."""
import re
import pytest
from test_cli_error_paths import run_command, fitness_input


@pytest.mark.parametrize('kind,scale,ref,power', [
    ('energy', 'engevl', 'engref', 'engpow'),
    ('force', 'forceevl', 'forceref', 'forcepow'),
    ('stress', 'stressevl', 'stressref', 'stresspow')])
@pytest.mark.parametrize('field,value', [('scale','-50'),('scale','-1'),('scale','0'),
    ('scale','nan'),('scale','inf'),('ref','nan'),('ref','inf'),('power','nan'),('power','inf')])
def test_invalid_enabled_parameters_preserve_output(tmp_path, fitness_input, kind, scale, ref, power, field, value):
    (tmp_path/'output.cfg').write_text('preserve me\n')
    option = {'scale':scale, 'ref':ref, 'power':power}[field]
    result=run_command('calc_fitness', fitness_input+['--select_by_energy','false',
        '--select_by_'+kind,'true','--'+option,value],tmp_path)
    assert result.returncode == 1, result.stdout+result.stderr
    assert 'fitness requires' in result.stderr
    assert (tmp_path/'output.cfg').read_text() == 'preserve me\n'


@pytest.mark.parametrize('option', ['engevl','forceevl','stressevl','engref','forceref','stressref',
                                    'engpow','forcepow','stresspow'])
def test_disabled_parameters_are_irrelevant(tmp_path, fitness_input, option):
    result=run_command('calc_fitness',fitness_input+['--select_by_energy','false','--'+option,'nan'],tmp_path)
    assert result.returncode == 0, result.stdout+result.stderr
    assert float(re.search(r'Feature\s+Fitness\s+(\S+)',(tmp_path/'output.cfg').read_text())[1]) == 1


@pytest.mark.parametrize('value',['nan','inf','-inf'])
def test_threshold_must_be_finite(tmp_path, fitness_input, value):
    (tmp_path/'output.cfg').write_text('original')
    result=run_command('calc_fitness',fitness_input+['--threshold',value],tmp_path)
    assert result.returncode == 1
    assert (tmp_path/'output.cfg').read_text() == 'original'


@pytest.mark.parametrize('power',[0,-1,1,1000])
def test_formula_retains_zero_negative_powers_and_zero_weight(tmp_path, fitness_input, power):
    result=run_command('calc_fitness',fitness_input+['--engpow',str(power),'--threshold','-2'],tmp_path)
    assert result.returncode == 0, result.stdout+result.stderr
    expected=0 if power==1000 else 10/(10+50**power)
    value=float(re.search(r'Feature\s+Fitness\s+(\S+)',(tmp_path/'output.cfg').read_text())[1])
    assert value == pytest.approx(expected,abs=1e-6)


def test_combined_energy_force_stress_formula(tmp_path, fitness_input):
    (tmp_path/'input.xyz').write_text('1\nLattice="10 0 0 0 10 0 0 0 10" '
        'Properties=species:S:1:pos:R:3:forces:R:3 energy=-40 virial="2 0 0 0 1 0 0 0 1"\nH 0 0 0 2 0 0\n')
    result=run_command('calc_fitness',fitness_input+['--select_by_force','true','--select_by_stress','true'],tmp_path)
    assert result.returncode == 0, result.stdout+result.stderr
    value=float(re.search(r'Feature\s+Fitness\s+(\S+)',(tmp_path/'output.cfg').read_text())[1])
    assert value == pytest.approx(.5*(1/5)*(100/102),abs=1e-6)


def test_bad_later_frame_preserves_existing_output(tmp_path, fitness_input):
    with (tmp_path/'input.xyz').open('a') as stream:
        stream.write('2\nProperties=species:S:1:pos:R:3 energy=0\nH 0 0 0\n')
    (tmp_path/'output.cfg').write_text('original')
    result=run_command('calc_fitness',fitness_input,tmp_path)
    assert result.returncode == 1
    assert (tmp_path/'output.cfg').read_text() == 'original'
