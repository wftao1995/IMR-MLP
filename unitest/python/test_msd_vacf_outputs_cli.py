"""All enabled MSD/VACF derived paths are checked before either writer runs."""
import os
import pytest
from test_cli_error_paths import run_command


def setup(tmp_path, settings):
    (tmp_path/'index.map').write_text('H 0\nHe 0\n')
    (tmp_path/'input.xyz').write_text(''.join(
        '2\nLattice="20 0 0 0 20 0 0 0 20" pbc="T T T" '
        'Properties=species:S:1:pos:R:3:velocities:R:3\n'
        f'H {1+.1*f} 2 3 .1 0 0\nHe {2+.2*f} 3 4 .2 0 0\n' for f in range(6)))
    (tmp_path/'TRACKID').write_text('1\n1\n')
    (tmp_path/'GROUPID').write_text('1\n[ first 0 1 ]\n0\n')
    (tmp_path/'PROJDIR').write_text('2\n1 0 0\n0 1 0\n')
    (tmp_path/'settings.inp').write_text('msd:drift_cm false\nvacf:dfreq 1\n'+settings)


def run(tmp_path, *flags):
    return run_command('imrmlp',['calc_prop','--input','input.xyz','--input-format','extxyz',
        '--settings','settings.inp','--num_threads','1',*flags],tmp_path)


@pytest.mark.parametrize('settings,path',[
    ('vacf:output_header C\nmsd:output_total C_total\n','C_total.dat'),
    ('vacf:types true\nvacf:output_header C\nmsd:total false\nmsd:types true\nmsd:output_types C_type\n','C_type_0.dat'),
    ('vacf:groups true\nvacf:output_header C\nmsd:total false\nmsd:groups true\nmsd:output_group C_group\n','C_group_0.dat'),
    ('vacf:track true\nvacf:output_header C\nmsd:total false\nmsd:track true\nmsd:output_track C_track\n','C_track_1.dat'),
    ('vacf:projection true\nvacf:noiso true\nvacf:output_header C\nmsd:output_total C_proj_2_total\n','C_proj_2_total.dat'),
    ('vacf:vdos true\nvacf:output_header C\nmsd:output_total C_vdos_total\n','C_vdos_total.dat'),
    ('msd:types true\nmsd:output_total M_0\nmsd:output_types M\n','M_0.dat'),
])
def test_derived_output_collisions_preserve_files(tmp_path,settings,path):
    setup(tmp_path,settings)
    (tmp_path/path).write_text('keep this file\n')
    result=run(tmp_path,'--vacf','--msd')
    assert result.returncode == 1, result.stdout+result.stderr
    assert 'output collision' in result.stdout+result.stderr
    assert (tmp_path/path).read_text() == 'keep this file\n'
    assert not (tmp_path/'VACF_total.dat').exists()


@pytest.mark.parametrize('alias',['relative','symlink','hardlink'])
def test_path_aliases_are_collisions(tmp_path,alias):
    (tmp_path/'sub').mkdir()
    prefix='sub/../C_total' if alias=='relative' else 'alias'
    setup(tmp_path,f'vacf:output_header C\nmsd:output_total {prefix}\n')
    target=tmp_path/'C_total.dat';target.write_text('original')
    if alias=='symlink': (tmp_path/'alias.dat').symlink_to(target)
    if alias=='hardlink': os.link(target,tmp_path/'alias.dat')
    result=run(tmp_path,'--vacf','--msd')
    assert result.returncode == 1, result.stdout+result.stderr
    assert 'output collision' in result.stdout+result.stderr
    assert target.read_text() == 'original'


@pytest.mark.parametrize('flags',[('--msd',),('--vacf',),('--msd','--vacf')])
def test_disabled_outputs_do_not_cause_false_collisions(tmp_path,flags):
    setup(tmp_path,'vacf:output_header C\nmsd:output_total C_vdos_total\nvacf:vdos false\n')
    result=run(tmp_path,*flags)
    assert result.returncode == 0,result.stdout+result.stderr


def test_joint_results_match_individual_runs_for_all_categories(tmp_path):
    settings='msd:types true\nmsd:groups true\nmsd:track true\nvacf:types true\nvacf:groups true\nvacf:track true\nvacf:projection true\nvacf:vdos true\n'
    outputs={}
    for mode,flags in [('msd',['--msd']),('vacf',['--vacf']),('joint',['--msd','--vacf'])]:
        directory=tmp_path/mode;directory.mkdir();setup(directory,settings)
        result=run(directory,*flags)
        assert result.returncode == 0,result.stdout+result.stderr
        outputs[mode]={p.name:p.read_text() for p in directory.glob('*.dat')}
    assert outputs['joint'] == outputs['msd']|outputs['vacf']
    assert len(outputs['msd'])==5
    assert len(outputs['vacf'])==30
