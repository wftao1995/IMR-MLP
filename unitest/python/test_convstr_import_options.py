"""Explicit content detection and source-unit import use the real convstr CLI."""
import gzip
import re
import pytest
from test_cli_error_paths import run_command

BOHR = .529177210903
HARTREE = 27.211396641308
REAL_ENERGY = 4.184 * 6.241506363094e21 / 6.02214076e23
ATOMIC_TIME = 2.4188843265857e-5


def frame(energy=2, tensor='virial="1 0 0 0 2 0 0 0 3"'):
    return ('1\nLattice="2 0 0 0 2 0 0 0 2" pbc="T T T" '
            'Properties=species:S:1:pos:R:3:forces:R:3:velocities:R:3 '
            f'energy={energy} fenergy=3 {tensor}\nH 3 .5 .25 1 2 3 4 5 6\n')


def invoke(tmp_path, files, extra=(), outfile='output.xyz'):
    (tmp_path/'index.map').write_text('H 0\n')
    return run_command('convstr', ['--inputfiles',*files,'--outputfile',outfile,
        '--output-format','extxyz','--index','index.map',*extra],tmp_path)


def read_output(path):
    lines=path.read_text().splitlines();result=[];offset=0
    while offset<len(lines):
        count=int(lines[offset]);header=lines[offset+1]
        tags={m[0]:(m[1] if m[1] else m[2]) for m in
              re.findall(r'(\w+)=(?:"([^"]*)"|(\S+))',header)}
        properties=tags['Properties'].split(':');columns={};start=0
        for key,kind,width in zip(properties[::3],properties[1::3],properties[2::3]):
            columns[key]=(start,int(width));start+=int(width)
        row=lines[offset+2].split()
        def values(key, row=row, columns=columns):
            begin,width=columns[key];return list(map(float,row[begin:begin+width]))
        result.append((tags,values));offset+=count+2
    return result


@pytest.mark.parametrize('mode', [(),('--loadrate','1'),('--outrate','1'),
                                   ('--loadrate','1','--outrate','1')])
@pytest.mark.parametrize('preset,length,energy,time', [
    ('atomic',BOHR,HARTREE,ATOMIC_TIME),('real',1.,REAL_ENERGY,.001),
    ('canonical',1.,1.,1.),('metal',1.,1.,1.)])
def test_explicit_units_all_io_paths(tmp_path,mode,preset,length,energy,time):
    original=frame();(tmp_path/'input.xyz').write_text(original)
    result=invoke(tmp_path,['input.xyz'],['--input-format','extxyz','--input-unit-system',preset,*mode])
    assert result.returncode == 0,result.stdout+result.stderr
    tags,values=read_output(tmp_path/'output.xyz')[0]
    assert list(map(float,tags['Lattice'].split())) == pytest.approx([2*length,0,0,0,2*length,0,0,0,2*length])
    assert values('pos') == pytest.approx([3*length,.5*length,.25*length])
    assert values('forces') == pytest.approx([energy/length,2*energy/length,3*energy/length])
    assert values('velocities') == pytest.approx([4*length/time,5*length/time,6*length/time])
    assert float(tags['energy']) == pytest.approx(2*energy)
    assert float(tags['fenergy']) == pytest.approx(3*energy)
    assert list(map(float,tags['virial'].split())) == pytest.approx([energy,0,0,0,2*energy,0,0,0,3*energy])
    assert (tmp_path/'input.xyz').read_text() == original


@pytest.mark.parametrize('mode',[(),('--loadrate','1','--outrate','1')])
def test_extxyz_stress_uses_source_volume_before_normalization(tmp_path,mode):
    (tmp_path/'input.xyz').write_text(frame(tensor='stress="1 0 0 0 2 0 0 0 3"'))
    result=invoke(tmp_path,['input.xyz'],['--input-format','extxyz','--input-unit-system','atomic',*mode])
    assert result.returncode == 0,result.stdout+result.stderr
    tags,_=read_output(tmp_path/'output.xyz')[0]
    # Source volume is 8 Bohr^3. W=-P*V is then energy-dimensional Hartree.
    assert list(map(float,tags['virial'].split())) == pytest.approx([-8*HARTREE,0,0,0,-16*HARTREE,0,0,0,-24*HARTREE])


def test_unit_import_precedes_wrap_and_sort(tmp_path):
    (tmp_path/'input.xyz').write_text(frame(3)+frame(1))
    result=invoke(tmp_path,['input.xyz'],['--input-format','extxyz','--input-unit-system','atomic',
                                         '--wrap','true','--sort-by','energy'])
    assert result.returncode == 0,result.stdout+result.stderr
    output=read_output(tmp_path/'output.xyz')
    assert [float(tags['energy']) for tags,_ in output] == pytest.approx([HARTREE,3*HARTREE])
    for _,values in output:
        assert values('pos') == pytest.approx([BOHR,.5*BOHR,.25*BOHR])


@pytest.mark.parametrize('suffix',['', '.bad', '.gz'])
def test_opt_in_detects_misnamed_and_compressed_extxyz(tmp_path,suffix):
    name='misnamed'+suffix
    if suffix=='.gz':
        with gzip.open(tmp_path/name,'wt') as stream:stream.write(frame())
    else:(tmp_path/name).write_text(frame())
    result=invoke(tmp_path,[name],['--detect-format'])
    assert result.returncode == 0,result.stdout+result.stderr
    assert float(read_output(tmp_path/'output.xyz')[0][0]['energy']) == 2.


def test_explicit_format_has_priority_over_detection(tmp_path):
    # This CP2K comment is deliberately not auto-detected as plain EXTXYZ.
    (tmp_path/'input.unknown').write_text('1\ni = 0, time = 0, E = 2\nH 1 2 3\n')
    result=invoke(tmp_path,['input.unknown'],['--detect-format','--input-format','cp2kxyz'])
    assert result.returncode == 0,result.stdout+result.stderr
    assert read_output(tmp_path/'output.xyz')[0][1]('pos') == [1.,2.,3.]


def test_detection_checks_every_input_before_opening_output(tmp_path):
    (tmp_path/'one.data').write_text(frame())
    (tmp_path/'two.data').write_text('ATOMS\nH 0 0 0\n')
    (tmp_path/'output.xyz').write_text('preserve existing output')
    result=invoke(tmp_path,['one.data','two.data'],['--detect-format','--overwrite','--loadrate','1','--outrate','1'])
    assert result.returncode == 1,result.stdout+result.stderr
    assert 'formats differ' in result.stdout+result.stderr
    assert (tmp_path/'output.xyz').read_text() == 'preserve existing output'


@pytest.mark.parametrize('extra,filename,text',[
    (['--detect-format'],'input.data','unrecognized content\n'),
    (['--detect-format'],'input.data',''),
    (['--detect-format'],'input.data',frame()+'ATOMS\nH 0 0 0\n'),
    (['--input-format','extxyz','--input-unit-system','mystery'],'input.data',frame()),
    (['--input-format','xsf','--input-unit-system','atomic'],'input.data','ATOMS\nH 0 0 0\n'),
    (['--input-format','cp2kxyz','--input-unit-system','canonical'],'input.data','1\ni = 0, E = 0\nH 0 0 0\n'),
])
def test_invalid_import_options_preserve_output(tmp_path,extra,filename,text):
    (tmp_path/filename).write_text(text);(tmp_path/'output.xyz').write_text('original')
    result=invoke(tmp_path,[filename],[*extra,'--overwrite','--loadrate','1','--outrate','1'])
    assert result.returncode == 1,result.stdout+result.stderr
    assert (tmp_path/'output.xyz').read_text() == 'original'


def test_detect_and_units_work_for_consistent_multiple_inputs(tmp_path):
    (tmp_path/'one.data').write_text(frame(1));(tmp_path/'two.data').write_text(frame(2))
    result=invoke(tmp_path,['one.data','two.data'],['--detect-format','--input-unit-system','atomic'])
    assert result.returncode == 0,result.stdout+result.stderr
    assert [float(t['energy']) for t,_ in read_output(tmp_path/'output.xyz')] == pytest.approx([HARTREE,2*HARTREE])


def test_default_and_explicit_canonical_outputs_match(tmp_path):
    (tmp_path/'input.xyz').write_text(frame())
    result=invoke(tmp_path,['input.xyz'],['--input-format','extxyz'],outfile='default.xyz')
    assert result.returncode == 0,result.stdout+result.stderr
    result=invoke(tmp_path,['input.xyz'],['--input-format','extxyz','--input-unit-system','canonical'])
    assert result.returncode == 0,result.stdout+result.stderr
    assert (tmp_path/'output.xyz').read_bytes() == (tmp_path/'default.xyz').read_bytes()


def test_unit_override_validates_all_inferred_input_formats_before_output(tmp_path):
    (tmp_path/'one.xyz').write_text(frame())
    (tmp_path/'two.xsf').write_text('ATOMS\nH 0 0 0\n')
    (tmp_path/'output.xyz').write_text('original')
    result=invoke(tmp_path,['one.xyz','two.xsf'],['--input-unit-system','atomic','--overwrite'])
    assert result.returncode == 1,result.stdout+result.stderr
    assert (tmp_path/'output.xyz').read_text() == 'original'


def test_unit_override_supports_inferred_extxyz_in_sequential_mode(tmp_path):
    (tmp_path/'input.xyz').write_text(frame())
    result=invoke(tmp_path,['input.xyz'],['--input-unit-system','atomic','--loadrate','1','--outrate','1'])
    assert result.returncode == 0,result.stdout+result.stderr
    assert float(read_output(tmp_path/'output.xyz')[0][0]['energy']) == pytest.approx(2*HARTREE)
