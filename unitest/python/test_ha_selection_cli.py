"""HA category selection prunes classification while preserving global fractions."""
import pytest

from test_cli_error_paths import run_command


def execute(root, name, threads, selection=None, all_indices=True, unbond=False, pair_cutoff=False, voro=False):
    directory = root/name
    directory.mkdir()
    (directory/'index.map').write_text('H 1\nHe 1\n')
    frames = [
        [('H',0,0,0),('H',1,0,0),('H',.5,.866025403784,0),('He',.5,.288675134595,.816496580928)],
        [('H',0,0,0),('He',1,0,0),('H',1,1,0),('He',0,1,0)],
    ]
    text = ''
    for rows in frames:
        text += f'{len(rows)}\nLattice="8 0 0 0 8 0 0 0 8" pbc="T T T" Properties=species:S:1:pos:R:3\n'
        text += ''.join(f'{symbol} {x} {y} {z}\n' for symbol,x,y,z in rows)
    (directory/'input.xyz').write_text(text)
    settings = f'ha_index:unbond {str(unbond).lower()}\n'
    if selection:
        settings += 'ha_index:indices '+selection+'\n'
    if pair_cutoff:
        settings += 'pair_cutoff:H-H 1.1\npair_cutoff:H-He 1.1\npair_cutoff:He-He 1.1\n'
    (directory/'ha.ini').write_text(settings)
    args = ['calc_prop','--ha-index','--input','input.xyz','--input-format','extxyz',
            '--settings','ha.ini','--cutoff','1.5' if pair_cutoff else '1.1','--num_threads',str(threads)]
    if all_indices:
        args += ['--output-all-indices']
    if pair_cutoff:
        args += ['--pair-cutoff']
    if voro:
        args += ['--voro-neighbor']
    result = run_command('imrmlp',args,directory)
    assert result.returncode == 0, result.stdout+result.stderr
    return directory


def table(path):
    result = {}
    order = []
    for line in path.read_text().splitlines():
        row = line.split()
        if len(row) != 3 or (row[0] != 'Total' and not row[0].isdigit()):
            continue
        result[row[0]] = (int(row[1]),float(row[2]))
        order.append(row[0])
    return result,order


@pytest.mark.parametrize('threads', [1,2])
@pytest.mark.parametrize('unbond', [False,True])
@pytest.mark.parametrize('pair_cutoff', [False,True])
def test_selected_total_partial_and_omp_match_full_post_filter(tmp_path, threads, unbond, pair_cutoff):
    full = execute(tmp_path,'full',threads,unbond=unbond,pair_cutoff=pair_cutoff)
    selected = execute(tmp_path,'selected',threads,'1551 1211 1001 1551',unbond=unbond,pair_cutoff=pair_cutoff)
    default_formatter = execute(tmp_path,'common-flag',threads,'1551 1211 1001 1551',False,unbond,pair_cutoff)
    names = ['out_total.txt','Bond_H_H.txt','Bond_H_He.txt','Bond_He_He.txt']
    for name in names:
        baseline,_ = table(full/name)
        result,order = table(selected/name)
        assert order == ['1001','1211','1551','Total']
        for code in order[:-1]:
            assert result[code] == baseline.get(code,(0,0.))
        assert result['Total'][0] == sum(result[code][0] for code in order[:-1])
        assert result['Total'][1] == pytest.approx(sum(result[code][1] for code in order[:-1]),abs=2e-6)
        assert (selected/name).read_text() == (default_formatter/name).read_text()
    # Global denominator, including excluded square/common-neighbor categories,
    # is retained; partial H-He fractions are not normalized to one.
    rows,_ = table(selected/'Bond_H_He.txt')
    assert rows['1211'][1] < 1.


def test_omitted_selection_keeps_default_and_voronoi_keeps_selection(tmp_path):
    baseline = execute(tmp_path,'default',1)
    empty = execute(tmp_path,'empty',1,'')
    for path in baseline.glob('*.txt'):
        assert path.read_text() == (empty/path.name).read_text()
    full = execute(tmp_path,'voro-all',1,voro=True)
    selected = execute(tmp_path,'voro-selected',2,'1421 1551',False,voro=True)
    for name in ['out_total_voro_neighbor.txt','Bond_H_H_voro.txt','Bond_H_He_voro.txt','Bond_He_He_voro.txt']:
        expected,_ = table(full/name)
        actual,order = table(selected/name)
        assert order == ['1421','1551','Total']
        for code in ['1421','1551']:
            assert actual[code] == expected.get(code,(0,0.))
