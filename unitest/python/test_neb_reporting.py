"""Opt-in timing and independent symmetry release preserve the main CLI result."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
from test_neb_parallel import inputs, run, compare, binary


@pytest.mark.parametrize('mpi,mode,ranks', [(False,'images',1),(True,'images',4),(True,'atoms',3)])
def test_profile_preserves_path_and_accounts_for_all_images(tmp_path, mpi, mode, ranks):
    binary(mpi)
    for name in ['base','profile']:
        work=tmp_path/name;inputs(work,mode)
        if name=='profile':
            with (work/'neb.ini').open('a') as out: out.write('neb:profile profile.json\n')
        result=run(work,ranks,2,mpi)
        assert result.returncode in (0,2),result.stdout+result.stderr
    compare(tmp_path/'base',tmp_path/'profile')
    assert not (tmp_path/'base/profile.json').exists()
    profile=json.loads((tmp_path/'profile/profile.json').read_text())
    assert len(profile['ranks'])==ranks
    assert profile['omp_threads']==2
    for row in profile['ranks']:
        for scope in ['main','symmetry_release']:
            assert all(np.isfinite(v) and v>=0 for v in row[scope].values())
        assert row['symmetry_release']['image_evaluations']==0
    count=sum(r['main']['image_evaluations'] for r in profile['ranks'])
    text=(tmp_path/'profile/neb.dat').read_text()
    assert f'# image_force_evaluations {int(count)}\n' in text


def symmetric_inputs(work):
    inputs(work)
    for i,x,y in [(0,1.38,1.19),(2,1.68,1.47)]:
        (work/f'{i}.xyz').write_text('2\nLattice="6 0 0 0 7 0 0 0 8" pbc="T T T" '
            f'Properties=species:S:1:pos:R:3\nH {x} {y} 1.52\nH {6-x} {y} 1.52\n')
    with (work/'neb.ini').open('a') as out:
        out.write('neb:keep_symm true\ngtol 100\nneb:symmetry_release release\n'
                  'neb:symmetry_release_trials 1\nneb:symmetry_release_steps 2\n'
                  'neb:symmetry_release_evaluations 2\nneb:profile profile.json\n')


@pytest.mark.parametrize('mpi', [False,True])
def test_release_separate_counts_and_baseline(tmp_path, mpi):
    binary(mpi)
    work=tmp_path/'run';symmetric_inputs(work)
    result=run(work,3,2,mpi)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads((work/'release/report.json').read_text())
    assert report['status'] in ('no_lower_barrier_observed','lower_barrier_found')
    assert len(report['trials'])==2
    profile=json.loads((work/'profile.json').read_text())
    main=sum(r['main']['image_evaluations'] for r in profile['ranks'])
    release=sum(r['symmetry_release']['image_evaluations'] for r in profile['ranks'])
    assert release==sum(t['image_force_evaluations'] for t in report['trials'])
    assert f'# image_force_evaluations {int(main)}\n' in (work/'neb.dat').read_text()
    assert release>0
    for i in range(3):
        np.testing.assert_allclose(np.loadtxt(work/f'nebimage_{i}_final.vasp',skiprows=8),
                                   np.loadtxt(work/f'release/image-{i}.vasp',skiprows=8),atol=1e-12)


@pytest.mark.parametrize('failure', ['existing_directory','unconverged','profile_output'])
def test_reporting_failures_do_not_hang(tmp_path,failure):
    binary(True)
    work=tmp_path/'run';symmetric_inputs(work)
    if failure=='existing_directory': (work/'release').mkdir()
    if failure=='unconverged':
        p=work/'neb.ini';p.write_text(p.read_text().replace('gtol 100','gtol 1e-15')+'neb:max_force_evaluations 1\n')
    if failure=='profile_output':
        p=work/'neb.ini';p.write_text(p.read_text().replace('neb:profile profile.json','neb:profile missing/profile.json'))
    result=run(work,3,2,True)
    if failure=='profile_output': assert result.returncode==1
    else:
        assert result.returncode in (0,2)
        if failure=='unconverged': assert json.loads((work/'release/report.json').read_text())['status']=='skipped'
    assert (work/'neb.dat').exists()


def test_release_worker_failure_preserves_converged_main_and_exits(tmp_path):
    binary(True)
    work=tmp_path/'run';symmetric_inputs(work)
    script=work/'calculate.sh'
    script.write_text('echo call >> "$1/calls"\n'
        'case "$1" in *mpi-rank-1) [ "$(wc -l < "$1/calls")" -gt 1 ] && exit 7;; esac\n'
        'cat > "$1/result.xyz" <<EOF\n2\nLattice="6 0 0 0 7 0 0 0 8" '
        'Properties=species:S:1:pos:R:3:forces:R:3 energy=0 virial="0 0 0 0 0 0 0 0 0"\n'
        'H 1.38 1.19 1.52 0 0 0\nH 4.62 1.19 1.52 0 0 0\nEOF\n')
    p=work/'neb.ini'
    p.write_text(p.read_text().replace('pot:pottype lj612','pot:pottype shellfile').replace('neb:climb true','neb:climb false')+
        f'pot:shellfile:runscript {script}\npot:shellfile:index index.map\n'
        'pot:shellfile:read_outfile result.xyz\npot:shellfile:read_outformat extxyz\n')
    result=run(work,3,2,True)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads((work/'release/report.json').read_text())
    assert report['status']=='failed'
    assert all(t['status']=='failed' for t in report['trials'])
    assert all(t['attempted_image_evaluations']==2 for t in report['trials'])
    assert '# status converged' in (work/'neb.dat').read_text()
