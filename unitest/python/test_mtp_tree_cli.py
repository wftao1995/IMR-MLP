"""Full-tree initialization, migration and composite energy-offset responses."""
from pathlib import Path
import numpy as np
import pytest
from test_gpm_workflow_cli import prepare_fixture, run
from test_cli_error_paths import run_command

FAMILIES=['les','sog','mtpm_sog','nn1','nn2','edq','edqrd','hybrid','scale','hmtp','linear']


def prepare(path, family):
    if family in ['edq','edqrd'] and not (Path(__file__).resolve().parents[2]/'potentials/potential_mtp_edqrd.h').exists():
        pytest.skip('EDQ/EDQRd model family is absent in this repository')
    prepare_fixture(path)
    text=(path/'base.mtp').read_text()
    text=text[:text.index('radial_coeffs')]+text[text.index('alpha_moments_count'):]
    text=text[:text.index('species_coeffs')]
    (path/'template.mtp').write_text(text)
    common='pot:potfile template.mtp\n'
    if family in ['les','sog']:
        common+='pot:pottype mtp_les\npot:les_kernel '+('sog' if family=='sog' else 'coulomb')+'\n'
    elif family=='mtpm_sog':common+='pot:pottype mtp_m\npot:extmodel mtp_m_n2\npot:n_input 2\npot:n_neuron 3\npot:n_neuron2 2\npot:lr_les true\npot:les_kernel sog\n'
    elif family in ['nn1','nn2']:common+=f'pot:pottype mtpnn1\npot:nnlay {1 if family=="nn1" else 2}\npot:nnhidden 3\npot:nnhidden2 2\n'
    elif family in ['edq','edqrd']:common+=f'pot:pottype mtp_{family}\npot:sr_potfile template.mtp\npot:charge_potfile template.mtp\n'
    elif family in ['hybrid','scale','hmtp']:
        common+='pot:pottype '+{'hybrid':'hybrid','scale':'hybrid_scale','hmtp':'hmtp'}[family]+'\npot:npot 2\npot:pot1:pottype mtp\npot:pot1:potfile template.mtp\n'
        if family=='hybrid':common+='pot:pot2:pottype zbl\npot:pot2:r0 0.5\npot:pot2:r1 2\n'
        else:common+='pot:pot2:pottype mtp\npot:pot2:potfile template.mtp\n'
        if family=='scale':common+='pot:feature min_distance\npot:boundary1 1 4\n'
    else:common+='pot:pottype descriptor_linear\n'
    if family in ['sog','mtpm_sog']:
        common+='pot:sog_channels 2\npot:sog_weights 1 2 1.5 0.5\npot:sog_log_bandwidths -0.5 0.2 -0.2 0.3\n'
    (path/'tree.ini').write_text(common)
    (path/'periodic.xyz').write_text('3\nLattice="12 0 0 0 12 0 0 0 12" pbc="T T T" Properties=species:S:1:pos:R:3\nH 1 1 1\nH 2.3 1.2 1\nH 1.4 3 1.3\n')
    run(path,'initmtpm','--input=template.mtp','--settings=tree.ini','--elements','H','--output=tree.gpm','--randomize=false')


def labels(path,model,output,input='periodic.xyz'):
    (path/'empty.ini').write_text('')
    run(path,'efs',input,output,'--settings=empty.ini','--index=missing.map','--potfile='+model,'--pottype=auto','--input-format=extxyz','--output-format=mlpcfg')
    values=[]
    for block in (path/output).read_text().split('BEGIN_CFG')[1:]:
        lines=block.splitlines();n=int(lines[lines.index('Size')+1]);a=next(i for i,l in enumerate(lines) if l.startswith('AtomData:'));keys=lines[a].split()[1:]
        forces=[[float(row.split()[keys.index(k)]) for k in ['fx','fy','fz']] for row in lines[a+1:a+1+n]]
        stress=next(i for i,l in enumerate(lines) if l.startswith('PlusStress:'))
        values.extend([float(lines[lines.index('Energy')+1]),*np.array(forces).ravel(),*map(float,lines[stress+1].split())])
    return np.array(values)


@pytest.mark.parametrize('family',FAMILIES)
def test_initialize_reuse_reinitialize_and_adjust_full_model_tree(tmp_path,family):
    prepare(tmp_path,family)
    run(tmp_path,'model_validate','--input=tree.gpm')
    expected=labels(tmp_path,'tree.gpm','expected.cfg')
    run(tmp_path,'initmtpm','--input=tree.gpm','--output=reused.gpm')
    np.testing.assert_allclose(labels(tmp_path,'reused.gpm','reused.cfg'),expected,rtol=1e-9,atol=1e-10)
    run(tmp_path,'initmtpm','--input=tree.gpm','--output=reset.gpm','--reinitialize','--randomize=false')
    run(tmp_path,'model_validate','--input=reset.gpm')
    run(tmp_path,'adjustmtp','--input=tree.gpm','--output=expanded.gpm','--append-elements','He','--new-species-init=clone','--clone-from','0')
    np.testing.assert_allclose(labels(tmp_path,'expanded.gpm','expanded.cfg'),expected,rtol=1e-8,atol=1e-9)
    run(tmp_path,'adjustmtp','--input=expanded.gpm','--output=reordered.gpm','--neworder','1','0','--basis-order=level-group')
    np.testing.assert_allclose(labels(tmp_path,'reordered.gpm','reordered.cfg'),expected,rtol=1e-8,atol=1e-9)
    run(tmp_path,'adjustmtp','--input=tree.gpm','--output=random.gpm','--append-elements','He','--new-species-init=random')
    np.testing.assert_allclose(labels(tmp_path,'random.gpm','random.cfg'),expected,rtol=1e-8,atol=1e-9)
    run(tmp_path,'model_convert','--input=tree.gpm','--legacy','--profile=inference','--output=native.model')
    run(tmp_path,'initmtpm','--input=native.model','--elements','H','--output=from-native.gpm')
    np.testing.assert_allclose(labels(tmp_path,'from-native.gpm','native.cfg'),expected,rtol=1e-8,atol=1e-9)
    run(tmp_path,'adjustmtp','--input=native.model','--output=native-adjusted.gpm','--basis-order=level')
    np.testing.assert_allclose(labels(tmp_path,'native-adjusted.gpm','native-adj.cfg'),expected,rtol=1e-8,atol=1e-9)
    # The complete GPM is the sole remaining model/structure source.
    for name in ['tree.ini','template.mtp','base.mtp','index.map','tree.gpm','expanded.gpm']:(tmp_path/name).unlink()
    run(tmp_path,'model_validate','--input=reordered.gpm')
    np.testing.assert_allclose(labels(tmp_path,'reordered.gpm','independent.cfg'),expected,rtol=1e-8,atol=1e-9)


@pytest.mark.parametrize('family,target',[('hybrid','pot1'),('scale','pot2'),('edq','short_range'),('edqrd','short_range')])
def test_composite_offset_fit_uses_complete_model_response(tmp_path,family,target):
    prepare(tmp_path,family)
    # Three geometries with a known +0.4 eV per-atom correction.
    if family=='scale':
        baseline=labels(tmp_path,'tree.gpm','baseline.cfg')[0]
        (tmp_path/'teacher.xyz').write_text((tmp_path/'periodic.xyz').read_text().replace(' pbc=',f' energy={baseline+0.8:.17g} pbc='))
        run(tmp_path,'initmtpm','--input=tree.gpm','--output=teacher.gpm','--dataset=teacher.xyz','--format=extxyz','--fit-species-offsets','--offset-target='+target)
    frames=[]
    for distance in [1.3,1.7,2.1]:
        (tmp_path/'periodic.xyz').write_text(f'2\nLattice="12 0 0 0 12 0 0 0 12" pbc="T T T" Properties=species:S:1:pos:R:3\nH 1 1 1\nH {1+distance} 1 1\n')
        result=labels(tmp_path,'teacher.gpm' if family=='scale' else 'tree.gpm','frame.cfg')
        frames.append((tmp_path/'periodic.xyz').read_text().replace(' pbc=',f' energy={result[0]+(0 if family=='scale' else 0.8):.17g} pbc='))
    (tmp_path/'fit.xyz').write_text(''.join(frames))
    run(tmp_path,'initmtpm','--input=tree.gpm','--output=fitted.gpm','--dataset=fit.xyz','--format=extxyz','--fit-species-offsets','--offset-target='+target)
    after=labels(tmp_path,'fitted.gpm','after.cfg',input='fit.xyz')
    # Every two-atom frame contains E, 6 forces and 6 virials.
    energies=after.reshape(-1,13)[:,0]
    wanted=[float(frame.split('energy=')[1].split()[0]) for frame in frames]
    np.testing.assert_allclose(energies,wanted,atol=1e-8,rtol=1e-8)


def test_bad_charge_offset_target_preserves_existing_output(tmp_path):
    prepare(tmp_path,'edqrd');(tmp_path/'old.gpm').write_bytes(b'keep')
    result=run_command('imrmlp',['initmtpm','--input=tree.gpm','--output=old.gpm','--dataset=periodic.xyz','--format=extxyz','--fit-species-offsets','--offset-target=charge'],tmp_path)
    assert result.returncode!=0
    assert 'energy branches' in result.stderr
    assert (tmp_path/'old.gpm').read_bytes()==b'keep'


def test_composite_settings_entry_and_combined_statistics(tmp_path):
    prepare(tmp_path,'hybrid')
    run(tmp_path,'initmtpm','--settings=tree.ini','--elements','H','--output=settings.gpm')
    expected=labels(tmp_path,'tree.gpm','reference.cfg')[0]
    (tmp_path/'labeled.xyz').write_text((tmp_path/'periodic.xyz').read_text().replace(' pbc=',f' energy={expected+0.8:.17g} pbc='))
    run(tmp_path,'initmtpm','--input=tree.gpm','--output=normalized.gpm','--dataset=labeled.xyz','--format=extxyz','--normalize-descriptors','--fit-species-offsets','--offset-target=pot1')
    assert labels(tmp_path,'normalized.gpm','normalized.cfg')[0]==pytest.approx(expected+0.8)
    run(tmp_path,'initmtpm','--input=tree.gpm','--output=direct.model')
    run(tmp_path,'initmtpm','--input=direct.model','--output=direct.gpm')
    np.testing.assert_allclose(labels(tmp_path,'direct.gpm','direct.cfg'),labels(tmp_path,'tree.gpm','tree.cfg'),atol=1e-9,rtol=1e-8)


def test_absent_public_model_family_fails_before_publication(tmp_path):
    if (Path(__file__).resolve().parents[2]/'potentials/potential_mtp_edqrd.h').exists():
        pytest.skip('Only applies to repositories without EDQ/EDQRd')
    prepare(tmp_path,'hybrid');(tmp_path/'old.gpm').write_bytes(b'keep')
    result=run_command('imrmlp',['initmtpm','--input=template.mtp','--type=mtp_edqrd','--output=old.gpm'],tmp_path)
    assert result.returncode!=0
    assert 'not implemented in this repository' in result.stderr
    assert (tmp_path/'old.gpm').read_bytes()==b'keep'


@pytest.mark.parametrize('family',['sog','nn2','scale'])
def test_mpi_full_tree_restore_and_migration(tmp_path,family):
    import os
    import subprocess
    executable=os.environ.get('MTP_WORKFLOW_MPI_EXECUTABLE')
    if not executable:pytest.skip('Set MTP_WORKFLOW_MPI_EXECUTABLE to an MPI CLI')
    prepare(tmp_path,family)
    expected=labels(tmp_path,'tree.gpm','before.cfg')
    executable=Path(executable).resolve();env=os.environ.copy()
    env['LD_LIBRARY_PATH']=str(executable.parent)+':'+env.get('LD_LIBRARY_PATH','')
    def launch(*args):
        return subprocess.run(['mpiexec','-n','2',str(executable),*args],cwd=tmp_path,env=env,capture_output=True,text=True,timeout=30)
    result=launch('initmtpm','--input=tree.gpm','--output=mpi-rebuilt.gpm','--reinitialize','--randomize=false')
    assert result.returncode==0,result.stdout+result.stderr
    run(tmp_path,'model_validate','--input=mpi-rebuilt.gpm')
    result=launch('adjustmtp','--input=tree.gpm','--output=mpi-adjusted.gpm','--append-elements','He','--new-species-init=clone','--clone-from','0','--basis-order=level-group')
    assert result.returncode==0,result.stdout+result.stderr
    np.testing.assert_allclose(labels(tmp_path,'mpi-adjusted.gpm','mpi.cfg'),expected,atol=1e-9,rtol=1e-8)
