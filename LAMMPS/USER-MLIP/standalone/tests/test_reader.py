#!/usr/bin/env python3
"""Exercise the exported reader in isolated subprocesses, without IMR-MLP."""
import argparse
import json
import re
from pathlib import Path
import subprocess
import tempfile
from fixtures import HEADS, write_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',required=True,type=Path)
    parser.add_argument('--compatibility-root',action='append',default=[],type=Path,
                        help='Scan *.mtp fixtures recursively and record acceptance/diagnostics')
    parser.add_argument('--report',type=Path,help='Write fixture compatibility results as JSON')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='mtp-reader-') as temporary:
        root=Path(temporary); write_models(root)
        passed=0
        def check(name,text,good):
            nonlocal passed
            path=root/(name+'.mtp'); path.write_bytes(text if isinstance(text,bytes) else text.encode())
            run=subprocess.run([str(args.probe.resolve()),str(path)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=20)
            if (run.returncode==0)!=good: raise AssertionError(f'{name}: returncode={run.returncode}\n{run.stdout}')
            if not good and run.returncode!=1: raise AssertionError(f'{name}: reader crashed, rather than rejecting input\n{run.stdout}')
            passed+=1
        for head in HEADS:
            original=(root/(head+'.mtp')).read_text()
            check(head+'-valid',original,True)
            check(head+'-mvs',original.encode()+b'#MVS_v1.1\n\x00\xff\xfe\x01\n',True)
        base=(root/'q2.mtp').read_text()
        check('truncated',base.rsplit('model_coeffs =',1)[0]+'model_coeffs = {0.5,\n',False)
        check('nan',base.replace('scaling = 1','scaling = nan'),False)
        check('count-overflow',base.replace('alpha_moments_count = 2','alpha_moments_count = 2147483648'),False)
        check('index-out-of-range',base.replace('alpha_moment_mapping = {0, 1, 1}','alpha_moment_mapping = {0, 1, 2}'),False)
        check('undefined-node',base.replace('{{0, 0, 1, 1}, {0, 0, 1, 1}}','{{1, 0, 1, 1}, {0, 0, 1, 1}}'),False)
        check('coefficient-count',base.replace('"model_coeffs_count":5','"model_coeffs_count":6'),False)
        check('negative-unused-hidden',base+'# mtp_hidden1 = -1\n',False)
        check('oversized-unused-hidden',base+'# mtp_hidden2 = 2147483647\n',False)
        check('analytic-unused-positive-hidden',base+'# mtp_hidden1 = 1\n',False)
        check('nn1-unused-positive-hidden',(root/'n1.mtp').read_text()+'# mtp_hidden2 = 1\n',False)
        check('conflicting-footer',base+'# mtp_model = mtp_m_fs\n',False)
        check('legacy-les',base+'les_species_coeffs = {0.1,0.2}\nles_moment_coeffs = {0.1,0.2,0.3}\n',False)
        check('legacy-sog',base+'sog_weights = {0.1}\n',False)
        check('unknown-radial',base.replace('RBChebyshev','UnsupportedRadial'),False)
        start,end=base.index('radial_basis_type'),base.index('radial_funcs_count')
        def radial(name,extra='',size=2):
            return f'radial_basis_type = {name}\nmin_dist = 0.5\nmax_dist = 4.8\nradial_basis_size = {size}\n'+extra
        for name in ('RBPolyB','RBPolyBCos','RBChebyshevPoly4','RBChebyshevPoly5'):
            extra='rb_b = -0.7\n' if 'ChebyshevPoly' in name else ''
            check(name,base[:start]+radial(name,extra)+base[end:],True)
        single=base[:start]+radial('RBChebyshev',size=1)+base[end:]
        first,last=single.index('radial_coeffs'),single.index('alpha_moments_count')
        single=single[:first]+re.sub(r'\{([^{}]+)\}',lambda m:'{'+m.group(1).split(',')[0]+'}',single[first:last])+single[last:]
        check('single-basis',single,True)
        check('zero-basis',single.replace('radial_basis_size = 1','radial_basis_size = 0'),False)
        check('negative-min',base.replace('min_dist = 0.5','min_dist = -0.5'),False)
        check('duplicate-cache',base+'# standalone_radial_v1 = {}\n# standalone_radial_v1 = {}\n',False)
        check('unknown-cache-version',base+'# standalone_radial_v2 = {}\n',False)

        compatibility=[]
        paths=set()
        for directory in args.compatibility_root:
            if not directory.exists(): parser.error(f'Fixture root does not exist: {directory}')
            paths.update([directory] if directory.is_file() else directory.rglob('*.mtp'))
        for path in sorted(paths):
            run=subprocess.run([str(args.probe.resolve()),str(path.resolve())],stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,text=True,timeout=20)
            if run.returncode not in (0,1):
                raise AssertionError(f'Compatibility fixture crashed: {path}, returncode={run.returncode}\n{run.stdout}')
            compatibility.append({'path':str(path.resolve()),'accepted':run.returncode==0,'diagnostic':run.stdout.strip()})
        report={'reader_cases_passed':passed,'compatibility':compatibility}
        if args.report: args.report.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'reader_cases_passed':passed,'compatibility_scanned':len(compatibility),
                          'compatibility_accepted':sum(item['accepted'] for item in compatibility)}))

if __name__=='__main__': main()
