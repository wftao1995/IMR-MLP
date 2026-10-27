#!/usr/bin/env python3
"""Independently reconstruct VCNEB enthalpy gradients and low saddle modes.

Uses the native potential only as a single-structure E/F/stress calculator;
coordinate differentiation is independent of the NEB implementation. It never
optimizes inputs, changes atom correspondence, or certifies a global mechanism.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'

import numpy as np
from ase.io import read
from scipy.linalg import null_space
from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('vc_oracle', ROOT/'scripts/validation/vcneb_oracle.py')
oracle_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle_module)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def read_settings(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            key, value = line.split(None, 1)
            result.setdefault(key, value)
    return result


def registry(frames, mapping):
    groups = np.asarray(mapping['ql_ids'])
    sublayers = np.asarray(mapping['sublayer_ids'])
    if len(groups) != len(frames[0]) or list(mapping['symbols']) != frames[0].get_chemical_symbols():
        raise ValueError('Semantic mapping does not match atom ordering/species')
    rows = []
    for image, atoms in enumerate(frames):
        cell = atoms.cell.array
        centers = []
        for group in sorted(set(groups)):
            indices = np.flatnonzero((groups == group) & (sublayers == 2))
            if not len(indices):
                raise ValueError('Mapping lacks central Se sublayer')
            # Keep labeled trajectories continuous; do not individually wrap atoms.
            centers.append(atoms.positions[indices].mean(axis=0))
        centers = np.array(centers)
        gaps = np.roll(centers, -1, axis=0)-centers
        gaps[-1] += cell[2]
        inplane = gaps @ np.linalg.pinv(cell[:2])
        rows.append(dict(image=image, central_Se_centers_A=centers.tolist(),
                         adjacent_QL_vectors_A=gaps.tolist(),
                         registry_in_supercell_basis=inplane.tolist(),
                         periodic_registry=(inplane-np.rint(inplane)).tolist()))
    return {'coordinate_note': 'Current supercell a/b coefficients, not inferred A/B/C paper labels',
            'atom_mapping_sha256': hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest(),
            'images': rows}


def low_modes(oracle, saddle, output, max_products):
    y = oracle.encode(saddle)
    translations = np.zeros((len(y), 3))
    for j in range(3):
        translations[6+j::3, j] = 1/np.sqrt(len(saddle))
    Q = null_space(translations.T)
    count = min(8, Q.shape[1]-1)
    if count < 2:
        raise ValueError('Saddle has too few nontranslation degrees of freedom')
    records, vectors = [], []
    products = 0
    for h in [1e-4, 5e-5]:
        def product(direction):
            nonlocal products
            products += 1
            if products > max_products:
                raise RuntimeError('Hessian-vector product budget exhausted')
            full = Q@direction
            plus = oracle.evaluate(oracle.decode(y+h*full, saddle))['gradient']
            minus = oracle.evaluate(oracle.decode(y-h*full, saddle))['gradient']
            return Q.T@((plus-minus)/(2*h))
        operator = LinearOperator((Q.shape[1], Q.shape[1]), matvec=product, dtype=float)
        try:
            values, modes = eigsh(operator, k=count, which='SA', tol=1e-7,
                                  ncv=min(40, Q.shape[1]), maxiter=1000,
                                  v0=np.random.default_rng(20261021).normal(size=Q.shape[1]))
        except (ArpackNoConvergence, RuntimeError) as error:
            return dict(status='INDETERMINATE', reason=str(error), products=products)
        order = np.argsort(values);values=values[order];modes=modes[:, order]
        try:
            residuals = [float(np.linalg.norm(product(modes[:, j])-values[j]*modes[:, j]))
                         for j in range(count)]
            symmetry = []
            rng = np.random.default_rng(20261021)
            for _ in range(3):
                u, v = rng.normal(size=(2, Q.shape[1]));u/=np.linalg.norm(u);v/=np.linalg.norm(v)
                symmetry.append(float(abs(u@product(v)-v@product(u))))
        except RuntimeError as error:
            return dict(status='INDETERMINATE', reason=str(error), products=products)
        records.append(dict(step=h, eigenvalues=values.tolist(), eigenpair_residuals=residuals,
                            sampled_antisymmetry=symmetry))
        vectors.append(Q@modes)
    threshold = max(1e-4,
                    10*float(np.abs(np.array(records[0]['eigenvalues'])-records[1]['eigenvalues']).max()),
                    10*max(x for r in records for x in r['eigenpair_residuals']),
                    10*max(x for r in records for x in r['sampled_antisymmetry']))
    for record in records:
        values = np.array(record['eigenvalues'])
        record.update(negative_low_modes=int((values < -threshold).sum()),
                      uncertain_low_modes=int((np.abs(values) <= threshold).sum()))
    status = 'PASS' if all(r['negative_low_modes'] == 1 and r['uncertain_low_modes'] == 0 for r in records) else 'FAIL'
    if any(r['uncertain_low_modes'] for r in records):
        status = 'INDETERMINATE'
    np.savez_compressed(output/'low_modes.npz', modes_h=vectors[0], modes_h2=vectors[1], translations=translations)
    return dict(status=status, dimension=len(y), removed_translation_modes=3,
                checked_lowest_modes=count, threshold=threshold, products=products,
                steps=records, limitation='Low-spectrum local check; no global path minimality claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mapping', type=Path)
    parser.add_argument('--max-products', type=int, default=3000)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve();args.output.mkdir(parents=True, exist_ok=False)
    config = read_settings(args.run_dir/'neb.ini')
    for key in ['index', 'pot:potfile', 'pot:typemap']:
        if key in config and not Path(config[key]).is_absolute():
            config[key]=str(args.run_dir/config[key])
    meta, scales, _, _ = oracle_module.parse_summary(args.run_dir/config.get('summary', 'neb.dat'))
    frames = read(args.run_dir/config.get('mepout', 'mep.xyz'), ':')
    result = dict(status='NOT_RUN', numerical_status=meta.get('status'),
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 [Path(__file__), ROOT/'scripts/validation/vcneb_oracle.py']},
                  summary_sha256=hashlib.sha256((args.run_dir/config.get('summary', 'neb.dat')).read_bytes()).hexdigest())
    if args.mapping:
        dump(args.output/'registry.json', registry(frames, json.loads(args.mapping.read_text())))
    ci = int(meta.get('climbing_image', '-1'))
    if ci < 0:
        result.update(status='NOT_APPLICABLE', reason='No selected internal climbing image')
    elif (meta.get('status') != 'converged' or float(meta['max_neb_force']) > .003):
        result.update(status='INDETERMINATE', reason='CI requires converged generalized residual <=0.003')
    elif meta.get('symm_strain') != 'true' or meta.get('joint_metric') != 'vcneb-frobenius-v1':
        result.update(status='INDETERMINATE', reason='Verifier requires the symmetric VCNEB chart')
    else:
        from imrmlp import Potential
        from imrmlp.calculator import GPMCalculator
        elements = [line.split()[0] for line in Path(config['index']).read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith('#')]
        potential_settings={k[4:]: v for k, v in config.items() if k.startswith('pot:')}
        potential_settings.setdefault('typemap',config['index'])
        potential = Potential.from_settings(potential_settings)
        oracle = oracle_module.Oracle.__new__(oracle_module.Oracle)
        oracle.R = np.loadtxt(args.run_dir/'neb_reference_cell.data');oracle.Rinv=np.linalg.inv(oracle.R)
        oracle.scales=np.array([scales[j] for j in range(6)])
        oracle.recast=meta.get('recast_atomic_force') == 'true'
        oracle.p=float(meta['pressure_GPa'])/160.21766208
        oracle.calc=GPMCalculator(potential=potential, elements=elements);oracle.times=[];oracle.calls=0
        raw=oracle.evaluate(frames[ci])
        fmax=float(np.linalg.norm(raw['forces'], axis=1).max())
        smax=float(np.abs((raw['stress']+oracle.p*np.eye(3))*160.21766208).max())
        result.update(climbing_image=ci, max_force_eV_A=fmax, max_stress_residual_GPa=smax)
        checks=oracle_module.directional_checks(oracle, frames[ci])
        dump(args.output/'gradient_checks.json', checks)
        if fmax > .01 or smax > .02 or not checks['pass']:
            result.update(status='INDETERMINATE', reason='Independent physical stationarity/gradient checks failed')
        else:
            result['hessian']=low_modes(oracle, frames[ci], args.output, args.max_products)
            result['status']=result['hessian']['status']
        result.update(single_structure_efs=oracle.calls,
                      potential_sha256=hashlib.sha256(Path(config['pot:potfile']).read_bytes()).hexdigest())
    libraries={Path(line.split()[-1]) for line in Path('/proc/self/maps').read_text().splitlines()
               if '/libimrmlp' in line or '/_imrmlp.' in line}
    result['runtime_sha256']={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in libraries if p.is_file()}
    dump(args.output/'result.json', result)
    print(json.dumps(result, ensure_ascii=False))
    if result['status'] not in ['PASS', 'NOT_APPLICABLE']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
