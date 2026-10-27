#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT
"""Install the CPU MTP interfaces in a LAMMPS src directory without overwrites."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lammps_src', type=Path)
    parser.add_argument('--with-omp', action='store_true')
    parser.add_argument('--uninstall', action='store_true')
    args = parser.parse_args()
    target = args.lammps_src.resolve()
    if not all((target / name).is_file() for name in ('atom.h', 'pair.h')):
        parser.error('lammps_src must contain atom.h and pair.h')
    manifest = target / '.imrmlp-manifest.json'

    if args.uninstall:
        if not manifest.is_file():
            parser.error('no IMR-MLP installation manifest found')
        entries = json.loads(manifest.read_text())
        for name, expected in entries.items():
            if Path(name).name != name or name.startswith('.'):
                parser.error(f'unsafe manifest entry: {name}')
            path = target / name
            if path.is_symlink() or not path.is_file() or digest(path) != expected:
                parser.error(f'{name} changed or is missing; no files removed')
        for name in entries:
            (target / name).unlink()
        manifest.unlink()
        print(f'Removed IMR-MLP CPU interfaces from {target}')
        return

    if manifest.exists():
        parser.error('already installed; uninstall before updating')
    if args.with_omp and not any((target / name).is_file()
                                for name in ('thr_omp.h', 'OPENMP/thr_omp.h')):
        parser.error('--with-omp requires LAMMPS OPENMP sources')
    package = Path(__file__).resolve().parent
    sources = [package / 'pair_MLIP.cpp', package / 'pair_MLIP.h']
    for folder in ('opt', 'opt-les', 'transport', 'hyb-ti'):
        sources += sorted(p for p in (package / folder).iterdir()
                          if p.suffix in ('.cpp', '.h') and
                          (args.with_omp or '_omp.' not in p.name))
    sources += [package / name for name in
                ('mlip_pair_metadata.h', 'mlip_legacy_reader.h', 'mlip_model_mpi.h', 'mlip_restart.h')]
    core = package.parents[1] / 'runtime/mtp'
    sources += [core / name for name in
                ('mlip.cpp', 'mlip.h', 'radial_basis.h', 'radial_cache.h',
                 'radial_basis_setup.cpp', 'cpu/mtp_cpu.cpp', 'cpu/mtp_cpu.h',
                 'gpu-direct/radial_gpu_math.h', 'gpu-direct/mtp_gpu_direct.h',
                 'gpu-direct/gpu_backend.h', 'gpu-direct/sog_gpu.h', 'gpu-direct/gpu_cache.h')]
    sources += [package.parents[1] / 'tools' / name for name in ('mtp_metadata.h', 'picojson.h', 'gpm_container.h', 'gpm_mtp.h')]
    sources += [package.parents[1] / 'potentials' / name
                for name in ('sog_model.h', 'reciprocal_phases.h')]
    sources += [package.parent / 'COPYING.GPL-2.0']
    for source in sources:
        dest = target / source.name
        if dest.exists() or dest.is_symlink():
            parser.error(f'{dest} already exists; no files copied')
        if not source.is_file():
            parser.error(f'missing source: {source}')
    # Validate all paths before installing, and retain hashes for guarded removal.
    for source in sources:
        shutil.copyfile(source, target / source.name)
    manifest.write_text(json.dumps({p.name: digest(p) for p in sources}, indent=2) + '\n')
    print(f'Installed {len(sources)} files in {target}; reconfigure and rebuild LAMMPS.')
    print('Add the Eigen include directory to CMAKE_CXX_FLAGS (for example -I/usr/include/eigen3).')
    if args.with_omp:
        print('Enable PKG_OPENMP=ON in the LAMMPS build.')


if __name__ == '__main__':
    main()
