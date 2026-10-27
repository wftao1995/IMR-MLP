#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: GPL-2.0-only
# Adapted for the IMR-MLP public source layout, 2026-09-22.

"""Build the regression runner against an existing static CMake LAMMPS build.

The CMake build must use Unix Makefiles (link.txt), MPI and a static liblammps.a.
No files in the supplied LAMMPS source/build directories are changed.
"""
import argparse
from pathlib import Path
import shlex
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--lammps-source', required=True, type=Path, help='LAMMPS src directory')
parser.add_argument('--lammps-build', required=True, type=Path)
parser.add_argument('--work', required=True, type=Path)
parser.add_argument('--mpicxx', default='mpicxx')
parser.add_argument('--all-cpu', action='store_true',
                    help='Also compile standard, optimized, spline, OPENMP and transport styles')
args = parser.parse_args()
root = Path(__file__).resolve().parents[4]
base = Path(__file__).resolve().parent.parent
work = args.work.resolve()
work.mkdir(parents=True, exist_ok=True)
build = args.lammps_build.resolve()
link = shlex.split((build/'CMakeFiles/lmp.dir/link.txt').read_text())
libraries = link[link.index('liblammps.a'):]
libraries = [str(build/x) if not x.startswith('-') and (build/x).exists() else x for x in libraries]
command = [args.mpicxx, '-O2', '-std=c++17', '-I'+str(args.lammps_source.resolve()),
           '-I'+str(base), '-I'+str(base.parent), '-I'+str(root/'tools'), '-I'+str(root/'potentials'), str(base/'tests/lammps_runner.cpp'),
           str(base/'pair_MLIP_opt_les.cpp'), str(base/'compute_mlip_les_charge.cpp'),
           '-o', str(work/'lmp-sog')] + libraries
if args.all_cpu:
    package = base.parent
    sources = [package/'pair_MLIP.cpp', root/'runtime/mtp/mlip.cpp',
               root/'runtime/mtp/radial_basis_setup.cpp', root/'runtime/mtp/cpu/mtp_cpu.cpp']
    sources += sorted((package/'opt').glob('*.cpp'))
    sources += sorted((package/'transport').glob('*.cpp'))
    command[1:1] = ['-DIMRMLP_TEST_ALL_CPU', '-fopenmp', '-I/usr/include/eigen3',
                    '-I'+str(root/'runtime/mtp'),
                    '-I'+str(args.lammps_source.resolve()/'OPENMP'),
                    '-I'+str(root/'tools'), '-I'+str(package), '-I'+str(package/'opt'),
                    '-I'+str(package/'transport')] + [str(p) for p in sources]
with (work/'build.log').open('w') as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

# Reuse the checkout's compiler/ABI/dependency flags for the IMRMLP reference.
recipe = subprocess.run(['make', '-n', '-B', 'build/release/unitest/ut_sog.test'],
                        cwd=root, capture_output=True, text=True, check=True)
line = next(line for line in recipe.stdout.splitlines() if ' unitest/ut_sog.cpp ' in line)
command = shlex.split(line)
command = [str(base/'tests/generate_reference.cpp') if x == 'unitest/ut_sog.cpp' else x
           for x in command if x not in ('-lCatch2Main', '-lCatch2', 'build/release/unitest/main_init.o')]
command[command.index('-o')+1] = str(work/'reference')
with (work/'reference-build.log').open('w') as log:
    subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
print(f'Built {work}/lmp-sog and {work}/reference')
