#!/usr/bin/env python3
"""Check a registered harmonic/reference style against a two-atom spring oracle."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lmp', type=Path)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--mpiexec', default='mpiexec')
    args = parser.parse_args()
    command = [str(args.lmp.resolve())]
    if args.ranks > 1:
        command = [args.mpiexec, '-np', str(args.ranks)] + command
    env = dict(os.environ)
    env.setdefault('OMP_NUM_THREADS', '2')
    with tempfile.TemporaryDirectory(prefix='harmonic-reference-') as tmp:
        work = Path(tmp)
        reference = '''TI_HARMONIC_REFERENCE 1
units metal
natoms 2
energy0 -1
cell 10 0 0 0 10 0 0 0 10
atoms
1 1 2 2 2
2 1 7 2 2
blocks 3
1 1 2 0 0 0 2 0 0 0 2
1 2 -2 0 0 0 -2 0 0 0 -2
2 2 2 0 0 0 2 0 0 0 2
end
'''
        (work / 'reference').write_text(reference)
        setup = '''units metal
atom_style atomic
boundary p p p
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 2.1 2 2
create_atoms 1 single 7 2 2
mass 1 1
'''
        count = 0
        for mode in ('auto', 'replicated', 'distributed'):
            for scale in (1.0, 0.25):
                for translation in (0, 9):
                    pair = f'pair_style harmonic/reference reference {mode}\npair_coeff * *\n'
                    if scale != 1:
                        pair = (f'pair_style hybrid/scaled {scale} harmonic/reference reference {mode} '
                                '0.75 zero 1.0\npair_coeff * * harmonic/reference\npair_coeff * * zero\n')
                    text = setup + pair + f'''displace_atoms all move {translation} 0 0 units box
compute ea all pe/atom
thermo_style custom step pe
thermo_modify norm no
run 0
variable energy equal pe
print "ORACLE_ENERGY $(v_energy:%.16g)" file energy
write_dump all custom forces id fx fy fz c_ea modify sort id
'''
                    (work / 'in.test').write_text(text)
                    result = subprocess.run(command + ['-in', 'in.test', '-log', 'none'],
                                            cwd=work, env=env, text=True, capture_output=True)
                    if result.returncode:
                        raise RuntimeError(result.stdout + result.stderr)
                    energy = float((work / 'energy').read_text().split()[-1])
                    assert math.isclose(energy, -.99 * scale, abs_tol=1e-12), energy
                    dump = (work / 'forces').read_text().split('ITEM: ATOMS')[1].splitlines()[1:]
                    for line, expected_force in zip(dump, (-.2, .2), strict=True):
                        _, fx, fy, fz, ea = map(float, line.split())
                        assert math.isclose(fx, expected_force * scale, abs_tol=1e-10), line
                        assert abs(fy) < 1e-12 and abs(fz) < 1e-12, line
                        assert math.isclose(ea, -.495 * scale, abs_tol=1e-10), line
                    count += 1
        # The parser must reject an off-diagonal block outside upper-triangle order.
        (work / 'reference').write_text(reference.replace('1 2 -2', '2 1 -2'))
        result = subprocess.run(command + ['-in', 'in.test', '-log', 'none'],
                                cwd=work, env=env, text=True, capture_output=True)
        assert result.returncode != 0 and 'unordered FC block' in result.stdout + result.stderr
        print(f'PASS: {count} energy/force/per-atom cases and invalid-block rejection; ranks={args.ranks}')


if __name__ == '__main__':
    main()
