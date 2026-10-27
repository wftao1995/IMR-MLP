# MTP + ZBL short-range coupling

[中文](README_zh.md)

This example evaluates the existing unary Al MTP together with a fixed ZBL
short-range correction. The structure, element map, and demonstration
potential are included in this directory.

After `make imrmlp` from the repository root, enter this directory and run:

```bash
../../build/release/imrmlp calc_efs --settings calc-efs.ini \
  --input POSCAR --output output-efs.cfg
```

Adjust the executable path if a different build mode is in use. The result is
written to `output-efs.cfg`.

The total energy is

```text
E_total = E_MTP + sum_{i<j} S(r_ij) V_ZBL(r_ij)
```

Here `S=1` at and below `r0=1.0 Angstrom`, changes smoothly from one to zero
between `r0` and `r1=2.0 Angstrom`, and is zero at and above `r1`. The supplied
Al crystal has no pair this short, so this input also checks that adding ZBL
does not perturb the normal-distance prediction.

For another material:

1. Replace `pot:pot2:potfile`, and set `index` and `pot:typemap` to your element map.
2. Set `pot:spcount` to the number of species.
3. List atomic numbers in `pot:pot1:z` in exactly the same order as `index.map`.
4. Choose `r0` and `r1` from pair-distance scans; always keep `0 < r0 < r1`.

If the fitting data contain pairs inside the switching interval, fit through
the same `hybrid` configuration with `trainnlpot`. That makes MTP learn the
DFT-minus-ZBL residual and avoids counting the short-range interaction twice.
The bundled potential is only a low-precision functional
test potential, not a production Al model.
