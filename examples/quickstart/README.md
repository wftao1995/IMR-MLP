# A self-contained MTP workflow

[中文](README_zh.md) · [Build prerequisites](../../docs/guides/installation.md)

This example generates 20 three-atom configurations and labels them with a
small synthetic MTP: 16 for training and four held out for validation. The
species label `Al` demonstrates mapping only. Neither the reference nor the
fitted model describes real aluminium.

## Run

Complete the CPU build and runtime setup in the [installation guide](../../docs/guides/installation.md),
then run from the repository root. If installation already created a successful
`build/quickstart/`, skip to the next section to inspect it:

```sh
export PATH="$PWD/build/release:$PATH"
export LD_LIBRARY_PATH="$PWD/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 examples/quickstart/run.py
```

The script uses Python's standard library; the Python extension, ASE, LAMMPS
and external datasets are not required. It refuses to overwrite a nonempty
output directory. To choose another CLI or output location:

```sh
python3 examples/quickstart/run.py \
  --imrmlp /absolute/path/to/imrmlp --work /tmp/imrmlp-demo
```

The default output directory is `build/quickstart/` under this checkout.

## What runs

The script prepares `reference.mtp`, `template.mtp`, `index.map`, input
configurations and `fit.ini`, then runs the following commands **from the
output directory**. Here `imrmlp` means the executable selected above:

```sh
imrmlp calc_efs train-input.cfg train.cfg --potfile reference.mtp --index index.map
imrmlp calc_efs valid-input.cfg valid.cfg --potfile reference.mtp --index index.map
imrmlp trainmtp fit.ini --linear-fit
cp _Trained_Potential_.mtp fitted.mtp
imrmlp calc_err --input valid.cfg --potfile fitted.mtp --index index.map --output-summary errors.txt
imrmlp calc_efs valid-input.cfg predicted.cfg --potfile fitted.mtp --index index.map
```

This fits the linear coefficients while keeping the radial basis fixed.
The starting linear coefficients differ from the reference coefficients.
`trainmtp` always writes its final snapshot to `_Trained_Potential_.mtp` in
the working directory; `fit:save_to` controls trainer checkpoints separately.
The script copies the final snapshot to `fitted.mtp` explicitly.

Each stage prints `OK`. Inspect `errors.txt` and the per-stage `.log` files.
For this exact synthetic fit, the printed E/F/stress RMS errors should be
near numerical precision (the report prints six decimal places). Real data
will have finite fitting errors; a small training error is not a validation
of transferability.

| File | Source / purpose |
| --- | --- |
| `train-input.cfg`, `valid-input.cfg` | Generated 16/4-frame geometries without reference labels |
| `train.cfg`, `valid.cfg` | Training/validation data labeled with synthetic `reference.mtp`; do not substitute unlabeled inputs |
| `template.mtp`, `fit.ini`, `index.map` | Starting model, runnable training settings and species map |
| `fitted.mtp` | Fitted model for subsequent CLI/Python/LAMMPS exercises |
| `predicted.cfg`, `errors.txt` | Four validation predictions and their errors against `valid.cfg` |
| Per-stage `.log` files | First place to inspect when a script stage fails |

All files are inside `--work`. Enter that directory before running the manual
commands above; substitute your actual path for `build/quickstart/` on other
pages if you chose a custom output directory.

## Nonlinear fitting and next steps

To try ALGLIB L-BFGS from the same template, run in the generated directory:

```sh
imrmlp trainmtp fit.ini --do-lbfgs
```

This overwrites the default final snapshot; copy it to a distinct filename
before comparing runs. The main [training guide](../../docs/guides/workflows.md)
explains optimizer selection, data splits and model deployment. See the
[Python guide](../../README-python.md) to evaluate `fitted.mtp` from Python/ASE,
and the [LAMMPS guide](../../LAMMPS/README.md) for `pair_style mlip/opt`.

To practice finding new geometries to label, continue with the self-contained
[MaxVol selection exercise](../active-learning/README.md), which prints grades
and saves candidates from a generated exploration pool.
For relaxation, NEB and properties, use the [calculation walkthrough](../basic-calculations/README.md),
which supplies a separate LJ potential and all inputs for a stable teaching crystal.
For basis selection, fitting/validation and calculations in one MTP route,
use the [MTP walkthrough](../mtp-workflow/README.md), whose trained model supports
the teaching crystal and vacancy tasks directly.
