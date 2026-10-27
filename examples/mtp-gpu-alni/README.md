# Al-Ni MTP GPU fitting example

[中文](README_zh.md)

This example uses an L1₂-Ni3Al structure to demonstrate GPU energy, force and
stress evaluation together with fitting to synthetic labels. Supply your own
Al/Ni MTP model whose element order matches `index.map`.

## Files

- `POSCAR.L12-Ni3Al`: reference structure.
- `POSCAR.L12-Ni3Al-displaced`: displaced structure used to generate the
  reference labels.
- `index.map`: model element order, Al then Ni.
- `train/fit-gpu.ini`: CUDA fitting settings.
- `train/make_synthetic_training_set.py`: multiplies the reference E/F/S labels
  by 1.02 so the fit has a non-zero loss.
- `train/submit-gpu.slurm`: runs labeling and fitting in a separate run
  directory.

## Running

Build the CUDA program from the repository root:

```bash
make -j8 BUILD_MODE=cuda CUDA_HOME=/path/to/cuda CUDA_ARCH=sm_80 imrmlp
```

Select `CUDA_ARCH` for your GPU, using an nvcc architecture name such as
`sm_80`. The output is `build/cuda/imrmlp-cu`. Export
the program and model paths, enter this example directory, then adjust the
partition, node and GPU resources in the Slurm script for your cluster:

```bash
export GPMAIN="$PWD/build/cuda/imrmlp-cu"
export MTP_POTENTIAL=/absolute/path/to/alni.mtp
cd examples/mtp-gpu-alni
```

Submit with:

```bash
sbatch train/submit-gpu.slurm
```

The script first computes reference labels with `calc_efs --device=cuda`, then
builds the synthetic training set and finally runs `trainnlpot --do-lbfgs`.
Each run is written to `MTP_RUN_ROOT/train-JOB_ID`; the default run root is
`mtp-gpu-runs` under the submission directory. `fit-gpu.ini` limits training to
three iterations; inspect `train.log` and the saved model rather than treating
a successful job as a converged fit. This supplied script targets CUDA; HIP
requires a HIP build and corresponding device settings in both commands.

The synthetic labels only validate the evaluation and training path. A real fit
needs separate labeled training and validation sets. See the
[GPU development guide](../../docs/development/cuda.md) for the backend
interface and settings, and the [GPU runtime](../../runtime/mtp/gpu-direct/README.md)
for LAMMPS deployment.
