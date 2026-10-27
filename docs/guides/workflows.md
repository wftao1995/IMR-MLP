# Training, evaluation and deployment {#guide_workflows}

[中文](workflows_zh.md) · [Runnable quickstart](../../examples/quickstart/README.md)

@tableofcontents

This page is for training a model. If a suitable model is already available,
choose evaluation, relaxation or NEB in [getting started](getting-started.md)
without repeating training first.

**First training run with no data?** Complete the [self-contained quickstart](../../examples/quickstart/README.md)
and use its generated `fit.ini`, `train.cfg`, `valid.cfg` and `template.mtp`.
This page's three-way EXTXYZ workflow is for your reference data; the quickstart
has only training/validation CFG sets, not a separate final test set; keep its generated settings.

For basis generation → reference labeling → fitting/validation selection → final
testing → calculations with the same model, use the
[self-contained MTP walkthrough](../../examples/mtp-workflow/README.md), including all model/structure sources.

The commands below assume `imrmlp` is on `PATH`, or has been replaced with the
absolute path to the intended build. Run each calculation in its own working
directory. Relative paths in settings refer to the process working directory,
not automatically to the settings file's directory.

## 1. Establish a species map and labeled data

Create `index.map` with **one element symbol per line**, in the model's species
order. For an Al/Ni model whose indices are 0 and 1:

```text
Al
Ni
```

Use the same order in all preprocessing, training and deployment steps.
LAMMPS atom types start at 1, while numeric model species indices start
at 0. New LAMMPS inputs should prefer element-name mappings. An incorrect map can produce plausible but physically wrong results.

Convert each reference calculation explicitly, then concatenate CFG records:

```sh
imrmlp convstr OUTCAR reference.cfg --index index.map
imrmlp convstr vasprun.xml reference-xml.cfg --index index.map
cat reference.cfg reference-xml.cfg > dataset.cfg
```

These inputs are placeholders for user-supplied reference calculations.
Geometry-only trajectories such as XDATCAR do not supply energy, force or
stress labels merely by conversion. Verify electronic convergence and label
units in the reference calculations; do not confuse ionic stopping criteria
with electronic/force accuracy.

Positions and cell vectors use Å, total energy uses eV, and force uses eV/Å.
The native extensive stress representation is in eV; `calc_err` additionally
reports volume-normalized virial stresses in GPa. Preserve the documented IO
conventions when importing labels from another code.

## 2. Separate training, validation and final testing

Hold out structures by trajectory, phase, composition or generating calculation
where possible. Adjacent MD frames should not be treated as independent train
and test samples. Use validation to choose model/settings and reserve a final
test set for evaluation after those choices.

For a simple demonstration split:

```sh
imrmlp split_set dataset.cfg split_ --nsplit 10 --shuffle yes
cat split_0.xyz split_1.xyz split_2.xyz split_3.xyz split_4.xyz \
    split_5.xyz split_6.xyz split_7.xyz > train.xyz
cp split_8.xyz valid.xyz
cp split_9.xyz test.xyz
```

`split_set` writes extxyz files. Equal mode assigns structures round-robin after
the optional shuffle. Alternatively, create approximately 80% / 10% / 10% subsets
directly in a separate run directory:

```sh
imrmlp split_set dataset.cfg split_ --ratios=8,1,1 --shuffle yes
cp split_0.xyz train.xyz
cp split_1.xyz valid.xyz
cp split_2.xyz test.xyz
```

Ratio mode takes consecutive segments after any shuffle; integer subset sizes
use largest-remainder rounding. The CLI has no seed option; retain the actual split files
for reproducibility. Do not rename extxyz files to `.cfg` and rely on extension
based format detection.

## 3. Choose a compatible MTP template

Start from a template with the intended species count/order, radial basis,
cutoff and moment basis. Keep an untouched copy. The
[quickstart](../../examples/quickstart/README.md) ships a tiny synthetic template
for learning the workflow. The `mtmbasis` generator and radial-basis settings
are documented in [configuration](../reference/configuration.md); generation settings are
not a substitute for a complete deployable model.

`initmtpm` initializes the MTP-M nonlinear-head model; it is not a required
initialization step for ordinary MTP training. Do not use it blindly on a
standard template.

## 4. Train explicitly

Save `train.ini` using whitespace-separated values:

```text
index index.map
potfile template.mtp
trainset train.xyz
validset valid.xyz
fit:force_weight 0.05
fit:stress_weight 0.001
fit:weight_L2 1e-8
fit:save_to checkpoint.mtp
```

The weights are illustrative, relative to the default energy weight of 1.
Choose them according to label quality, normalization and target observables.

```sh
# ALGLIB L-BFGS: optimize radial and linear coefficients
imrmlp trainmtp train.ini --do-lbfgs
cp _Trained_Potential_.mtp fitted.mtp

# Alternative: refit only the linear coefficients with fixed radial functions
imrmlp trainmtp train.ini --linear-fit
cp _Trained_Potential_.mtp fitted-linear.mtp
```

**Select an optimizer or `--linear-fit` explicitly.** A bare `trainmtp
train.ini` does not select a nonlinear optimization method. `trainmtp` writes
its final snapshot as `_Trained_Potential_.mtp` in the working directory.
`fit:save_to` names trainer checkpoints, not this final command-level output.
Separate run directories prevent overwriting earlier models.

Optional linear refits during nonlinear optimization are described in the
[linear-refit guide](../advanced/linear-refit.md). `trainnlpot` supports additional
model and device paths, while `train_linear` is a separate generic linear fitting
command. These commands accept different training settings; check the selected
command's `--help` before reusing a configuration.

## 5. Validate and retain provenance

```sh
imrmlp calc_err --input valid.xyz --potfile fitted.mtp --index index.map \
  --output-summary validation.txt
imrmlp calc_err --input test.xyz --potfile fitted.mtp --index index.map \
  --output-summary test.txt
imrmlp calc_efs test.xyz predicted.xyz --potfile fitted.mtp --index index.map
imrmlp --version > version.txt
sha256sum fitted.mtp template.mtp train.xyz valid.xyz test.xyz > inputs.sha256
```

Report energy per atom, force and stress errors separately; inspect outliers,
phases/compositions and target properties. Check the model on the intended
range of volumes, temperatures and defects. Validation E/F errors alone do
not establish stable long MD trajectories. Retain the mapping, settings,
model, split files, logs, build mode and dependency versions.

### Small example: read an error report

In the [self-contained quickstart](../../examples/quickstart/README.md),
`errors.txt` should count four validation configurations for energy and twelve
atoms for forces. Errors approach numerical precision because the chosen model
can represent this synthetic problem exactly; real material errors should not
be expected to vanish. Check sample counts and label completeness before
interpreting the statistics.

For example, an `Energy per atom` RMS of `0.004 eV` means `4 meV/atom`, and
a `Forces` RMS of `0.08 eV/Å` means `80 meV/Å`. `Forces` summarizes force-error
vectors, whereas `Forces components` summarizes components. Compare the same
row on the same validation set across models. Low averages with a large maximum
call for inspecting the responsible phase, defect or high-temperature structures
and deciding whether more reference data are needed; lower training errors do
not replace this check.

## 6. Calculate with the fitted model

Start with a [single-point calculation](getting-started.md) and compare E/F/stress
on the same geometry. Then choose the target workflow:

- [Structure relaxation](relaxation.md): fixed/variable cells, trajectories, exit codes and convergence.
- [NEB paths](neb.md): endpoints, atom correspondence, initialization, barriers and restart.
- [Crystal properties](properties.md): phonons, elasticity, EOS and numerical convergence.

Average validation errors do not replace stability and property checks under
the target calculation conditions.

## 7. Deploy the same model

For LAMMPS, first [install the interfaces](../../LAMMPS/README.md), then use:

```text
units metal
newton on
pair_style mlip/opt
pair_coeff * * fitted.mtp Al Ni
```

The example maps two LAMMPS atom types to model species Al/Ni. A one-species
Al model takes only `Al`. Older models without element metadata need verified
numeric species mappings. Place this fragment after box/atom setup; the interface
guide has a complete input. Use `run 0` first and compare energy/forces with CLI
predictions on the same geometry, cell, map and boundary conditions. Choose
`mlip/spline` only for spline models; LES/SOG has dedicated styles.
Heat-current observables require the transport pair styles.

For Python/ASE, use the [Python guide](../reference/python.md) and an explicit `elements`
list in model order. Start with the [active-learning cycle](active-learning.md);
[MaxVol theory](../theory/maxvol.md) explains the method.
Selection grades describe extrapolation relative to a chosen model/active set;
they are not a calibrated error estimate by themselves.

### Transition paths and variable-cell calculations

Start paths with the [NEB guide](neb.md); variable cells and other saddle methods
are in the [stationary-point guide](../advanced/saddle-workflows.md). Prepare and
relax the endpoints under the same pressure and constraints before constructing
a path. Converge the image count and check the saddle independently; a small
projected NEB force alone does not establish a transition state.

## Additional fitting and basis selection

- [fastfit cost selection](../advanced/fastfit-cost-selection.md)
- [HVP labels and fitting](../advanced/hvp-training.md)
- [MTP variable projection](../advanced/mtp-native-varpro.md)
