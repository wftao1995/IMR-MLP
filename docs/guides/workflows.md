# Training, evaluation and deployment {#guide_workflows}

[中文](workflows_zh.md)

Train an ordinary MTP on your own reference data, split validation and final test
sets, evaluate, and deploy the same model to the CLI, LAMMPS or Python.

Out of scope: energy, relaxation, path and property calculations with an existing
model (see the [single-command calculation](single-point.md) pages) and generating
or trimming a basis pool (see [basis selection](basis-selection.md)).

## Required inputs

| File | Content | Source |
| --- | --- | --- |
| `index.map` | Model species order | Convention described below |
| `reference.cfg` etc. | Reference data with E/F/stress labels | Reference calculations converted with `convstr` |
| `template.mtp` | Template model with the intended species, radial basis, cutoff and moment basis | Prepare it yourself or take it from an example |
| `train.xyz`, `valid.xyz`, `test.xyz` | Split training, validation and final test sets | Split from the reference data |

The commands assume `imrmlp` is on `PATH`, or has been replaced with the absolute
path to the intended build. Run each calculation in its own working directory;
relative paths in settings refer to the process working directory, see
[input files and settings conventions](conventions.md).

## Species map and reference labels

Create `index.map` with one element symbol per line, in the model's species order.
For an Al/Ni model whose indices are 0 and 1:

```text
Al
Ni
```

Use the same order in all preprocessing, training and deployment steps. LAMMPS
atom types start at 1, while numeric model species indices start at 0. New LAMMPS
inputs should prefer element-name mappings. An incorrect map can produce plausible
but physically wrong results.

Convert each reference calculation explicitly, then concatenate CFG records:

```sh
imrmlp convstr OUTCAR reference.cfg --index index.map
imrmlp convstr vasprun.xml reference-xml.cfg --index index.map
cat reference.cfg reference-xml.cfg > dataset.cfg
```

These filenames are placeholders for user-supplied reference calculations.
Geometry-only trajectories such as XDATCAR do not supply energy, force or stress
labels merely by conversion. Verify electronic convergence and label units in the
reference calculations; do not confuse ionic stopping criteria with
electronic/force accuracy.

Positions and cell vectors use Å, total energy uses eV, and force uses eV/Å. The
native extensive stress representation is in eV; `calc_err` additionally reports
volume-normalized stresses in GPa. Preserve the documented IO conventions when
importing labels from another code.

## Dataset split

Hold out structures by trajectory, phase, composition or generating calculation
where possible. Adjacent MD frames should not be treated as independent training
and test samples. Use validation to choose models and settings, and reserve a final
test set for evaluation after those choices. One command produces 80% / 10% / 10%:

```sh
imrmlp split_set dataset.cfg split_ --ratios=8,1,1 --shuffle yes
cp split_0.xyz train.xyz
cp split_1.xyz valid.xyz
cp split_2.xyz test.xyz
```

`split_set` writes extxyz files. Ratio mode takes consecutive segments after the
optional shuffle, with integer subset sizes rounded by largest remainder; without
`--shuffle`, adjacent structures in the file stay together in one subset, matching
the requirement to hold out whole trajectories. To obtain N equally sized subsets
instead (parallel labelling, per-shard inspection), pass `--nsplit N` without
`--ratios`: it assigns round-robin and defaults to 10.

The CLI has no seed option; retain the actual split files for reproducibility. Do
not rename extxyz files to `.cfg` and rely on extension-based format detection.

The self-contained quickstart provides only training and validation CFGs, not a
prepared final test set; it verifies the workflow rather than replacing the split
described here.

## Template selection

Start from a template with the intended species count and order, radial basis,
cutoff and moment basis, and keep an untouched copy. Generating a pool with
`mtmbasis` and trimming it with `fastfit` is covered in
[basis selection](basis-selection.md); generator and radial-basis settings are in
the [configuration reference](../reference/configuration.md). Generation settings
are not a substitute for a complete deployable model.

A template does not have to be built from scratch.
[MTM-new](https://github.com/wftao1995/MTM-new) is a set of optional preset bases
arranged by size; each directory holds an already optimised `potential.mtp_0` in the
format `mtmbasis` writes — one species, `RBChebyshev`, 2–5 Å, 8 radial basis
functions, no fitted coefficients.

```sh
git clone https://github.com/wftao1995/MTM-new
```

Copy a directory of suitable size to `template.mtp` and it becomes the training
starting point. Directories come in two kinds:

| Directory | Contents |
| --- | --- |
| `ps-1/`, `ps-2/` | Results under two contraction-search parameter sets; the second number in a directory name equals `alpha_scalar_moments` in the file |
| `shapeev/N` | Shapeev's original basis at level N, with `max_lev = N` in `settings.ini` |

These files declare `species_count = 1`. Multi-element systems require editing
that line on an untrained copy and keeping the order in `index.map` consistent.
The `settings.ini` in each directory holds `mtmbasis` contraction-search settings,
not training settings. The scalar count in a directory name measures basis size,
not accuracy: `shapeev/30` carries about 4000 scalars, and its cost has to be
measured with a `fastfit` baseline.

`initmtpm` initializes the MTP-M nonlinear-head model; it is not a required
initialization step for ordinary MTP training. Do not use it blindly on a standard
template.

## Training settings and commands

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

The weights are illustrative, relative to the default energy weight of 1. Choose
them according to label quality, normalization and target observables.

```sh
# ALGLIB L-BFGS: optimize radial and linear coefficients
imrmlp trainmtp train.ini --do-lbfgs
cp _Trained_Potential_.mtp fitted.mtp

# Alternative: refit only the linear coefficients with fixed radial functions
imrmlp trainmtp train.ini --linear-fit
cp _Trained_Potential_.mtp fitted-linear.mtp
```

**Select an optimizer or `--linear-fit` explicitly**: a bare `trainmtp train.ini`
does not select a nonlinear optimization method. The final snapshot is always
written as `_Trained_Potential_.mtp` in the working directory; `fit:save_to` names
trainer checkpoints, not this command-level output. Separate run directories
prevent overwriting earlier models.

Optional linear refits during nonlinear optimization are described in the
[linear-refit guide](../advanced/linear-refit.md). `trainnlpot` supports additional
model and device paths, while `train_linear` is a separate generic linear fitting
command. These commands accept different training settings; check the selected
command's `--help` before reusing a configuration.

## Evaluation and provenance

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
phases, compositions and target properties. Check the model on the intended range
of volumes, temperatures and defects. Validation E/F errors alone do not establish
stable long MD trajectories. Retain the mapping, settings, model, split files,
logs, build mode and dependency versions.

### Reading an error report

In the quickstart `errors.txt`, energy statistics cover four validation
configurations and force statistics cover twelve atoms. Errors approach numerical
precision because the chosen model can represent this synthetic problem exactly;
real material errors should not be expected to vanish. Check sample counts and
label completeness before interpreting the statistics.

An `Energy per atom` RMS of `0.004 eV` means `4 meV/atom`, and a `Forces` RMS of
`0.08 eV/Å` means `80 meV/Å`. `Forces` summarizes force-error vectors, whereas
`Forces components` summarizes components; compare the same row on the same
validation set across models. Low averages with a large maximum call for
inspecting the responsible phase, defect or high-temperature structures and
deciding whether more reference data are needed; lower training errors do not
replace this check.

## Deployment

### LAMMPS

First [install the interfaces](../../LAMMPS/README.md), then use:

```text
units metal
newton on
pair_style mlip/opt
pair_coeff * * fitted.mtp Al Ni
```

The example maps two LAMMPS atom types to model species Al/Ni; a one-species Al
model takes only `Al`. Older models without element metadata need verified numeric
species mappings. Place this fragment after box and atom setup; the interface guide
has a complete input. Use `run 0` first and compare energy and forces with CLI
predictions on the same geometry, cell, map and boundary conditions. Choose
`mlip/spline` only for spline models; LES/SOG has dedicated styles. Heat-current
observables require the transport pair styles.

### Python and ASE

Use the [Python interface](../reference/python.md) and an explicit `elements` list
in model order. The active-learning selection cycle is described in
[active learning](active-learning.md); [MaxVol theory](../theory/maxvol.md) explains
the method. Selection grades describe extrapolation relative to a chosen model and
active set; they are not a calibrated error estimate by themselves.

## Common errors

| Symptom | Cause | Action |
| --- | --- | --- |
| Training finishes without the expected iterations | `--do-lbfgs` or `--linear-fit` was not given | Select the optimization mode explicitly; a bare command does not choose one |
| The model file you named is missing | The final snapshot is always `_Trained_Potential_.mtp` | Copy and rename after training, or use a separate run directory |
| Training errors are low but calculations deviate | Species map error or leakage across splits | Check `index.map` and split by trajectory, phase or composition |
| Validation error suddenly improves | The validation set was reused while tuning | Report final metrics on `test.xyz`, which takes no part in selection |
| Format detection fails | extxyz renamed to `.cfg` | Keep the extension or specify the format explicitly in the settings |
| LAMMPS energy disagrees with the CLI | Type order, mapping or boundary conditions differ | Compare with `run 0` under identical conditions |

## Related

- [Basis selection](basis-selection.md), [fastfit cost selection](../advanced/fastfit-cost-selection.md)
- [HVP labels and fitting](../advanced/hvp-training.md), [MTP variable projection](../advanced/mtp-native-varpro.md)
- Calculations after training: [single-point](single-point.md), [relaxation](relaxation.md), [NEB](neb.md), [crystal properties](properties.md)
