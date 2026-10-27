# MTP Walkthrough: Basis Generation, Training, Validation and Calculations

[中文](README_zh.md) · [CPU installation](../../docs/guides/installation.md) · [LJ calculation exercise](../basic-calculations/README.md)

Generate two ordinary MTP candidate bases, fit synthetic reference data, select
by validation errors and use **the same trained MTP** for relaxation, NEB, force
constants, elasticity and EOS. Only Python's standard library and the CPU CLI
are required, except for optional Phonopy postprocessing. No external model,
private dataset or Python extension is needed. All inputs originate in the
supplied [prepare.py](prepare.py).

This controlled exercise learns a known pair reference using an ordinary MTP's
**two-body subset** with fixed radial parameters. It is not a general many-body
metal potential or an aluminium model. `Al` supplies a single-species label and
mass. The pair reference motivates choosing this restricted subset. Complete
the workflow before adapting it to your own reference calculations.

## 1. Where do inputs and reference labels come from?

| Item | Source and purpose |
| --- | --- |
| `reference/force-shifted-lj.table` | Generated energy table of a force-shifted LJ function; CLI interpolation provides reference E/F/stress, **not DFT labels** |
| `train-input.cfg`, `valid-input.cfg`, `test-input.cfg` | Independently generated geometry using seeds 1701, 1702 and 1704, without reference labels |
| `basis-8/`, `basis-16/` | Basis-generation and training settings; `mtmbasis` generates topology and fitting supplies coefficients |
| `calculations/` | Separately generated four-atom FCC, 31-atom vacancy endpoints and task settings; trained models are deployed later |
| `provenance.json` | Reference parameters, table grid, seeds, structure counts, units and candidate-basis records |

The reference uses `epsilon=0.1 eV`, `sigma=2.5 Å` and `rc=6 Å`:

@f$ U(r)=4\epsilon[(\sigma/r)^{12}-(\sigma/r)^6] @f$ and
@f$ \phi(r)=U(r)-U(r_c)-(r-r_c)U'(r_c) @f$ for @f$r<r_c@f$, zero outside.
Energy and its first derivative vanish at the cutoff; the second derivative
need not be continuous there. The table spans 1.8–6 Å in 0.001 Å increments,
4201 points; native table interpolation supplies derivatives. This differs
from the LJ exercise's directly truncated `lj612`, so do not reuse its numbers.

| Dataset | Dimers | Perturbed FCC, 4 atoms | Perturbed vacancy paths, 31 atoms | Total frames |
| --- | --- | --- | --- | --- |
| Training | 60 | 24 | 16 | 100 |
| Validation | 16 | 8 | 4 | 28 |
| Final test | 16 | 8 | 4 | 28 |

Dimer separations span 2.1–5.95 Å. Training uses 60 equally spaced dimer distances
including both interval ends; validation/test distances are random. FCC lattice
parameters span 3.72–4.08 Å, with displacements up to ±0.04 Å per component.
Vacancy structures remove the
origin atom from 2×2×2 FCC and move a neighbor into that site, using lattice
parameters 3.84–3.96 Å, random path fractions and ±0.02 Å component perturbations.
All splits cover the same controlled structural families: they test interpolation,
not extrapolation to unseen phases, temperatures or elements.

## 2. Prepare environment, geometry and settings

Complete installation and oneMKL setup, then run from the repository root containing `Makefile`:

```sh
imrmlp_root="$PWD"
export PATH="$imrmlp_root/build/release:$PATH"
export LD_LIBRARY_PATH="$imrmlp_root/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 examples/mtp-workflow/prepare.py
imrmlp_mtp="$imrmlp_root/build/mtp-workflow"
```

The script refuses a nonempty directory. For another run, add
`--work build/mtp-workflow-2` and update `imrmlp_mtp`. Re-enable the environment
and absolute path variables in a new terminal. Calculations below should return
`0`; if an immediate `echo $?` is nonzero, inspect the log before continuing.

## 3. Reference-label the three splits

```sh
cd "$imrmlp_mtp"
imrmlp calc_efs train-input.cfg train.cfg --potfile reference/force-shifted-lj.table \
  --pottype pair_table --index index.map > label-train.log 2>&1
echo $?
imrmlp calc_efs valid-input.cfg valid.cfg --potfile reference/force-shifted-lj.table \
  --pottype pair_table --index index.map > label-valid.log 2>&1
echo $?
imrmlp calc_efs test-input.cfg test.cfg --potfile reference/force-shifted-lj.table \
  --pottype pair_table --index index.map > label-test.log 2>&1
echo $?
```

Only the resulting `train.cfg`, `valid.cfg` and `test.cfg` have reference E/F/stress.
Splits were fixed before labeling. Training reads training and validation data;
reserve final test evaluation until model selection is complete. CFG `type=0`
means model species 0, corresponding to the first `index.map` entry, `Al`.

## 4. Generate and complete two basis templates

The generator is `mtmbasis`; the similarly named `mtpbasis` compares bases and
remaps coefficients rather than generating them.

```sh
cd "$imrmlp_mtp/basis-8"
imrmlp mtmbasis basis.ini > basis.log 2>&1
echo $?
cd "$imrmlp_mtp/basis-16"
imrmlp mtmbasis basis.ini > basis.log 2>&1
echo $?
python3 "$imrmlp_root/examples/mtp-workflow/prepare.py" \
  --work "$imrmlp_mtp" --complete-templates
```

Each directory first receives `potential.mtp_0`, a topology that is **not a
trained model**. The final command completes both as `template.mtp`, retaining
the originals; it does not label or fit. It checks expected scalar counts and
refuses to overwrite existing templates.

The 16-term candidate's `basis.ini` is:

```text
max_lev 32
max_dim 1
max_mu 15
max_nv 0
mode 3
seed 17
maxiter 100
```

The 8-term candidate uses `max_lev 16`, `max_mu 7`, otherwise identical settings.
These are **top-level keys**, without `pot:`. `max_nv=0` and `max_dim=1` retain
only `M(mu,0)`, producing 8 or 16 scalars. These counts are not those of a full
level-16/level-32 basis with angular and higher-body terms. `mode` and `maxiter`
control contraction-layout search, not the fitting optimizer or training steps.

Template completion sets one species, `RBChebyshev`, a 2–6 Å radial interval
and 8/16 radial basis functions. Radial coefficients form an identity matrix;
species offsets and scalar coefficients start at zero. Thus
@f$g_\mu(r)=(6-r)^2T_\mu((r-4)/2)@f$ inside the cutoff. Fixed radial functions
leave 9/17 linear parameters: one species offset and 8/16 scalar coefficients.
No reference-potential coefficients are embedded in the MTP. See
[MTP theory](../../docs/theory/mtp.md) and the
[configuration reference](../../docs/reference/configuration.md) for basis definitions
and general generator settings.

## 5. Fit, validate and choose the basis

Use each isolated directory's `fit.ini`, explicitly choosing linear fitting:

```sh
cd "$imrmlp_mtp/basis-8"
imrmlp trainmtp fit.ini --linear-fit > train.log 2>&1
echo $?
cp _Trained_Potential_.mtp fitted.mtp
imrmlp calc_err --input ../valid.cfg --potfile fitted.mtp \
  --index ../index.map --output-summary validation.txt
echo $?
cd "$imrmlp_mtp/basis-16"
imrmlp trainmtp fit.ini --linear-fit > train.log 2>&1
echo $?
cp _Trained_Potential_.mtp fitted.mtp
imrmlp calc_err --input ../valid.cfg --potfile fitted.mtp \
  --index ../index.map --output-summary validation.txt
echo $?
```

Both configurations use the same data and weights: `energy_weight=1`,
`force_weight=0.1`, `stress_weight=0.001`, `weight_L2=1e-12`, with random
initialization and radial orthogonalization disabled. Complete settings are
already supplied; do not substitute another trainer's configuration. The final
snapshot is `_Trained_Potential_.mtp`; `checkpoint.mtp` is a separate checkpoint.

For this teaching problem, prescribe validation energy RMS no greater than
1 meV/atom and force-vector RMS no greater than 0.001 eV/Å, choosing the smaller
candidate meeting both. Typical results are:

| Candidate | Linear parameters | Validation energy RMS (meV/atom) | Validation force RMS (eV/Å) | Passes? |
| --- | --- | --- | --- | --- |
| 8 scalars | 9 | About 8.03 | About 0.0952 | No |
| 16 scalars | 17 | About 0.024 | About 0.000324 | Yes |

Read **Energy per atom** and **Forces** RMS in `validation.txt`; the latter
differs from `Forces components`. Each report should count 28 structures and
188 atoms. These are exercise selection criteria, not universal material
accuracy requirements. If your results fail, inspect inputs/logs before
continuing rather than deploying without checking validation.

After choosing 16 terms, evaluate training and final test errors and deploy
one identical file:

```sh
cd "$imrmlp_mtp"
cp basis-16/fitted.mtp model.mtp
imrmlp calc_err --input train.cfg --potfile model.mtp \
  --index index.map --output-summary training.txt
echo $?
imrmlp calc_err --input test.cfg --potfile model.mtp \
  --index index.map --output-summary test.txt
echo $?
for task in relax neb fc elastic eos phonopy; do
  cp model.mtp "calculations/$task/model.mtp"
done
sha256sum model.mtp calculations/*/model.mtp
```

Hashes should match. The loop distributes the chosen model without refitting.
`test.txt` should likewise count 28 structures and 188 atoms: typical energy RMS
is 0.024 meV/atom and force RMS is 0.000297 eV/Å. Maximum force error is about
0.00142 eV/Å; passing RMS does not put every atom below the same threshold, so
inspect outliers too. Repeated tuning against test errors requires a new final
test set.

## 6. Relax with the trained MTP and prepare property inputs

```sh
cd "$imrmlp_mtp/calculations/relax"
imrmlp relax relax.ini > relax.log 2>&1
echo $?
imrmlp relax bulk.ini > bulk.log 2>&1
echo $?
imrmlp convstr bulk.xyz bulk.vasp --index index.map
echo $?
cp bulk.vasp ../fc/relaxed.vasp
cp bulk.vasp ../elastic/relaxed.vasp
cp bulk.vasp ../eos/relaxed.vasp
cp bulk.vasp ../phonopy/relaxed.vasp
```

The four-atom FCC starts at 3.9 Å, with its first atom displaced by
`(0.03, −0.02, 0.01) Å`. `relax.ini` optimizes atoms at fixed cell; `bulk.ini`
then reads `relaxed.xyz` and optimizes the zero-pressure cell. Final `bulk.vasp`
edges are near 3.90045 Å; all property inputs copy it. Relaxation exit `2` means
unconverged: stop and inspect as explained in the
[relaxation guide](../../docs/guides/relaxation.md).

## 7. Relax endpoints and run NEB with the same model

```sh
cd "$imrmlp_mtp/calculations/neb"
imrmlp relax initial.ini > initial.log 2>&1
echo $?
imrmlp convstr initial.xyz initial.vasp --index index.map
imrmlp relax final.ini > final.log 2>&1
echo $?
imrmlp convstr final.xyz final.vasp --index index.map
imrmlp neb neb.ini > neb.log 2>&1
echo $?
cat neb.dat
```

A separate 2×2×2 FCC cell loses its origin atom, leaving 31 atoms. Moving a
neighbor into the vacancy on the same atom row defines the final state, with
both endpoints in a common 7.8 Å cell. Relax endpoints first, then run seven-image
fixed-cell CI-NEB. Expect `converged`, full `max_neb_force <= 0.02 eV/Å` and
maximum `ent_diff` near 0.551 eV, with equal endpoint energies. `mep.xyz` holds
seven final frames. See the [NEB guide](../../docs/guides/neb.md) for atom identity,
convergence and restarts.

## 8. Force constants, elasticity and EOS

```sh
cd "$imrmlp_mtp/calculations/fc"
imrmlp calc_fc --potfile model.mtp \
  --input relaxed.vasp --index index.map --supercell 2 2 2 \
  --reduced=false > fc.log 2>&1
echo $?
head -n 3 FORCE_CONSTANTS
cd "$imrmlp_mtp/calculations/elastic"
imrmlp calc_elastic --potfile model.mtp --input relaxed.vasp \
  --index index.map --settings elastic.ini \
  --elastic-output elastic.out > elastic.log 2>&1
echo $?
head -n 8 elastic.out
cd "$imrmlp_mtp/calculations/eos"
imrmlp calc_eos --settings eos.ini --input relaxed.vasp --index index.map \
  --output ev.dat --output-eos eos.dat > eos.log 2>&1
echo $?
cat eos.dat
```

`FORCE_CONSTANTS` starts with `32 32`, the full expanded force-constant matrix,
with blocks in eV/Å². Elastic output uses GPa: typical `C11≈88.24`,
`C12≈C44≈49.50`. EOS gives `V0≈14.835 Å³/atom`, `E0≈−0.6415 eV/atom`,
`B0≈0.3905 eV/Å³` (about 62.56 GPa); `ev.dat` energy is per atom.
See [crystal properties](../../docs/guides/properties.md) for numerical convergence
and output definitions.

E/F validation should be followed by target-property checks. Running the
reference potential independently through this workflow in the checked
environment gave a lattice parameter near 3.90040 Å, `C11≈88.25 GPa` and a
vacancy barrier near 0.55086 eV, close to the MTP. This establishes agreement
for the synthetic reference problem, not real aluminium accuracy. You can also
relabel the final MTP path with the reference and check its model errors:

```sh
cd "$imrmlp_mtp"
imrmlp calc_efs calculations/neb/mep.xyz path-reference.xyz \
  --potfile reference/force-shifted-lj.table --pottype pair_table --index index.map
echo $?
imrmlp calc_err --input path-reference.xyz --potfile model.mtp \
  --index index.map --output-summary path-errors.txt
echo $?
```

Expect seven frames and 217 atoms. Reference labels come from the reference
potential, rather than using MTP predictions to validate themselves.

## 9. Optional: Phonopy DOS

After additionally installing Phonopy 4, use the prepared separate directory:

```sh
cd "$imrmlp_mtp/calculations/phonopy"
phonopy-init -d --dim="2 2 2" -c relaxed.vasp
echo $?
imrmlp phonopyfc --potfile model.mtp --disp phonopy_disp.yaml \
  --index-file index.map --settings-file settings.ini > forces.log 2>&1
echo $?
phonopy --mesh="10 10 10" --dos phonopy_disp.yaml > dos.log 2>&1
echo $?
head -n 5 total_dos.dat
```

Inspect `FORCE_SETS`, `total_dos.dat` and logs; do not copy another supercell's
force constants from `fc/`. The [LJ walkthrough](../basic-calculations/README.md)
explains DOS columns, negative frequency grid points and supercell convergence.

## What changes for your own material?

Replace synthetic labeling with converged DFT or other reference calculations,
splitting by trajectory/phase/composition. Do not generate reference labels
with the MTP being deployed. Check units, stress convention and species order.
Real many-body materials commonly need angular and higher-body terms, beyond
increasing this example's radial count. Its template-completion helper supports
only the supplied 8/16-term settings, not arbitrary MTP initialization.
`min_dist=2` is not a hard wall against shorter atom separations.

Choose basis, cutoff and radial parameters using validation, target properties
and cost. Use matching nonlinear settings when training radial parameters.
See the [training guide](../../docs/guides/workflows.md),
[MTP theory](../../docs/theory/mtp.md) and [linear refits](../../docs/advanced/linear-refit.md).
To expand coverage, use [active learning](../../docs/guides/active-learning.md)
to find candidates, obtain new reference labels, retrain and rebuild matching state.
