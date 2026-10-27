# Command Cheat Sheet {#cheatsheet}

[中文](cheatsheet_zh.md)

Quick reference for the most common `imrmlp` commands.
For full documentation see @ref guide_commands. Filenames below are placeholders
for your own inputs. Run from the directory containing those files and keep the
same `index.map` throughout. Options and short aliases differ by command; check
`imrmlp COMMAND --help` before adapting an example.

For complete inputs and interpretation, use [getting started](../guides/getting-started.md),
[relaxation](../guides/relaxation.md), [NEB](../guides/neb.md),
[properties](../guides/properties.md) and [active learning](../guides/active-learning.md).

---

## Training & Fitting

| Goal | Command | Settings |
|------|---------|----------|
| Full MTP training (radial + linear) | `imrmlp trainmtp train.ini --do-lbfgs` | `[fit]`, top-level `potfile` |
| MTP linear coefficients only | `imrmlp trainmtp train.ini --linear-fit` | `[fit]`, top-level `potfile` |
| Linear descriptor potential | `imrmlp train_linear --settings=linear_train.ini -in train.cfg -o linear.pot` | `[fit]`, `[pot]` |
| Basis selection and fast fitting | `imrmlp fastfit settings.ini` | Top-level `mode`: 0 for spectra, 1 for fitting; selection in further modes |
| NN-MTP training | `imrmlp trainnlpot train.ini --do-lbfgs` | `[fit]`, `[pot]` |
| Initialize an MTP-M nonlinear head | `imrmlp initmtpm --settings=init.ini --dataset=train.cfg` | MTP-M `[pot]` settings and training data |
| Create NN-MTP template | `imrmlp nnmtp -pot base.mtp -d train.cfg -n 10 -o nn` | — |

`nnmtp` writes `nn.mtp`, `nn.nn`, `nn.sc` and normalization statistics from the
`nn` output prefix. `initmtpm` is specific to MTP-M; ordinary MTP training starts
from a compatible template.

## EFS & Validation

| Goal | Command | Example |
|------|---------|---------|
| Compute energy/forces/stress | `imrmlp efs -pot pot.mtp -in str.cfg -out out.cfg` | — |
| Compare prediction vs reference | `imrmlp err -pot pot.mtp -in ref.cfg` | — |
| Check potential derivatives | `imrmlp testpot --settings=test.ini` | — |
| Compute descriptors | `imrmlp calc_descriptors --settings=settings.ini` | Top-level `dtype` and type-specific settings |

## Relaxation & Saddle Points

| Goal | Command | Settings |
|------|---------|----------|
| Geometry relaxation | `imrmlp relax relax.ini` | `[relax]`, `[pot]` |
| NEB (minimum energy path) | `imrmlp neb neb.ini` | `[neb]`, `[pot]` |
| Mode following | `imrmlp modefollowing --settings=mf.ini --input=str.cfg` | — |

## Active Learning

| Goal | Command | Example |
|------|---------|---------|
| MaxVol select subset | `imrmlp maxvol_select --pool=pool.cfg --potfile=pot.mtp` | `--threshold=1.001` |
| MaxVol grade structures | `imrmlp maxvol_sample --potfile=pot.mtp --input=pool.cfg --state-file=pot.mtp.sel` | — |
| MaxVol augment set | `imrmlp maxvol_extend --potfile=pot.mtp --state-file=pot.mtp.sel --candidates=pool.cfg` | — |
| FPS / CUR diversity select | `imrmlp diverse_select --n-select=50 --pool=pool.xyz --potfile=pot.mtp` | `--method=cur` |

MaxVol selection writes its reusable active set to
`pot.mtp.sel`. Diversity selection defaults to MTP-based descriptors and therefore
also needs a compatible potential. These selection scores are not error bounds.

## Elastic Constants & Phonons

| Goal | Command | Example |
|------|---------|---------|
| Analytic elastic constants | `imrmlp elastic --method=analytical --potfile=pot.mtp --input=POSCAR` | — |
| Numerical elastic constants | `imrmlp elastic --method=numerical -pot pot.mtp -in POSCAR` | `--strain-magnitude=0.005` |
| Force constants | `imrmlp fc -pot pot.mtp -in POSCAR -dim 3 3 3` | `--reduced=false --hessian-method=analytic` |
| Phonopy FORCE_SETS | `imrmlp mlpphdisp -pot pot.mtp -d phonopy_disp.yaml` | — |
| EOS (energy-volume) | `imrmlp ev --input=POSCAR --settings=eos.ini -ub 0.05 -lb -0.05` | `[pot]`, `[relax]` (or default `input.ini`) |

## Analysis

| Goal | Command | Example |
|------|---------|---------|
| RDF, bond angles, SRO, etc. | `imrmlp calc_prop --rdf --bondangle --input=str.cfg` | `--cutoff=5.0` |
| Trajectory average | `imrmlp calc_trjavg traj.cfg avg.cfg` | `--start=100 --end=500` |
| Free energy error | `imrmlp calc_ferr --input=dft.cfg --prediction=mlp.cfg --temperature=300` | — |
| Find interstitial sites | `imrmlp findint --input=POSCAR --insert=Li` | `--min-dist=1.0` |

## Structure Manipulation

| Goal | Command | Example |
|------|---------|---------|
| Convert formats | `imrmlp convstr POSCAR output.xyz --index=index.map` | `-if POSCAR -of extxyz` |
| Make supercell | `imrmlp mk_super --input=POSCAR --supercell 2 2 2 --output=POSCAR-2x2x2` | — |
| Make lower-triangular cell | `imrmlp mk_lower --input=POSCAR` | — |
| Find (nearly) orthogonal cell | `imrmlp mk_orth --input=str.cfg` | — |
| Search orthogonal cell | `imrmlp mk_orth_search --input=str.cfg` | `--tol=1e-4` |
| Reduce to primitive | `imrmlp primi_reduce --input=POSCAR` | — |
| Reduce translational mult. | `imrmlp trans_reduce --input=POSCAR` | `--tol=1e-4` |
| Swap lattice axes | `imrmlp swap_axis --order=132 --input=POSCAR` | — |
| Check equivalence | `imrmlp sameq --input1=s1.cfg --input2=s2.cfg` | `--tol=0.1` |
| Split dataset | `imrmlp split_set ds.cfg split_ --nsplit=2 --shuffle yes` | — |
| Remove duplicate atoms | `imrmlp rmdups --input=overlap.cfg --rtol=0.5` | — |
| Remove duplicate structures | `imrmlp unique --input=ds.cfg --sym-prec=1e-3` | — |

## Potential Utilities

| Goal | Command | Example |
|------|---------|---------|
| MTP → spline (LAMMPS) | `imrmlp mtp2spline pot.mtp spline_out` | — |
| Reorder species | `imrmlp adjustmtp --input=in.mtp --output=out.mtp --neworder 0 2 1` | — |
| Compare/remap MTP bases | `imrmlp mtpbasis -s source.mtp -t target.mtp [-o remapped.mtp]` | exact item/span test |
| Generate MTP basis | `imrmlp mtmbasis --settings=mtmbasis.ini` | writes `potential.mtp_<iteration>` |
| Export tabulated potential | `imrmlp pot2table potential.sw sw --pair --threebody` | — |

## Misc

| Command | Description |
|---------|-------------|
| `imrmlp smooth_traj --input=traj.cfg --output=smooth.cfg` | Low-pass filter trajectory |

---

## Common option names

| Flag | Effect |
|------|--------|
| `--settings` | Load command settings where supported; `phonopyfc` uses `--settings-file` / `-i` |
| `--potfile` / `-pot` | Potential file |
| `--input` / `-in` | Input structure file |
| `--output` / `-o` | Output file |
| `--input-format` / `-if` | Input file format |
| `--output-format` / `-of` | Output file format |
| `--index` / `-id` | Species index file |
| `--verbose` | Verbose output (only a few commands) |
| `--help` | Command-specific help |
