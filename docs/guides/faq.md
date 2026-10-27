# Frequently asked questions {#guide_faq}

[中文](faq_zh.md) · [Command reference](../reference/commands.md) · [Configuration](../reference/configuration.md)

Short answers to the questions that come up before a first run. Symptom-oriented
diagnostics live in [troubleshooting](troubleshooting.md); this page answers
"which one should I use" and "how is it defined" questions.

## Which command trains a potential?

| Goal | Command | Notes |
|---|---|---|
| Moment Tensor Potential | `trainmtp` | Pass `--do-lbfgs` for nonlinear optimization or `--linear-fit` for the linear coefficients; neither is implied by default. |
| NN-MTP / HMTP and other nonlinear forms | `trainnlpot` | The optimizer must be selected explicitly; the potential type comes from `[pot]`. |
| Linear descriptor potential | `train_linear` | Least squares with optional ridge regularization. |
| Fast linear fit and basis selection | `fastfit` | Mode 0 eigenvalue spectra, mode 1 weighted least squares, further modes for basis/NSGA-II selection. |

These commands share the whitespace-separated settings syntax, but accept
different keys and potential definitions. Use the settings for the selected
command; do not copy them between trainers unchanged. The final `trainmtp` model is
`_Trained_Potential_.mtp` in the working directory; `fit:save_to` controls trainer
checkpoints.

## How do I discover commands and options?

```sh
imrmlp --help                # list every registered command
imrmlp trainmtp --help       # options of one command
```

Use the current executable's help output for available commands; nicknames such as `efs`, `err`, `elastic`, `neb`,
`dimer` and `fps_select` map onto canonical commands and behave identically.
The [command reference](../reference/commands.md) documents every option table,
and the [cheatsheet](../reference/cheatsheet.md) groups them by task.

## Why is my settings entry ignored?

Settings files separate keys and values with whitespace: `key value`. An equals
sign becomes part of the value, so `cutoff = 5.0` does not set `cutoff`. The
`.mtp` model grammar is separate and does use equals signs. See
[configuration](../reference/configuration.md) for the `[section]` rules, duplicate
keys and relative path resolution.

## How is the species order determined?

Species order is explicit and never inferred from filenames or from the order in
which structures happen to appear:

- The top-level `index` setting points to an `index.map` file with one element
  per line. Use `pot:typemap` too when the selected potential requires a map.
- CFG species are indices into that map, not atomic numbers.
- LAMMPS types are 1-based; numeric model-species mappings are 0-based.
  New inputs should prefer element names in `pair_coeff`.
- Python/ASE needs an explicit `elements` list in model order.

## Which units are used?

Energies are eV, lengths Å and forces eV/Å. External-stress and pressure settings
use GPa. The native extensive stress is in eV, and the ASE calculator returns a
volume-normalized stress in eV/Å³. Temperatures and timesteps follow the
units of the calling driver (for example LAMMPS `units metal`).

## Which selection command should I use?

| Situation | Command |
|---|---|
| Build a first, maximally independent training set | `maxvol_select` |
| Grade new structures against an existing active set | `maxvol_sample` |
| Grow an existing active set incrementally | `maxvol_extend` |
| Diverse subset by FPS or CUR leverage scores | `diverse_select` (alias `fps_select`) |

For very large production pools prefer the CLI commands; the Python equivalents
(`MaxVol`, `ActiveSelector`, `select_from_pool`, `cur_select`) are convenient for
in-memory matrices and custom loops.

## Which potential type should I choose?

Start with ordinary `mtp`; see [MTP_M](../theory/mtp-m.md) for trainable channels
and nonlinear heads. LES/SOG belongs to `mtp_les`,
and `hybrid`/`hybrid_scale` combines a short-range
potential with `zbl` or another sub-potential. The
[potential type overview](../reference/potential-types.md) lists all 21 registered
types with their input artifacts and build switches.

## Are MPI and GPU supported?

Yes, as separate build modes: `make BUILD_MODE=mpi imrmlp` produces
`imrmlp-mpi`/`libimrmlp-mpi.so`. CUDA uses `BUILD_MODE=cuda` with
`CUDA_HOME`/`CUDA_ARCH`; ROCm HIP uses `BUILD_MODE=hip` with `HIP_ROOT`/`HIP_ARCH`.
Combined modes are `mpi-cuda` and `mpi-hip`. For evaluation commands that expose
`--device`, select `cpu`, `cuda` or `hip` (`dcu` is an alias for `hip`). See
[installation](installation.md) and [GPU support](../development/cuda.md). A GPU build does not change the CLI
surface; unsupported combinations fail at potential construction time.

## Can CLI and Python be mixed?

Yes. Both bind the same library, so a model trained with `trainmtp` loads
unchanged in Python, and structures written by `convstr` are readable by
`imrmlp.read`. Check the potential type, species map and I/O format in both
interfaces. `Potential(...)` defaults to `pottype="auto"`: GPM uses its embedded
type, while legacy files fall back to `mtp`. Specify other legacy types
explicitly; extensions are not used to guess the model family.
See the [Python API](../reference/python.md).

## How do I make results reproducible?

Use one build, pin the thread counts (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`) and
set the seeds exposed by the feature you use, for example `dimer:random_seed`.
For the Dimer option, `random_seed=0` is reproducible within a given build,
while `-1` requests nondeterministic initialization. Run directories should be separate so that
`_Trained_Potential_.mtp` and logs are not overwritten.
