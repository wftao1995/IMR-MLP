# Example input files

[中文](README_zh.md)

Settings templates and command-line references for `imrmlp` subcommands. Each command usually has a
`*-simple.ini` starting point and a fuller `*.ini` variant. Files use
whitespace-separated keys (no `=`), with subsystem prefixes such as `fit:`,
`relax:`, `pot:`, `neb:`, or `builder:`.

Every deck is cut into blocks separated by one blank line: input and output,
potential, parallelism, images or frames, optimizer, convergence, then the blocks
specific to the command. `# ===== ... =====` headers name each block and values
sit in one column, so a key can be found without reading the whole file. The order
does not affect parsing, and a deck of a handful of keys is left undivided.

A commented-out key marks an option that stays at its default. Write either a key
with a value or a commented key, never a bare key: the loader reports
`missing setting value` and stops.

| Area | Decks |
|---|---|
| EFS and validation | `calc-efs*.ini`, `calc-err*.ini`, `testpot*.ini` |
| Training | `fit*.ini`, `fastfit*.ini`, `initmtpm*.ini`, `nnmtp*.ini` |
| Long-range training | `fit-sog.ini`, `fit-sog-multichannel.ini` |
| Relaxation and saddle points | `relax*.ini`, `neb*.ini`, `dimer*.ini`, `modefollowing*.ini`, `nroptimize.ini` |
| Active learning | `select*.ini`, `diverse-select*.ini`, `calc-fitness*.ini` |
| Structure utilities | `convstr*.ini`, `mk-*.ini`, `swap-axis*.ini`, `sameq*.ini`, `findint*.ini`, `trans-reduce*.ini`, `primi-reduce*.ini`, `split-set*.ini`, `remove-duplicate-*.ini`, `smooth-traj*.ini` |
| Potential utilities | `adjustmtp*.ini`, `mtp2spline*.ini`, `mtpbasis*.ini`, `mtmbasis*.ini`, `pot2table*.ini` |
| Properties and phonons | `prop*.ini`, `phonopyfc*.ini`, `calc-trjavg*.ini`, `ev*.ini`, `calc-elastic*.ini`, `calc-numelastic*.ini`, `calc-fc*.ini`, `calc-ferr*.ini` |
| Hybrid example | `zbl-mtp-hybrid-al/calc-efs.ini` |

These are templates, not self-contained calculations. Copy a suitable file to a
new run directory, replace the input/model paths and check the species map.
Relative paths resolve from the directory where you launch `imrmlp`, even when
the settings file is elsewhere. For example, from the repository root:

```sh
mkdir -p build/my-efs-run
cp example-inputs/calc-efs-simple.ini build/my-efs-run/calc-efs.ini
```

Edit the copied settings, place the required structure, model and `index.map` in
that run directory (or use absolute paths), then run:

```sh
cd build/my-efs-run
../release/imrmlp calc_efs --settings=calc-efs.ini --input POSCAR --output output-efs.cfg
```

The simple EFS template expects `POSCAR`, `potential.mtp` and `index.map` by default and
writes `output-efs.cfg`. For a first calculation with supplied inputs, use the
[self-contained quickstart](../examples/quickstart/README.md).
For relaxation, NEB and property calculations with complete inputs, use the
[calculation walkthrough](../examples/basic-calculations/README.md), which
generates models, structures and settings and passes outputs to the next step.

Files marked “CLI reference only” describe command-line options and cannot be
loaded as settings. These include many structure utilities, `adjustmtp`,
`mtp2spline`, and `nroptimize`. `phonopyfc` uses `--settings-file` instead of
`--settings`; `pot2table` reads only potential settings from its file and requires
its export options on the command line.

`nroptimize.ini` is a parameter reference rather than a directly loadable template.
`dimer.ini` is the complete Dimer example; `dimer-simple.ini` is a minimal
starting point. Command options are documented in
[`docs/reference/commands.md`](../docs/reference/commands.md), and settings keys
in [`docs/reference/configuration.md`](../docs/reference/configuration.md).
