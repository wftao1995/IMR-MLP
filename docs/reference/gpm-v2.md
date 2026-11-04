# GPM v2 model bundle {#reference_gpm_v2}

[中文](gpm-v2_zh.md)

GPM stores the effective potential, its dependencies, and optionally an active-learning session in one file. Loading a potential never starts candidate collection. Legacy potential and `.sel` input remain available.


For ordinary use, convert, load and validate the model without editing its
binary contents. Conversion preserves model behavior; it does not add training
or backend support. Install any external runtimes or programs required by the
model on the receiving machine.

## CLI and Python


```sh
imrmlp model_convert --settings input.ini --output model.gpm
imrmlp model_convert --input old.mtp --type mtp --index index.map --output model.gpm
imrmlp model_convert --settings with-al.ini --output resume.gpm --profile resume
imrmlp model_info --input model.gpm
imrmlp model_validate --input resume.gpm
```

`with-al.ini` uses the existing `pot:` and `al:` configuration keys. To restore from the same model, set `al:source embedded`; it is mutually exclusive with `al:state_file` and `al:committee_prefix`. The simulation explicitly initializes AL against its potential before use.

```python
from imrmlp import Potential, ModelBundle

potential = Potential("model.gpm")
potential.calculate(atoms)              # inference only
potential.save("copy.gpm")              # preserve embedded AL by default
potential.save("deploy.gpm", profile="inference")

bundle = ModelBundle.load("resume.gpm")
session = bundle.create_al(export_file="candidates.xyz")
session.check(atoms)                     # explicit scoring and collection
session.save("continued.gpm")

# Create a bundle from existing settings; attach an already selected pool.
bundle = ModelBundle.from_settings({"pottype": "mtp", "potfile": "old.mtp"})
bundle.attach_selector(selector)
bundle.save("selected.gpm", profile="resume")
```

`preserve` is the default save profile; `inference` explicitly removes AL; `resume` requires AL. A changed model with old AL cannot be saved under `preserve`. Rebuild and attach a new selection, or explicitly choose `inference`. Attaching a session snapshots its state; use `session.save` for the latest pool and counters.

LAMMPS native uses the shared factory and can infer the type from GPM. Standalone shares the C++17 reader and supports its existing short-range MTP/MTP-M executors; other families and long-range contributions are rejected. Enable standalone `MTP_ZSTD=ON` to read compressed blocks. Format support does not add LAMMPS AL execution.

## Binary contract

GPM v2 rejects major version 1; no legacy-GPM reader or implicit conversion is provided.
The magic remains `GPMLMODL` and the JSON schema remains `gpml.model`.
All binary integers are little endian. The preamble is 64 bytes:

| Offset | Bytes | Meaning |
|---|---:|---|
| 0 | 8 | `GPMLMODL` |
| 8 | 2 | Major version, 2 |
| 10 | 2 | Minor version, 0 |
| 12 | 4 | Required flags, currently 0 |
| 16 | 8 | JSON byte count |
| 24 | 8 | Absolute start of binary data |
| 32 | 32 | Reserved, written as zero; no checksum |

The schema is `gpml.model`; required members are `units`, `species`, `root_model`,
`models`, and `blocks`. Units are angstrom, eV, charge e, and internal virial_eV;
the stress value is not pressure or volume-normalized stress. Container major
version 2 is independent of model nodes' `version: 1`. Unknown required features
and preamble flags are rejected.

JSON follows immediately; binary data starts at `align64(64 + json_length)`.
Block offsets are relative to binary data and aligned to 64 bytes. Nonempty blocks
cannot overlap. Directory entries contain `id`, `dtype`, `shape`, `offset`,
`stored_size`, `raw_size`, and `codec`. There is no `sha256` field. Dtypes are
`f64`, `i32`, `u32`, `i64`, `u64`, and `u8`; matrices use row order. Dimensions,
offsets and byte counts are unsigned decimal strings. Shape products must match
raw byte lengths. Codecs are `none` and `zstd`; writers default to `none`.

Ordinary arrays use `{ "block": "id" }`. A structure-property slice uses
`{ "block": "id", "start": "0", "shape": ["16", "3", "1"] }`, where `start`
is an element offset. Empty arrays store only `dtype` and `shape`, with no block.
Finite scalar state values are inline JSON; negative zero, infinity and NaN use
an explicit `{ "float": "-0" }`, `inf`, `-inf`, or `nan` tag. Binary arrays retain
IEEE FP64 values. Vectors of matrices are flattened with shape `[count, rows, cols]`;
ragged matrix vectors retain individual shapes.

Each structure collection packs one contiguous numeric block per property/dtype,
including all frames' coordinates, forces and cells. A small JSON record per frame
stores slices, presence flags, attributes, labels and topology; its directory is a
single `u8` block. The decoder caches each collection's numeric blocks only while
restoring that collection. Selected snapshots remain independent of history.

Readers check versions, duplicate keys, references, dtypes, dimensions, lengths,
slice bounds, offsets, overlap and decompression. No file/header/block checksum is
computed or verified; arbitrary bit changes that leave the structure valid are
not detected. Inference leaves AL payloads unread. Full `model_validate` decodes
all blocks/models and restores embedded AL. Default limits are 64 MiB JSON and
8 GiB per decoded block.

Writers stage in the destination directory, check write/close errors, fsync, and
atomically rename. They do not reopen and hash the completed file. Publication
failure preserves the previous destination. Use one writer per destination;
existing rank-zero save sites and collective CLI error propagation still apply.

## Model nodes

`models` maps IDs to nodes with `type`, `version: 1`, `species`, `species_mode`, `common`, and `children`. Nodes add `configuration`, `parameters`, `state`, or `resources` as appropriate. Child references are model IDs, never input filenames. Required references and cycles are checked when traversing the graph.

Species IDs are zero based. Each indexed type has an entry `{ "id": 0 }`, with `symbol` added only when known. `mindist` declares `species_mode: "type_independent"`. Explicit external element maps must agree with known identities. Internal ordered mappings are preserved even when chemical names were never supplied.

The v1 codecs cover `mtp`, `mtp_m`, `mtpnn1`, `mtp_les`, `hmtp`, `hybrid`, `hybrid_scale`, `eamalloy`, `eamfs`, `eampoly`, `lj612`, `zbl`, `sw`, `tersoff`, `pair_table`, `threebody_table`, `mindist`, `descriptor_linear`, `openkim`, `acelmp`, and `shellfile`.

MTP nodes store radial definitions, per-basis scaling, moment rules, explicit coefficient order, and named head/kernel fields. `mtp_m` metadata distinguishes li/q2/fs/n1/n2 and describes network dimensions and activation. Native numeric arrays are independent of Eigen or object memory layout. Existing MTP parsers receive an in-memory rendering of these fields; no original model is reopened.

PolyB and PolyBCos radial definitions can include the prepared orthogonalization matrix and normalization values. Readers restore these values, including for ordered split bases; older GPM files without these blocks remain readable and initialize the basis locally.

Other codecs store effective spline tables, function parameters, descriptor settings, NN normalization and weights, or ordered children. MTM descriptor models embed their MTP dependency. Hybrid training flags and feature transition parameters are preserved. These codecs retain existing computation and training restrictions.

`EAMZhou2004::SaveParameterSet` / `LoadParameterSet` persist the separate `eam_zhou2004_parameters` object, including active elements and named parameters. This is a parameter-set artifact, not a new runnable potential factory type. `model_info` and `model_validate` accept it.

KIM nodes declare their model identifier and store effective parameter arrays. ACE nodes embed the loaded basis resource and require the PACE runtime. Shell nodes embed captured scripts and explicitly listed input resources; declare required programs with `shellfile:dependencies`, and inputs with `shellfile:resource:relative/name source/path`. Resource names cannot escape the job directory. Extracted scripts belong to the loaded potential's lifetime. A shell model still executes its declared scripts when explicitly evaluated.

## Active learning

`active_learning` has its own `version: 1`, `method`, `target_model`, and one `binding_signature`. This public build supports `maxvol`. `policy` stores selection/break thresholds, interval, step counter, abort flag, sparsification method, and target size. Export paths are runtime overrides. `candidate_pool` points to versioned structure records.

Structure records preserve cell, coordinates, species, PBC, physical properties, labels, masks, topology, per-atom/global typed attributes, and string metadata. Their directory is itself a `u8` JSON block; their numerical fields use packed property arrays shared by the collection. They never contain native `Atoms` objects or pointers.

MaxVol stores builder settings, E/F/virial/site weights, row scaling, initialization, active rows, row provenance, selected configurations, ordered history, progress, selection mode, and termination status. The single AL signature covers effective model data, species order, builder configuration, and the row controls `weight`, `pow_scale`, `init_mode`, and `init_value`. It excludes history/candidate contents, block IDs, offsets, and compression. Inference files have no signatures; top-level and duplicate model/builder fingerprints are removed. Full and exact streaming retain the history needed for rescanning. B matrices, factorizations, thread buffers, and mmap files are reconstructed. Legacy `.sel` conversion rebuilds source rows with the supplied model and compares them to the active rows before binding.


A changed target configuration, parameter array, or element order invalidates the bound session. Matching parameter counts are insufficient. Scoring and saving a restored session check this binding. Optimizer, MD integrator, and labeling scheduler checkpoints are outside this format.

## Single-file selection and training output

For model-dependent builders, `maxvol_select` defaults to `<potential>.maxvol.gpm`.
Use `--state-output` to choose another GPM path or a legacy `.sel` path.
Independent descriptors continue to default to `.sel`.

```sh
imrmlp maxvol_select --potfile model.gpm --dtype pot_efs --pool train.cfg --state-output session.gpm
imrmlp maxvol_sample --state-file session.gpm --input candidates.xyz
imrmlp maxvol_sample --state-file session.gpm --dump-state true
imrmlp maxvol_sample --state-file session.gpm --extract-configs history.xyz
imrmlp maxvol_extend --potfile session.gpm --candidates candidates.xyz --state-output continued.gpm
```

Restoration uses the embedded model and species order without the original model or
`index.map`. Explicit model, species, and selection overrides must agree with the
stored state. Continuation retains session policy, candidate pools, and optional
extensions. Changing the model requires rebuilding selection.

`trainmtp`, `trainnlpot`, and `train_linear` default to `_Trained_Potential_.gpm`.
Ordinary training outputs and default checkpoints contain only the model, dropping
obsolete AL state from the input. Add `--save-maxvol-state` to rebuild MaxVol from
the complete training set and embed it in the final GPM. Configure this rebuild
with `builder:` and `selector:` settings. A rebuild failure preserves the previous
final file. Explicit native paths such as `--output model.mtp` and native
`fit:save_to` paths remain supported. `--save-maxvol-state` requires a `.gpm` output.

## Legacy persistence mode

Pass `--legacy` to `trainmtp`, `trainnlpot`, `train_linear`, `maxvol_select`,
`maxvol_sample`, or `maxvol_extend` to restore the pre-GPM persistence defaults.
The equivalent top-level settings key is `legacy true`; an explicit
`--legacy=false` overrides it. The switch is local to each command invocation.

| Command | Behavior with `--legacy` |
|---|---|
| `trainmtp` | Final `_Trained_Potential_.mtp`; default `fit:save_to potential.mtp` |
| `trainnlpot`, `train_linear` | Final `_Trained_Potential_`; default `fit:save_to potential_fitted` |
| Training checkpoints | Original native checkpoint filenames, including each optimizer's iteration naming |
| `maxvol_select` | Default `<potential>.sel`, using the existing v10 binary writer |
| `maxvol_sample` | Default `<potential>.sel`, even if a GPM sidecar or embedded MaxVol state exists |
| `maxvol_extend --state-file old.sel` | Restore the external state with a separate model, then overwrite that `.sel` unless `--state-output` is explicit |
| `maxvol_extend --train train.cfg` | Original three-stage workflow; no automatic state output, unless `--state-output old.sel` is explicit |

Explicit native output and checkpoint paths are retained. Legacy mode rejects
GPM state paths and `.gpm` training output/checkpoint paths instead of writing
legacy data under a GPM filename. It also rejects `--save-maxvol-state`, which
requests GPM embedding: run `maxvol_select --legacy` after training to produce
an external `.sel`. Empty extension selections leave the structure output alone,
while an explicitly resumed state is still updated.

```sh
imrmlp trainmtp --legacy --settings fit.ini --do-lbfgs
imrmlp maxvol_select --legacy --potfile _Trained_Potential_.mtp --pool train.cfg
imrmlp maxvol_sample --legacy --potfile _Trained_Potential_.mtp --input candidates.xyz
imrmlp maxvol_extend --legacy --potfile _Trained_Potential_.mtp --state-file _Trained_Potential_.mtp.sel --candidates candidates.xyz
```

This mode changes persistence conventions, not fitting or selection algorithms.
GPM major version 1 is not supported.

## Export GPM to native model files

`model_convert --legacy` converts a complete GPM model to its original native
format. It does not initialize blank MTP templates or require training data.
Without `--legacy`, conversion continues to produce GPM.

```sh
imrmlp model_convert --input session.gpm --legacy --output model.mtp
imrmlp model_convert --input session.gpm --legacy --output model.mtp --profile inference
```

The default `preserve` profile also exports embedded MaxVol state as
`model.mtp.sel`. `resume` requires compatible MaxVol state; `inference` exports
only the model and removes an old `model.mtp.sel` at the same destination.
Other AL methods cannot be represented by `.sel` and fail unless `inference`
is selected. AL policy, candidate queues and GPM extensions are not exported.

The command writes `model.mtp.index.map` when the GPM contains a complete element
table, and `model.mtp.ini` with the native reload settings. Run commands using
this settings file from the exported directory. MTPNN1 uses `.mtp`, `.nn` and
`.sc` suffixes on the output prefix; HMTP uses `_subN`. MTP-M requires the
`extmodel` option recorded in the companion settings file.

`hybrid` and `hybrid_scale` export a manifest at the requested output path and
all child files as `<output>.potN`, including fixed children. The manifest uses
relative child paths and preserves training flags and mixing configuration.
Move the entire exported file set together. An MTM-backed `descriptor_linear`
also exports `<output>.descriptor.mtp`, referenced relative to its model file.

Native export supports `mtp`, `mtp_m`, `mtp_les`, `mtpnn1`,
`hmtp`, `hybrid`, `hybrid_scale`, `eam_trainable`, `eampoly`,
`lj612`, `zbl`, `mindist`, `sw`, `tersoff`, `pair_table`, `threebody_table`
and `descriptor_linear`. Families without a complete reloadable native writer
(including MEAM, EAM alloy/FS tables, committees and external runtime models)
fail explicitly. Export does not approximate or refit models. Reloaded inference
data must match the source, so data without an exact native representation
(such as customized prepared polynomial matrices) is rejected. Training
regularizers and per-call accumulation flags are not part of this comparison.

For a `pot_efs` MaxVol state, use the exported native model and state together:

```sh
imrmlp maxvol_sample --potfile model.mtp --pottype mtp --dtype pot_efs \
  --state-file model.mtp.sel --index model.mtp.index.map --input candidates.xyz
```

The companion settings also records the restored builder, selector and state path:

```sh
imrmlp maxvol_sample --settings model.mtp.ini --input candidates.xyz
imrmlp maxvol_extend --settings model.mtp.ini --candidates candidates.xyz
```

Match the original builder family when using the legacy CLI; a composite model
also needs its original `--pottype`. Native output paths must not end in `.gpm`.
Files are staged and reloaded before publication. Reported publication failures
roll back existing outputs; concurrent readers should wait for conversion to finish.
