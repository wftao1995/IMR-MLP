# GPM v1 model bundle {#reference_gpm_v1}

[中文](gpm-v1_zh.md)

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

All integers are unsigned little-endian unless a block dtype says otherwise. The 64-byte preamble is:

| Byte offset | Size | Meaning |
|---|---:|---|
| 0 | 8 | ASCII `GPMLMODL` |
| 8 | 2 | Major version, 1 |
| 10 | 2 | Minor version, 0 |
| 12 | 4 | Required flags, currently 0 |
| 16 | 8 | UTF-8 JSON byte count |
| 24 | 8 | Absolute start of the binary region |
| 32 | 32 | SHA-256 of the exact JSON bytes |

JSON immediately follows the preamble. The binary region starts at `align64(64 + json_length)`. Padding contains zeros. Every block offset is relative to that region and is a multiple of 64. Empty blocks may share an offset. Nonempty blocks cannot overlap.

`schema` is `gpml.model`. Required top-level members are `units`, `species`, `root_model`, `models`, and `blocks`. Optional members include `active_learning`, `structures`, `resources`, `provenance`, and `extensions`. `model_fingerprint` identifies the effective root graph. Unknown `required_features` and preamble flags fail loading; optional extension values survive an unchanged save.

Units are `{ "length": "angstrom", "energy": "eV", "charge": "e", "stress": "virial_eV" }`. The stress field is the project's internal virial, with its existing sign convention; it is not pressure or virial divided by volume.

Each block directory record contains `id`, `dtype`, `shape`, `offset`, `stored_size`, `raw_size`, `codec`, and `sha256`. Dtypes are `f64`, `i32`, `u32`, `i64`, `u64`, and `u8`. FP64 uses IEEE-754 bit patterns. Matrices use row order. Shape dimensions, byte counts, and offsets are canonical unsigned decimal **strings**. Their products must exactly equal `raw_size / sizeof(dtype)`. The checksum covers uncompressed bytes. `codec` is `none` or `zstd`; default writers use `none`. The low-level writer accepts a codec per block. Builds without zstd reject a requested compressed block explicitly.

A numeric reference is `{ "block": "unique/id" }`. The reader checks the JSON checksum, duplicate keys, dimensions, offsets, overlap, and required version before model construction. It checks block SHA-256 when a block is read. Ordinary inference leaves AL history and candidate payloads unread. Full validation reads every block. Default limits are 64 MiB for JSON and 8 GiB for an individual decoded block.

Writers stage data in the destination directory, validate the result, fsync it, and atomically rename it. Failure before rename preserves the old destination. The C++ writer is a local operation, suitable for existing rank-zero save sites; the conversion CLI executes on rank zero and broadcasts failure to other ranks. Callers must not invoke independent writers on the same destination concurrently.


Here `zstd` is an internal GPM block codec. This is separate from outer `.zst`
compression of structure, settings or text-potential files, which is unsupported.
Ordinary CLI conversion uses uncompressed GPM blocks by default.

## Model nodes

`models` maps IDs to nodes with `type`, `version: 1`, `species`, `species_mode`, `common`, and `children`. Nodes add `configuration`, `parameters`, `state`, or `resources` as appropriate. Child references are model IDs, never input filenames. Required references and cycles are checked when traversing the graph.

Species IDs are zero based. Each indexed type has an entry `{ "id": 0 }`, with `symbol` added only when known. `mindist` declares `species_mode: "type_independent"`. Explicit external element maps must agree with known identities. Internal ordered mappings are preserved even when chemical names were never supplied.

The v1 codecs cover `mtp`, `mtp_m`, `mtpnn1`, `mtp_les`, `hmtp`, `hybrid`, `hybrid_scale`, `eamalloy`, `eamfs`, `eampoly`, `lj612`, `zbl`, `sw`, `tersoff`, `pair_table`, `threebody_table`, `mindist`, `descriptor_linear`, `openkim`, `acelmp`, and `shellfile`.

MTP nodes store radial definitions, per-basis scaling, moment rules, explicit coefficient order, and named head/kernel fields. `mtp_m` metadata distinguishes li/q2/fs/n1/n2 and describes network dimensions and activation. Native numeric arrays are independent of Eigen or object memory layout. Existing MTP parsers receive an in-memory rendering of these fields; no original model is reopened.

Other codecs store effective spline tables, function parameters, descriptor settings, NN normalization and weights, or ordered children. MTM descriptor models embed their MTP dependency. Hybrid training flags and feature transition parameters are preserved. These codecs retain existing computation and training restrictions.

`EAMZhou2004::SaveParameterSet` / `LoadParameterSet` persist the separate `eam_zhou2004_parameters` object, including active elements and named parameters. This is a parameter-set artifact, not a new runnable potential factory type. `model_info` and `model_validate` accept it.

KIM nodes declare their model identifier and store effective parameter arrays. ACE nodes embed the loaded basis resource and require the PACE runtime. Shell nodes embed captured scripts and explicitly listed input resources; declare required programs with `shellfile:dependencies`, and inputs with `shellfile:resource:relative/name source/path`. Resource names cannot escape the job directory. Extracted scripts belong to the loaded potential's lifetime. A shell model still executes its declared scripts when explicitly evaluated.

## Active learning

`active_learning` has its own `version: 1`, `method`, `target_model`, and `model_fingerprint`. This public build supports `maxvol`. `policy` stores selection/break thresholds, interval, step counter, abort flag, sparsification method, and target size. Export paths are runtime overrides. `candidate_pool` points to versioned structure records.

Structure records preserve cell, coordinates, species, PBC, physical properties, labels, masks, topology, per-atom/global typed attributes, and string metadata. Their directory is itself a `u8` JSON block; their numerical fields occupy independent blocks. They never contain native `Atoms` objects or pointers.

MaxVol stores builder settings, E/F/virial/site weights, row scaling, initialization, active rows, row provenance, selected configurations, ordered history, progress, selection mode, and termination status. The builder binding has a separate fingerprint covering its configuration and numeric blocks. Full and exact streaming retain the history needed for rescanning. B matrices, factorizations, thread buffers, and mmap files are reconstructed. Legacy `.sel` conversion rebuilds source rows with the supplied model and compares them to the active rows before binding.


A changed target configuration, parameter array, or element order invalidates the bound session. Matching parameter counts are insufficient. Scoring and saving a restored session check this binding. Optimizer, MD integrator, and labeling scheduler checkpoints are outside this format.
