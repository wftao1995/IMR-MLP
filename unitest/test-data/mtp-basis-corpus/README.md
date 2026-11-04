# Real MTP basis regression corpus

These eight coefficient-free templates come from the user-provided `MTM-correct`
collection. They contain topology and radial definitions, without atomic datasets,
learned coefficients, or generator logs. `sources.json` records the relative source
paths, original SHA-256 digests and structural inventory for all 27 source files.
The fixture copies are byte-identical to their source templates. The local
`.gitattributes` preserves the original trailing space on empty `potential_tag`
headers; the source checksum check still protects every fixture byte.

| Fixture | Scalars | Product rules | Dependency depth | Purpose |
|---|---:|---:|---:|---|
| `shapeev-02.mtp` | 1 | 0 | 0 | No product rules |
| `shapeev-06.mtp` | 5 | 5 | 2 | Small graph with scalar-coordinate gaps |
| `shapeev-16.mtp` | 92 | 741 | 3 | Rank-six contractions and interleaved sums |
| `shapeev-20.mtp`, `shapeev-20-t.mtp` | 288 each | 4665 / 4526 | 3 | Same scalar space, different contraction graphs/order |
| `ps1-7b-517.mtp` | 517 | 842 | 5 | Deep dependencies |
| `ps1-5b-5323.mtp` | 5323 | 16715 | 4 | Large scalar/coefficient permutation |
| `ps2-7b-4464.mtp` | 4464 | 12172 | 5 | Deep graph with higher angular rank |

`ut_mtp_basis_corpus.test` initializes deterministic two-species ordinary MTP,
two-layer NN MTP-M and MTP+ZBL models. It checks scalar columns using unique
coefficient tags (all latent channels), E/F/S on asymmetric, periodic and strained
geometries, repeated and alternating `level`/`level-group` ordering, transient graph
level caches, species cloning/permutation, and GPM/native reloads. ZBL is active on
the chosen structures. Inputs are removed after initialization. The alternative
contraction test compares the scalar-value multisets and E/F/S independently of
scalar numbering, using shared radial parameters and uniform scalar weights.

Build and run from the repository root (set `MKLROOT` as required):

```sh
make build/release/unitest/ut_mtp_basis_corpus.test
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
LD_LIBRARY_PATH="$PWD/build/release:$LD_LIBRARY_PATH" \
build/release/unitest/ut_mtp_basis_corpus.test
```

Use the complete external collection for the extended gate:

```sh
python3 scripts/validation/audit_mtm_basis_corpus.py \
  /home/atao/workspace/MTM-correct --output build/mtm-correct-audit.json

MTP_BASIS_CORPUS_ROOT=/home/atao/workspace/MTM-correct \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
LD_LIBRARY_PATH="$PWD/build/release:$LD_LIBRARY_PATH" \
build/release/unitest/ut_mtp_basis_corpus.test
```

The external gate uses all 27 templates, including both `shapeev/30` contraction
graphs (approximately 190,000 rules each). It adds the level-30 cross-graph scalar
and E/F/S comparison. An explicitly specified missing/empty corpus fails. Normal
`make test` uses only the portable fixture copies and requires no external path.
These are numerical regression inputs, not fitted material potentials.

## Optional MPI/CUDA checks

Set the matching freshly built CLI paths. Each MPI case uses two ranks. CUDA
cases compare both the reordered model on CPU and its GPU predictions against
the original model; the input GPM must remain byte-identical. The largest case
uses `shapeev/30` when the external corpus is supplied.

```sh
MTP_BASIS_CORPUS_ROOT=/home/atao/workspace/MTM-correct \
MTP_WORKFLOW_MPI_imrmlpCUTABLE="$PWD/build/mpi/imrmlp-mpi" \
MTP_WORKFLOW_GPU_imrmlpCUTABLE="$PWD/build/cuda/imrmlp-cu" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
LD_LIBRARY_PATH="$PWD/build/release:$LD_LIBRARY_PATH" \
pytest -q unitest/python/test_mtp_basis_corpus_cli.py
```
