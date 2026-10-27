# Analysis and Descriptor API {#api_analysis}

[中文](api-analysis_zh.md)

This page is for C++ users adding analyses or descriptors. For existing analysis
commands, start with the [command reference](../reference/commands.md).
Build the library and follow [API examples](cpp-examples.md) to compile the complete
bond-order program below.


@tableofcontents

## Structure and trajectory analysis

| Class | Header | Purpose |
|---|---|---|
| `ComputeRDF` | `analyse/compute_rdf.h` | Pair counts and radial distribution functions |
| `ComputeVoro` | `analyse/compute_voro.h` | Voronoi volumes, faces and coordination |
| `ComputeAngles` | `analyse/compute_bondangle.h` | Bond-angle and dihedral distributions |
| `ComputeMSD` | `analyse/compute_msd.h` | Mean squared displacement |
| `ComputeBondOrder` | `analyse/compute_boo.h` | Bond orientation order |
| `ComputeSRO` | `analyse/compute_sro.h` | Short-range order |
| `ComputeCNA` | `analyse/compute_cna.h` | Common-neighbor analysis |
| `ComputeHA` | `analyse/compute_ha.h` | Honeycutt–Andersen indices |
| `ComputeVACF` | `analyse/compute_vacf.h` | Velocity autocorrelation |

Use `GPSettings` constructors for configurable analyses. For CLI options and
output files, see [calc_prop](../reference/commands.md) and the
[configuration reference](../reference/configuration.md).

`ComputePosavg` (`analyse/compute_posavg.h`) provides streaming averages through
`init`, `accumulate` and `compute`. Select fractional averaging for changing
cells, or mass-weighted drift removal for a fixed cell. The shared implementation
backs `calc_prop --avgpos` and `calc_trjavg`; their drift conventions are described
in the [command reference](../reference/commands.md).

`ComputeSmoothFFT` (`analyse/compute_fftfilter.h`) backs `smooth_traj`.
`compute2(trajectory, timestep_fs, window)` requires uniformly sampled frames;
`ComputeSmoothFFT::SUPERGAUSSIAN` selects the super-Gaussian window.

## Model Hessians and Phonopy

`ComputeHessian` (`analyse/compute_hessian.h`) computes model Hessians.
`compute_fc` returns a dense atomic Hessian; `compute_fc_blocks` returns
`Potentials::HessianMatrix` blocks. Select `auto`, `analytic`, `force_fd`,
`energy_fd` or `hybrid` with `set_method`.

`calc_fc` exports force constants, and `phonopyfc` generates `FORCE_SETS` from
Phonopy displacement files. The [Python adapter](../reference/python.md#phonopy-integration)
passes ASE calculator forces to Phonopy for force-constant fitting, dispersion,
DOS and thermodynamics.

## Atomic descriptors

`Descriptors` (`descriptors/descriptors.h`) defines `feature_size()`,
`compute(Neighborhood&, Eigen::VectorXd&)` and `clone()`. Implementations include
ACSF (`DescriptorBehler`), bond order (`DescriptorBondOrder`) and MTP scalar
moments (`DescriptorMTM`).

This complete example evaluates bond-order features for one atom:

```cpp
#include "base/neighborhood.h"
#include "descriptors/descriptor_BondOrder.h"
#include <iostream>

int main() {
    Atoms atoms;
    atoms.cell_ = 20 * Eigen::Matrix3d::Identity();
    atoms.pos_ = {{1, 1, 1}, {2.2, 1.4, 1.8}, {1.7, 3.1, 2.5}};
    atoms.chemmap_ = {0, 0, 0};
    Neighborhoods neighbors(atoms, 5.0);
    DescriptorBondOrder descriptor(8);
    Eigen::VectorXd features;
    descriptor.compute(neighbors[0], features);
    std::cout << features.transpose() << '\n';
}
```

Bond-order output is `[Q_2, Q_4, ..., Q_maxl, W_2, W_4, ..., W_maxl]`.
`maxl` must be even and at least 2. ACSF and MTP descriptors also accept
`ComputeOptions` with a `DescriptorGradient`; gradients have shape
`(feature_size, neighbor_count, 3)`. A `DescriptorMTM` references its loaded
`Potentials::PotMTP` model, which must outlive the descriptor.

## Symmetry and structure comparison

`external/spglib_driver.h` exposes `Symmetry3D::spacegroup_number`,
`standardize_cell` and `get_sym_dataset`. The returned `symdataset` owns its
symmetry operations, Wyckoff indices and atom mappings.

For structure matching, first call `prepare_comparison_structure` on both
inputs with the same tolerance, then `equivalent_structures` with
`StructureComparisonOptions`. The shared matcher preserves species and finds
one-to-one atom mappings under periodic translations and cell symmetries.
`sameq` and structure deduplication use this kernel.

## See also

- @ref analyse — Analysis module
- @ref descriptors — Descriptor module
- @ref symmetry — Symmetry module
