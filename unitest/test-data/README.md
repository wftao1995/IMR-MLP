# Repository test fixtures

These small fixtures ship with the repository. C++ tests access this directory
through the temporary test runner's `unitest/test-data` link; Python tests resolve
it relative to the checkout. No sibling `tests` checkout or machine-specific path
is needed. Run:

```sh
LD_LIBRARY_PATH="$PWD/build/release:${LD_LIBRARY_PATH:-}" make USE_MKL=0 test
LD_LIBRARY_PATH="$PWD/build/release:${LD_LIBRARY_PATH:-}" pytest
```

## Models and structures

The compact model definitions and enumlib surface samples were retained from
the project's historical test fixtures. They are regression inputs, not validated production
potentials. `portable-mtp` contains the small plain, nonlinear, LES seed, and
hybrid basis definitions; the case directories retain the filenames used by the
corresponding regression tests. `test-nnmtp` includes the matching neural-network
and scaling payloads needed to exercise the legacy multi-file reader.

`potential.mtp` and `case-descriptors/potential.mtp` use the compact 92-moment
nonlinear fixture instead of the historical 82 MB model. Tests of descriptor
normalization use five synthetic four-atom frames in `test-nnmtp/sub.cfg`.
Positions are deliberately varied; their zero forces,
zero virials, and constant energies are synthetic labels, not reference physics.
No external training dataset is included.

## LES seed update

`case-mtp-les/les.mtp` is a short-range seed with no `les_species_coeffs` or
`les_moment_coeffs` blocks. Its dormant `lr_k_cutoff` and `les_output_scaling`
fields were removed because the current reader explicitly rejects them. Tests
initialize their own LES v2 or SOG v3/v4 coefficients and kernel settings before
evaluating the long-range model. This edit is not a conversion of a fitted legacy
LES potential and does not assert equivalence of old and new LES functionals.

Tests intentionally rejecting obsolete LES fields still construct those fields
in memory; the parser's rejection policy is unchanged.
