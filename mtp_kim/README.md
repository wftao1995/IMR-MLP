# MTP OpenKIM model driver

This directory contains a dependency-light MTP evaluator and an OpenKIM API
2.x model driver. The evaluator uses only the C++ standard library; the driver
adds the KIM API dependency.

## Supported radial bases

The KIM evaluator supports the radial bases accepted by the main MTP basis
factory:

- `RBChebyshevConfigurable`
- `RBChebyshev`, `RBChebyshevPoly4`, `RBChebyshevPoly5`
- `RBPolyB`, `RBPolyBCos`
- `RBChebyCos`

The configurable basis supports all cutoff, transform, and basis-function
forms documented in `docs/reference/radial-basis.md`.

For existing KIM parameterized models, the standalone evaluator also accepts
these legacy names even though the main basis factory no longer does:

- `RBChebyshevI2`, `RBChebyshevPoly2`, `RBChebyshevPoly3`,
  `RBChebyshevPoly5BSide`, `RBChebyshevPoly7BSideC`
- `RBChebyshevFlip`, `RBChebyshevShiftedFlip`, `RBConstant`
- `RBChebyCosShiftedScaled`
- `RBChebyDoubleCosLeft`, `RBChebyDoubleCosShiftedScaled`,
  `RBChebyDoubleCosShiftedScaledLeft`
- `RBChebyDoubleSinMiddle`,
  `RBChebyDoubleSinDoubleScaledShiftedMiddle`

Both a shared basis and `radial_split`/`radial_basis_split` pair-specific bases
are supported. Pair-specific bases must have the same `radial_basis_size`, as
required by MTP's radial-coefficient layout.

## Parameter files

A portable model using this driver must provide:

1. One `.mtp` potential file in MTP format version `1.1.0`.
2. One `.species` text file listing the chemical symbols in MTP species-index
   order. Symbols can be separated by whitespace or newlines; text after `#`
   is ignored.

For example, a three-species MTP whose indices mean Ti, O, and Al uses:

```text
Ti
O
Al
```

The species file is required because an `.mtp` file stores integer species
indices but no portable chemical-name mapping.

The `.mtp` numbers are interpreted as angstrom/eV. At creation time the driver
converts coordinates, cutoffs, energies, forces, and virials to the unit system
requested through the KIM API.

## Build and test

```sh
cmake -S mtp_kim -B build/mtp-kim -DCMAKE_BUILD_TYPE=Release
cmake --build build/mtp-kim --parallel
ctest --test-dir build/mtp-kim --output-on-failure
```

`MTP_KIM_DRIVER_NAME` defaults to the development placeholder
`MTP__MD_000000000000_000`. Set it to the assigned OpenKIM model-driver ID
when preparing a distributable item:

```sh
cmake -S mtp_kim -B build/mtp-kim \
  -DMTP_KIM_DRIVER_NAME=MTP__MD_XXXXXXXXXXXX_000
```

The repository-level radial regression test is
`unitest/ut_mtp_kim_radial.cpp`; it compares values and first derivatives
against the main IMRMLP implementations.
