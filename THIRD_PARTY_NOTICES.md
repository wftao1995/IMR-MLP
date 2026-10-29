# Third-party notices

The MIT license at the repository root covers original IMR-MLP source code.
It does not replace any third-party license. This source tree is not a claim
that a linked binary, or every bundled file, is MIT-only.

| Component | Location | License and attribution |
| --- | --- | --- |
| LBFGS++ (local fork) | `deplibs/LBFGSpp` | MIT; Jorge Nocedal, Naoaki Okazaki, Yixuan Qiu, Dirk Toewe; see `LICENSE.md` |
| picojson v1.3.0 | `tools/picojson.h` | BSD-2-Clause; Cybozu Labs, Inc. and Kazuho Oku; see `tools/picojson.LICENSE` and the header |
| fmt | `deplibs/fmt` | MIT; Victor Zverovich and fmt contributors; see `LICENSE` |
| spglib | `deplibs/spglib` | BSD-3-Clause; Atsushi Togo and contributors; see `LICENSE` and per-file notices |
| TinyXML-2 | `deplibs/tinyxml2` | Zlib; Lee Thomason; see `LICENSE.txt` and source headers |
| Voro++ | `deplibs/voro++` | BSD-3-Clause; Regents of the University of California; see `LICENSE` |
| Eigen standalone headers | `LAMMPS/USER-MLIP/standalone/vendor/eigen3` | MPL-2.0 and per-file licenses; see `COPYING.MPL2` and `SOURCE.txt` |
| LAMMPS-derived adapters | `LAMMPS/USER-MLIP` | GPL-2.0; original Sandia/LAMMPS notices retained; see `LAMMPS/COPYING.GPL-2.0` and per-file headers |

LBFGS++ was imported from the existing local fork at commit
`608471b61e4b2e82ac4432b6de3af83cc297deec`. Its force-based extensions are retained.
Other bundled libraries were imported from the original checkout with their
source notices intact. No binary archives or object files are distributed.

`config/license-files.txt` lists the notices included in binary installations.
Python wheels install them under `imrmlp/licenses`; `make install`, `updatelib`
and `updinst` install them under `$(PREFIX)/share/licenses/imrmlp`.

## External build and runtime dependencies

ALGLIB and FFTW remain external dependencies. The installed ALGLIB free edition
has a GPL license; ALGLIB also offers a separately licensed commercial edition.
FFTW has GPL/commercial licensing. Linking a GPL edition affects the license
obligations for redistribution of the resulting executable or Python extension.
Moving a dependency outside this source tree does not remove those obligations.
A wholly permissive binary distribution would require compatible licenses or
replacement implementations; this cleanup does not perform those replacements.

The main library uses system Eigen; the standalone LAMMPS package vendors a pinned header copy with its notices.

Eigen (MPL-2.0), optional Intel oneMKL, OpenMP, optional OpenKIM, zlib,
Catch2 and pybind11 are provided by the build environment and keep their own
licenses. Consult the actual dependency versions and their license files before
redistributing compiled artifacts.

## Documentation dependencies

The optional manual build installs Sphinx (BSD-2-Clause), MyST-Parser (MIT),
Furo (MIT) and jieba (MIT) into a local virtual environment; their code is not
vendored or included in the native library or Python wheel. See
`docs/requirements.txt` for the versions inherited from the atoms-agent manual.
MathJax 3.2.2 (Apache-2.0; MathJax Consortium) is installed through
`docs/package-lock.json`. Its runtime and LICENSE are copied into each generated
manual under `assets/mathjax/`; retain that license when publishing the manual.

## Removed material

The LAMMPS-ported MEAM implementation and its public interfaces were removed
from this public source branch.

The non-redistributable Shapeev BFGS files and all associated training entry
points were removed. The old LAMMPS XRD data header and
`thirdorder_common.py` are not included. MTP LAMMPS adapters are retained under
their GPLv2 terms, separately from the independent MIT MTP runtime and GPU
kernels in `runtime/mtp`. See `LAMMPS/README.md` for supported styles and builds.
The GPL adapters and combined LAMMPS executables are not MIT-only.
PTM, phonon Green functions, and their obsolete examples and tests were removed.
Positive-definite harmonic-reference fitting and export are available through
`imrmlp ti_fit_fc`; the LAMMPS evaluator is in `LAMMPS/USER-MLIP/hyb-ti`.

Mathematical references to Shapeev's MTP formulation and basis levels are kept:
those identify published scientific definitions, not the removed optimizer.
