# Scalar metadata expressions in convstr

[中文](convstr-scalar-labels_zh.md)

Use `--label-expr` to compute numeric labels while converting structures:

```sh
imrmlp convstr --inputfiles trajectory.xyz --input-format extxyz \
  --outputfile labelled.xyz --output-format extxyz \
  --label-expr 'energy_per_atom=energy/size' 'energy_density=energy/volume'
```

Quote each assignment as one shell argument. Multiple assignments follow the
option. Targets are identifiers containing letters, digits and underscores,
starting with a letter or underscore. Physical and built-in field names such as
`energy`, `fenergy` (including the `free_energy` input alias), `fitness`,
`volume`, `size`, `id`, `forces` and `total_charge`
are reserved as targets. Duplicate targets and using both `--label` and
`--label-expr` for one target are errors.

Expressions support numbers (including scientific notation), parentheses, unary
`+`/`-`, `+ - * /`, constant powers with `^`, and the one-argument functions
`sqrt`, `exp`, `log`, `sin`, `cos`. Powers associate to the right and bind more
tightly than a leading sign: `-2^2` is `-4`, `2^-2` is `0.25`. Exponents must
contain no frame variables; `energy^2` is valid, `2^size` is not.

Variables read the frame's `energy`, `fenergy`, `fitness`, `volume`, `size`, `id`,
and global double/integer attributes. Missing energy or free-energy data is an
error. Text labels are not parsed as numbers, even when their contents look
numeric. Arrays and per-atom properties are outside this feature.

All assignments read the same original frame after explicit source-unit
conversion and before wrapping, displacement or sorting. An existing ordinary
metadata key may be recomputed. For example, with original `offset=3`,
`'offset=offset+1' 'saved=offset'` produces `offset=4`, `saved=3`.
A newly created output is not available to another assignment in the same call.
The target's old integer/text aliases are replaced by a double; an existing
array target is rejected. Sorting can use the derived numeric attributes:

```sh
imrmlp convstr --inputfiles trajectory.xyz --outputfile sorted.xyz \
  --label-expr 'energy_per_atom=energy/size' --sort-by energy_per_atom
```

Batch, sequential-input, sequential-output and fully streaming paths use the
same assignment function. Sorting still requires a path that keeps frames in
memory. `--label note --value 'energy/size'` retains its original literal-text
behavior and does not evaluate an expression.

EXTXYZ and BIN preserve computed labels as numeric scalar attributes. CFG writes
them through its existing `Feature name number` text representation. Reading
arbitrary CFG Feature text does not promote it to numeric metadata, so further
arithmetic should use an EXTXYZ or BIN intermediate. This option does not add
metadata preservation to formats that do not already support such labels.

Invalid syntax and target conflicts are checked before opening output. Missing
or nonnumeric variables, zero divisors, function-domain failures and nonfinite
results report the frame ID and assignment. In streaming mode a later failure
retains the already written valid prefix; it does not roll back the entire file.
