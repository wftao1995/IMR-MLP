# Selecting HA categories

[中文](ha-selected-indices_zh.md)

`calc_prop --ha-index` can classify only requested Honeycutt–Andersen category codes:

```text
ha_index:indices 1421 1551 1661
```

Save this as `ha.ini`, prepare a trajectory and its species-ordered `index.map`,
then run:

```sh
imrmlp calc_prop --settings ha.ini --input trajectory.xyz \
  --input-format extxyz --ha-index
```

These integers are HA categories, not atom IDs or neighbor slots. An omitted or empty
`indices` selection retains the existing complete calculation and output behavior.
In a settings file, omit the key to select all categories; the shared settings parser
does not accept a key without a value. C++ callers may explicitly pass an empty set.
Duplicates are ignored.

With an explicit selection, the total and chemical-pair files contain just those codes
in ascending order, including zero-count rows, followed by their `Total`. This selection
takes precedence over `--output-all-indices` for HA; the flag still controls Voronoi
output independently. Existing radial/Voronoi neighbors, `unbond`, pair cutoffs and
frame/thread aggregation remain in effect.

Every fraction retains the full number of owned candidate pairs as its denominator,
including pairs outside the selection. Chemical-pair fractions use that same global
denominator. Thus the selected `Total` fraction can be less than one.

The calculator skips later classification stages when a pair's bonded category or
common-neighbor count cannot match the selected ordinary codes. Pair-cutoff filtering
still applies to the entire bond graph. Periodic images retain their original identity
and pair ownership; the fourth HA classifier is unchanged. Extended category codes use
complete classification followed by exact selection, preserving the existing encoding.
Not every workload becomes faster: the gain follows the selected class's share of the bond
graph, and selecting the class present on every FCC bond saves
little work.
