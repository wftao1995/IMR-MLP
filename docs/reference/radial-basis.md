# Configurable Chebyshev radial basis

[中文](radial-basis_zh.md)

Use this page when configuring or migrating the radial basis of an MTP
template. Existing trained models normally need no edits here; changing the
basis changes predictions and requires retraining or verification. `min_dist`,
`max_dist`, `cutoff_r1` and `cutoff_r2` are in Å.

`RBChebyshevConfigurable` evaluates

\[
Q_n(r)=\mathrm{scaling}\,f_\mathrm{cut}(r)P_n(s(r)).
\]

`basis_function_type = chebyshev` uses \(P_n=T_n\). The
`one_minus_chebyshev` form uses \(P_n=1-T_{n+1}\), which covers the former
Flip variants.

## Potential-file format

Tags have a fixed order. `basis_function_type` is optional when reading old
configurable files and defaults to `chebyshev`.

```text
radial_basis_type = RBChebyshevConfigurable
    min_dist = 0.5
    max_dist = 5.0
    radial_basis_size = 8
    basis_function_type = chebyshev
    cutoff_type = poly5
    cutoff_b = -0.3333333333333333
    shift_type = exponential
    shift_lambda = 1.0
```

Let \(t=(r-r_{\min})/(r_{\max}-r_{\min})\). Available cutoff types are:

| `cutoff_type` | Envelope |
|---|---|
| `quadratic` | \((r_{\max}-r)^2\) |
| `normalized_quadratic` | \((1-t)^2\) |
| `cosine` | \([1+\cos(\pi t)]/2\) |
| `cosine_left` | \([1+\cos(\pi r/r_{\max})]/2\) |
| `cosine_unscaled` | \(1+\cos(\pi t)\) |
| `cosine_squared` | \([1+\cos(\pi t)]^2\) |
| `sine_squared` | \(\sin^2(\pi t)\) |
| `poly3` | \((1-t)^2[1+(b+2)t]\) |
| `poly4` | \((1-t)^3[1+(b+3)t]\) |
| `poly5` | \((1-t)^4[1+(b+4)t]\) |
| `smoothstep5` | fifth-order two-sided smoothstep |
| `smoothstep7` | seventh-order rise, plateau, and fall |

`poly3`, `poly4`, and `poly5` require `cutoff_b`.
`smoothstep7` requires `cutoff_r1` and `cutoff_r2`, with
`min_dist < cutoff_r1 <= cutoff_r2 < max_dist`. `radial_basis_size` must be a
positive integer.

Available transforms are:

| `shift_type` | Meaning |
|---|---|
| `linear` | \(2t-1\) |
| `exponential` | endpoint-normalized exponential |
| `exponential_left` | exponential normalized by `max_dist` |
| `i2` | \(2[(r-r_{\max})/(r_{\max}-r_{\min})]^2-1\) |
| `exponential_tanh` | endpoint-normalized exponential followed by normalized tanh |

The exponential transforms require `shift_lambda`.
`exponential_tanh` additionally requires `shift_scale_lambda`.
Zero parameters use their analytic linear limits.

Values and derivatives through second order are zero outside
`[min_dist,max_dist]`.

## Legacy types

Specialized classes retained by the C++ API are `RBChebyshev`,
`RBChebyCos`, `RBChebyshevPoly4`, `RBChebyshevPoly5`, `RBPolyB`, and
`RBPolyBCos`. Removed specialized radial-basis names are not accepted as
potential-file aliases; rewrite them explicitly as `RBChebyshevConfigurable`.
