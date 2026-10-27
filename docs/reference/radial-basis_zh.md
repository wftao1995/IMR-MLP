# 可配置 Chebyshev 径向基

[English](radial-basis.md)

此页用于配置或迁移 MTP 模板中的径向基。使用已有训练模型进行预测时通常无需
修改这些字段；修改径向基会改变模型预测，需要重新训练或验证。`min_dist`、
`max_dist`、`cutoff_r1` 和 `cutoff_r2` 的单位均为 Å。

`RBChebyshevConfigurable` 计算

\[
Q_n(r)=\mathrm{scaling}\,f_\mathrm{cut}(r)P_n(s(r)).
\]

`basis_function_type = chebyshev` 使用 \(P_n=T_n\)；
`one_minus_chebyshev` 使用 \(P_n=1-T_{n+1}\)，覆盖原来的 Flip 类型。

## 势文件格式

标签顺序固定。读取旧的 configurable 文件时，
`basis_function_type` 可以省略，默认值为 `chebyshev`。

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

令 \(t=(r-r_{\min})/(r_{\max}-r_{\min})\)，可选 cutoff 为：

| `cutoff_type` | 包络 |
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
| `smoothstep5` | 五次双边 smoothstep |
| `smoothstep7` | 七次上升、平台和下降包络 |

`poly3`、`poly4`、`poly5` 需要 `cutoff_b`；
`smoothstep7` 需要 `cutoff_r1` 和 `cutoff_r2`，满足
`min_dist < cutoff_r1 <= cutoff_r2 < max_dist`。`radial_basis_size` 必须为正整数。

可选 transform 为：

| `shift_type` | 含义 |
|---|---|
| `linear` | \(2t-1\) |
| `exponential` | 端点归一化指数变换 |
| `exponential_left` | 按 `max_dist` 归一化的指数变换 |
| `i2` | \(2[(r-r_{\max})/(r_{\max}-r_{\min})]^2-1\) |
| `exponential_tanh` | 端点归一化指数变换后接归一化 tanh |

指数变换需要 `shift_lambda`；`exponential_tanh` 还需要
`shift_scale_lambda`。参数为零时使用解析线性极限。

在 `[min_dist,max_dist]` 外，函数值及一阶、二阶导数均为零。

## 旧类型

C++ API 中保留的专用类为 `RBChebyshev`、`RBChebyCos`、
`RBChebyshevPoly4`、`RBChebyshevPoly5`、`RBPolyB` 和
`RBPolyBCos`。已删除的专用径向基名称不再作为势文件别名接受，
需要显式改写为 `RBChebyshevConfigurable`。
