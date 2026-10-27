# MTP + ZBL 短程耦合

[English](README.md)

本例将单元素 Al MTP 与固定的 ZBL 短程修正叠加。本目录已包含结构、元素映射和
演示势。在仓库根目录执行 `make imrmlp` 后，进入本目录运行：

```bash
../../build/release/imrmlp calc_efs --settings calc-efs.ini \
  --input POSCAR --output output-efs.cfg
```

使用其他构建模式时修改程序路径。结果写入 `output-efs.cfg`。

总能量为：

```text
E_total = E_MTP + sum_{i<j} S(r_ij) V_ZBL(r_ij)
```

`r <= r0=1.0 Å` 时 `S=1`，在 `r0` 与 `r1=2.0 Å` 之间平滑降到零，
`r >= r1` 时为零。所附 Al 晶体没有这么短的原子对，因此也用于检查 ZBL 修正
不会改变正常间距下的预测。

用于其他材料时：

1. 替换 `pot:pot2:potfile`，并将 `index` 和 `pot:typemap` 指向自己的元素映射。
2. 将 `pot:spcount` 设为物种数。
3. 按 `index.map` 的相同顺序，在 `pot:pot1:z` 中填写原子序数。
4. 根据原子对距离扫描选择 `r0`、`r1`，始终满足 `0 < r0 < r1`。

训练数据含有切换区间内的原子对时，应通过相同的 `hybrid` 配置用 `trainnlpot`
拟合，使 MTP 学习 DFT 减去 ZBL 后的残差，避免短程相互作用被重复计算。
所附势仅用于低精度功能测试，不能作为生产 Al 势使用。
