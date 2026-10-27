# 简谐参考势与热力学积分 {#advanced_harmonic_ti}

[English](harmonic-ti.md)

机器学习势与参考体系之间的自由能差通过热力学积分（TI）获得。这需要一个
自由能可解析、且相空间与目标体系重叠的参考体系：即按目标温度拟合的
正定简谐参考势。

IMR-MLP 提供该构造的两个部分：

- `imrmlp ti_fit_fc`：由 MD 轨迹或自洽系综拟合有限温度正定有效力常数（FC），
  并导出为版本 1 的参考文件。
- `pair_style harmonic/reference`（位于 `LAMMPS/USER-MLIP/hyb-ti/`，由 CPU
  安装脚本安装）：在 LAMMPS 中求值该参考势，配合
  `pair_style hybrid/scaled` 与 MTP 按耦合参数混合。

参考势是 TI 的拟合与采样产物，不是材料模型：它只复现目标体系在有限温度下
的简谐部分，且只有与拟合时的结构、温度和 FC 一起才有意义。

## 用 `ti_fit_fc` 拟合参考势

必须显式选择工作流：`md`、`ensemble`、`md-ensemble` 或 `export`。

```bash
imrmlp ti_fit_fc --workflow md-ensemble \
  --traj md.extxyz --traj-format extxyz \
  --potfile potential.mtp --index index.map \
  --temperature 600 --output-prefix BETA_600K
```

| 工作流 | 含义 |
| --- | --- |
| `md` | 按给定温度拟合提供的 MD 轨迹 |
| `ensemble` | 从当前参考势采样的自洽高斯系综 |
| `md-ensemble` | 先 MD 拟合再做自洽精修；通常选择此项 |
| `export` | 不做拟合，仅把已有力常数转换为 LAMMPS 参考文件 |

命令强制的输入要求：

- 轨迹必须是定胞 NVT。变胞（NPT）轨迹会被拒绝：先取 NPT 平均晶胞，
  再在该晶胞下跑定胞 NVT。
- 不指定 `--input` 时，MD 工作流会展开轨迹、去除质量加权质心漂移，
  并以其质心作为参考结构。
- 默认用所选势重新计算力。要消费轨迹中已存储的力，需显式传
  `--force-source trajectory`。
- 周期像在去除共同平移后对齐，包括质心漂移跨越半个晶胞的情形。
  原子间的相对晶体运动必须小于半个周期。

完整选项见[命令参考](../reference/commands_zh.md)的 `ti_fit_fc` 一节，
或运行 `imrmlp ti_fit_fc --help`。

## 拟合过程保证了什么

拟合与采样都在质量加权内部子空间中进行，因此纯平移被构造性地排除。

- 默认的分阶段拟合先更新模态刚度，再用正平方根表示更新耦合的软模式与近
  简并块。`--root-representation` 可选 `normal`、`sqrt`（默认）或 `root4`。
- 两个阶段都最小化加权均方力残差，并加 `--ridge` 惩罚项向当前 FC 收缩。
  重复样本不会改变惩罚强度。
- `--initial-floor-thz`（默认 1 THz）是每次自洽迭代使用的物理采样刚度；
  `--pd-floor-thz`（默认 1e-4 THz）只是拟合 FC 的严格数值正定下限。
  两者不是同一个量。
- 自洽迭代采样对立（antithetic）高斯系综、计算真实 MTP 力，并在根空间做
  阻尼更新（`--mixing`）。只有在模态预热（`--warmup-iterations`）结束后
  才统计收敛。
- 违反位移或最小距离限制的候选会作为完整的 `+q/-q` 对整体拒绝并替换。
  `--max-sampling-attempts=0` 时预算为所请求系综大小的 20 倍；预算耗尽会
  报告拒绝原因与几何极值并失败，绝不会悄悄降低采样下限或用不足的样本拟合。

## 输出与验收

一次运行以 `--output-prefix` 为前缀写出：

| 文件 | 内容 |
| --- | --- |
| `<prefix>_FORCE_CONSTANTS` | 拟合得到的力常数 |
| `<prefix>_LAMMPS_REFERENCE` | 供 LAMMPS `pair_style harmonic/reference` 使用的版本 1 参考文件 |
| 质心/参考结构 | 构建参考势所用的结构 |
| 迭代 CSV 与诊断信息 | 每次迭代的收敛情况、采样尝试、接受率、几何极值 |
| 验证系综 | 仅用于验证的独立系综 |
| 报告 | 训练/留出误差、下限值与 LAMMPS 模式诊断 |

如何解读误差：

- 训练误差描述的是去中心化后的拟合目标。
- 留出误差、`final_md_force_rmse` 与 `harmonic_validation_force_rmse` 使用
  真实简谐力残差，且不去除验证集的力偏移。单位均为质量加权内部力坐标
  （eV/Å/√amu）。
- `lammps_*` 诊断与 `# lammps_mode` 表始终描述写入
  `<prefix>_LAMMPS_REFERENCE` 的确切 FC。TI 验收与解析简谐自由能应使用这些值，
  而不是拟合误差。
- 接受率持续偏低说明参考势或采样下限不合适，不应据此放宽安全限制。

导出给 LAMMPS 时可用稀疏形式截断原子对的 3×3 块
（`--lammps-fc-block-threshold`、`--lammps-fc-cutoff`），截断后现场块
（onsite）会被重建以强制声学求和规则；需要时还可用最小的各向同性稳定化
把最低内部频率抬到目标值（`--lammps-fc-min-frequency-thz`）。
务必检查报告中的保留块比例、Hessian 相对变化、稳定化前后的频率与 ASR 误差：
截断 `fitted` 的 FC 可能显著改变参考势；若要保留原始局域 FC，
使用 `--lammps-fc-source initial`。

```bash
imrmlp ti_fit_fc --workflow export \
  --input REFERENCE.xyz --input-format extxyz \
  --initial-fc FORCE_CONSTANTS --index index.map \
  --lammps-fc-source initial --lammps-fc-block-threshold 0.01 \
  --lammps-fc-min-frequency-thz 0.5 \
  --write-validation-ensemble false --output-prefix TI_SPARSE
```

## 在 LAMMPS 中使用 `pair_style harmonic/reference`

该风格属于 [LAMMPS 接口](../../LAMMPS/README_zh.md)，由
`python3 LAMMPS/USER-MLIP/install.py` 安装（`hyb-ti` 目录），
也可以手动复制到干净的 LAMMPS `src` 目录。不需要链接 `libimrmlp`。

```text
units metal
# 先创建/读取模拟盒子、原子和质量
pair_style harmonic/reference BETA_600K_LAMMPS_REFERENCE auto
pair_coeff * *
```

要求：周期性边界、固定晶胞、原子 ID 为 `1..N` 且与参考文件一致、
原子类型匹配、每种类型质量为正（`atom_style atomic`）。该风格会去除
质量加权的全局平移位移，不请求近邻表。不支持 restart 序列化：
读取 restart 后需重新指定 pair 风格与参考文件。

可选的存储参数为 `auto`（默认）、`replicated` 或 `distributed`。
replicated 模式在每个 MPI rank 上保存全部 FC 行；distributed 模式分块保存
并汇总力响应。当估算的稠密 FC 块超过每 rank 256 MiB 且 rank 数大于 1 时，
auto 会选择 distributed。两种模式都在每个 rank 上保存位置与位移缓冲。
行内核在编译了 OpenMP 时使用 OpenMP；请把 `OMP_NUM_THREADS` 设为分配的核数。

## 与 MTP 做热力学积分

参考势与目标势按耦合参数 λ 混合：λ = 0 为简谐参考，λ = 1 为完整 MTP。

```text
variable lambda equal 0.5
variable one equal 1.0-v_lambda
pair_style hybrid/scaled v_one harmonic/reference BETA_600K_LAMMPS_REFERENCE &
                         v_lambda mlip/opt
pair_coeff * * harmonic/reference
pair_coeff * * mlip/opt potential.mtp Al Ni
```

MTP 的 `pair_coeff` 请按[LAMMPS 元素映射](../../LAMMPS/README_zh.md)使用元素名。

每个 λ 点是一次独立的 LAMMPS 运行。IMR-MLP 提供的是参考势与可混合的
pair 风格；λ 取值方案、对 \(\langle U_{\mathrm{MTP}} - U_{\mathrm{harm}}\rangle_\lambda\)
沿 λ 的积分以及误差估计属于使用者的工作流。由于参考势是在某一温度下拟合的，
只应在该温度、且只在拟合所用的晶胞下复用。

## 参考文件格式

```text
TI_HARMONIC_REFERENCE 1
units metal
natoms 2
energy0 -1
cell 10 0 0 0 10 0 0 0 10
atoms
1 1 2 2 2
2 1 7 2 2
blocks 3
1 1 2 0 0 0 2 0 0 0 2
1 2 -2 0 0 0 -2 0 0 0 -2
2 2 2 0 0 0 2 0 0 0 2
end
```

晶胞矢量与每个 3×3 FC 块均为行主序；晶胞矢量遵循 LAMMPS 的受限三斜约定。
原子行格式为 `id type x y z`。块索引从 1 开始，要求 `i <= j` 且严格升序；
下三角块由转置重建，只需存储非零块。单位为 Å 与 eV（FC 为 eV/Å²）。
参考文件应满足声学求和规则并具有稳定的非平移模态——适配器不会拟合或
稳定化 Hessian。由 `atoms-agent` 的 `gpmain ti_fit_fc` 写出的文件仍然兼容。

## 验证与回归

- 单元测试：`unitest/ut_ti_harmonic_reference.cpp` 与
  `unitest/ut_ti_reference_io.cpp` 覆盖拟合与参考文件 I/O。
- LAMMPS 侧回归（针对包含该风格的可执行文件）：

```bash
python3 LAMMPS/USER-MLIP/hyb-ti/tests/run_regression.py /path/to/lmp
python3 LAMMPS/USER-MLIP/hyb-ti/tests/run_regression.py /path/to/lmp --ranks 2
```

测试检查解析双原子弹簧的能量、力与逐原子能量、周期平移、全部存储模式、
`hybrid/scaled` 以及非法 FC 输入。

## 限制

- 参考势在单一温度与晶胞下拟合，不可迁移。
- LAMMPS 适配器既不拟合也不稳定化 Hessian；非正定或违反 ASR 的参考文件
  会按原样使用。
- 不支持 pair 状态的 restart。
- 历史上的 `debye/opt` 与 `ti/debye` 适配器不包含在本发布中。
- 简谐参考的 GPU 执行需要单独的设备验证；`mlip/gpudirect` 不覆盖该风格。
