# 固定玻恩有效电荷的周期外电场示例

[English](README.md)

本例需要外部模型与响应数据，运行前请替换下面的占位路径。
验证新安装请先运行[快速入门](../quickstart/README_zh.md)。

原 Coulomb LES 的训练、解析 BEC 与扩胞验证脚本见本页末尾。它们用于测量动态响应误差，与下面的固定 DFT BEC 场力入口分别使用。

本示例将独立 DFT 玻恩有效电荷（BEC）与已有零场势叠加，支持 `units metal`、
三维周期性边界、固定晶胞和恒定宏观电场。此入口应使用 DFT 响应张量。本示例没有
验证 LES 有限波矢响应可作为物理 BEC；其可积性与晶格换基不变性需要另行检查。

以下 Python 命令从仓库根目录执行。生成输入需要 NumPy；导出 DFT 数据还需要 ASE。
运行生成的输入，需要已安装对应 LES/SOG pair style 及运行库的 LAMMPS。

## 物理约定

输入 BEC 的行是极化／电场方向，列是位移／力方向，数值以基本电荷 e 归一化。电场输入单位 V/Å、坐标 Å，输出附加力 eV/Å、能量 eV：

```text
F_i = Z_i^T E
Delta mu = sum_i Z_i (r_i_unwrapped - r_i_reference)
U_field = -E . Delta mu
```

`U_field` 是相对于所给参考构型的电场能量，省略常数 `-E . mu_reference`。因此不同相的输出不能直接用于比较外场自由能；绝对参考极化另需提供。这里使用已经是物理响应的 DFT BEC，不再乘除 `sqrt(epsilon_infinity)`。电场指体相内部的宏观 E，未求解电极、外部电压与内部场之间的边界问题。

固定 BEC 是参考结构附近的小位移近似。原子穿越盒子边界时用 LAMMPS image flags 保持位移连续；不能用最小镜像反复重置参考位移。它不适用于把原子迁移整个晶格周期、扩散或极化翻转路径都视作常数 BEC。变胞响应、场致电子极化的高阶项和完整电场应力尚未实现，生成输入关闭附加 virial，并在初始化及每个 MD 步检查晶胞是否改变。

## 输入与生成

`fixed-bec-reference-v1` JSON 显式包含 `cell_A`（晶格矢量为行）、`positions_A`、`species`、逐原子 `masses_amu`、`bec_e`（N×3×3）以及 `bec_axes: "polarization,displacement"`。必须由物理响应数据生成，不能填入 SOG 隐变量数组。`export_dfpt.py` 只适配特定
VASP 目录结构：`--source` 下每个相均应有 `PHASE/high_cutoff/dfpt/`，其中包含
`POSCAR`、`INCAR`、`KPOINTS`、`OUTCAR`、`vasprun.xml` 和预先整理的 `response.json`。
该 JSON 必须含 `gap.indirect_gap_eV`、`BEC_e`、`epsilon_infinity`；脚本不会从任意
VASP 输出自动生成它。脚本读取最终 XML 张量，并与 `response.json` 交叉核验。

```sh
python3 examples/les-response/export_dfpt.py \
  --source /absolute/path/DFT_BEC \
  --output /absolute/path/new-reference-export
```

默认相目录为 `alpha_2H`、`beta_2H`、`alpha_3R`，可用 `--phases` 指定其他目录名。
导出要求带隙大于 0.01 eV、VASP 正常结束、张量有限且相互一致，并保存输入哈希及检查结果。
数据布局不同时，应直接准备上述参考 JSON，或按自己的数据修改导出适配器。

本地已有参考 JSON 后：

```sh
python3 examples/les-response/fixed_bec_field.py \
  --reference /absolute/path/alpha_2H.json \
  --work /absolute/path/new-field-case \
  --field 0 0 0.01 --replicate 2 2 2 \
  --model /absolute/path/sog3.mtp --type-order In Se
```

`--field` 在参考 JSON 的原 Cartesian 坐标系指定。程序将一般三斜胞转换为 LAMMPS restricted triclinic 形式，同时旋转位置、BEC、电场，变换矩阵写入 `manifest.json`。复制超胞时逐原子复制 BEC，不乘体积／复制数。`--type-order` 必须等于模型训练时的物种索引顺序；此例为模型物种 0=In、1=Se。

声学和规则默认要求原参考胞的 `max(abs(sum_i Z_i)) <= 1e-6 e`。若需要去除已确认的数值舍入残差，可显式 `--project-asr`；原残差、是否投影和结果均记录。不要用投影掩盖定义错误或未收敛计算。

输出：

- `reference.data`：原子 ID、元素、晶胞和参考位置。
- `init.inc`：零场势与模拟初始化；`--model` 使用 `mlip/opt-les`，省略时使用 `pair_style zero` 供接口测试。
- `field.inc`、六个 `*.atom`：按原子 ID 储存固定外场力和参考坐标，通过 `fix addforce` 施加力并计入一致能量。
- `in.field`：`run 0` 计算及力输出，作为已验证的最小入口。
- `manifest.json`：输入／模型哈希、类型映射、原始／旋转后电场、完整张量与参考坐标。

用包含本库 LES/SOG pair style 的 LAMMPS 在结果目录执行：

```sh
OMP_NUM_THREADS=1 mpirun -np 4 /absolute/path/lmp-sog -in in.field
```

固定晶胞的 MD 可在 `init.inc`、`field.inc` 之后增加 `fix nve` 或 `fix nvt` 及相应协议。重启时保留原子 ID、数目、image flags 和原参考文件，重新定义 `field.inc`；不可把重启时的位置当作新的参考而丢失已有电场做功。不能在加载这些逐原子数据后再 `replicate`、重排 ID 或增删原子。当前生成入口专用于固定晶胞 MD。

## 验证

```sh
python3 examples/les-response/validate_fixed_bec.py \
  --lmp /absolute/path/lmp-sog --work /absolute/path/new-checks \
  --references /absolute/path/reference-export \
  --model /absolute/path/sog3.mtp
python3 examples/les-response/validate_fixed_bec.py --dynamics-only \
  --lmp /absolute/path/lmp-sog --work /absolute/path/new-checks \
  --references /absolute/path/reference-export \
  --model /absolute/path/sog3.mtp
```

第一步验证非对称 BEC 的转置方向、三斜胞旋转、ASR 拒绝／显式投影、三相真实 SOG+DFT BEC 的 MPI 1/2/4 力能叠加，以及合成体系跨 PBC、4→2 进程重启和变胞拒绝。第二步补做三相 100 步、0.1 fs 步长的短 NVE。结果是接口正确性检查，不是长时间精度或相变预测验证。

LAMMPS 接入依据：[fix addforce](https://docs.lammps.org/fix_addforce.html)、[atomfile 变量](https://docs.lammps.org/variable.html)、[未折回坐标](https://docs.lammps.org/compute_property_atom.html)。

## 解析 BEC、扩胞及去平均力检查

该可选分析需要用 `--executable` 指定兼容的外部 `les_bec_evaluate` 辅助程序。
本仓库未提供该程序的源码和构建规则，构建主程序 `imrmlp` 不会生成它。
如果已有此辅助程序及匹配的 `libimrmlp.so`，可以运行：

```sh
python3 examples/les-response/check_les_convergence.py \
  --model /absolute/path/les-model.mtp \
  --executable /absolute/path/les_bec_evaluate \
  --references /absolute/path/dfpt-json-directory \
  --work /absolute/path/new-response-results
```

输入沿用上面的三相 `fixed-bec-reference-v1` 导出文件。使用标量
trace(ε∞)/3，不对各向异性作额外拟合；在最大 α-2H 超胞上选定一个整体符号，
然后固定用于全部相。α-2H 因此是符号锚点，三个构型属于参考点核查，不能当作
广泛独立验证集。

脚本逐项报告 1/2/3/4/6/8/12 倍各向复制、等价换基、对应副本一致性、ASR
及九分量 BEC 误差。原胞和 2 倍胞用独立中心差分核对解析导数。
对固定种子产生的 0.01 Å 小扰动构型，在 1/4/12 倍胞上计算局部力 Jacobian
的反对称部分和两种边长的矩形回路做功，同时记录原始力和减去全体系平均
附加力的结果。扰动构型没有 DFT 标签，不混入 BEC 精度统计。

回路只移动第一份副本中的原子，其他复制原子保持不动；这是局部自由度检查，
不是让所有副本相干位移。回路平面取原始力 Jacobian 的最大反对称分量，
投影后的回路沿相同平面比较。JSON 中的有限波矢 BEC 不直接用于场能量。
脚本的初步精度/尺寸筛选阈值不是普遍物理容差；通过也不能证明外场 MD 的精度。

`--work` 内的既有结果只在模型、动态库、程序和参考文件哈希均匹配时复用。
程序自动优先加载 `--executable` 同目录的 `libimrmlp.so`。应在构建／测试完成后
运行；若还需并行构建，可将该程序与动态库一起复制到独立运行目录，避免重链接
期间加载到不完整的库。
此流程不调用 DFT、不计算 Berry-phase 极化，也不执行外场动力学。
