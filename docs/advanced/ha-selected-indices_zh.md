# 选择 HA 类别

[English](ha-selected-indices.md)

`calc_prop --ha-index` 只统计指定的 Honeycutt–Andersen 类别码：

```text
ha_index:indices 1421 1551 1661
```

将以上设置保存为 `ha.ini`，准备轨迹和与其元素顺序一致的 `index.map` 后运行：

```sh
imrmlp calc_prop --settings ha.ini --input trajectory.xyz \
  --input-format extxyz --index index.map --ha-index
```

这些整数是 HA 类别，不是原子 ID 或近邻槽位。省略 `indices` 或将其置空
会保留原有的完整计算与输出行为。在设置文件中，省略该键即表示选择全部类别；
共享设置解析器不接受只有键名没有值的写法。C++ 调用方可以显式传入空集合，
重复项会被忽略。

显式选择时，总表与化学对文件只包含所选类别，按升序排列，包括计数为零的行，
并在其后附上各自的 `Total`。对 HA 而言，该选择优先于 `--output-all-indices`；
后者仍独立控制 Voronoi 输出。原有的径向/Voronoi 近邻、`unbond`、对截断以及
按帧和线程的汇总方式都不受影响。

每个分数的分母仍是本进程负责的全部候选对数，包含选择之外的对。化学对分数使用
同一个全局分母，因此所选类别的 `Total` 分数可能小于 1。

当某个对的成键类别或公共近邻数不可能匹配所选普通类别时，计算器会跳过后续分类
阶段；对截断过滤仍作用于整个键图。周期镜像保持原有身份与对归属，第四个 HA
分类器不变。扩展类别码先完整分类再精确选取，以保留原编码。并非所有负载都会变快：
若选择的类别在 FCC 的每条键上都出现，能省下的计算很少。
