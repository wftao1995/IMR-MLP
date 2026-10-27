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
共享设置解析器不接受只有键名没有值的写法。
