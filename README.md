# jax_101

## `xmap` with reduction

```bash
python xmap_reduce.py
```

## `xmap` with dot

This script requires 8 GPUs. It supports 5 different DP+TP modes: DP, TP_COL,
TP_ROW, DP_TP_COL, DP_TP_ROW.

### `xmap` with no-axis-name-awareness dot in different DP+TP combinations

```bash
python xmap_dot_psum.py --mode 3 --show_ir
```

### `xmap` with axis-name-aware dot in different DP+TP combinations

```bash
python xmap_dot.py --mode 3 --show_ir
```

