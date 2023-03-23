# jax_101

## `xmap` with dot

### `xmap` with no-axis-name-awareness dot in different DP+TP combinations

This script requires 8 GPUs. It supports 5 different DP+TP modes: DP, TP_COL,
TP_ROW, DP_TP_COL, DP_TP_ROW.

```bash
python xmap_dot_psum.py --mode 3 --show_ir
```

