# Parquet -> NumPy memmap tuning

## Config
- instruments: 6
- days: 252
- feats: 80
- missing_rate: 0.06

## Results
| mode | chunk_rows | write_sec | batches | load_scan_sec | load_materialize_sec | shape | nan_ratio | size_gb |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| loop | 100000 | 38.850 | 4 | 10.997 | 7.882 | (362880,6,80) | 0.059833 | 1.298 |
| loop | 200000 | 39.113 | 2 | 11.270 | 7.676 | (362880,6,80) | 0.059833 | 1.298 |
| reshape | 200000 | 45.291 | 2 | 10.759 | 7.477 | (362880,6,80) | 0.059833 | 1.298 |
| reshape | 100000 | 49.352 | 4 | 11.273 | 8.264 | (362880,6,80) | 0.059833 | 1.298 |

## Best
- mode: `loop`
- chunk_rows: 100000
- wall_time_sec: 38.850
- load_scan_sec: 10.997
- load_materialize_sec: 7.882

## Takeaways
- `reshape` is effectively metadata-only and not the bottleneck; conversion/scan dominates.
- Larger chunk sizes usually reduce Python-loop overhead and number of collect batches.
