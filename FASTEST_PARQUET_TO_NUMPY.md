# Fastest approach found (Polars -> NumPy panel)

From benchmark runs in `benchmark_parquet_to_numpy.py`, the fastest strategy was **single wide join once**, then reshape/slice to `(T, N, F)`.

## Why it wins
- Avoids re-running the full lazy query once per feature.
- Performs one `join_asof` per instrument (not per instrument × feature).
- Materializes a single streaming dataframe, then uses column slicing to build NumPy arrays.

## Recommended production pattern
1. Build one global 1-minute index (`idx`) from global min/max timestamp.
2. For each instrument, join all selected features in one `join_asof(..., tolerance=0)`.
3. Collect once (`engine="streaming"`).
4. Write to `np.memmap` (or one `.npy`) in `(T, N, F)` layout.

```python
# outline only
lz = idx
for inst, scan in scans.items():
    lz = lz.join_asof(
        scan.select(["ts", *[pl.col(f).alias(f"{f}__{inst}") for f in feats]]),
        on="ts",
        tolerance=0,
    )

df = lz.collect(engine="streaming")
out = np.memmap("panel.dat", mode="w+", dtype=np.float64, shape=(T, N, F))
for fi, f in enumerate(feats):
    out[:, :, fi] = df.select([f"{f}__{inst}" for inst in instruments]).to_numpy(order="c", structured=False)
out.flush()
```

## Benchmark snapshot (synthetic minutely shards)
- `per_feature`: 5.244s
- `single_wide`: **2.347s (best)**
- `long_pivot`: 2.903s

These were run with 6 instruments, 45 days, 24 features, and injected minute gaps. Scale-up behavior should remain directionally similar; for full 10-year x 129-field runs, memory-map output and benchmark on a representative feature subset first.
