from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


@dataclass
class BenchConfig:
    instruments: int = 6
    days: int = 252
    feats: int = 80
    missing_rate: float = 0.06
    seed: int = 7


def make_data(root: Path, cfg: BenchConfig) -> tuple[list[str], list[str], int, int]:
    rng = np.random.default_rng(cfg.seed)
    feat_cols = [f"f{i:03d}" for i in range(cfg.feats)]
    instrument_names = [f"inst{i}" for i in range(cfg.instruments)]

    start_us = int(__import__("datetime").datetime(2014, 1, 1).timestamp() * 1_000_000)
    step_us = 60 * 1_000_000
    mins_per_day = 24 * 60

    for inst in instrument_names:
        inst_dir = root / inst
        inst_dir.mkdir(parents=True, exist_ok=True)
        for d in range(cfg.days):
            base = start_us + d * mins_per_day * step_us
            ts = base + np.arange(mins_per_day, dtype=np.int64) * step_us
            keep = rng.random(mins_per_day) > cfg.missing_rate
            ts = ts[keep]
            data = {"ts": ts}
            for c in feat_cols:
                data[c] = rng.standard_normal(ts.size).astype(np.float64)
            pl.DataFrame(data).write_parquet(inst_dir / f"day_{d:04d}.parquet", compression="zstd")

    end_us = start_us + cfg.days * mins_per_day * step_us
    return instrument_names, feat_cols, start_us, end_us


def build_scans(root: Path, instruments: list[str]) -> dict[str, pl.LazyFrame]:
    return {inst: pl.scan_parquet(str(root / inst / "*.parquet")).set_sorted("ts") for inst in instruments}


def make_idx(start_us: int, end_us: int) -> pl.LazyFrame:
    step_us = 60 * 1_000_000
    return pl.LazyFrame({"ts": np.arange(start_us, end_us, step_us, dtype=np.int64)}).set_sorted("ts")


def build_single_wide_lazy(idx: pl.LazyFrame, scans: dict[str, pl.LazyFrame], feats: list[str]) -> pl.LazyFrame:
    lz = idx
    for inst, scan in scans.items():
        cols = ["ts"] + [pl.col(f).alias(f"{inst}__{f}") for f in feats]
        lz = lz.join_asof(scan.select(cols), on="ts", tolerance=0)
    return lz


def _write_batch_reshape(mm: np.memmap, row0: int, arr2d: np.ndarray, n: int, fcount: int) -> int:
    rows = arr2d.shape[0]
    mm[row0 : row0 + rows, :, :] = arr2d.reshape(rows, n, fcount)
    return rows


def _write_batch_loop(mm: np.memmap, row0: int, arr2d: np.ndarray, n: int, fcount: int) -> int:
    rows = arr2d.shape[0]
    for i in range(n):
        c0 = i * fcount
        mm[row0 : row0 + rows, i, :] = arr2d[:, c0 : c0 + fcount]
    return rows


def benchmark_memmap_load(out_path: Path, shape: tuple[int, int, int]) -> dict[str, float]:
    t0 = time.perf_counter()
    mm = np.memmap(out_path, mode="r", dtype=np.float64, shape=shape)
    map_open_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    checksum = float(np.nansum(mm))
    scan_sec = time.perf_counter() - t1

    t2 = time.perf_counter()
    arr = np.array(mm, copy=True)
    materialize_sec = time.perf_counter() - t2
    del arr
    del mm
    return {
        "load_map_open_sec": map_open_sec,
        "load_scan_sec": scan_sec,
        "load_materialize_sec": materialize_sec,
        "checksum": checksum,
    }


def strat_single_wide_memmap_batched(idx: pl.LazyFrame, scans: dict[str, pl.LazyFrame], feats: list[str], out_path: Path, chunk_rows: int, write_mode: str) -> dict[str, float]:
    lz = build_single_wide_lazy(idx, scans, feats)
    insts = list(scans.keys())
    t = idx.select(pl.len()).collect().item()
    n = len(insts)
    fcount = len(feats)

    mm = np.memmap(out_path, mode="w+", dtype=np.float64, shape=(t, n, fcount))
    writer = _write_batch_reshape if write_mode == "reshape" else _write_batch_loop

    row0 = 0
    batch_count = 0
    t0 = time.perf_counter()
    for batch in lz.collect_batches(chunk_size=chunk_rows, maintain_order=True):
        arr2d = batch.drop("ts").to_numpy(order="c", structured=False)
        row0 += writer(mm, row0, arr2d, n, fcount)
        batch_count += 1

    mm.flush()
    dt = time.perf_counter() - t0
    nan_ratio = float(np.isnan(mm).mean())
    size_gb = out_path.stat().st_size / 1024**3
    del mm
    if row0 != t:
        raise RuntimeError(f"wrote {row0} rows to memmap, expected {t}")
    load_stats = benchmark_memmap_load(out_path, (t, n, fcount))
    return {
        "sec": dt,
        "batches": batch_count,
        "nan_ratio": nan_ratio,
        "size_gb": size_gb,
        "rows": t,
        **load_stats,
    }


def run_tuning(cfg: BenchConfig, chunk_rows_list: list[int], modes: list[str]) -> tuple[list[dict[str, float]], Path]:
    td = Path(tempfile.mkdtemp(prefix="pq_np_bench_"))
    try:
        instruments, feats, start, end = make_data(td, cfg)
        scans = build_scans(td, instruments)
        idx = make_idx(start, end)

        rows = idx.select(pl.len()).collect().item()
        results: list[dict[str, float]] = []
        for mode in modes:
            for chunk_rows in chunk_rows_list:
                out_mm = td / f"panel_{mode}_{chunk_rows}.memmap"
                stats = strat_single_wide_memmap_batched(idx, scans, feats, out_mm, chunk_rows, mode)
                stats.update({"mode": mode, "chunk_rows": chunk_rows, "rows": rows, "N": len(instruments), "F": len(feats)})
                results.append(stats)
        return results, td
    except Exception:
        shutil.rmtree(td, ignore_errors=True)
        raise


def write_report(report_path: Path, cfg: BenchConfig, results: list[dict[str, float]]):
    best = min(results, key=lambda x: x["sec"])
    lines = [
        "# Parquet -> NumPy memmap tuning",
        "",
        "## Config",
        f"- instruments: {cfg.instruments}",
        f"- days: {cfg.days}",
        f"- feats: {cfg.feats}",
        f"- missing_rate: {cfg.missing_rate}",
        "",
        "## Results",
        "| mode | chunk_rows | write_sec | batches | load_scan_sec | load_materialize_sec | shape | nan_ratio | size_gb |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for r in sorted(results, key=lambda x: x["sec"]):
        shape = f"({int(r['rows'])},{int(r['N'])},{int(r['F'])})"
        lines.append(
            f"| {r['mode']} | {int(r['chunk_rows'])} | {r['sec']:.3f} | {int(r['batches'])} | {r['load_scan_sec']:.3f} | {r['load_materialize_sec']:.3f} | {shape} | {r['nan_ratio']:.6f} | {r['size_gb']:.3f} |"
        )

    lines += [
        "",
        "## Best",
        f"- mode: `{best['mode']}`",
        f"- chunk_rows: {int(best['chunk_rows'])}",
        f"- wall_time_sec: {best['sec']:.3f}",
        f"- load_scan_sec: {best['load_scan_sec']:.3f}",
        f"- load_materialize_sec: {best['load_materialize_sec']:.3f}",
        "",
        "## Takeaways",
        "- `reshape` is effectively metadata-only and not the bottleneck; conversion/scan dominates.",
        "- Larger chunk sizes usually reduce Python-loop overhead and number of collect batches.",
    ]
    report_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=252)
    ap.add_argument("--feats", type=int, default=80)
    ap.add_argument("--instruments", type=int, default=6)
    ap.add_argument("--chunks", type=str, default="100000,200000,500000,1000000")
    ap.add_argument("--modes", type=str, default="reshape,loop")
    ap.add_argument("--report", type=Path, default=Path("MEMMAP_BENCHMARK_RESULTS.md"))
    args = ap.parse_args()

    cfg = BenchConfig(instruments=args.instruments, days=args.days, feats=args.feats)
    chunk_rows_list = [int(x) for x in args.chunks.split(",") if x]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    results, temp_dir = run_tuning(cfg, chunk_rows_list, modes)
    write_report(args.report, cfg, results)
    for r in sorted(results, key=lambda x: x['sec']):
        print(r)
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
