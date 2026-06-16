#!/usr/bin/env python3
"""Summarize scalar/NEON solver benchmark runs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_last_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def read_ghia(path: Path, nx: int, re_value: int) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = {}
    prefix = f"rust_mac_N{nx}_Re{re_value}_"
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            stem = row["stem"]
            if stem.startswith(prefix):
                label = stem.removeprefix(prefix)
                rows[f"{label}:{row['profile']}"] = row
    return rows


def as_float(row: dict[str, str] | None, key: str) -> float:
    if not row:
        return math.nan
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--nt", type=int, required=True)
    parser.add_argument("--re", type=int, default=100)
    parser.add_argument("--baseline-csv", type=Path)
    parser.add_argument("--scalar-csv", type=Path, required=True)
    parser.add_argument("--simd-csv", type=Path)
    parser.add_argument("--ghia-csv", type=Path, default=Path("results/ghia_validation_summary.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/benchmark_comparison.csv"))
    args = parser.parse_args()

    ghia = read_ghia(args.ghia_csv, args.nx, args.re)
    rows: list[dict[str, str]] = []

    baseline = read_last_row(args.baseline_csv) if args.baseline_csv else None
    scalar = read_last_row(args.scalar_csv)
    simd = read_last_row(args.simd_csv) if args.simd_csv else None
    scalar_total = as_float(scalar, "total_s")
    baseline_total = as_float(baseline, "total_s")

    def append(label: str, row: dict[str, str] | None, validation_label: str | None) -> None:
        total = as_float(row, "total_s")
        u_err = as_float(ghia.get(f"{validation_label}:u_centerline") if validation_label else None, "rms_error")
        v_err = as_float(ghia.get(f"{validation_label}:v_centerline") if validation_label else None, "rms_error")
        rows.append(
            {
                "label": label,
                "nx": str(args.nx),
                "nt": str(args.nt),
                "total_s": f"{total:.9g}" if not math.isnan(total) else "",
                "ms_per_step": row.get("ms_per_step", "") if row else "",
                "speedup_vs_recorded_baseline": f"{baseline_total / total:.6g}"
                if baseline_total > 0.0 and total > 0.0
                else "",
                "speedup_vs_optimized_scalar": f"{scalar_total / total:.6g}"
                if scalar_total > 0.0 and total > 0.0
                else "",
                "u_centerline_rms_error": f"{u_err:.9g}" if not math.isnan(u_err) else "",
                "v_centerline_rms_error": f"{v_err:.9g}" if not math.isnan(v_err) else "",
                "divergence_rms_after_projection": row.get("divergence_rms_after_projection", "") if row else "",
                "pressure_residual_final": row.get("pressure_residual_final", "") if row else "",
            }
        )

    if baseline:
        append("recorded-scalar-baseline", baseline, None)
    append("optimized-scalar", scalar, "scalar")
    if simd:
        append("neon-simd", simd, "neon")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with args.out.open() as f:
        print(f.read().strip())


if __name__ == "__main__":
    main()
