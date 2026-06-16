#!/usr/bin/env python3
"""Plot compact runtime statistics from benchmark_comparison.csv."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else float("nan")


def grouped_by_grid(rows: list[dict[str, str]]) -> dict[int, dict[str, dict[str, str]]]:
    grouped: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[int(row["nx"])][row["label"]] = row
    return dict(sorted(grouped.items()))


def plot_grouped_bars(
    grouped: dict[int, dict[str, dict[str, str]]],
    key: str,
    ylabel: str,
    title: str,
    out: Path,
) -> None:
    labels = ["optimized-scalar", "neon-simd"]
    names = ["scalar", "NEON"]
    grids = list(grouped)
    x = list(range(len(grids)))
    width = 0.34

    plt.figure(figsize=(7.2, 4.4))
    for offset, label, name in [(-width / 2, labels[0], names[0]), (width / 2, labels[1], names[1])]:
        values = [as_float(grouped[n].get(label, {}), key) for n in grids]
        plt.bar([pos + offset for pos in x], values, width=width, label=name)

    plt.xticks(x, [str(n) for n in grids])
    plt.xlabel("grid size N")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_speedup(grouped: dict[int, dict[str, dict[str, str]]], out: Path) -> None:
    grids = []
    speedups = []
    for n, rows in grouped.items():
        neon = rows.get("neon-simd")
        if neon and neon.get("speedup_vs_optimized_scalar"):
            grids.append(n)
            speedups.append(float(neon["speedup_vs_optimized_scalar"]))

    plt.figure(figsize=(7.2, 4.4))
    plt.bar([str(n) for n in grids], speedups, color="#2a6f97")
    plt.axhline(1.0, color="black", linewidth=1)
    plt.xlabel("grid size N")
    plt.ylabel("speedup vs optimized scalar")
    plt.title("NEON Speedup By Grid Size")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "results" / "benchmark_comparison.csv")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    args = parser.parse_args()

    rows = read_rows(args.summary)
    grouped = grouped_by_grid(rows)
    args.figures.mkdir(parents=True, exist_ok=True)

    plot_grouped_bars(
        grouped,
        "total_s",
        "runtime (s)",
        "Solver Runtime By Grid Size",
        args.figures / "benchmark_runtime_total.png",
    )
    plot_grouped_bars(
        grouped,
        "ms_per_step",
        "ms per timestep",
        "Timestep Cost By Grid Size",
        args.figures / "benchmark_ms_per_step.png",
    )
    plot_speedup(grouped, args.figures / "benchmark_neon_speedup.png")

    print(f"Wrote runtime benchmark figures to {args.figures}")


if __name__ == "__main__":
    main()
