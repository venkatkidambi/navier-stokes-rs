#!/usr/bin/env python3
"""Validate MAC-grid cavity fields against Ghia et al. Re=100 centerlines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIELDS = RESULTS / "fields"
FIGURES = ROOT / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


GHIA_U = {
    100: [
        (1.0000, 1.00000),
        (0.9766, 0.84123),
        (0.9688, 0.78871),
        (0.9609, 0.73722),
        (0.9531, 0.68717),
        (0.8516, 0.23151),
        (0.7344, 0.00332),
        (0.6172, -0.13641),
        (0.5000, -0.20581),
        (0.4531, -0.21090),
        (0.2813, -0.15662),
        (0.1719, -0.10150),
        (0.1016, -0.06434),
        (0.0703, -0.04775),
        (0.0625, -0.04192),
        (0.0547, -0.03717),
        (0.0000, 0.00000),
    ]
}

GHIA_V = {
    100: [
        (1.0000, 0.00000),
        (0.9688, -0.05906),
        (0.9609, -0.07391),
        (0.9531, -0.08864),
        (0.9453, -0.10313),
        (0.9063, -0.16914),
        (0.8594, -0.22445),
        (0.8047, -0.24533),
        (0.5000, 0.05454),
        (0.2344, 0.17527),
        (0.2266, 0.17507),
        (0.1563, 0.16077),
        (0.0938, 0.12317),
        (0.0781, 0.10890),
        (0.0703, 0.10091),
        (0.0625, 0.09233),
        (0.0000, 0.00000),
    ]
}


def read_matrix(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if row:
                rows.append([float(x) for x in row])
    return rows


def interp1(xs: list[float], ys: list[float], x: float) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo = 0
    hi = len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return (1.0 - t) * ys[lo] + t * ys[hi]


def bilinear_at(field: list[list[float]], xs: list[float], ys: list[float], x: float, y: float) -> float:
    row_values = [interp1(xs, row, x) for row in field]
    return interp1(ys, row_values, y)


def load_cases(re_filter: set[int] | None) -> list[dict]:
    cases = []
    if not FIELDS.exists():
        return cases
    for meta_path in sorted(FIELDS.glob("*_metadata.json")):
        with meta_path.open() as f:
            meta = json.load(f)
        re_value = int(round(float(meta.get("Re", meta.get("re", 0)))))
        if re_filter is not None and re_value not in re_filter:
            continue
        stem = meta_path.name.removesuffix("_metadata.json")
        u_path = FIELDS / f"{stem}_u.csv"
        v_path = FIELDS / f"{stem}_v.csv"
        p_path = FIELDS / f"{stem}_p.csv"
        if not (u_path.exists() and v_path.exists() and p_path.exists()):
            print(f"Skipping {stem}: missing one of u/v/p field CSVs.")
            continue
        cases.append(
            {
                "stem": stem,
                "language": str(meta.get("language", stem.split("_")[0])),
                "run_label": str(meta.get("run_label", "")),
                "tolerance_strategy": str(meta.get("tolerance_strategy", "")),
                "re": re_value,
                "nx": int(meta["nx"]),
                "ny": int(meta["ny"]),
                "dt": float(meta["dt"]),
                "nt": int(meta["nt"]),
                "final_time": float(meta.get("final_time", float(meta["dt"]) * int(meta["nt"]))),
                "u": read_matrix(u_path),
                "v": read_matrix(v_path),
            }
        )
    return cases


def select_plot_cases(cases: list[dict], include_all: bool) -> list[dict]:
    if include_all:
        return cases
    selected: dict[tuple[int, str], dict] = {}
    for case in cases:
        label = case["run_label"] or case["tolerance_strategy"] or "run"
        key = (case["re"], label)
        current = selected.get(key)
        if current is None or (case["nx"], case["nt"]) > (current["nx"], current["nt"]):
            selected[key] = case
    return sorted(selected.values(), key=lambda c: (c["re"], c["run_label"] or c["tolerance_strategy"], c["nx"]))


def extract_profiles(case: dict) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    nx = case["nx"]
    ny = case["ny"]
    dx = 1.0 / nx
    dy = 1.0 / ny
    u = case["u"]
    v = case["v"]

    u_x = [i * dx for i in range(nx + 1)]
    u_y = [(j + 0.5) * dy for j in range(ny)]
    v_x = [(i + 0.5) * dx for i in range(nx)]
    v_y = [j * dy for j in range(ny + 1)]

    u_samples = []
    for y, _ in GHIA_U[case["re"]]:
        if y <= 0.0:
            value = 0.0
        elif y >= 1.0:
            value = 1.0
        else:
            value = bilinear_at(u, u_x, u_y, 0.5, y)
        u_samples.append((y, value))

    v_samples = []
    for x, _ in GHIA_V[case["re"]]:
        if x <= 0.0 or x >= 1.0:
            value = 0.0
        else:
            value = bilinear_at(v, v_x, v_y, x, 0.5)
        v_samples.append((x, value))

    return u_samples, v_samples


def error_metrics(reference: list[tuple[float, float]], sampled: list[tuple[float, float]]) -> dict:
    errors = [s[1] - r[1] for r, s in zip(reference, sampled)]
    l2 = math.sqrt(sum(e * e for e in errors))
    rms = l2 / math.sqrt(len(errors))
    max_abs = max(abs(e) for e in errors)
    return {"l2_error": l2, "rms_error": rms, "max_abs_error": max_abs}


def warn_if_transient(case: dict) -> None:
    if case["final_time"] < 10.0:
        print(
            "Warning: Centerline comparison to Ghia is a steady-state validation. "
            f"Current data may be under-converged in physical time ({case['stem']} final_time={case['final_time']:.3g})."
        )


def plot_profiles(re_value: int, case_results: list[dict]) -> None:
    if plt is None:
        print("matplotlib is not installed; skipping Ghia profile figures.")
        return
    FIGURES.mkdir(exist_ok=True)
    ref_u = GHIA_U[re_value]
    ref_v = GHIA_V[re_value]

    plt.figure(figsize=(6, 6))
    plt.plot([u for _, u in ref_u], [y for y, _ in ref_u], "ko", label="Ghia et al.")
    for result in case_results:
        samples = result["u_samples"]
        tag = result["run_label"] or result["tolerance_strategy"] or "run"
        plt.plot([u for _, u in samples], [y for y, _ in samples], marker=".", label=f"{tag} N={result['nx']}")
    plt.xlabel("u velocity")
    plt.ylabel("y")
    plt.title(f"Lid-Driven Cavity u Centerline, Re={re_value}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / f"ghia_u_centerline_Re{re_value}.png", dpi=220)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot([x for x, _ in ref_v], [v for _, v in ref_v], "ko", label="Ghia et al.")
    for result in case_results:
        samples = result["v_samples"]
        tag = result["run_label"] or result["tolerance_strategy"] or "run"
        plt.plot([x for x, _ in samples], [v for _, v in samples], marker=".", label=f"{tag} N={result['nx']}")
    plt.xlabel("x")
    plt.ylabel("v velocity")
    plt.title(f"Lid-Driven Cavity v Centerline, Re={re_value}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / f"ghia_v_centerline_Re{re_value}.png", dpi=220)
    plt.close()


def plot_error_summary(re_value: int, rows: list[dict]) -> None:
    if plt is None:
        print("matplotlib is not installed; skipping Ghia error figure.")
        return
    selected = [r for r in rows if int(r["re"]) == re_value]
    labels = [f"{r['run_label'] or r['tolerance_strategy'] or 'run'}\nN={r['nx']}\n{r['profile']}" for r in selected]
    values = [float(r["rms_error"]) for r in selected]
    plt.figure(figsize=(max(7, 0.65 * len(selected)), 4.8))
    if selected:
        plt.bar(range(len(selected)), values)
        plt.xticks(range(len(selected)), labels, rotation=35, ha="right")
        plt.yscale("log")
    else:
        plt.text(0.5, 0.5, "No validation rows found", ha="center", va="center", transform=plt.gca().transAxes)
    plt.ylabel("RMS error")
    plt.title(f"Ghia Centerline Error Summary, Re={re_value}")
    plt.tight_layout()
    plt.savefig(FIGURES / f"ghia_error_summary_Re{re_value}.png", dpi=220)
    plt.close()


def write_summary(rows: list[dict]) -> None:
    RESULTS.mkdir(exist_ok=True)
    columns = [
        "language",
        "stem",
        "run_label",
        "tolerance_strategy",
        "re",
        "nx",
        "ny",
        "nt",
        "final_time",
        "profile",
        "l2_error",
        "rms_error",
        "max_abs_error",
    ]
    with (RESULTS / "ghia_validation_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--re", type=int, action="append", help="Reynolds number to validate. Default: 100")
    parser.add_argument("--generate-rust-fields", action="store_true", help="If fields are missing, run one Rust validation case and export fields.")
    parser.add_argument("--nx", type=int, default=256, help="Grid size for optional single-case field generation.")
    parser.add_argument("--nt", type=int, default=1000, help="Timestep count for optional single-case field generation.")
    parser.add_argument("--plot-all", action="store_true", help="Plot every exported field case instead of one representative case per label.")
    args = parser.parse_args()
    re_filter = set(args.re or [100])

    unsupported = sorted(re for re in re_filter if re not in GHIA_U or re not in GHIA_V)
    if unsupported:
        print(f"No built-in Ghia data for Re={unsupported}; skipping those values.")
        re_filter -= set(unsupported)

    cases = load_cases(re_filter)
    if not cases and args.generate_rust_fields:
        re_value = sorted(re_filter)[0]
        subprocess.run(
            [
                "cargo",
                "run",
                "--release",
                "--",
                "--nx",
                str(args.nx),
                "--ny",
                str(args.nx),
                "--re",
                str(re_value),
                "--nt",
                str(args.nt),
                "--poisson-solver",
                "multigrid",
                "--mode",
                "optimized_tolerance",
                "--write-fields",
            ],
            cwd=ROOT,
            check=True,
        )
        cases = load_cases(re_filter)

    if not cases:
        print("No final velocity fields found in results/fields/.")
        print("Centerline validation requires exported MAC u/v/p fields; summaries alone are not enough.")
        print("Generate one Rust field case with:")
        print("  cargo run --release -- --nx 256 --ny 256 --re 100 --nt 1000 --poisson-solver multigrid --mode optimized_tolerance --write-fields")
        print("Or explicitly generate from this script with:")
        print("  python scripts/validate_ghia_centerlines.py --generate-rust-fields")
        return

    plot_cases = select_plot_cases(cases, args.plot_all)
    plot_stems = {case["stem"] for case in plot_cases}
    rows = []
    by_re: dict[int, list[dict]] = {}
    for case in cases:
        warn_if_transient(case)
        u_samples, v_samples = extract_profiles(case)
        u_err = error_metrics(GHIA_U[case["re"]], u_samples)
        v_err = error_metrics(GHIA_V[case["re"]], v_samples)
        result = {**case, "u_samples": u_samples, "v_samples": v_samples}
        if case["stem"] in plot_stems:
            by_re.setdefault(case["re"], []).append(result)
        for profile, metrics in [("u_centerline", u_err), ("v_centerline", v_err)]:
            rows.append(
                {
                    "language": case["language"],
                    "stem": case["stem"],
                    "run_label": case["run_label"],
                    "tolerance_strategy": case["tolerance_strategy"],
                    "re": case["re"],
                    "nx": case["nx"],
                    "ny": case["ny"],
                    "nt": case["nt"],
                    "final_time": f"{case['final_time']:.12e}",
                    "profile": profile,
                    "l2_error": f"{metrics['l2_error']:.12e}",
                    "rms_error": f"{metrics['rms_error']:.12e}",
                    "max_abs_error": f"{metrics['max_abs_error']:.12e}",
                }
            )

    write_summary(rows)
    for re_value, case_results in sorted(by_re.items()):
        plot_profiles(re_value, case_results)
        plot_error_summary(re_value, rows)

    print(f"Wrote {RESULTS / 'ghia_validation_summary.csv'}")
    if plt is not None:
        print(f"Wrote Ghia validation figures to {FIGURES}")


if __name__ == "__main__":
    main()
