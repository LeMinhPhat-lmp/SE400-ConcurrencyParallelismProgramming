#!/usr/bin/env python3
"""Read raw benchmark CSV, compute statistics and metrics, draw figures.

Metrics: median/mean/std/CV, speedup, efficiency, throughput, effective
bandwidth, Karp-Flatt serial fraction.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend: no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ELEM_BYTES = 8  # uint64_t

# Bytes moved between CPU and memory per element (approximate model).
BYTES_PER_ELEM = {
    ("seq", "reduce"): 1,
    ("par", "reduce"): 1,
    ("seq", "scan_inc"): 2,  # read in + write out
    ("par", "scan_inc"): 3,  # read in (phase 1) + read in (phase 3) + write out
}


def load(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    bad = df[df["correct"] != 1]
    if not bad.empty:
        raise SystemExit(f"Found {len(bad)} incorrect measurements; aborting analysis.")
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["algo", "op", "dtype", "n", "threads"]
    g = df.groupby(group_cols)["time_ns"]
    s = g.agg(
        reps="count",
        median="median",
        mean="mean",
        std="std",
        min="min",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    ).reset_index()
    s["cv"] = s["std"] / s["mean"]

    # Sequential baseline (median) for the same (op, n).
    base = (
        s[s["algo"] == "seq"][["op", "n", "median"]]
        .rename(columns={"median": "t_seq"})
    )
    s = s.merge(base, on=["op", "n"], how="left")

    # Speedup, efficiency (relative to the best sequential implementation).
    s["speedup"] = s["t_seq"] / s["median"]
    s["efficiency"] = s["speedup"] / s["threads"]
    # Error band of speedup from the IQR (a larger time -> a smaller speedup).
    s["speedup_lo"] = s["t_seq"] / s["q75"]
    s["speedup_hi"] = s["t_seq"] / s["q25"]

    # Throughput (Gelem/s) and effective bandwidth (GB/s). time in ns:
    # elements / ns == Gelem/s ; bytes / ns == GB/s.
    s["throughput_gelems"] = s["n"] / s["median"]
    bpe = s.apply(lambda r: BYTES_PER_ELEM[(r["algo"], r["op"])], axis=1)
    s["bandwidth_gbs"] = s["n"] * ELEM_BYTES * bpe / s["median"]

    # Parallel overhead p*T_p - T_seq (ns).
    s["overhead_ns"] = s["threads"] * s["median"] - s["t_seq"]

    # Karp-Flatt: e = (1/S - 1/p) / (1 - 1/p), defined for p > 1.
    p = s["threads"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        kf = (1.0 / s["speedup"] - 1.0 / p) / (1.0 - 1.0 / p)
    s["karp_flatt"] = np.where(p > 1, kf, np.nan)
    return s


# ---------------------------------------------------------------- figures --

def fig_speedup(s: pd.DataFrame, n: int, outdir: Path) -> None:
    """E1: strong scaling at fixed n."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for op, marker in [("reduce", "o"), ("scan_inc", "s")]:
        d = s[(s.algo == "par") & (s.op == op) & (s.n == n)].sort_values("threads")
        if d.empty:
            continue
        yerr = [d["speedup"] - d["speedup_lo"], d["speedup_hi"] - d["speedup"]]
        ax.errorbar(d["threads"], d["speedup"], yerr=np.abs(yerr), marker=marker,
                    capsize=3, label=op)
    pmax = s[(s.algo == "par") & (s.n == n)]["threads"].max()
    ax.plot([1, pmax], [1, pmax], "k--", linewidth=1, label="ideal (S = p)")
    ax.set_xlabel("threads p")
    ax.set_ylabel("speedup S = T_seq / T_p")
    ax.set_title(f"Strong scaling, n = {n:,}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "speedup_vs_threads.png", dpi=150)
    plt.close(fig)


def fig_efficiency(s: pd.DataFrame, n: int, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for op, marker in [("reduce", "o"), ("scan_inc", "s")]:
        d = s[(s.algo == "par") & (s.op == op) & (s.n == n)].sort_values("threads")
        ax.plot(d["threads"], d["efficiency"], marker=marker, label=op)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("threads p")
    ax.set_ylabel("efficiency E = S / p")
    ax.set_title(f"Parallel efficiency, n = {n:,}")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "efficiency_vs_threads.png", dpi=150)
    plt.close(fig)


def fig_size_effect(s: pd.DataFrame, op: str, outdir: Path) -> None:
    """E2: how n affects throughput and speedup (crossover point)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    seq = s[(s.algo == "seq") & (s.op == op)].sort_values("n")
    axes[0].plot(seq["n"], seq["throughput_gelems"], "k--o", label="sequential")
    for p in sorted(s[s.algo == "par"]["threads"].unique()):
        d = s[(s.algo == "par") & (s.op == op) & (s.threads == p)].sort_values("n")
        axes[0].plot(d["n"], d["throughput_gelems"], marker="o", label=f"p={p}")
        axes[1].plot(d["n"], d["speedup"], marker="o", label=f"p={p}")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("n (elements)")
    axes[0].set_ylabel("throughput [Gelem/s]")
    axes[0].set_title(f"Throughput ({op})")
    axes[1].axhline(1.0, color="k", linestyle="--", linewidth=1)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("n (elements)")
    axes[1].set_ylabel("speedup")
    axes[1].set_title(f"Speedup vs n ({op}); S < 1 means slower than sequential")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"size_effect_{op}.png", dpi=150)
    plt.close(fig)


def fig_bandwidth(s: pd.DataFrame, n: int, outdir: Path) -> None:
    """E4: effective memory bandwidth."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for op, marker in [("reduce", "o"), ("scan_inc", "s")]:
        d = s[(s.algo == "par") & (s.op == op) & (s.n == n)].sort_values("threads")
        ax.plot(d["threads"], d["bandwidth_gbs"], marker=marker, label=f"par {op}")
        seq = s[(s.algo == "seq") & (s.op == op) & (s.n == n)]
        if not seq.empty:
            ax.axhline(seq["bandwidth_gbs"].iloc[0], linestyle=":", linewidth=1,
                       label=f"seq {op}")
    ax.set_xlabel("threads p")
    ax.set_ylabel("effective bandwidth [GB/s]")
    ax.set_title(f"Memory bandwidth, n = {n:,}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "bandwidth_vs_threads.png", dpi=150)
    plt.close(fig)


def fig_karp_flatt(s: pd.DataFrame, n: int, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for op, marker in [("reduce", "o"), ("scan_inc", "s")]:
        d = s[(s.algo == "par") & (s.op == op) & (s.n == n) & (s.threads > 1)]
        d = d.sort_values("threads")
        ax.plot(d["threads"], d["karp_flatt"], marker=marker, label=op)
    ax.set_xlabel("threads p")
    ax.set_ylabel("Karp-Flatt serial fraction e(p)")
    ax.set_title(f"Karp-Flatt metric, n = {n:,}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "karp_flatt.png", dpi=150)
    plt.close(fig)


def fig_time_boxplot(df: pd.DataFrame, n: int, outdir: Path) -> None:
    """Distribution of raw measurements: shows noise and outliers."""
    d = df[(df.n == n) & (df.op == "scan_inc")].copy()
    d["label"] = d["algo"] + "/p" + d["threads"].astype(str)
    order = ["seq/p1"] + [f"par/p{p}" for p in sorted(d[d.algo == "par"]["threads"].unique())]
    data = [d[d.label == lab]["time_ns"] / 1e6 for lab in order]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(data, labels=order, showfliers=True)
    ax.set_ylabel("time [ms]")
    ax.set_title(f"Scan time distribution, n = {n:,}")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outdir / "scan_time_boxplot.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw.csv")
    ap.add_argument("--summary", default="results/summary.csv")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--n", type=int, default=0,
                    help="n used for fixed-size figures (default: largest n)")
    args = ap.parse_args()

    outdir = Path(args.figdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load(args.raw)
    s = summarise(df)
    s.to_csv(args.summary, index=False)

    n_fixed = args.n if args.n else int(s["n"].max())

    fig_speedup(s, n_fixed, outdir)
    fig_efficiency(s, n_fixed, outdir)
    fig_bandwidth(s, n_fixed, outdir)
    fig_karp_flatt(s, n_fixed, outdir)
    fig_time_boxplot(df, n_fixed, outdir)
    for op in ["reduce", "scan_inc"]:
        fig_size_effect(s, op, outdir)

    noisy = s[s["cv"] > 0.05]
    if not noisy.empty:
        print(f"Warning: {len(noisy)} configurations have CV > 5% (noisy measurements).")
    print(f"Wrote {args.summary} and figures in {outdir}/")


if __name__ == "__main__":
    main()