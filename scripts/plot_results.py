"""Generate result plots from benchmark CSV (distributions, not single numbers)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(csv_path: str, out_prefix: str):
    df = pd.read_csv(csv_path)
    methods = sorted(df["method"].unique())
    depths = sorted(df["depth"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    # ---- panel 1: P(optimal) distributions ---------------------------------
    ax = axes[0]
    data, labels, colors = [], [], []
    palette = {"xy": "#3b7dd8", "soft": "#d8813b"}
    for m in methods:
        for d in depths:
            vals = df[(df.method == m) & (df.depth == d)]["p_optimal"]
            data.append(vals.values)
            labels.append(f"{m}\np={d}")
            colors.append(palette[m])
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("P(sample optimal state)")
    ax.set_title("Optimal-state probability\n(per run; higher is better)")
    ax.grid(axis="y", alpha=0.4)

    # ---- panel 2: feasibility ----------------------------------------------
    ax = axes[1]
    for m in methods:
        means = [df[(df.method == m) & (df.depth == d)]["p_feasible"].mean() for d in depths]
        stds = [df[(df.method == m) & (df.depth == d)]["p_feasible"].std() for d in depths]
        x = np.arange(len(depths)) + (0.18 if m == "xy" else -0.18)
        ax.bar(x, means, yerr=stds, width=0.32,
               label=m, color=palette[m], alpha=0.85, capsize=3)
    ax.set_xticks(np.arange(len(depths)))
    ax.set_xticklabels([f"p={d}" for d in depths])
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_ylabel("feasibility probability of sampled states")
    ax.set_title("Feasibility of sampled assignments\n(XY guaranteed by construction)")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)

    # ---- panel 3: mean feasible objective relative to optimum --------------
    ax = axes[2]
    for m in methods:
        ratios = []
        for d in depths:
            sub = df[(df.method == m) & (df.depth == d)]
            ratios.append((sub["mean_feasible_objective"] / sub["f_opt"]).values)
        bp = ax.boxplot(ratios, positions=np.arange(len(depths)) + (0.18 if m == "xy" else -0.18),
                        widths=0.3, patch_artist=True, manage_ticks=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(palette[m])
            patch.set_alpha(0.7)
    sa_ratio = df.dropna(subset=["sa_objective"]).groupby("instance_seed").agg(
        {"sa_objective": "first", "f_opt": "first"})
    greedy_ratio = df.dropna(subset=["greedy_objective"]).groupby("instance_seed").agg(
        {"greedy_objective": "first", "f_opt": "first"})
    ax.axhline((sa_ratio.sa_objective / sa_ratio.f_opt).mean(), color="green", ls="--",
               label=f"SA ({(sa_ratio.sa_objective/sa_ratio.f_opt).mean():.2f})")
    ax.axhline((greedy_ratio.greedy_objective / greedy_ratio.f_opt).mean(), color="red", ls="--",
               label=f"Greedy ({(greedy_ratio.greedy_objective/greedy_ratio.f_opt).mean():.1f})")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(depths)))
    ax.set_xticklabels([f"p={d}" for d in depths])
    ax.set_ylabel("typical objective / optimum  (log)")
    ax.set_title("Typical solution quality vs classical\n(lower is better; QAOA uses mean feasible)")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)

    fig.suptitle("XY-mixer QAOA vs Soft-QUBO - 6 jobs x 3 nodes, "
                 f"{df.instance_seed.nunique()} instances x {df.qaoa_seed.nunique()} seeds (exact simulation)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = f"{out_prefix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "results/bench_6x3_full.csv"
    prefix = sys.argv[2] if len(sys.argv) > 2 else "results/benchmark_plots"
    main(csv, prefix)
