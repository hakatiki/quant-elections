"""Text summaries, matplotlib plots, and HTML report for simulation results.

All plot functions save to output_dir and return the figure path(s).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive for pipeline use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from lib.poll_model import PARTY_COLORS, PARTY_LABELS, PARTIES
from lib.transfer_model import (
    MAIN_PARTIES_2026, SOURCES, TARGETS, Q_PRIOR,
    aggregate_results, prior_vs_calibrated, transfer_matrix_report,
)

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         11,
})

COLORS_2026 = {
    "Fidesz-KDNP": "#fd8204",
    "TISZA":        "#1B4D8E",
    "Baloldal":     "#e6331a",
    "Mi Hazank":    "#215b2c",
    "MKKP":         "#888888",
    "Other":        "#aaaaaa",
}

LABELS_2026 = {
    "Fidesz-KDNP": "Fidesz-KDNP",
    "TISZA":        "TISZA",
    "Baloldal":     "Baloldal",
    "Mi Hazank":    "Mi Hazank",
    "MKKP":         "MKKP",
    "Other":        "Other/EM",
}


# ── Text reports ──────────────────────────────────────────────────────────────

def print_seat_summary(agg: dict) -> None:
    n = agg["n_sim"]
    print("=" * 82)
    print(f"  SEAT PROJECTION  ({n:,} simulations)")
    print("=" * 82)
    print(f"  {'Party':<16s} {'OEVK med':>9} {'List med':>9} {'Total med':>10} "
          f"{'5th%':>7} {'95th%':>7} {'P(>0)':>7}")
    print("  " + "-" * 72)
    for p in agg["parties"]:
        ts    = agg["total_seats"][p]
        o_med = np.median(agg["oevk_seats"][p])
        l_med = np.median(agg["list_seats"][p])
        t_med = np.median(ts)
        p5    = np.percentile(ts, 5)
        p95   = np.percentile(ts, 95)
        ppos  = (ts > 0).mean() * 100
        label = LABELS_2026.get(p, p)
        print(f"  {label:<16s} {o_med:>8.0f}  {l_med:>8.0f}  {t_med:>9.0f}  "
              f"{p5:>6.0f}  {p95:>6.0f}  {ppos:>6.0f}%")
    print("  " + "-" * 72)
    td = agg["total_seats_per_draw"]
    print(f"  {'Draw total (med)':<16s} {' ':>9} {' ':>9} {np.median(td):>9.0f}  "
          f"{np.percentile(td, 5):>6.0f}  {np.percentile(td, 95):>6.0f}  "
          f"  (should be ~199)")
    print("=" * 82)


def print_outcome_probabilities(agg: dict) -> None:
    o = agg["outcomes"]
    print("=" * 55)
    print("  OUTCOME PROBABILITIES")
    print("=" * 55)
    print(f"  TISZA largest party:           {o['p_tisza_largest']*100:>6.1f}%")
    print(f"  TISZA majority (>=100 seats):  {o['p_tisza_majority']*100:>6.1f}%")
    print(f"  Fidesz largest party:          {o['p_fidesz_largest']*100:>6.1f}%")
    print(f"  Fidesz majority (>=100 seats): {o['p_fidesz_majority']*100:>6.1f}%")
    print(f"  Fidesz supermajority (>=133):  {o['p_fidesz_supermajority']*100:>6.1f}%")
    print(f"  Tied:                          {o['p_tied']*100:>6.1f}%")
    print("  " + "-" * 42)
    print(f"  Median TISZA seats:  {o['median_tisza']:.0f}  "
          f"[{o['p5_tisza']:.0f}-{o['p95_tisza']:.0f}]  (90% CI)")
    print(f"  Median Fidesz seats: {o['median_fidesz']:.0f}  "
          f"[{o['p5_fidesz']:.0f}-{o['p95_fidesz']:.0f}]  (90% CI)")
    print("=" * 55)


def print_poll_summary(current_est: dict, parties: Optional[list[str]] = None) -> None:
    if parties is None:
        parties = [p for p in PARTIES if p in current_est]
    print("=" * 60)
    print("  CURRENT POLLING ESTIMATE")
    print("=" * 60)
    print(f"  {'Party':<20s} {'Estimate':>10} {'90% CI':>20}")
    print("  " + "-" * 54)
    for p in parties:
        lo, med, hi = current_est[p]
        if np.isnan(med):
            continue
        label = PARTY_LABELS.get(p, p)
        print(f"  {label:<20s} {med:>9.1f}%  [{lo:>5.1f}% – {hi:>5.1f}%]")
    print("=" * 60)


# ── Plots: polling ────────────────────────────────────────────────────────────

def plot_polling_trend(
    df: pd.DataFrame,
    trend: dict,
    parties: Optional[list[str]] = None,
    out_path: str = "output/plots/polling_trend.png",
) -> str:
    if parties is None:
        parties = [p for p in PARTIES if p in df.columns]

    fig, ax = plt.subplots(figsize=(16, 7))

    for p in parties:
        if p not in df.columns:
            continue
        color = PARTY_COLORS.get(p, "#555")
        mask = df[p].notna()
        ax.scatter(df.loc[mask, "mid_date"], df.loc[mask, p],
                   color=color, alpha=0.3, s=18, zorder=2)
        if p not in trend:
            continue
        td = trend[p]
        if td["dates"]:
            ax.fill_between(td["dates"], td["lo"], td["hi"], alpha=0.15, color=color)
            ax.plot(td["dates"], td["med"], color=color, linewidth=2.5,
                    label=PARTY_LABELS.get(p, p), zorder=3)

    ax.axhline(5, color="red", linestyle=":", alpha=0.5, linewidth=1, label="5% threshold")
    ax.set_ylabel("Support (%)", fontsize=12)
    ax.set_title("Hungarian Polling Trend with 90% Bootstrap CI", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, 65)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_house_effects(
    house_effects: pd.DataFrame,
    parties: Optional[list[str]] = None,
    out_path: str = "output/plots/house_effects.png",
) -> str:
    if parties is None:
        parties = ["Fidesz", "TISZA"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(house_effects.index))
    width = 0.35
    for i, p in enumerate(parties[:2]):
        vals = house_effects[p].astype(float).values
        bars = ax.bar(x + i * width, vals, width, label=PARTY_LABELS.get(p, p),
                      color=PARTY_COLORS.get(p, "#888"), alpha=0.8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(house_effects.index, rotation=30, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("House effect (pp)", fontsize=12)
    ax.set_title("Pollster House Effects: Fidesz vs TISZA", fontweight="bold", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Plots: seats ──────────────────────────────────────────────────────────────

def plot_seat_distributions(
    agg: dict,
    out_path: str = "output/plots/seat_distributions.png",
) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Total seat histograms
    ax = axes[0]
    for p in ["Fidesz-KDNP", "TISZA", "Mi Hazank"]:
        arr = agg["total_seats"][p]
        med = np.median(arr)
        ax.hist(arr, bins=range(0, 200), alpha=0.6, color=COLORS_2026[p],
                label=f"{LABELS_2026[p]} (med={med:.0f})",
                density=True, edgecolor="white", linewidth=0.3)
    ax.axvline(100, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Majority (100)")
    ax.set_xlabel("Total seats"); ax.set_ylabel("Density")
    ax.set_title("Total Seat Distribution", fontweight="bold")
    ax.legend(fontsize=9); ax.set_xlim(0, 199)

    # 2. OEVK seat distributions
    ax2 = axes[1]
    for p in ["Fidesz-KDNP", "TISZA"]:
        arr = agg["oevk_seats"][p]
        med = np.median(arr)
        ax2.hist(arr, bins=range(0, 107), alpha=0.6, color=COLORS_2026[p],
                 label=f"{LABELS_2026[p]} (med={med:.0f})",
                 density=True, edgecolor="white", linewidth=0.3)
    ax2.axvline(53, color="red", linestyle="--", linewidth=1.5, alpha=0.5)
    ax2.set_xlabel("OEVK seats"); ax2.set_title("District Seat Distribution", fontweight="bold")
    ax2.legend(fontsize=9); ax2.set_xlim(0, 106)

    # 3. Win probability donut
    ax3 = axes[2]
    o = agg["outcomes"]
    sizes = [o["p_tisza_largest"], o["p_fidesz_largest"], o["p_tied"]]
    lbls  = [f"TISZA\n{o['p_tisza_largest']*100:.0f}%",
             f"Fidesz\n{o['p_fidesz_largest']*100:.0f}%",
             f"Tied\n{o['p_tied']*100:.0f}%"]
    colors_pie = [COLORS_2026["TISZA"], COLORS_2026["Fidesz-KDNP"], "#dddddd"]
    wedges, _ = ax3.pie(
        [max(s, 0.001) for s in sizes], labels=lbls, colors=colors_pie,
        startangle=90, textprops={"fontsize": 12, "fontweight": "bold"},
    )
    ax3.add_artist(plt.Circle((0, 0), 0.5, fc="white"))
    ax3.set_title("P(Largest Party)", fontweight="bold")

    n = agg["n_sim"]
    fig.suptitle(f"Seat Projection — {n:,} simulations", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_seat_scatter(
    agg: dict,
    out_path: str = "output/plots/seat_scatter.png",
) -> str:
    fidesz = agg["total_seats"]["Fidesz-KDNP"]
    tisza  = agg["total_seats"]["TISZA"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(fidesz, tisza, alpha=0.06, s=6, color="#555")
    ax.axhline(100, color=COLORS_2026["TISZA"], linestyle="--", alpha=0.5, label="TISZA majority (100)")
    ax.axvline(100, color=COLORS_2026["Fidesz-KDNP"], linestyle="--", alpha=0.5, label="Fidesz majority (100)")
    ax.plot([0, 199], [199, 0], color="grey", linestyle=":", alpha=0.3, label="Sum = 199")
    ax.set_xlabel("Fidesz-KDNP total seats", fontsize=12)
    ax.set_ylabel("TISZA total seats", fontsize=12)
    ax.set_title("Seat Outcome Space (each dot = 1 simulation)", fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 199); ax.set_ylim(0, 199)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_transfer_matrix(
    Q: np.ndarray,
    Q_prior: Optional[np.ndarray] = None,
    out_path: str = "output/plots/transfer_matrix.png",
) -> str:
    """Plot the calibrated transfer matrix, optionally alongside the prior."""
    n_panels = 2 if Q_prior is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(10 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    def _draw_matrix(ax, M, title):
        n_rows, n_cols = M.shape
        im = ax.imshow(M * 100, cmap="YlOrRd", aspect="auto", vmin=0, vmax=85)
        ax.set_xticks(range(n_cols)); ax.set_yticks(range(n_rows))
        ax.set_xticklabels(TARGETS, rotation=35, ha="right", fontsize=10)
        ax.set_yticklabels(SOURCES, fontsize=10)
        for i in range(n_rows):
            for j in range(n_cols):
                val = M[i, j] * 100
                c = "white" if val > 40 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=c)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Transfer probability (%)", fontsize=10)
        ax.set_title(title, fontweight="bold", fontsize=12)

    _draw_matrix(axes[0], Q, "Calibrated Transfer Matrix (Q_cal)")
    if Q_prior is not None:
        _draw_matrix(axes[1], Q_prior, "Prior Transfer Matrix (Q_prior)")

    fig.suptitle("Voter Transfer: 2022 source → 2026 target", fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_competitive_districts(
    sim_results: list[dict],
    districts: list[str],
    out_path: str = "output/plots/competitive_districts.png",
    top_n: int = 30,
) -> str:
    n = len(sim_results)
    fidesz_wins = np.zeros(len(districts))
    tisza_wins  = np.zeros(len(districts))

    for r in sim_results:
        for d_idx, dist in enumerate(districts):
            winner = r["district_seats"].get(dist, "")
            if winner == "Fidesz-KDNP":
                fidesz_wins[d_idx] += 1
            elif winner == "TISZA":
                tisza_wins[d_idx] += 1

    fidesz_rate = fidesz_wins / n
    tisza_rate  = tisza_wins  / n
    gap = np.abs(fidesz_rate - tisza_rate)

    # Sort by competitiveness (smallest gap first)
    order = np.argsort(gap)[:top_n]

    fig, ax = plt.subplots(figsize=(14, max(6, top_n * 0.28)))
    y_pos = np.arange(len(order))
    dist_labels = [districts[i] for i in order]

    ax.barh(y_pos, [tisza_rate[i]  * 100 for i in order],
            color=COLORS_2026["TISZA"],        alpha=0.7, label="TISZA win %")
    ax.barh(y_pos, [-fidesz_rate[i] * 100 for i in order],
            color=COLORS_2026["Fidesz-KDNP"], alpha=0.7, label="Fidesz win %")

    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(dist_labels, fontsize=9)
    ax.set_xlabel("Win rate (% simulations)")
    ax.set_title(f"Most Competitive Districts (top {top_n} by smallest win-rate gap)",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Sensitivity plot ──────────────────────────────────────────────────────────

def plot_sensitivity(
    sweep_results: dict,
    out_path: str = "output/plots/sensitivity.png",
) -> str:
    """Plot sensitivity sweep results.

    sweep_results: {sweep_name: [(value, outcomes_dict), ...]}
    """
    n_panels = len(sweep_results)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, (sweep_name, sweep_data) in zip(axes, sweep_results.items()):
        xs = [v for v, _ in sweep_data]
        ys_largest  = [o["p_tisza_largest"]  * 100 for _, o in sweep_data]
        ys_majority = [o["p_tisza_majority"] * 100 for _, o in sweep_data]

        ax.plot(xs, ys_largest,  "o-", color="#1B4D8E", linewidth=2,
                markersize=7, label="P(TISZA most seats)")
        ax.plot(xs, ys_majority, "s--", color="#5b8fcc", linewidth=1.5,
                markersize=6, label="P(TISZA majority ≥100)")
        ax.set_xlabel(sweep_name, fontsize=11)
        ax.set_ylabel("Probability (%)", fontsize=11)
        ax.set_title(sweep_name, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        for x, y in zip(xs, ys_largest):
            ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8, color="#1B4D8E")

    fig.suptitle("Sensitivity Analysis: P(TISZA outcomes) vs Model Parameters",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_transfer_sensitivity_heatmap(
    grid_results: np.ndarray,
    x_vals: list,
    y_vals: list,
    x_label: str = "conc_national",
    y_label: str = "sigma_district",
    metric: str = "P(TISZA largest) %",
    out_path: str = "output/plots/sensitivity_heatmap.png",
) -> str:
    """2D sensitivity heatmap for two transfer-model parameters."""
    fig, ax = plt.subplots(figsize=(10, 7))

    im = ax.imshow(grid_results, cmap="RdYlGn", aspect="auto",
                   vmin=0, vmax=100, origin="lower")
    ax.set_xticks(range(len(x_vals)))
    ax.set_yticks(range(len(y_vals)))
    ax.set_xticklabels([str(v) for v in x_vals], fontsize=10)
    ax.set_yticklabels([str(v) for v in y_vals], fontsize=10)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f"{metric} — 2D Parameter Sensitivity", fontweight="bold", fontsize=13)

    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            val = grid_results[i, j]
            c = "black" if 30 < val < 70 else "white"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=c)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(metric, fontsize=11)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── JSON report ───────────────────────────────────────────────────────────────

def save_json_report(
    agg: dict,
    current_est: Optional[dict] = None,
    out_path: str = "output/reports/simulation_report.json",
) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "n_sim": agg["n_sim"],
        "outcomes": agg["outcomes"],
        "seat_medians": {
            p: {
                "oevk":  int(np.median(agg["oevk_seats"][p])),
                "list":  int(np.median(agg["list_seats"][p])),
                "total": int(np.median(agg["total_seats"][p])),
                "p5":    int(np.percentile(agg["total_seats"][p], 5)),
                "p95":   int(np.percentile(agg["total_seats"][p], 95)),
            }
            for p in agg["parties"]
        },
        "total_seats_draw_median": int(np.median(agg["total_seats_per_draw"])),
    }

    if current_est is not None:
        report["poll_estimates"] = {
            p: {"lo": round(lo, 2), "med": round(med, 2), "hi": round(hi, 2)}
            for p, (lo, med, hi) in current_est.items()
            if not np.isnan(med)
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return out_path
