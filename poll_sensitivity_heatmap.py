"""Poll sensitivity heatmap: winner win-probability over a Fidesz x TISZA vote-share grid.

Each cell shows the probability of the leading party winning (TISZA blue side,
Fidesz orange side). Custom party-colour diverging colormap.

Usage:
    python poll_sensitivity_heatmap.py --n-sim 200 --step 2.0   # quick draft
    python poll_sensitivity_heatmap.py --n-sim 500 --step 1.0   # higher resolution
"""

from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from lib.config import SimConfig
from lib.transfer_model import (
    Q_PRIOR,
    load_baseline_and_matrix,
    load_nowcast,
    calibrate_transfer_matrix,
    run_simulation,
    aggregate_results,
)

# Party brand colours
FIDESZ_ORANGE = "#F4831F"   # Fidesz official orange
TISZA_BLUE    = "#003F87"   # TISZA / Magyar Peter blue

MINOR_KEYS = ["MH26", "MKKP26", "Bal26", "Other26"]


def _make_party_cmap() -> mcolors.LinearSegmentedColormap:
    """Diverging colormap: Fidesz orange (0%) -- white (50%) -- TISZA blue (100%)."""
    colors_list = [FIDESZ_ORANGE, "#FFFFFF", TISZA_BLUE]
    return mcolors.LinearSegmentedColormap.from_list(
        "FideszTISZA", colors_list, N=512
    )


def build_shares(
    f_raw: float,
    t_raw: float,
    minor_shares: dict[str, float],
) -> dict[str, float]:
    total = f_raw + t_raw + sum(minor_shares.values())
    shares: dict[str, float] = {"Fidesz26": f_raw / total, "TISZA26": t_raw / total}
    for k, v in minor_shares.items():
        shares[k] = v / total
    return shares


def compute_grid(
    V22: np.ndarray,
    districts: list[str],
    minor_shares: dict[str, float],
    fidesz_vals: np.ndarray,
    tisza_vals: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    """Return grid of shape (nT, nF) with P(TISZA largest) in [0, 100]."""
    nF, nT = len(fidesz_vals), len(tisza_vals)
    grid = np.full((nT, nF), np.nan)
    flat_se = {k: 0.02 for k in ["Fidesz26", "TISZA26"] + MINOR_KEYS}
    total_cells = nF * nT
    done = 0

    for i, t_raw in enumerate(tisza_vals):
        for j, f_raw in enumerate(fidesz_vals):
            done += 1
            shares_cell = build_shares(f_raw, t_raw, minor_shares)
            print(
                f"  [{done:3d}/{total_cells}] F={f_raw*100:.1f}%  T={t_raw*100:.1f}%  ...",
                end=" ", flush=True,
            )
            Q_cal = calibrate_transfer_matrix(
                V22, shares_cell, flat_se, Q_PRIOR, cfg.lam_prior
            )
            sim_results = run_simulation(V22, Q_cal, districts, cfg, verbose=False)
            agg = aggregate_results(sim_results)
            p = agg["outcomes"]["p_tisza_largest"] * 100.0
            grid[i, j] = p
            winner = "TISZA" if p >= 50 else "Fidesz"
            disp = p if p >= 50 else 100 - p
            print(f"{winner} {disp:.1f}%")

    return grid


def plot_heatmap(
    grid: np.ndarray,
    fidesz_vals: np.ndarray,
    tisza_vals: np.ndarray,
    current_f: float,
    current_t: float,
    out_path: str,
) -> None:
    cmap = _make_party_cmap()

    f_step = float(fidesz_vals[1] - fidesz_vals[0]) if len(fidesz_vals) > 1 else 0.01
    t_step = float(tisza_vals[1] - tisza_vals[0])  if len(tisza_vals)  > 1 else 0.01

    extent = [
        fidesz_vals[0] * 100 - f_step * 50,
        fidesz_vals[-1] * 100 + f_step * 50,
        tisza_vals[0] * 100 - t_step * 50,
        tisza_vals[-1] * 100 + t_step * 50,
    ]

    fig, ax = plt.subplots(figsize=(14, 11))

    im = ax.imshow(
        grid,
        cmap=cmap,
        vmin=0, vmax=100,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="bilinear",   # smooth gradient between grid points
    )

    # --- Cell annotations: show winner's probability ---
    for i, t_raw in enumerate(tisza_vals):
        for j, f_raw in enumerate(fidesz_vals):
            val = grid[i, j]
            if np.isnan(val):
                continue
            if val >= 50:
                label = f"T {val:.0f}%"
                # on blue side: white text when dark, dark when near white
                text_color = "white" if val > 65 else "#003F87"
            else:
                label = f"F {100-val:.0f}%"
                text_color = "white" if val < 35 else "#F4831F"

            ax.text(
                f_raw * 100, t_raw * 100, label,
                ha="center", va="center",
                fontsize=7, fontweight="bold",
                color=text_color,
            )

    # --- 50% contour (toss-up line) ---
    X = fidesz_vals * 100
    Y = tisza_vals * 100
    try:
        CS50 = ax.contour(X, Y, grid, levels=[50],
                          colors=["#333333"], linewidths=[2.5], linestyles=["-"])
        ax.clabel(CS50, inline=True, fontsize=10, fmt="50%% toss-up")

        # Additional contours at 75% and 90% for each side
        CS_t = ax.contour(X, Y, grid, levels=[75, 90],
                          colors=[TISZA_BLUE, TISZA_BLUE],
                          linewidths=[1.4, 1.0], linestyles=["--", ":"])
        ax.clabel(CS_t, inline=True, fontsize=8, fmt=lambda v: f"T {v:.0f}%")

        CS_f = ax.contour(X, Y, grid, levels=[25, 10],
                          colors=[FIDESZ_ORANGE, FIDESZ_ORANGE],
                          linewidths=[1.4, 1.0], linestyles=["--", ":"])
        ax.clabel(CS_f, inline=True, fontsize=8, fmt=lambda v: f"F {100-v:.0f}%")
    except Exception:
        pass

    # --- Current polling position star ---
    ax.plot(
        current_f * 100, current_t * 100,
        marker="*", markersize=22,
        color="gold", markeredgecolor="black", markeredgewidth=1.3,
        zorder=10, linestyle="none",
        label=f"Current polls  Fidesz={current_f*100:.1f}%  TISZA={current_t*100:.1f}%",
    )

    # --- Axes ---
    ax.set_xlabel("Fidesz vote share (%)", fontsize=13, labelpad=8)
    ax.set_ylabel("TISZA vote share (%)", fontsize=13, labelpad=8)
    ax.set_title(
        "Election outcome sensitivity  --  winner win-probability by polling scenario",
        fontweight="bold", fontsize=13, pad=12,
    )

    ax.set_xticks(fidesz_vals * 100)
    ax.set_yticks(tisza_vals * 100)
    ax.set_xticklabels(
        [f"{v*100:.1f}%" for v in fidesz_vals], rotation=45, ha="right", fontsize=8
    )
    ax.set_yticklabels([f"{v*100:.1f}%" for v in tisza_vals], fontsize=8)

    # --- Colorbar with party labels ---
    cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("P(TISZA largest party)  [0%=Fidesz certain, 100%=TISZA certain]",
                   fontsize=10)
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.set_ticklabels(["Fidesz\ncertain", "F 75%", "Toss-up\n50-50", "T 75%",
                         "TISZA\ncertain"])

    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Poll sensitivity heatmap")
    p.add_argument("--n-sim", type=int, default=500, help="MC draws per cell")
    p.add_argument("--step", type=float, default=1.0,
                   help="Grid step in percentage points (default 1.0pp)")
    p.add_argument("--f-min", type=float, default=33.0)
    p.add_argument("--f-max", type=float, default=48.0)
    p.add_argument("--t-min", type=float, default=40.0)
    p.add_argument("--t-max", type=float, default=55.0)
    p.add_argument("--output-dir", default="output")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimConfig(n_sim=args.n_sim, seed=args.seed)

    print("Loading 2022 baseline (Excel files)...")
    V22, districts = load_baseline_and_matrix()
    print(f"  {len(districts)} districts loaded")

    print("Loading nowcast...")
    nowcast_shares, nowcast_se, _ = load_nowcast(cfg.nowcast_json)
    current_f = nowcast_shares["Fidesz26"]
    current_t = nowcast_shares["TISZA26"]
    print(f"  Fidesz={current_f*100:.2f}%  TISZA={current_t*100:.2f}%  (renormalized)")

    minor_shares = {k: nowcast_shares[k] for k in MINOR_KEYS}
    minor_sum = sum(minor_shares.values())
    print(f"  Minor parties fixed at {minor_sum*100:.2f}% total")

    step = args.step / 100.0
    fidesz_vals = np.arange(args.f_min / 100.0, (args.f_max + step * 0.1) / 100.0, step)
    tisza_vals  = np.arange(args.t_min / 100.0, (args.t_max + step * 0.1) / 100.0, step)

    total_cells = len(fidesz_vals) * len(tisza_vals)
    print(f"\nGrid: {len(fidesz_vals)} Fidesz x {len(tisza_vals)} TISZA = {total_cells} cells")
    print(f"n_sim={cfg.n_sim} per cell  ({total_cells * cfg.n_sim:,} total MC draws)\n")

    grid = compute_grid(V22, districts, minor_shares, fidesz_vals, tisza_vals, cfg)

    grid_path = Path(args.output_dir) / "sensitivity_grid.npz"
    np.savez(
        grid_path,
        grid=grid,
        fidesz_vals=fidesz_vals,
        tisza_vals=tisza_vals,
        minor_Bal26=minor_shares.get("Bal26", 0.0),
        minor_MH26=minor_shares.get("MH26", 0.0),
        minor_MKKP26=minor_shares.get("MKKP26", 0.0),
        minor_Other26=minor_shares.get("Other26", 0.0),
    )
    print(f"Grid saved: {grid_path}")

    out_path = str(Path(args.output_dir) / "poll_sensitivity_heatmap.png")
    plot_heatmap(grid, fidesz_vals, tisza_vals, current_f, current_t, out_path)


if __name__ == "__main__":
    main()
