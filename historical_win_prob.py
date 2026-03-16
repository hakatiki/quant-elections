"""Historical win-probability time series.

For each day in nowcast_daily.csv between START_DATE and END_DATE, use the
rolling nowcast vote shares to calibrate the transfer matrix and run a small
Monte Carlo seat simulation.  Records P(TISZA largest) and P(Fidesz largest)
as a time series and plots the result.

Usage:
    python historical_win_prob.py [--replot]

    --replot  Skip simulation and just re-plot output/historical_win_prob.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from lib.config import SimConfig
from lib.transfer_model import (
    Q_PRIOR,
    aggregate_results,
    calibrate_transfer_matrix,
    load_baseline_and_matrix,
    run_simulation,
)

# ── Constants ─────────────────────────────────────────────────────────────────

START_DATE = "2024-11-01"
END_DATE = "2026-02-22"

# Mapping from nowcast_daily.csv party prefix → calibration target key
PARTY_MAP = {
    "Fidesz": "Fidesz26",
    "TISZA": "TISZA26",
    "Bal": "Bal26",
    "MH": "MH26",
    "MKKP": "MKKP26",
    "EM": "Other26",
}

Z_80 = 1.2816  # normal quantile for 80% CI (used to back-compute SE)

OUTPUT_DIR = Path("output")
OUTPUT_CSV = OUTPUT_DIR / "historical_win_prob.csv"
OUTPUT_PNG = OUTPUT_DIR / "historical_win_prob.png"

NOWCAST_CSV = "nowcast_daily.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_shares_se(row: pd.Series) -> tuple[dict, dict]:
    """Extract nowcast_shares and nowcast_se dicts from one nowcast_daily row."""
    shares: dict[str, float] = {}
    se: dict[str, float] = {}

    for party, target in PARTY_MAP.items():
        hat = row.get(f"{party}_hat", np.nan)
        lo = row.get(f"{party}_lo80", np.nan)
        hi = row.get(f"{party}_hi80", np.nan)

        # Fill missing small parties with 0
        if pd.isna(hat):
            hat, lo, hi = 0.0, 0.0, 0.0

        shares[target] = hat / 100.0
        # Back-compute SE from 80% CI; floor at 0.5 pp to avoid zero weights
        if pd.isna(lo) or pd.isna(hi) or hi <= lo:
            se_val = 0.005  # fallback 0.5 pp
        else:
            se_val = (hi - lo) / (2 * Z_80 * 100.0)
        se[target] = max(se_val, 0.001)

    # Renormalise shares to sum to 1
    total = sum(shares.values())
    if total > 0:
        shares = {k: v / total for k, v in shares.items()}

    return shares, se


# ── Main ──────────────────────────────────────────────────────────────────────

def run_simulation_loop(df: pd.DataFrame) -> pd.DataFrame:
    """Calibrate + simulate for each date row, return results DataFrame."""
    print(f"Loading 2022 baseline...", flush=True)
    V22, districts = load_baseline_and_matrix()

    cfg = SimConfig(
        n_sim=200,
        sigma_district=0.15,
        conc_national=80.0,
        lam_prior=5.0,
        seed=2026,
        include_nationality_seat=True,
    )

    records = []
    n = len(df)
    print(f"Running {n} daily nowcast -> simulation steps ({cfg.n_sim} draws each)...", flush=True)
    print(f"Estimated time: {n * 0.6 / 60:.0f}-{n * 1.2 / 60:.0f} min\n", flush=True)

    for i, (_, row) in enumerate(df.iterrows()):
        date = row["date"]

        if i % 30 == 0:
            pct = 100 * i / n
            print(f"  [{pct:5.1f}%] {date.date()} ...", flush=True)

        shares, se = build_shares_se(row)

        # Skip dates where Fidesz or TISZA have no data
        if shares.get("Fidesz26", 0) == 0 or shares.get("TISZA26", 0) == 0:
            continue

        try:
            Q_cal = calibrate_transfer_matrix(V22, shares, se, Q_PRIOR, cfg.lam_prior)
            sim = run_simulation(V22, Q_cal, districts, cfg, verbose=False)
            agg = aggregate_results(sim)
            out = agg["outcomes"]

            records.append({
                "date": date,
                "p_tisza_largest": out["p_tisza_largest"],
                "p_fidesz_largest": out["p_fidesz_largest"],
                "tisza_median_seats": out["median_tisza"],
                "fidesz_median_seats": out["median_fidesz"],
            })
        except Exception as exc:
            print(f"  WARNING: {date.date()} failed: {exc}", file=sys.stderr)
            continue

    print(f"\nDone. {len(records)} dates processed.", flush=True)
    return pd.DataFrame(records)


def plot_results(results: pd.DataFrame) -> None:
    """Plot the win-probability time series."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        results["date"],
        results["p_tisza_largest"] * 100,
        color="#1f77b4",
        linewidth=2,
        label="P(TISZA largest)",
    )
    ax.plot(
        results["date"],
        results["p_fidesz_largest"] * 100,
        color="#d62728",
        linewidth=2,
        label="P(Fidesz largest)",
    )
    ax.axhline(50, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax.set_title(
        "P(win plurality) over time -- rolling nowcast -> MC simulation (n=200/day)",
        fontsize=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Win probability")
    ax.legend(loc="center left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150)
    print(f"Plot saved to {OUTPUT_PNG}", flush=True)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical win-probability time series")
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Skip simulation; re-plot from existing output/historical_win_prob.csv",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.replot:
        if not OUTPUT_CSV.exists():
            sys.exit(f"ERROR: {OUTPUT_CSV} not found. Run without --replot first.")
        print(f"Loading existing results from {OUTPUT_CSV}")
        results = pd.read_csv(OUTPUT_CSV, parse_dates=["date"])
    else:
        # Load and filter nowcast daily
        df = pd.read_csv(NOWCAST_CSV, parse_dates=["date"])
        df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].reset_index(drop=True)
        print(f"Loaded {len(df)} rows from {NOWCAST_CSV} ({START_DATE} to {END_DATE})")

        results = run_simulation_loop(df)

        results.to_csv(OUTPUT_CSV, index=False)
        print(f"Results saved to {OUTPUT_CSV}", flush=True)

    plot_results(results)


if __name__ == "__main__":
    main()
