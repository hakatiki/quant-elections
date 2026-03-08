"""End-to-end Hungarian election seat simulation pipeline.

Usage:
    python pipeline.py                          # defaults
    python pipeline.py --n-sim 10000            # more draws
    python pipeline.py --no-nationality-seat    # disable German minority fix
    python pipeline.py --sigma-district 0.20    # increase district noise
    python pipeline.py --output-dir out2        # custom output directory

The pipeline:
  1. Run nowcast (refreshes nowcast_latest.json from data.csv)
  2. Load poll data and estimate house effects
  3. Compute current polling estimate with bootstrap CIs
  4. Load 2022 OEVK baseline and build source-bloc matrix
  5. Calibrate voter transfer matrix to match nowcast
  6. Monte Carlo seat simulation (with German minority fix)
  7. Aggregate results and print reports
  8. Save plots and JSON report
"""

from __future__ import annotations

import argparse
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hungarian election seat simulation pipeline")
    p.add_argument("--n-sim",            type=int,   default=5000)
    p.add_argument("--sigma-district",   type=float, default=0.15)
    p.add_argument("--conc-national",    type=float, default=80.0)
    p.add_argument("--lam-prior",        type=float, default=5.0)
    p.add_argument("--seed",             type=int,   default=2026)
    p.add_argument("--output-dir",       default="output")
    p.add_argument("--no-nationality-seat", action="store_true",
                   help="Disable German minority nationality seat (reproduces notebook bug)")
    p.add_argument("--skip-nowcast",     action="store_true",
                   help="Skip nowcast refresh (use existing nowcast_latest.json)")
    p.add_argument("--skip-plots",       action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from lib.config import SimConfig
    cfg = SimConfig(
        n_sim=args.n_sim,
        sigma_district=args.sigma_district,
        conc_national=args.conc_national,
        lam_prior=args.lam_prior,
        seed=args.seed,
        output_dir=args.output_dir,
        include_nationality_seat=not args.no_nationality_seat,
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    (out_dir / "reports").mkdir(exist_ok=True)

    print("=" * 65)
    print("  HUNGARIAN ELECTION SEAT SIMULATION PIPELINE")
    print("=" * 65)
    print(f"  n_sim={cfg.n_sim:,}  sigma={cfg.sigma_district}  conc={cfg.conc_national}")
    print(f"  lam_prior={cfg.lam_prior}  seed={cfg.seed}")
    print(f"  nationality_seat={cfg.include_nationality_seat}")
    print()

    # ── Step 1: Refresh nowcast ───────────────────────────────────────────────
    if not args.skip_nowcast:
        print("Step 1 | Refreshing nowcast...")
        try:
            result = subprocess.run(
                [sys.executable, "nowcast.py", "--csv", cfg.data_csv],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                print(f"  WARNING: nowcast.py returned {result.returncode}")
                print(result.stderr[:500])
            else:
                # Print the nowcast summary lines
                for line in result.stdout.splitlines():
                    if line.strip():
                        print(f"  {line}")
        except Exception as e:
            print(f"  WARNING: could not run nowcast.py: {e}")
    else:
        print("Step 1 | Skipping nowcast refresh.")

    # ── Step 2: Load polls and house effects ──────────────────────────────────
    print("\nStep 2 | Loading poll data and estimating house effects...")
    from lib.poll_model import (
        load_polls, estimate_house_effects, bootstrap_estimate,
        compute_rolling_trend, PARTIES, PARTY_LABELS,
    )
    df_polls = load_polls(cfg.data_csv)
    parties = [p for p in PARTIES if p in df_polls.columns]
    print(f"  {len(df_polls)} polls  |  "
          f"{df_polls['mid_date'].min():%Y-%m-%d} to {df_polls['mid_date'].max():%Y-%m-%d}")
    print(f"  Pollsters: {', '.join(sorted(df_polls['pollster'].dropna().unique()))}")

    house_effects = estimate_house_effects(
        df_polls, parties,
        since=cfg.house_effect_since,
        min_polls=cfg.house_effect_min_polls,
    )
    print("\n  Pollster house effects (pp, vs period mean):")
    print(f"  {'Pollster':<15s}", "  ".join(f"{p:>6s}" for p in parties[:5]))
    for pollster in house_effects.index:
        row = "  ".join(
            f"{house_effects.loc[pollster, p]:>+6.1f}" if pd.notna(house_effects.loc[pollster, p]) else f"{'—':>6s}"
            for p in parties[:5]
        )
        print(f"  {pollster:<15s}  {row}")

    # ── Step 3: Current polling estimate ─────────────────────────────────────
    print("\nStep 3 | Computing current polling estimate...")
    current_est = bootstrap_estimate(
        df_polls, parties,
        halflife_days=cfg.halflife_days,
        max_age_days=cfg.max_age_days,
        house_effects=house_effects,
    )
    from lib.reports import print_poll_summary
    print()
    print_poll_summary(current_est, parties)

    # ── Step 4: Load 2022 baseline ────────────────────────────────────────────
    print("\nStep 4 | Loading 2022 OEVK baseline...")
    from lib.transfer_model import (
        load_baseline_and_matrix, load_nowcast,
        calibrate_transfer_matrix, run_simulation, aggregate_results,
        check_calibration, transfer_matrix_report, prior_vs_calibrated,
        Q_PRIOR,
    )
    V22, districts = load_baseline_and_matrix()
    print(f"  {len(districts)} districts  |  {V22.sum():,.0f} total 2022 OEVK votes")
    print("  Source bloc totals:")
    from lib.transfer_model import SOURCES
    for j, src in enumerate(SOURCES):
        pct = V22[:, j].sum() / V22.sum() * 100
        print(f"    {src:<12s} {V22[:, j].sum():>10,.0f}  ({pct:.1f}%)")

    # ── Step 5: Calibrate transfer matrix ─────────────────────────────────────
    print("\nStep 5 | Calibrating voter transfer matrix...")
    nowcast_shares, nowcast_se = load_nowcast(cfg.nowcast_json)
    Q_cal = calibrate_transfer_matrix(V22, nowcast_shares, nowcast_se, Q_PRIOR, cfg.lam_prior)

    transfer_matrix_report(Q_cal, "Calibrated transfer matrix Q_cal")
    print()
    prior_vs_calibrated(Q_PRIOR, Q_cal)
    print()
    check_calibration(V22, Q_cal, nowcast_shares)

    # ── Step 6: Monte Carlo simulation ────────────────────────────────────────
    print(f"\nStep 6 | Running {cfg.n_sim:,} MC simulations...")
    if cfg.include_nationality_seat:
        print("  German minority nationality seat: ENABLED (correct)")
    else:
        print("  German minority nationality seat: DISABLED (reproduces notebook)")
    sim_results = run_simulation(V22, Q_cal, districts, cfg, verbose=True)

    # ── Step 7: Aggregate and report ──────────────────────────────────────────
    print("\nStep 7 | Aggregating results...")
    agg = aggregate_results(sim_results)

    print()
    from lib.reports import print_seat_summary, print_outcome_probabilities
    print_seat_summary(agg)
    print()
    print_outcome_probabilities(agg)

    # Close district win analysis
    print("\nCompetitive districts (win rate 40–60%):")
    fidesz_w = np.zeros(len(districts))
    tisza_w  = np.zeros(len(districts))
    for r in sim_results:
        for d_idx, dist in enumerate(districts):
            w = r["district_seats"].get(dist, "")
            if w == "Fidesz-KDNP": fidesz_w[d_idx] += 1
            elif w == "TISZA":      tisza_w[d_idx]  += 1
    fr = fidesz_w / len(sim_results)
    tr = tisza_w  / len(sim_results)
    competitive = [(districts[i], fr[i], tr[i]) for i in range(len(districts))
                   if 0.40 <= max(fr[i], tr[i]) <= 0.60]
    competitive.sort(key=lambda x: abs(x[1] - x[2]))
    print(f"  {'District':>10s}  {'Fidesz':>8s}  {'TISZA':>8s}  {'Gap':>6s}")
    for dist, f, t in competitive[:20]:
        print(f"  {dist:>10s}  {f:>7.0%}  {t:>7.0%}  {abs(f-t):>5.0%}")
    print(f"  ... {len(competitive)} competitive districts total")

    # ── Step 8: Save outputs ──────────────────────────────────────────────────
    if not args.skip_plots:
        print("\nStep 8 | Generating plots...")
        from lib.reports import (
            plot_polling_trend, plot_house_effects,
            plot_seat_distributions, plot_seat_scatter,
            plot_transfer_matrix, plot_competitive_districts,
            save_json_report,
        )

        print("  Computing rolling trend (this takes ~30s)...")
        trend = compute_rolling_trend(
            df_polls, parties=["Fidesz", "TISZA", "MH"],
            start="2024-06-01",
            house_effects=house_effects,
            halflife_days=cfg.halflife_days,
            max_age_days=cfg.max_age_days,
            n_boot=300,
        )

        paths = []
        paths.append(plot_polling_trend(
            df_polls, trend, parties=["Fidesz", "TISZA", "MH", "MKKP", "Bal"],
            out_path=str(out_dir / "plots" / "polling_trend.png"),
        ))
        paths.append(plot_house_effects(
            house_effects,
            out_path=str(out_dir / "plots" / "house_effects.png"),
        ))
        paths.append(plot_seat_distributions(
            agg, out_path=str(out_dir / "plots" / "seat_distributions.png"),
        ))
        paths.append(plot_seat_scatter(
            agg, out_path=str(out_dir / "plots" / "seat_scatter.png"),
        ))
        paths.append(plot_transfer_matrix(
            Q_cal, Q_PRIOR,
            out_path=str(out_dir / "plots" / "transfer_matrix.png"),
        ))
        paths.append(plot_competitive_districts(
            sim_results, districts,
            out_path=str(out_dir / "plots" / "competitive_districts.png"),
        ))

        report_path = save_json_report(
            agg, current_est,
            out_path=str(out_dir / "reports" / "simulation_report.json"),
        )
        paths.append(report_path)

        print(f"\n  Saved {len(paths)} output files to {out_dir}/")
        for p in paths:
            print(f"    {p}")
    else:
        print("\nStep 8 | Skipping plots (--skip-plots).")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
