"""Sensitivity analysis for the Hungarian election seat simulation.

Sweeps key model parameters one at a time and also runs 2D grid searches.
The transfer matrix is the most sensitive component — this script provides
detailed diagnostics on how each transfer-matrix prior assumption affects
the seat outcomes.

Usage:
    python sensitivity.py [--n-sim 1000] [--output-dir output]
"""

from __future__ import annotations

import argparse
import json
import sys

# Force UTF-8 output on Windows (avoids cp1250 encoding errors for Greek/math chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from copy import deepcopy
from pathlib import Path

import numpy as np

from lib.config import SimConfig
from lib.transfer_model import (
    Q_PRIOR, SOURCES, TARGETS,
    load_baseline_and_matrix, load_nowcast,
    calibrate_transfer_matrix, run_simulation, aggregate_results,
    transfer_matrix_report, prior_vs_calibrated,
)
from lib.reports import (
    plot_sensitivity, plot_transfer_sensitivity_heatmap,
    print_outcome_probabilities,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_pipeline(
    V22: np.ndarray,
    districts: list[str],
    nowcast_shares: dict,
    nowcast_se: dict,
    cfg: SimConfig,
    q_prior: np.ndarray = Q_PRIOR,
    verbose: bool = False,
) -> dict:
    """Calibrate + simulate and return aggregated outcomes."""
    Q_cal = calibrate_transfer_matrix(V22, nowcast_shares, nowcast_se, q_prior, cfg.lam_prior)
    results = run_simulation(V22, Q_cal, districts, cfg, verbose=verbose)
    return aggregate_results(results)["outcomes"]


def _modify_prior_row(q_prior: np.ndarray, source_idx: int, target_idx: int, new_val: float) -> np.ndarray:
    """Return a copy of q_prior with q_prior[source_idx, target_idx] = new_val.

    The remaining columns of that row are rescaled proportionally to preserve row-sum = 1.
    """
    q = q_prior.copy()
    old = q[source_idx, target_idx]
    if abs(new_val - old) < 1e-10:
        return q
    remaining_old = 1.0 - old
    remaining_new = 1.0 - new_val
    if remaining_old > 1e-10:
        scale = remaining_new / remaining_old
    else:
        scale = 0.0
    q[source_idx] *= scale
    q[source_idx, target_idx] = new_val
    q[source_idx] /= q[source_idx].sum()  # re-normalise for numerical safety
    return q


# ── 1D sweeps ─────────────────────────────────────────────────────────────────

SWEEP_CONFIGS = {
    # (label, param_name, values, baseline_value)
    "sigma_district": (
        "District noise sigma",
        [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30],
        0.15,
    ),
    "conc_national": (
        "National Q concentration",
        [20, 40, 60, 80, 100, 150, 200, 400],
        80.0,
    ),
    "lam_prior": (
        "KL regularization lam",
        [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
        5.0,
    ),
}

# Transfer-matrix row/cell sweeps (most impactful):
PRIOR_SWEEPS = {
    # Key: readable label. Value: (source_idx, target_idx, values)
    "Fidesz22 -> Fidesz26 retention": (
        0, 0,   # Fidesz22 row, Fidesz26 col
        [0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90],
    ),
    "Opp22 -> TISZA26 transfer": (
        1, 1,   # Opp22 row, TISZA26 col
        [0.45, 0.50, 0.55, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80],
    ),
    "Fidesz22 -> TISZA26 defection": (
        0, 1,   # Fidesz22 row, TISZA26 col
        [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15],
    ),
    "MH22 -> MH26 retention": (
        2, 3,   # MH22 row, MH26 col
        [0.35, 0.40, 0.45, 0.50, 0.52, 0.55, 0.60, 0.65],
    ),
    "Opp22 -> Abstain (disillusionment)": (
        1, 6,   # Opp22 row, Abstain26 col
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
    ),
}


def run_1d_sweeps(
    V22: np.ndarray,
    districts: list[str],
    nowcast_shares: dict,
    nowcast_se: dict,
    base_cfg: SimConfig,
) -> dict:
    """Run all 1D hyperparameter sweeps. Returns {sweep_label: [(val, outcomes), ...]}."""
    results = {}

    # A. Standard config sweeps
    for param, (label, values, _baseline) in SWEEP_CONFIGS.items():
        print(f"\n=== 1D sweep: {label} ===")
        sweep_data = []
        for v in values:
            cfg = base_cfg.copy(**{param: v})
            out = _run_pipeline(V22, districts, nowcast_shares, nowcast_se, cfg)
            pct = out["p_tisza_largest"] * 100
            print(f"  {param}={v:<8}  P(TISZA largest)={pct:.1f}%  "
                  f"P(TISZA majority)={out['p_tisza_majority']*100:.1f}%")
            sweep_data.append((v, out))
        results[label] = sweep_data

    # B. Transfer-matrix prior sweeps
    for label, (src_idx, tgt_idx, values) in PRIOR_SWEEPS.items():
        print(f"\n=== Transfer prior sweep: {label} ===")
        sweep_data = []
        for v in values:
            q_mod = _modify_prior_row(Q_PRIOR, src_idx, tgt_idx, v)
            out = _run_pipeline(V22, districts, nowcast_shares, nowcast_se, base_cfg, q_prior=q_mod)
            pct = out["p_tisza_largest"] * 100
            print(f"  Q[{SOURCES[src_idx]}, {TARGETS[tgt_idx]}]={v:.2f}  "
                  f"P(TISZA largest)={pct:.1f}%")
            sweep_data.append((v, out))
        results[label] = sweep_data

    return results


# ── 2D grid: conc_national × sigma_district ───────────────────────────────────

def run_2d_grid(
    V22: np.ndarray,
    districts: list[str],
    nowcast_shares: dict,
    nowcast_se: dict,
    base_cfg: SimConfig,
    conc_vals: list = None,
    sigma_vals: list = None,
) -> tuple[np.ndarray, list, list]:
    """Run 2D grid over conc_national × sigma_district.

    Returns (grid, conc_vals, sigma_vals) where grid[i,j] = P(TISZA largest) %.
    """
    if conc_vals is None:
        conc_vals = [20, 40, 80, 150, 300]
    if sigma_vals is None:
        sigma_vals = [0.05, 0.10, 0.15, 0.20, 0.30]

    grid = np.zeros((len(sigma_vals), len(conc_vals)))
    total = len(conc_vals) * len(sigma_vals)
    done = 0

    print(f"\n=== 2D grid: conc_national × sigma_district ({total} points) ===")
    for i, sigma in enumerate(sigma_vals):
        for j, conc in enumerate(conc_vals):
            cfg = base_cfg.copy(sigma_district=sigma, conc_national=conc)
            out = _run_pipeline(V22, districts, nowcast_shares, nowcast_se, cfg)
            grid[i, j] = out["p_tisza_largest"] * 100
            done += 1
            print(f"  [{done}/{total}] conc={conc:<5} sigma={sigma:.2f}  "
                  f"P(TISZA largest)={grid[i,j]:.1f}%")

    return grid, conc_vals, sigma_vals


# ── Scenario analysis ─────────────────────────────────────────────────────────

SCENARIOS = {
    "Baseline": {},
    "High Fidesz retention (82%->90%)": {
        "prior_mod": (0, 0, 0.90),
    },
    "TISZA gets more ex-Opp voters (62%->72%)": {
        "prior_mod": (1, 1, 0.72),
    },
    "High Fidesz defection (4%->10% to TISZA)": {
        "prior_mod": (0, 1, 0.10),
    },
    "High opposition disillusionment (20%->35% abstain)": {
        "prior_mod": (1, 6, 0.35),
    },
    "Low nat'l Q certainty (conc=30)": {
        "cfg_mod": {"conc_national": 30.0},
    },
    "High nat'l Q certainty (conc=200)": {
        "cfg_mod": {"conc_national": 200.0},
    },
}


def run_scenarios(
    V22: np.ndarray,
    districts: list[str],
    nowcast_shares: dict,
    nowcast_se: dict,
    base_cfg: SimConfig,
) -> list[dict]:
    """Run named scenarios and return summary rows."""
    rows = []
    print("\n=== Scenario analysis ===")
    for name, mods in SCENARIOS.items():
        q_prior = Q_PRIOR.copy()
        cfg = base_cfg

        if "prior_mod" in mods:
            src, tgt, val = mods["prior_mod"]
            q_prior = _modify_prior_row(Q_PRIOR, src, tgt, val)
        if "cfg_mod" in mods:
            cfg = base_cfg.copy(**mods["cfg_mod"])

        out = _run_pipeline(V22, districts, nowcast_shares, nowcast_se, cfg, q_prior)
        row = {"scenario": name, **out}
        rows.append(row)
        print(f"  {name:<50s}  P(TISZA largest)={out['p_tisza_largest']*100:>5.1f}%  "
              f"Med seats TISZA={out['median_tisza']:.0f}  Fidesz={out['median_fidesz']:.0f}")

    return rows


def print_scenario_table(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("  SCENARIO COMPARISON")
    print("=" * 100)
    print(f"  {'Scenario':<52s} {'P(T lrg)':>8} {'P(T maj)':>8} "
          f"{'P(F lrg)':>8} {'P(F maj)':>8} {'Med T':>6} {'Med F':>6}")
    print("  " + "-" * 96)
    for r in rows:
        print(f"  {r['scenario']:<52s} "
              f"{r['p_tisza_largest']*100:>7.1f}%  "
              f"{r['p_tisza_majority']*100:>7.1f}%  "
              f"{r['p_fidesz_largest']*100:>7.1f}%  "
              f"{r['p_fidesz_majority']*100:>7.1f}%  "
              f"{r['median_tisza']:>5.0f}  "
              f"{r['median_fidesz']:>5.0f}")
    print("=" * 100)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sensitivity analysis for Hungarian election sim")
    p.add_argument("--n-sim",      type=int,   default=1000)
    p.add_argument("--output-dir", default="output")
    p.add_argument("--skip-2d",    action="store_true", help="Skip 2D grid (slow)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    base_cfg = SimConfig(n_sim=args.n_sim, output_dir=str(out_dir))

    print("Loading baseline and nowcast...")
    V22, districts = load_baseline_and_matrix()
    nowcast_shares, nowcast_se = load_nowcast(base_cfg.nowcast_json)
    print(f"  {len(districts)} districts, {V22.sum():,.0f} 2022 OEVK votes")
    print(f"  Nowcast: TISZA={nowcast_shares.get('TISZA26',0)*100:.1f}%  "
          f"Fidesz={nowcast_shares.get('Fidesz26',0)*100:.1f}%")

    # ── 1D sweeps ─────────────────────────────────────────────────────────────
    sweep_results = run_1d_sweeps(V22, districts, nowcast_shares, nowcast_se, base_cfg)
    plot_path = plot_sensitivity(
        sweep_results,
        out_path=str(out_dir / "plots" / "sensitivity_1d.png"),
    )
    print(f"\nSaved 1D sensitivity plot: {plot_path}")

    # ── 2D grid ───────────────────────────────────────────────────────────────
    if not args.skip_2d:
        grid, conc_vals, sigma_vals = run_2d_grid(
            V22, districts, nowcast_shares, nowcast_se, base_cfg
        )
        hm_path = plot_transfer_sensitivity_heatmap(
            grid, conc_vals, sigma_vals,
            x_label="conc_national", y_label="sigma_district",
            out_path=str(out_dir / "plots" / "sensitivity_2d_heatmap.png"),
        )
        print(f"Saved 2D heatmap: {hm_path}")
    else:
        print("Skipping 2D grid.")

    # ── Scenarios ─────────────────────────────────────────────────────────────
    scenario_rows = run_scenarios(V22, districts, nowcast_shares, nowcast_se, base_cfg)
    print_scenario_table(scenario_rows)

    # Save scenario JSON
    scenario_path = out_dir / "reports" / "scenarios.json"
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scenario_path, "w", encoding="utf-8") as f:
        json.dump(scenario_rows, f, indent=2)
    print(f"Saved scenario report: {scenario_path}")


if __name__ == "__main__":
    main()
