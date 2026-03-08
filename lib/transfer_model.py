"""Voter transfer matrix model for the 2026 Hungarian parliamentary seat projection.

Pipeline
--------
1. Load 2022 OEVK district votes and collapse into 5 source blocs.
2. Define an informative prior transfer matrix Q_PRIOR (source → 2026 target).
3. Calibrate Q to match the current nowcast via L-BFGS-B optimisation.
4. Monte Carlo: perturb Q (Dirichlet), add district noise (logistic-normal),
   draw vote counts (multinomial), call seat_calculator → store result.
5. Aggregate into outcome probabilities and seat distributions.

Key fixes vs. transfer_sim.ipynb
---------------------------------
* German minority (Magyarországi Németek) nationality seat now included by default.
  Ignoring it over-allocates 1 list seat to parties (93 instead of 92).
  The German minority has ~24 k votes, well above the ≈ 12 k quarter-quota threshold,
  so this seat is won in essentially every simulation draw.
* Per-draw total seat count is tracked (sum of medians ≠ 199 is a display artifact).
* Surplus votes are clamped to zero (max(0, ...)) — defensive, margin always ≥ 1.
* LIST_OEVK_RATIO uses the exact 2022 empirical value (0.9964).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

# ── Imports from project root ─────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from seats import load_2022_baseline, seat_calculator, FIDESZ, EEGY  # noqa: E402
from lib.config import SimConfig  # noqa: E402

# ── Source / target definitions ───────────────────────────────────────────────

SOURCES = ["Fidesz22", "Opp22", "MH22", "MKKP22", "Other22"]
TARGETS = ["Fidesz26", "TISZA26", "Bal26", "MH26", "MKKP26", "Other26", "Abstain26"]

S = len(SOURCES)   # 5
T = len(TARGETS)   # 7

_MH_FULL   = "MI HAZÁNK MOZGALOM"
_MKKP_FULL = "MAGYAR KÉTFARKÚ KUTYA PÁRT"

GERMAN_MINORITY_LIST = "MAGYARORSZÁGI NÉMETEK ORSZÁGOS ÖNKORMÁNYZATA"

# Approximate ratio of domestic party-list votes to OEVK votes in 2022.
# (5 356 391 / 5 375 688 = 0.9964)  Some OEVK voters leave the list blank;
# mail-in voters are NOT modelled here (they affect list totals by ~+4%).
LIST_OEVK_RATIO: float = 0.9964

# ── Prior transfer matrix ─────────────────────────────────────────────────────
# Row s: probability that a 2022 voter of source bloc s ends up in each 2026 target.
# Rows must sum to 1.  Calibration (step 3) adjusts these while staying nearby.
#
#              Fidesz26 TISZA26  Bal26   MH26   MKKP26  Other26  Abstain26
Q_PRIOR = np.array([
    [0.82,     0.04,    0.01,   0.03,   0.01,   0.01,    0.08],  # Fidesz22
    [0.03,     0.62,    0.08,   0.02,   0.03,   0.02,    0.20],  # Opp22
    [0.12,     0.08,    0.02,   0.52,   0.04,   0.02,    0.20],  # MH22
    [0.08,     0.28,    0.04,   0.04,   0.28,   0.03,    0.25],  # MKKP22
    [0.15,     0.20,    0.05,   0.08,   0.05,   0.12,    0.35],  # Other22
], dtype=np.float64)

assert Q_PRIOR.shape == (S, T), "Q_PRIOR shape mismatch"
assert np.allclose(Q_PRIOR.sum(axis=1), 1.0), "Q_PRIOR rows must sum to 1"

# ── Nowcast key → target name mapping ────────────────────────────────────────
NOWCAST_MAP: dict[str, str] = {
    "Fidesz": "Fidesz26",
    "TISZA":  "TISZA26",
    "Bal":    "Bal26",
    "MH":     "MH26",
    "MKKP":   "MKKP26",
    "EM":     "Other26",
}

# 2026 party name used inside seat_calculator
TARGET_TO_LIST_NAME: dict[str, str] = {
    "Fidesz26": "Fidesz-KDNP",
    "TISZA26":  "TISZA",
    "Bal26":    "Baloldal",
    "MH26":     "Mi Hazank",
    "MKKP26":   "MKKP",
    "Other26":  "Other",
}

# Thresholds for 2026 list ballots
LIST_META_2026: dict[str, dict] = {
    "Fidesz-KDNP": {"threshold": 0.10},  # Fidesz+KDNP = 2-party → 10%
    "TISZA":        {"threshold": 0.05},
    "Baloldal":     {"threshold": 0.05},
    "Mi Hazank":    {"threshold": 0.05},
    "MKKP":         {"threshold": 0.05},
    "Other":        {"threshold": 0.05},
}

# Parties tracked in aggregate results
MAIN_PARTIES_2026 = list(TARGET_TO_LIST_NAME.values())  # 6 parties


# ── Data loading ──────────────────────────────────────────────────────────────

def load_baseline_and_matrix(
    data_dir: Optional[Path] = None,
) -> tuple[np.ndarray, list[str]]:
    """Load 2022 szavazókör Excel files and build V22 (D × S source-bloc matrix).

    Returns
    -------
    V22 : ndarray, shape (D, S)
        OEVK votes per district per 2022 source bloc.
    districts : list of str
        Sorted district IDs (e.g. "01-1", ..., "20-4").
    """
    baseline = load_2022_baseline(data_dir)
    oevk_votes = baseline["oevk_votes"]
    districts = sorted(oevk_votes.keys())
    D = len(districts)

    V22 = np.zeros((D, S), dtype=np.float64)
    for i, dist in enumerate(districts):
        for party, votes in oevk_votes[dist].items():
            if party == FIDESZ:
                j = SOURCES.index("Fidesz22")
            elif party == EEGY:
                j = SOURCES.index("Opp22")
            elif party == _MH_FULL:
                j = SOURCES.index("MH22")
            elif party == _MKKP_FULL:
                j = SOURCES.index("MKKP22")
            else:
                j = SOURCES.index("Other22")
            V22[i, j] += votes

    return V22, districts


def load_nowcast(path: str = "nowcast_latest.json") -> tuple[dict[str, float], dict[str, float]]:
    """Load nowcast_latest.json.

    Returns
    -------
    shares : {target: float} vote shares (0-1) among decided voters
    se     : {target: float} uncertainty = half 80% CI (as proportion)
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    shares: dict[str, float] = {}
    se: dict[str, float] = {}
    for poll_key, target in NOWCAST_MAP.items():
        if poll_key in data.get("estimates", {}):
            e = data["estimates"][poll_key]
            shares[target] = e["point"] / 100.0
            se[target] = (e["hi80"] - e["lo80"]) / 2.0 / 100.0

    return shares, se


# ── Calibration ───────────────────────────────────────────────────────────────

def _q_from_logit(x: np.ndarray) -> np.ndarray:
    """Map unconstrained R^(S*(T-1)) → S×T probability matrix via softmax."""
    raw = x.reshape(S, T - 1)
    raw_full = np.hstack([raw, np.zeros((S, 1))])
    return softmax(raw_full, axis=1)


def _objective(
    x: np.ndarray,
    V22: np.ndarray,
    nowcast_shares: dict[str, float],
    nowcast_se: dict[str, float],
    Q_prior: np.ndarray,
    lam_prior: float,
) -> float:
    Q = _q_from_logit(x)
    V26 = V22 @ Q                   # (D, T)
    national = V26.sum(axis=0)      # (T,)
    non_abstain = national[:-1].sum()
    if non_abstain <= 0:
        return 1e12

    shares_model = national[:-1] / non_abstain

    # Fit term: weighted squared error vs nowcast targets
    loss = 0.0
    for j, t in enumerate(TARGETS[:-1]):
        if t in nowcast_shares:
            sigma = max(nowcast_se.get(t, 0.02), 0.005)
            loss += ((shares_model[j] - nowcast_shares[t]) / sigma) ** 2

    # Regularization: KL toward prior
    eps = 1e-10
    for i in range(S):
        for j in range(T):
            if Q_prior[i, j] > eps and Q[i, j] > eps:
                loss += lam_prior * Q_prior[i, j] * np.log(Q_prior[i, j] / Q[i, j])

    return loss


def calibrate_transfer_matrix(
    V22: np.ndarray,
    nowcast_shares: dict[str, float],
    nowcast_se: dict[str, float],
    Q_prior: np.ndarray = Q_PRIOR,
    lam_prior: float = 5.0,
) -> np.ndarray:
    """Calibrate Q to match the nowcast while staying close to Q_prior.

    Optimises the KL-regularised weighted least-squares objective via L-BFGS-B.
    Returns Q_cal: (S, T) calibrated row-stochastic transfer matrix.
    """
    log_prior = np.log(Q_prior + 1e-10)
    x0 = (log_prior[:, :-1] - log_prior[:, -1:]).flatten()

    res = minimize(
        _objective, x0,
        args=(V22, nowcast_shares, nowcast_se, Q_prior, lam_prior),
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-12},
    )

    Q_cal = _q_from_logit(res.x)

    # Verify calibration quality
    V26_cal = V22 @ Q_cal
    national = V26_cal.sum(axis=0)
    non_abstain = national[:-1].sum()
    max_err_pp = 0.0
    if non_abstain > 0:
        for j, t in enumerate(TARGETS[:-1]):
            if t in nowcast_shares:
                err = abs(national[j] / non_abstain - nowcast_shares[t]) * 100
                max_err_pp = max(max_err_pp, err)

    if max_err_pp > 0.5:
        print(f"  WARNING: calibration max error = {max_err_pp:.2f} pp (threshold 0.5 pp)")

    return Q_cal


def check_calibration(
    V22: np.ndarray,
    Q_cal: np.ndarray,
    nowcast_shares: dict[str, float],
) -> None:
    """Print calibration fit diagnostics."""
    V26 = V22 @ Q_cal
    national = V26.sum(axis=0)
    non_abstain = national[:-1].sum()
    print("Calibration fit (model vs nowcast):")
    print(f"  {'Target':<12s} {'Model':>7s} {'Nowcast':>8s} {'Error':>7s}")
    print("  " + "-" * 38)
    for j, t in enumerate(TARGETS[:-1]):
        model_pct = national[j] / non_abstain * 100
        target_pct = nowcast_shares.get(t, float("nan")) * 100
        err = model_pct - target_pct
        print(f"  {t:<12s} {model_pct:>6.1f}%  {target_pct:>6.1f}%  {err:>+6.2f}pp")
    print(f"  Implied turnout vs 2022 pool: {non_abstain / national.sum() * 100:.1f}%")


# ── Simulation ────────────────────────────────────────────────────────────────

def _sample_district_votes(
    V22: np.ndarray,
    Q_national: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample 2026 district votes from the transfer model with district-level noise.

    For each (district, source bloc) pair:
    1. Perturb Q_national[source] by logistic-normal noise in log-space.
    2. Draw votes from multinomial(n_voters, q_perturbed).

    Returns V26: (D, T) vote counts including Abstain column.
    """
    D, _ = V22.shape
    V26 = np.zeros((D, T), dtype=np.float64)
    log_q = np.log(Q_national + 1e-10)

    for d in range(D):
        for s in range(S):
            n_s = int(V22[d, s])
            if n_s < 1:
                continue
            noise = rng.normal(0.0, sigma, size=T)
            q_ds = softmax(log_q[s] + noise)
            V26[d] += rng.multinomial(n_s, q_ds).astype(np.float64)

    return V26


def run_one_sim(
    V22: np.ndarray,
    Q_cal: np.ndarray,
    districts: list[str],
    rng: np.random.Generator,
    cfg: SimConfig,
) -> dict:
    """One Monte Carlo draw: perturb Q, sample votes, run seat_calculator.

    Returns the full seat_calculator result dict, augmented with
    "total_seats_draw" (int: total mandates allocated in this draw).
    """
    # 1. Perturb national Q via Dirichlet
    Q_draw = np.zeros_like(Q_cal)
    for s in range(S):
        alpha = Q_cal[s] * cfg.conc_national + 1e-6
        Q_draw[s] = rng.dirichlet(alpha)

    # 2. Sample district votes
    V26 = _sample_district_votes(V22, Q_draw, cfg.sigma_district, rng)

    # 3. Build seat_calculator inputs (exclude Abstain column)
    n_party_targets = T - 1   # 6 parties
    party_targets = TARGETS[:n_party_targets]

    oevk_votes: dict[str, dict[str, int]] = {}
    for d_idx, dist in enumerate(districts):
        oevk_votes[dist] = {}
        for j, t in enumerate(party_targets):
            v = int(V26[d_idx, j])
            if v > 0:
                oevk_votes[dist][TARGET_TO_LIST_NAME[t]] = v

    # National OEVK totals per party (excluding Abstain)
    national_oevk = V26[:, :n_party_targets].sum(axis=0)
    total_oevk = float(national_oevk.sum())

    # Direct list votes: proxy as national OEVK × empirical list/OEVK ratio.
    # This approximates party-list ballot totals; mail-in votes (~4%) not modelled.
    direct_list_votes: dict[str, int] = {
        TARGET_TO_LIST_NAME[t]: int(national_oevk[j] * LIST_OEVK_RATIO)
        for j, t in enumerate(party_targets)
    }

    # 4. Nationality seat (German minority).
    # BUG FIX: not including this causes the party list pool to be 93 instead
    # of 92, over-allocating ~1 seat to parties in every draw.
    nat_votes: Optional[dict[str, int]] = None
    if cfg.include_nationality_seat:
        total_list = total_oevk * LIST_OEVK_RATIO
        kvota = total_list / cfg.total_list_seats
        quarter_kvota = kvota / 4.0
        german_v = max(0, int(rng.normal(cfg.german_minority_votes_2022, cfg.german_minority_vote_sigma)))
        if german_v >= quarter_kvota:
            nat_votes = {GERMAN_MINORITY_LIST: german_v}

    # 5. Run seat calculator
    result = seat_calculator(
        oevk_votes=oevk_votes,
        direct_list_votes=direct_list_votes,
        list_meta=LIST_META_2026,
        nationality_list_votes=nat_votes,
        total_list_seats=cfg.total_list_seats,
    )

    # Annotate with draw-level total (for correct total reporting)
    total = sum(result["total_seats"].values())
    result["total_seats_draw"] = total

    return result


def run_simulation(
    V22: np.ndarray,
    Q_cal: np.ndarray,
    districts: list[str],
    cfg: SimConfig,
    verbose: bool = True,
) -> list[dict]:
    """Run cfg.n_sim Monte Carlo draws and return list of result dicts."""
    rng = np.random.default_rng(cfg.seed)

    try:
        from tqdm.auto import tqdm
        iterator = tqdm(range(cfg.n_sim), desc="Simulating", disable=not verbose)
    except ImportError:
        if verbose:
            print(f"  Running {cfg.n_sim} simulations (install tqdm for progress bar)...")
        iterator = range(cfg.n_sim)

    return [run_one_sim(V22, Q_cal, districts, rng, cfg) for _ in iterator]


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_results(sim_results: list[dict]) -> dict:
    """Aggregate MC simulation results into summary statistics.

    Returns a dict with per-party seat arrays and outcome probability estimates.
    The "total_seats_per_draw" array gives the true per-draw mandate total
    (should be ~199 each draw when nationality seat is included).
    """
    n = len(sim_results)

    oevk_all = {p: np.zeros(n, dtype=int) for p in MAIN_PARTIES_2026}
    list_all  = {p: np.zeros(n, dtype=int) for p in MAIN_PARTIES_2026}
    total_all = {p: np.zeros(n, dtype=int) for p in MAIN_PARTIES_2026}
    total_seats_per_draw = np.zeros(n, dtype=int)

    for i, r in enumerate(sim_results):
        for p in MAIN_PARTIES_2026:
            oevk = r["district_seat_counts"].get(p, 0)
            lst  = r["list_seat_counts"].get(p, 0)
            tot  = oevk + lst
            oevk_all[p][i]  = oevk
            list_all[p][i]  = lst
            total_all[p][i] = tot
        total_seats_per_draw[i] = r.get("total_seats_draw", sum(r["total_seats"].values()))

    fidesz = total_all["Fidesz-KDNP"]
    tisza  = total_all["TISZA"]
    maj    = 100
    sup    = 133

    return {
        "n_sim":   n,
        "parties": MAIN_PARTIES_2026,
        "oevk_seats":          oevk_all,
        "list_seats":          list_all,
        "total_seats":         total_all,
        "total_seats_per_draw": total_seats_per_draw,
        "outcomes": {
            "p_tisza_largest":        float((tisza > fidesz).mean()),
            "p_tisza_majority":       float((tisza >= maj).mean()),
            "p_fidesz_largest":       float((fidesz > tisza).mean()),
            "p_fidesz_majority":      float((fidesz >= maj).mean()),
            "p_fidesz_supermajority": float((fidesz >= sup).mean()),
            "p_tied":                 float((tisza == fidesz).mean()),
            "median_tisza":           float(np.median(tisza)),
            "median_fidesz":          float(np.median(fidesz)),
            "p5_tisza":               float(np.percentile(tisza, 5)),
            "p95_tisza":              float(np.percentile(tisza, 95)),
            "p5_fidesz":              float(np.percentile(fidesz, 5)),
            "p95_fidesz":             float(np.percentile(fidesz, 95)),
        },
    }


# ── Transfer matrix diagnostics ───────────────────────────────────────────────

def transfer_matrix_report(Q: np.ndarray, label: str = "Transfer matrix") -> None:
    """Print a formatted transfer matrix."""
    print(f"\n{label}:")
    header = " ".join(f"{t:>9s}" for t in TARGETS)
    print(f"  {'':>12s}  {header}")
    for i, src in enumerate(SOURCES):
        row = " ".join(f"{Q[i, j]:>9.1%}" for j in range(T))
        print(f"  {src:>12s}  {row}")


def prior_vs_calibrated(Q_prior: np.ndarray, Q_cal: np.ndarray) -> None:
    """Print delta Q = Q_cal - Q_prior to highlight where calibration moved the prior."""
    print("\nCalibration deltas (Q_cal - Q_prior), pp:")
    header = " ".join(f"{t:>9s}" for t in TARGETS)
    print(f"  {'':>12s}  {header}")
    for i, src in enumerate(SOURCES):
        deltas = (Q_cal[i] - Q_prior[i]) * 100
        row = " ".join(f"{d:>+9.1f}" for d in deltas)
        print(f"  {src:>12s}  {row}")
