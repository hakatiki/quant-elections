"""Central configuration dataclass for the Hungarian election simulation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimConfig:
    """All tunable parameters for the end-to-end pipeline.

    Grouped by concern so each section can be overridden independently.
    """

    # ── Poll averaging ────────────────────────────────────────────────────────
    halflife_days: int = 21       # recency half-life for weighting
    max_age_days: int = 90        # discard polls older than this
    min_sample: int = 300         # minimum sample size to include
    tau_pp: float = 2.0           # systematic (non-sampling) error floor in pp
    house_effect_since: str = "2024-06-01"  # period for estimating house effects
    house_effect_min_polls: int = 2

    # ── Transfer model ────────────────────────────────────────────────────────
    # sigma_district: logistic-normal noise added per source-bloc per district.
    #   0.05 = tight, near-deterministic redistribution
    #   0.15 = moderate uncertainty (default)
    #   0.30 = high uncertainty (blurs district-level variation)
    sigma_district: float = 0.15

    # conc_national: Dirichlet concentration for per-draw Q perturbation.
    #   High (>200) = very little draw-to-draw variation in national Q
    #   Low  (<40)  = lots of variation → wider seat distribution
    #   MOST SENSITIVE PARAMETER — see sensitivity.py
    conc_national: float = 80.0

    # lam_prior: KL divergence regularization weight toward Q_PRIOR.
    #   High (>20) = calibrated Q forced close to prior
    #   Low  (<1)  = calibration can deviate freely from prior
    lam_prior: float = 5.0

    # ── Simulation ────────────────────────────────────────────────────────────
    n_sim: int = 5000
    seed: int = 2026

    # ── Seat rules ────────────────────────────────────────────────────────────
    total_list_seats: int = 93    # total list mandates (party + nationality)
    majority_threshold: int = 100  # seats needed for parliamentary majority
    supermajority_threshold: int = 133  # 2/3 supermajority

    # ── Nationality seat ──────────────────────────────────────────────────────
    # The German minority (Magyarországi Németek) won 1 seat in 2022 with 24,630 votes.
    # Ignoring this makes the party list pool 93 instead of 92, over-allocating
    # list seats to parties by 1. Enable this to correctly model 92 party seats.
    include_nationality_seat: bool = True
    german_minority_votes_2022: int = 24_630
    german_minority_vote_sigma: float = 2_000  # year-to-year variation

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_csv: str = "data.csv"
    nowcast_json: str = "nowcast_latest.json"
    output_dir: str = "output"

    def copy(self, **overrides) -> "SimConfig":
        """Return a copy with fields overridden."""
        from dataclasses import replace
        return replace(self, **overrides)
