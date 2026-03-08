"""Weighted polling average with house-effect correction and bootstrap CIs.

Key design decisions vs. the original election_estimates.ipynb:
- Party column names use "MH" (matching data.csv and nowcast.py), not "MiHazank"
- Bootstrap CI level is configurable
- house_effects is passed in, not computed internally (separation of concerns)
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

# Canonical party keys (match data.csv column headers exactly)
PARTIES = ["Fidesz", "TISZA", "Bal", "MH", "MKKP", "EM"]

PARTY_COLORS = {
    "Fidesz": "#fd8204",
    "TISZA": "#1B4D8E",
    "Bal": "#e6331a",
    "MH": "#215b2c",
    "MKKP": "#888888",
    "EM": "#7b2d8e",
}

PARTY_LABELS = {
    "Fidesz": "Fidesz-KDNP",
    "TISZA": "TISZA",
    "Bal": "Baloldal",
    "MH": "Mi Hazank",
    "MKKP": "MKKP",
    "EM": "Egys. Mo.-ert",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_polls(path: str = "data.csv") -> pd.DataFrame:
    """Load and clean the tab-separated poll CSV.

    Handles both Hungarian column names (Kezdet/Vég/…) and already-cleaned
    column names. Column "MH" is kept as-is (do NOT rename to MiHazank).
    """
    raw = pd.read_csv(path, sep="\t", encoding="utf-8")

    rename_map = {
        "Kezdet": "start_date",
        "Vég": "end_date",
        "Adatgazda": "pollster",
        "Mód": "method",
        "Minta": "sample_size",
    }
    df = raw.rename(columns=rename_map)

    # Drop pre-computed smoothed columns if present
    drop_cols = {"kod", "FideszV", "TiszaV", "DKV", "MiHazankV", "MKKPV", "EMV", "week_code"}
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Parse dates
    for col in ["start_date", "end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%Y.%m.%d", errors="coerce")

    # Midpoint date: prefer midpoint of period, fall back to end or start
    has_start = df["start_date"].notna()
    has_end = df["end_date"].notna()
    df["mid_date"] = pd.NaT
    df.loc[has_start & has_end, "mid_date"] = (
        df.loc[has_start & has_end, "start_date"]
        + (df.loc[has_start & has_end, "end_date"]
           - df.loc[has_start & has_end, "start_date"]) / 2
    )
    df.loc[~has_start & has_end, "mid_date"] = df.loc[~has_start & has_end, "end_date"]
    df.loc[has_start & ~has_end, "mid_date"] = df.loc[has_start & ~has_end, "start_date"]
    df = df.dropna(subset=["mid_date"])

    # Parse numeric columns
    for col in PARTIES + ["sample_size"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("mid_date").reset_index(drop=True)
    return df


# ── House effects ─────────────────────────────────────────────────────────────

def estimate_house_effects(
    df: pd.DataFrame,
    parties: Optional[list[str]] = None,
    since: str = "2024-06-01",
    min_polls: int = 2,
) -> pd.DataFrame:
    """Estimate pollster house effects as deviation from the period mean.

    Positive = pollster over-estimates that party vs. average.
    Rows = pollsters, columns = parties. NaN if too few polls.
    """
    if parties is None:
        parties = [p for p in PARTIES if p in df.columns]

    recent = df[df["mid_date"] >= since].copy()
    overall_means = recent[parties].mean()

    effects = pd.DataFrame(
        index=recent["pollster"].dropna().unique(), columns=parties, dtype=float
    )

    for pollster in effects.index:
        pmask = recent["pollster"] == pollster
        if pmask.sum() < min_polls:
            continue
        for party in parties:
            vals = recent.loc[pmask, party].dropna()
            if len(vals) >= min_polls:
                effects.loc[pollster, party] = vals.mean() - overall_means[party]

    return effects.dropna(how="all")


# ── Poll average ──────────────────────────────────────────────────────────────

def weighted_poll_average(
    df: pd.DataFrame,
    parties: Optional[list[str]] = None,
    as_of_date=None,
    halflife_days: int = 14,
    max_age_days: int = 90,
    house_effects: Optional[pd.DataFrame] = None,
) -> dict[str, float]:
    """Recency- and sample-weighted polling average with optional house-effect correction.

    Weights:
      - Recency: exp(-ln2 * age / halflife)
      - Sample size: sqrt(n / 1000)

    Returns {party: weighted_mean} or {party: nan} if no data.
    """
    if parties is None:
        parties = [p for p in PARTIES if p in df.columns]
    if as_of_date is None:
        as_of_date = df["mid_date"].max()
    as_of_date = pd.Timestamp(as_of_date)

    work = df[
        (df["mid_date"] <= as_of_date)
        & (df["mid_date"] >= as_of_date - timedelta(days=max_age_days))
    ].copy()

    if len(work) == 0:
        return {p: np.nan for p in parties}

    days_ago = (as_of_date - work["mid_date"]).dt.total_seconds() / 86400.0
    weights = (
        np.exp(-np.log(2) * days_ago / halflife_days)
        * np.sqrt(work["sample_size"].fillna(1000) / 1000.0)
    )

    results: dict[str, float] = {}
    for party in parties:
        vals = work[party].copy()

        # Apply house-effect correction
        if house_effects is not None and party in house_effects.columns:
            for pollster in work["pollster"].unique():
                if pollster in house_effects.index:
                    he = house_effects.loc[pollster, party]
                    if pd.notna(he):
                        mask = work["pollster"] == pollster
                        vals.loc[mask] = vals.loc[mask] - he

        valid = vals.notna()
        if valid.sum() == 0:
            results[party] = np.nan
        else:
            results[party] = float(np.average(vals[valid], weights=weights[valid]))

    return results


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_estimate(
    df: pd.DataFrame,
    parties: Optional[list[str]] = None,
    as_of_date=None,
    halflife_days: int = 14,
    max_age_days: int = 90,
    house_effects: Optional[pd.DataFrame] = None,
    n_boot: int = 2000,
    ci_level: float = 0.90,
    seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    """Bootstrap confidence intervals for the polling average.

    Returns {party: (lo, median, hi)} where lo/hi correspond to ci_level.
    E.g. ci_level=0.90 → 5th and 95th percentiles.
    """
    if parties is None:
        parties = [p for p in PARTIES if p in df.columns]
    if as_of_date is None:
        as_of_date = df["mid_date"].max()
    as_of_date = pd.Timestamp(as_of_date)

    work = df[
        (df["mid_date"] <= as_of_date)
        & (df["mid_date"] >= as_of_date - timedelta(days=max_age_days))
    ].copy()

    if len(work) < 3:
        pt = weighted_poll_average(df, parties, as_of_date, halflife_days, max_age_days, house_effects)
        return {p: (pt[p], pt[p], pt[p]) for p in parties}

    rng = np.random.default_rng(seed)
    lo_q = (1 - ci_level) / 2 * 100
    hi_q = (1 + ci_level) / 2 * 100
    boot: dict[str, list[float]] = {p: [] for p in parties}

    for _ in range(n_boot):
        idx = rng.choice(len(work), size=len(work), replace=True)
        est = weighted_poll_average(
            work.iloc[idx], parties, as_of_date,
            halflife_days, max_age_days, house_effects,
        )
        for p in parties:
            boot[p].append(est[p])

    out: dict[str, tuple[float, float, float]] = {}
    for p in parties:
        arr = np.array([x for x in boot[p] if not np.isnan(x)])
        if len(arr) == 0:
            out[p] = (np.nan, np.nan, np.nan)
        else:
            out[p] = (
                float(np.percentile(arr, lo_q)),
                float(np.median(arr)),
                float(np.percentile(arr, hi_q)),
            )
    return out


# ── Rolling trend ─────────────────────────────────────────────────────────────

def compute_rolling_trend(
    df: pd.DataFrame,
    parties: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    freq: str = "7D",
    house_effects: Optional[pd.DataFrame] = None,
    halflife_days: int = 14,
    max_age_days: int = 90,
    n_boot: int = 500,
    ci_level: float = 0.90,
    seed: int = 42,
) -> dict[str, dict]:
    """Compute rolling polling trend with CIs for plotting.

    Returns {party: {"dates": [...], "lo": [...], "med": [...], "hi": [...]}}.
    """
    if parties is None:
        parties = [p for p in PARTIES if p in df.columns]
    if start is None:
        start = df["mid_date"].min().strftime("%Y-%m-%d")
    if end is None:
        end = df["mid_date"].max().strftime("%Y-%m-%d")

    date_range = pd.date_range(start=start, end=end, freq=freq)
    trend: dict[str, dict] = {p: {"dates": [], "lo": [], "med": [], "hi": []} for p in parties}

    for d in date_range:
        est = bootstrap_estimate(
            df, parties, as_of_date=d,
            halflife_days=halflife_days, max_age_days=max_age_days,
            house_effects=house_effects, n_boot=n_boot, ci_level=ci_level, seed=seed,
        )
        for p in parties:
            lo, med, hi = est[p]
            if not np.isnan(med):
                trend[p]["dates"].append(d)
                trend[p]["lo"].append(lo)
                trend[p]["med"].append(med)
                trend[p]["hi"].append(hi)

    return trend
