"""
MVP v0 — Hungary poll-average nowcast

Reads data.csv, produces:
  - nowcast_daily.csv   (daily vote-share estimates + 80% intervals)
  - nowcast_latest.json (latest-day snapshot + metadata)
  - polls_clean.csv     (cleaned polls with computed weights)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────

PARTIES = ["Fidesz", "TISZA", "Bal", "MH", "MKKP", "EM"]
INPUT_PARTY_COLS = ["Fidesz", "TISZA", "Bal", "MH", "MKKP", "EM"]
DERIVED_COLS = ["kod", "FideszV", "TiszaV", "DKV", "MiHazankV", "MKKPV", "EMV"]

HALFLIFE_DAYS = 21
MAX_AGE_DAYS = 90
MIN_SAMPLE = 300
TAU = 0.02  # non-sampling error floor (in share units, i.e. 2 pp)
Z_80 = 1.2816  # normal quantile for 80% CI

MODE_WEIGHT = {
    "Telefonos": 1.00,
    "Hibrid": 0.95,
    "Online": 0.90,
    "Személyes": 0.95,
}


# ── A) Clean / standardise ──────────────────────────────────────────────────

def load_and_clean(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t", encoding="utf-8")

    df = raw.rename(columns={
        "Kezdet": "start_date",
        "Vég": "end_date",
        "Adatgazda": "pollster",
        "Mód": "method",
        "Minta": "sample_size",
    })

    for col in DERIVED_COLS:
        if col in df.columns:
            df = df.drop(columns=col)

    for col in ["start_date", "end_date"]:
        df[col] = pd.to_datetime(df[col], format="%Y.%m.%d", errors="coerce")

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

    for col in INPUT_PARTY_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sample_size"] = pd.to_numeric(df["sample_size"], errors="coerce")

    known_sum = df[INPUT_PARTY_COLS[:-1]].sum(axis=1)  # all except EM
    em_missing = df["EM"].isna()
    can_infer = em_missing & (known_sum <= 100) & (known_sum > 0)
    df.loc[can_infer, "EM"] = 100.0 - known_sum[can_infer]

    party_sum = df[INPUT_PARTY_COLS].sum(axis=1, min_count=1)
    df["party_sum"] = party_sum
    df["flag_sum"] = ~party_sum.between(95, 105)
    df["flag_sample"] = df["sample_size"] < MIN_SAMPLE

    n_before = len(df)
    df = df[df["sample_size"] >= MIN_SAMPLE].copy()
    n_after = len(df)
    if n_before != n_after:
        print(f"  dropped {n_before - n_after} poll(s) with sample < {MIN_SAMPLE}")

    df = df.sort_values("mid_date").reset_index(drop=True)
    return df


# ── B) Weighting ─────────────────────────────────────────────────────────────

def compute_weights(df: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    df["age_days"] = (asof - df["mid_date"]).dt.total_seconds() / 86400.0
    df = df[df["age_days"] >= 0].copy()

    df["w_time"] = 0.5 ** (df["age_days"] / HALFLIFE_DAYS)
    df["w_n"] = np.sqrt(df["sample_size"])
    df["w_mode"] = df["method"].map(MODE_WEIGHT).fillna(0.90)
    df["weight"] = df["w_time"] * df["w_n"] * df["w_mode"]
    return df


# ── C) Nowcast for a single day ──────────────────────────────────────────────

def nowcast_day(df: pd.DataFrame, d: pd.Timestamp):
    pool = df[(df["mid_date"] <= d)].copy()
    pool["age_days"] = (d - pool["mid_date"]).dt.total_seconds() / 86400.0
    pool = pool[pool["age_days"] <= MAX_AGE_DAYS]

    if len(pool) == 0:
        return None

    pool["w_time"] = 0.5 ** (pool["age_days"] / HALFLIFE_DAYS)
    pool["w_n"] = np.sqrt(pool["sample_size"])
    pool["w_mode"] = pool["method"].map(MODE_WEIGHT).fillna(0.90)
    pool["weight"] = pool["w_time"] * pool["w_n"] * pool["w_mode"]

    W = pool["weight"].values
    N = pool["sample_size"].values
    sum_w = W.sum()
    if sum_w == 0:
        return None
    sum_w2 = (W ** 2).sum()
    n_eff_polls = sum_w ** 2 / sum_w2

    row = {"date": d.strftime("%Y-%m-%d"), "n_polls": len(pool), "n_eff_polls": round(n_eff_polls, 1)}

    raw_hats = {}
    for party in PARTIES:
        vals = pool[party].values
        valid = ~np.isnan(vals)
        if valid.sum() == 0:
            raw_hats[party] = np.nan
            continue
        raw_hats[party] = np.average(vals[valid], weights=W[valid])

    tracked = {p: v for p, v in raw_hats.items() if not np.isnan(v)}
    total = sum(tracked.values())
    if total > 0:
        scale = 100.0 / total
    else:
        scale = 1.0

    for party in PARTIES:
        if party in tracked:
            p_hat = tracked[party] * scale / 100.0  # as proportion for variance calc
            p_pct = tracked[party] * scale           # percentage for output

            # Sampling variance of the weighted mean using each poll's sample size:
            #   Var(p_hat) = sum(w_i^2 * p(1-p)/n_i) / (sum w_i)^2
            vals = pool[party].values
            valid = ~np.isnan(vals)
            w_v = W[valid]
            n_v = N[valid]
            var_samp = np.sum(w_v ** 2 * p_hat * (1 - p_hat) / n_v) / (w_v.sum() ** 2)

            var_total = var_samp + TAU ** 2
            se = np.sqrt(var_total) * 100.0  # back to percentage points

            row[f"{party}_hat"] = round(p_pct, 2)
            row[f"{party}_lo80"] = round(max(p_pct - Z_80 * se, 0), 2)
            row[f"{party}_hi80"] = round(min(p_pct + Z_80 * se, 100), 2)
        else:
            row[f"{party}_hat"] = np.nan
            row[f"{party}_lo80"] = np.nan
            row[f"{party}_hi80"] = np.nan

    return row


# ── D) Run the full nowcast ──────────────────────────────────────────────────

def run(csv_path: Path, asof_date: pd.Timestamp, out_dir: Path):
    print(f"Loading {csv_path}")
    df = load_and_clean(csv_path)
    print(f"  {len(df)} polls, {df['mid_date'].min():%Y-%m-%d} to {df['mid_date'].max():%Y-%m-%d}")
    print(f"  pollsters: {', '.join(sorted(df['pollster'].unique()))}")

    df_weighted = compute_weights(df, asof_date)
    polls_out = df_weighted[
        ["start_date", "end_date", "mid_date", "pollster", "method", "sample_size"]
        + INPUT_PARTY_COLS
        + ["party_sum", "flag_sum", "age_days", "w_time", "w_n", "w_mode", "weight"]
    ].copy()
    polls_out.to_csv(out_dir / "polls_clean.csv", index=False)
    print(f"  wrote polls_clean.csv ({len(polls_out)} rows)")

    first_date = df["mid_date"].min().normalize()
    date_range = pd.date_range(first_date, asof_date, freq="D")

    rows = []
    for d in date_range:
        result = nowcast_day(df, d)
        if result is not None:
            rows.append(result)

    daily = pd.DataFrame(rows)
    daily.to_csv(out_dir / "nowcast_daily.csv", index=False)
    print(f"  wrote nowcast_daily.csv ({len(daily)} days)")

    latest = daily.iloc[-1].to_dict()

    pool_today = df_weighted[df_weighted["age_days"] <= MAX_AGE_DAYS]
    pollster_weights = (
        pool_today.groupby("pollster")["weight"]
        .sum()
        .sort_values(ascending=False)
    )
    weight_share = (pollster_weights / pollster_weights.sum() * 100).round(1).to_dict()

    meta = {
        "asof_date": asof_date.strftime("%Y-%m-%d"),
        "halflife_days": HALFLIFE_DAYS,
        "max_age_days": MAX_AGE_DAYS,
        "tau_pp": TAU * 100,
        "n_polls_in_window": int(latest["n_polls"]),
        "n_eff_polls": latest["n_eff_polls"],
        "pollster_weight_pct": weight_share,
        "estimates": {},
    }
    for party in PARTIES:
        hat_key = f"{party}_hat"
        if hat_key in latest and not (isinstance(latest[hat_key], float) and np.isnan(latest[hat_key])):
            meta["estimates"][party] = {
                "point": latest[hat_key],
                "lo80": latest[f"{party}_lo80"],
                "hi80": latest[f"{party}_hi80"],
            }

    with open(out_dir / "nowcast_latest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  wrote nowcast_latest.json")

    print("\n" + "=" * 60)
    print(f"  NOWCAST  —  {asof_date:%Y-%m-%d}")
    print("=" * 60)
    print(f"  {'Party':<12} {'Est':>7} {'80% CI':>18}")
    print("  " + "-" * 42)
    for party in PARTIES:
        e = meta["estimates"].get(party)
        if e:
            print(f"  {party:<12} {e['point']:>6.1f}%  [{e['lo80']:>5.1f}% – {e['hi80']:>5.1f}%]")
    print("=" * 60)
    print(f"  polls in window: {meta['n_polls_in_window']},  n_eff_polls: {meta['n_eff_polls']}")
    print(f"  weight share: {weight_share}")
    print()

    _sanity_check(daily)


def _sanity_check(daily: pd.DataFrame):
    hat_cols = [c for c in daily.columns if c.endswith("_hat")]
    last = daily.iloc[-1]

    issues = []
    for c in hat_cols:
        if last[c] < 0:
            issues.append(f"  FAIL: {c} = {last[c]} (negative)")

    hat_vals = [last[c] for c in hat_cols if not np.isnan(last[c])]
    total = sum(hat_vals)
    if not 99.5 <= total <= 100.5:
        issues.append(f"  WARN: party shares sum to {total:.1f} (expected ~100)")

    if issues:
        print("QA issues:")
        for i in issues:
            print(i)
    else:
        print("QA: all checks passed")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hungary poll-average nowcast (MVP v0)")
    parser.add_argument("--csv", default="data.csv", help="Path to poll CSV (tab-separated)")
    parser.add_argument("--asof", default=None, help="As-of date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--outdir", default=".", help="Output directory")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.asof:
        asof = pd.Timestamp(args.asof)
    else:
        asof = pd.Timestamp.now().normalize()  # tz-naive to match poll dates

    run(csv_path, asof, out_dir)
