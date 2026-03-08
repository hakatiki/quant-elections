"""
Hungarian parliamentary seat calculator.

Usage
-----
    from seats import seat_calculator, load_2022_baseline

    # Quick: reproduce 2022 from the Excel files
    baseline = load_2022_baseline()
    result = seat_calculator(**baseline)

    # Custom: pass your own district votes
    result = seat_calculator(
        oevk_votes={
            "01-1": {"Fidesz": 19000, "Opposition": 22000, "MiHazánk": 1500},
            "01-2": {"Fidesz": 22000, "Opposition": 28000, "MiHazánk": 2000},
            ...
        },
        direct_list_votes={"Fidesz": 2_800_000, "Opposition": 1_900_000, "MiHazánk": 330_000},
        list_meta={"Fidesz": {"threshold": 0.10}, "Opposition": {"threshold": 0.15}, "MiHazánk": {"threshold": 0.05}},
    )

    print(result["total_seats"])
    print(result["district_seat_counts"])
    print(result["list_seat_counts"])
"""

from __future__ import annotations

import glob
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Party name constants (2022 ballot names)
# ---------------------------------------------------------------------------
FIDESZ = "FIDESZ - MAGYAR POLGÁRI SZÖVETSÉG-KERESZTÉNYDEMOKRATA NÉPPÁRT"
EEGY = (
    "DEMOKRATIKUS KOALÍCIÓ-JOBBIK MAGYARORSZÁGÉRT MOZGALOM-"
    "MOMENTUM MOZGALOM-MAGYAR SZOCIALISTA PÁRT-"
    "LMP - MAGYARORSZÁG ZÖLD PÁRTJA-PÁRBESZÉD MAGYARORSZÁGÉRT PÁRT"
)

SHORT_NAMES = {
    FIDESZ: "Fidesz-KDNP",
    EEGY: "Egységben",
    "MI HAZÁNK MOZGALOM": "Mi Hazánk",
    "MAGYAR KÉTFARKÚ KUTYA PÁRT": "MKKP",
    "MEGOLDÁS MOZGALOM": "Megoldás",
    "NORMÁLIS ÉLET PÁRTJA": "Normális Élet",
}

# Fidesz+KDNP = 2-party → 10%; Opposition = 6-party → 15%; rest = 5%
DEFAULT_THRESHOLDS = {
    FIDESZ: 0.10,
    EEGY: 0.15,
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def dhondt(vote_totals: dict[str, int], n_seats: int) -> dict[str, int]:
    """d'Hondt (Jefferson) proportional allocation."""
    quotients = []
    for party, votes in vote_totals.items():
        for divisor in range(1, n_seats + 1):
            quotients.append((votes / divisor, party))
    quotients.sort(key=lambda x: -x[0])

    seats: dict[str, int] = defaultdict(int)
    for _, party in quotients[:n_seats]:
        seats[party] += 1
    return dict(seats)


def seat_calculator(
    oevk_votes: dict[str, dict[str, int]],
    direct_list_votes: dict[str, int],
    list_meta: dict[str, dict] | None = None,
    nationality_list_votes: dict[str, int] | None = None,
    total_list_seats: int = 93,
) -> dict:
    """
    Deterministic Hungarian parliamentary seat calculator.

    Parameters
    ----------
    oevk_votes : dict
        ``{district_id: {party: votes}}`` for each OEVK.
        Party names must match keys in *direct_list_votes* when the OEVK
        candidate belongs to a party that also runs a national list.
        Parties present here but absent from *direct_list_votes* are treated
        as having no list (their fractional votes are lost).
    direct_list_votes : dict
        ``{list_name: total_direct_votes}`` from the party-list ballot.
    list_meta : dict, optional
        ``{list_name: {"threshold": float}}``.  Defaults to 0.05 for any
        list not specified.
    nationality_list_votes : dict, optional
        ``{nationality_list_name: votes}``.  Preferential mandates are
        computed and subtracted from the party-list seat pool.
    total_list_seats : int
        Total list mandates (default 93).

    Returns
    -------
    dict
        Keys: ``district_seats``, ``district_seat_counts``,
        ``fractional_votes``, ``list_totals``, ``eligible_lists``,
        ``list_seat_counts``, ``nationality_seats``, ``total_seats``.
    """
    if list_meta is None:
        list_meta = {}

    parties_with_lists = set(direct_list_votes.keys())
    total_direct = sum(direct_list_votes.values())

    # ---- OEVK seats + fractional votes (töredékszavazat) ----
    district_winners: dict[str, str] = {}
    frac: dict[str, int] = defaultdict(int)

    for dist, votes_dict in oevk_votes.items():
        ranked = sorted(votes_dict.items(), key=lambda x: -x[1])
        winner_party, winner_v = ranked[0]
        runner_up_v = ranked[1][1] if len(ranked) > 1 else 0

        district_winners[dist] = winner_party

        # Surplus = votes beyond what was strictly needed to win.
        # Clamped to 0: margin of exactly 1 gives no surplus (not negative).
        surplus = max(0, winner_v - (runner_up_v + 1))
        if winner_party in parties_with_lists:
            frac[winner_party] += surplus

        for party, v in ranked[1:]:
            if party in parties_with_lists:
                frac[party] += v

    district_seat_counts: dict[str, int] = defaultdict(int)
    for p in district_winners.values():
        district_seat_counts[p] += 1

    # ---- List totals = direct + fractional ----
    list_totals = {
        lista: direct_list_votes[lista] + frac.get(lista, 0)
        for lista in direct_list_votes
    }

    # ---- Threshold check (on direct list votes) ----
    eligible = {}
    for lista in direct_list_votes:
        thr = list_meta.get(lista, {}).get("threshold", 0.05)
        if total_direct > 0 and direct_list_votes[lista] / total_direct >= thr:
            eligible[lista] = list_totals[lista]

    # ---- Nationality preferential mandates ----
    nat_seats: dict[str, int] = {}
    n_nat = 0
    if nationality_list_votes and total_direct > 0:
        kvota = total_direct / total_list_seats
        for nat, v in nationality_list_votes.items():
            if v >= kvota / 4:
                nat_seats[nat] = 1
                n_nat += 1

    # ---- d'Hondt ----
    n_party_seats = total_list_seats - n_nat
    list_seat_counts = dhondt(eligible, n_party_seats) if eligible else {}

    # ---- Totals ----
    all_parties = set(list(district_seat_counts.keys()) + list(list_seat_counts.keys()))
    total_seats: dict[str, int] = {}
    for p in all_parties:
        total_seats[p] = district_seat_counts.get(p, 0) + list_seat_counts.get(p, 0)
    for nat, n in nat_seats.items():
        total_seats[nat] = total_seats.get(nat, 0) + n

    return {
        "district_seats": district_winners,
        "district_seat_counts": dict(district_seat_counts),
        "fractional_votes": dict(frac),
        "list_totals": list_totals,
        "eligible_lists": eligible,
        "list_seat_counts": list_seat_counts,
        "nationality_seats": nat_seats,
        "total_seats": total_seats,
    }


# ---------------------------------------------------------------------------
# 2022 baseline loader
# ---------------------------------------------------------------------------

def load_2022_baseline(data_dir: str | Path | None = None) -> dict:
    """
    Load the 2022 szavazókör-level Excel files and return a kwargs dict
    ready for ``seat_calculator(**load_2022_baseline())``.

    Includes domestic list votes **and** mail-in (levélszavazat) votes.

    Parameters
    ----------
    data_dir : path, optional
        Directory containing the three .xlsx files.  Defaults to the
        directory where this module lives.

    Returns
    -------
    dict with keys ``oevk_votes``, ``direct_list_votes``, ``list_meta``,
    ``nationality_list_votes``.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent
    data_dir = Path(data_dir)

    xlsx_files = list(data_dir.glob("*.xlsx"))
    egyeni_file = next(f for f in xlsx_files if "Egy" in f.name)
    listas_file = next(f for f in xlsx_files if f.name.startswith("List"))
    level_file = next((f for f in xlsx_files if "Lev" in f.name), None)

    # ---- OEVK ----
    raw = pd.read_excel(egyeni_file)
    for col in ["MEGYEKÓD", "MEGYE", "OEVK"]:
        raw[col] = raw[col].ffill()

    cands = raw[raw["SZAVAZAT"].notna()].copy()
    cands["district"] = (
        cands["MEGYEKÓD"].astype(int).astype(str).str.zfill(2)
        + "-"
        + cands["OEVK"].astype(int).astype(str)
    )

    agg = (
        cands.groupby(["district", "SZERVEZET"])["SZAVAZAT"]
        .sum()
        .astype(int)
        .reset_index()
    )

    oevk_dict: dict[str, dict[str, int]] = {}
    for district in agg["district"].unique():
        dv = agg[agg["district"] == district]
        oevk_dict[district] = dict(zip(dv["SZERVEZET"], dv["SZAVAZAT"]))

    # ---- List votes ----
    raw_l = pd.read_excel(listas_file)
    vote_rows = raw_l[raw_l["SZAVAZAT"].notna()]

    is_nat = vote_rows["LISTA_TÍPUS"] == "Nemzetiségi"

    direct_dict = (
        vote_rows[~is_nat]
        .groupby("LISTA")["SZAVAZAT"]
        .sum()
        .astype(int)
        .to_dict()
    )

    # ---- Mail-in (levélszavazat) — diaspora list-only ballots ----
    if level_file is not None:
        raw_m = pd.read_excel(level_file)
        mail_rows = raw_m[raw_m["SZAVAZAT"].notna()]
        for _, row in mail_rows.iterrows():
            lista = row["LISTA"]
            votes = int(row["SZAVAZAT"])
            direct_dict[lista] = direct_dict.get(lista, 0) + votes

    nat_dict = (
        vote_rows[is_nat]
        .groupby("LISTA")["SZAVAZAT"]
        .sum()
        .astype(int)
        .to_dict()
    )

    # ---- Thresholds ----
    meta = {}
    for lista in direct_dict:
        thr = DEFAULT_THRESHOLDS.get(lista, 0.05)
        meta[lista] = {"threshold": thr}

    return {
        "oevk_votes": oevk_dict,
        "direct_list_votes": direct_dict,
        "list_meta": meta,
        "nationality_list_votes": nat_dict,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    baseline = load_2022_baseline()
    result = seat_calculator(**baseline)

    print("=== 2022 Seat Calculator Results ===\n")

    print("District seats (OEVK):")
    for p in sorted(result["district_seat_counts"], key=result["district_seat_counts"].get, reverse=True):
        print(f"  {SHORT_NAMES.get(p, p[:40]):<25s} {result['district_seat_counts'][p]:>3}")

    print(f"\nList seats (d'Hondt, {93 - len(result['nationality_seats'])} party + {len(result['nationality_seats'])} nationality):")
    for p in sorted(result["list_seat_counts"], key=result["list_seat_counts"].get, reverse=True):
        print(f"  {SHORT_NAMES.get(p, p[:40]):<25s} {result['list_seat_counts'][p]:>3}")
    for nat in result["nationality_seats"]:
        print(f"  {nat[:40]:<25s}   1  (nationality)")

    print("\nTotal seats:")
    for p in sorted(result["total_seats"], key=result["total_seats"].get, reverse=True):
        print(f"  {SHORT_NAMES.get(p, p[:40]):<25s} {result['total_seats'][p]:>3}")
    print(f"  {'TOTAL':<25s} {sum(result['total_seats'].values()):>3}")
