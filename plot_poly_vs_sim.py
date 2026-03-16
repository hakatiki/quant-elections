"""Parse poly.txt (Polymarket export) + plot against simulation win probs."""

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# ── 1. Parse poly.txt ─────────────────────────────────────────────────────────

poly_raw = Path("poly.txt").read_text(encoding="utf-8")

poly = {}
for line in poly_raw.strip().splitlines():
    # each line: "partyname{...json...}"
    brace = line.index("{")
    party = line[:brace].strip()
    history = json.loads(line[brace:])["history"]
    dates = [datetime.fromtimestamp(e["t"], tz=timezone.utc).date() for e in history]
    probs = [e["p"] for e in history]
    poly[party] = pd.DataFrame({"date": pd.to_datetime(dates), "p": probs})

# Save clean CSV
clean = poly["fidesz"].rename(columns={"p": "fidesz_p"}).merge(
    poly["tisza"].rename(columns={"p": "tisza_p"}), on="date", how="outer"
).sort_values("date")
clean.to_csv("poly_clean.csv", index=False)
print(f"Saved poly_clean.csv  ({len(clean)} rows, "
      f"{clean['date'].min().date()} to {clean['date'].max().date()})")

# ── 2. Load simulation results ─────────────────────────────────────────────────

sim = pd.read_csv("output/historical_win_prob.csv", parse_dates=["date"])

# ── 3. Plot ───────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 5))

# Simulation (solid)
ax.plot(sim["date"], sim["p_tisza_largest"] * 100,
        color="#1f77b4", linewidth=2, label="Sim: P(TISZA largest)")
ax.plot(sim["date"], sim["p_fidesz_largest"] * 100,
        color="#d62728", linewidth=2, label="Sim: P(Fidesz largest)")

# Polymarket (dashed)
ax.plot(poly["tisza"]["date"], poly["tisza"]["p"] * 100,
        color="#1f77b4", linewidth=1.5, linestyle="--", alpha=0.8,
        label="Polymarket: TISZA wins")
ax.plot(poly["fidesz"]["date"], poly["fidesz"]["p"] * 100,
        color="#d62728", linewidth=1.5, linestyle="--", alpha=0.8,
        label="Polymarket: Fidesz wins")

ax.axhline(50, color="black", linestyle=":", linewidth=0.8, alpha=0.4)

ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

ax.set_title("Win probability: MC simulation vs Polymarket")
ax.set_ylabel("Win probability")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
out = Path("output/poly_vs_sim.png")
fig.savefig(out, dpi=150)
print(f"Saved {out}")
plt.show()
