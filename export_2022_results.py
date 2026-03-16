"""
Export 2022 OEVK results from XLSX to JSON for the ETI news election map.
Outputs: src/assets/data/election-2026/oevk_results_2022.json
"""
import json
import os
import openpyxl
from collections import defaultdict

# Paths
XLSX_PATH = os.path.join(os.path.dirname(__file__), 'Egyéni_szavazás_szkjkv.xlsx')
GEOJSON_PATH = os.path.join(
    os.path.dirname(__file__),
    '..', 'ETI industries', 'eti-news',
    'src', 'assets', 'data', 'election-2026', 'hungary-counties.geojson'
)
OUT_PATH = os.path.join(
    os.path.dirname(__file__),
    '..', 'ETI industries', 'eti-news',
    'src', 'assets', 'data', 'election-2026', 'oevk_results_2022.json'
)

# Step 1: Build lookup from GeoJSON: (maz, evk) -> name
with open(GEOJSON_PATH, encoding='utf-8') as f:
    geojson = json.load(f)

name_lookup = {}
for feature in geojson['features']:
    props = feature['properties']
    key = (props['maz'].zfill(2), props['evk'].zfill(2))
    name_lookup[key] = props['name']

print(f"GeoJSON districts loaded: {len(name_lookup)}")

# Step 2: Aggregate XLSX data by (maz, evk)
# Column indices (0-based):
# 0: station_id, 1: protocol_id, 2: 'EGYÉNI', 3: maz, 4: county, 5: oevk,
# 6-10: location, 11: voters, 12: turnout, 13: in_box, 14: diff,
# 15: invalid, 16: valid_votes, 17: candidate, 18: org, 19: votes

district_data = defaultdict(lambda: {
    'valid_votes': 0,
    'fidesz_votes': 0,
    'opp_votes': 0,
})

current_maz = None
current_evk = None

wb = openpyxl.load_workbook(XLSX_PATH, read_only=True)
ws = wb.active

for i, row in enumerate(ws.iter_rows(values_only=True)):
    if i == 0:  # header
        continue

    station_id = row[0]
    if station_id is None:
        continue

    station_id = str(station_id)

    if station_id.endswith('F'):
        # Polling station aggregate row
        maz = row[3]
        evk = row[5]
        valid = row[16]
        if maz is not None and evk is not None:
            current_maz = str(maz).zfill(2)
            current_evk = str(evk).zfill(2)
            if valid is not None:
                district_data[(current_maz, current_evk)]['valid_votes'] += int(valid)
    elif station_id.endswith('T'):
        # Candidate vote row
        org = row[18]
        votes = row[19]
        if org is not None and votes is not None and current_maz is not None:
            org_upper = str(org).upper()
            votes_int = int(votes)
            if 'FIDESZ' in org_upper:
                district_data[(current_maz, current_evk)]['fidesz_votes'] += votes_int
            elif 'DEMOKRATIKUS KOAL' in org_upper:
                district_data[(current_maz, current_evk)]['opp_votes'] += votes_int

wb.close()

print(f"Districts found in XLSX: {len(district_data)}")

# Step 3: Compute percentages and build output
output = {}
fidesz_wins = 0
opp_wins = 0
missing = []

for key, data in district_data.items():
    name = name_lookup.get(key)
    if name is None:
        missing.append(key)
        continue

    valid = data['valid_votes']
    fidesz = data['fidesz_votes']
    opp = data['opp_votes']

    if valid == 0:
        continue

    fidesz_pct = round(fidesz / valid * 100, 2)
    opp_pct = round(opp / valid * 100, 2)
    other_pct = round(max(0.0, 100 - fidesz_pct - opp_pct), 2)
    margin = round(abs(fidesz_pct - opp_pct), 2)
    winner = 'fidesz' if fidesz >= opp else 'opposition'

    if winner == 'fidesz':
        fidesz_wins += 1
    else:
        opp_wins += 1

    output[name] = {
        'winner': winner,
        'fidesz_pct': fidesz_pct,
        'opp_pct': opp_pct,
        'other_pct': other_pct,
        'margin': margin,
        'fidesz_votes': fidesz,
        'opp_votes': opp,
        'valid_votes': valid,
    }

print(f"\n=== VERIFICATION ===")
print(f"Districts in output: {len(output)} (expected 106)")
print(f"Fidesz wins: {fidesz_wins} (expected ~88)")
print(f"Opposition wins: {opp_wins} (expected ~18)")
if missing:
    print(f"Keys in XLSX not found in GeoJSON: {missing[:10]}")

# Spot-check
for check in ['Budapest 1.', 'Budapest 5.', 'Budapest 9.']:
    if check in output:
        r = output[check]
        print(f"  {check}: winner={r['winner']}, fidesz={r['fidesz_pct']}%, opp={r['opp_pct']}%")

# Write output
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\nOutput written to: {OUT_PATH}")
