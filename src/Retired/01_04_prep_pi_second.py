# python

import pandas as pd
from math import log10
from pathlib import Path
import project_specifics as config
import json

# --- Set your parameters here ---
data_path = Path(config.filepath_data) /"processed"/ "merged_data.csv"
output_path = data_path.parent / "merged_data_with_pi_ratings.csv"
source = "goals"  # "goals" or "xg"
lam = 0.035
gamma = 0.7
b = 10
c = 3

# --- Read and harmonize data ---
df = pd.read_csv(data_path, parse_dates=["date"])
def tr(rating):
    v = b ** (abs(rating) / c) - 1
    return -v if rating < 0 else v


if {"goals_home", "goals_away"}.issubset(df.columns) is False:
    if {"home_goals", "away_goals"}.issubset(df.columns):
        df = df.rename(columns={"home_goals": "goals_home", "away_goals": "goals_away"})
    elif "score" in df.columns:
        df[["goals_home", "goals_away"]] = df["score"].str.extract(r"(\d+)[–-](\d+)").astype(float)
    else:
        raise ValueError("No goal columns found.")

if {"home_xg", "away_xg"}.issubset(df.columns) is False:
    if {"xg_home", "xg_away"}.issubset(df.columns):
        df = df.rename(columns={"xg_home": "home_xg", "xg_away": "away_xg"})
    else:
        raise ValueError("No xG columns found.")

df = df.dropna(subset=["home_xg", "away_xg"]).sort_values("date").reset_index(drop=True)

# --- Compute Π-ratings ---
src_h = "goals_home" if source == "goals" else "home_xg"
src_a = "goals_away" if source == "goals" else "away_xg"
home_ratings, away_ratings, exp_gds = [], [], []
print(df.shape)


for idx, row in df.iterrows():
    R = {}
    for _, r in df.iloc[:idx].iterrows():
        # every 100 rows, print progress
        if idx % 100 == 0:
            print(f"Processing row {idx}")

        h, a = r["home_team"], r["away_team"]
        if h not in R: R[h] = {"home": 0., "away": 0.}
        if a not in R: R[a] = {"home": 0., "away": 0.}
        ed = tr(R[h]["home"]) - tr(R[a]["away"])
        sd = r[src_h] - r[src_a]
        w = c * log10(1 + abs(sd - ed))
        dh = (1 if ed < sd else -1) * w * lam
        da = -dh
        R[h]["home"] += dh
        R[h]["away"] += dh * gamma
        R[a]["away"] += da
        R[a]["home"] += da * gamma
    home_ratings.append(R.get(row["home_team"], {}).get("home", 0.))
    away_ratings.append(R.get(row["away_team"], {}).get("away", 0.))
    exp_gds.append(tr(R.get(row["home_team"], {}).get("home", 0.)) - tr(R.get(row["away_team"], {}).get("away", 0.)))

prefix = "pi_goal_" if source == "goals" else "pi_xg_"
df[f"{prefix}home_pre_rating"] = home_ratings
df[f"{prefix}away_pre_rating"] = away_ratings
df[f"{prefix}exp_gd_pre"] = exp_gds

# --- Save result ---
df.to_csv(output_path, index=False)
# safe the pi ratings  in a json

output_path = output_path.with_suffix(".json")
df[[f"{prefix}home_pre_rating", f"{prefix}away_pre_rating", f"{prefix}exp_gd_pre"]].to_json(output_path, orient="records", lines=True)

print(f"Π-ratings written to {output_path.resolve()}")