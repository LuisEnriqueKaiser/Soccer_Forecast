import pandas as pd
import numpy as np
from pathlib import Path
import project_specifics as ps

leagues = [ps.england, ps.france, ps.spain, ps.germany, ps.italy]
baseline_path = Path(ps.filepath_00)

# Remove duplicates and strip whitespace
relevant_columns = list(dict.fromkeys([col.strip() for col in ps.relevant_columns + ps.relevant_odds]))
master_odds_data = pd.DataFrame(columns=relevant_columns)

for league in leagues:
    print(f"Processing {league}...")
    path_relevant = baseline_path / league
    for path in path_relevant.iterdir():
        if path.is_file() and path.suffix == '.csv':
            year = path.stem
            print(f"Loading data from {path}")
            odds_data = pd.read_csv(path, low_memory=False)
            odds_data.columns = odds_data.columns.str.strip()  # Strip whitespace
            odds_data["Season"] = year
            for col in relevant_columns:
                if col not in odds_data.columns:
                    odds_data[col] = np.nan
            odds_data = odds_data[relevant_columns]
            master_odds_data = pd.concat([master_odds_data, odds_data], ignore_index=True)

try:
    master_odds_data.to_csv(baseline_path / "combined_odds.csv", index=False)
    print(f"Wrote to {baseline_path / 'combined_odds.csv'}")
except Exception as e:
    print(f"Error saving combined odds data: {e}")