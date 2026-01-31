import pandas as pd
import os
import json
import matplotlib.pyplot as plt

# Load odds data
odds = pd.read_csv('/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/odds/combined_odds.csv')

# Load match data from processed files
path = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed"
df_combined = pd.DataFrame()

for filename in os.listdir(path):
    filepath = os.path.join(path, filename)
    if os.path.isfile(filepath) and 'updated' in filename:
        df = pd.read_csv(filepath)
        df_combined = pd.concat([df_combined, df], ignore_index=True)

# Convert date and create Season column
df_combined["date"] = pd.to_datetime(df_combined["date"], format="%Y-%m-%d", errors='coerce')
df_combined['Season'] = df_combined['date'].apply(lambda x: x.year if x.month < 8 else x.year + 1)

# Clean team names in both dataframes
df_combined['home_team'] = df_combined['home_team'].str.lower().str.strip()
df_combined['away_team'] = df_combined['away_team'].str.lower().str.strip()
odds['HomeTeam'] = odds['HomeTeam'].str.lower().str.strip()
odds['AwayTeam'] = odds['AwayTeam'].str.lower().str.strip()
odds = odds.rename(columns={'HomeTeam': 'home_team', 'AwayTeam': 'away_team'})

# Create lists of unique team names
list_of_teams_schedule = df_combined["home_team"].unique()
list_of_away_teams_schedule = df_combined["away_team"].unique()
list_of_teams_schedule = set(list_of_teams_schedule) | set(list_of_away_teams_schedule)

list_of_teams_odds = odds["home_team"].unique()
list_of_away_teams_odds = odds["away_team"].unique()
list_of_teams_odds = set(list_of_teams_odds) | set(list_of_away_teams_odds)

# Save lists of teams to text files
with open('/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/list_of_teams.txt', 'w') as f:
    for team in list_of_teams_schedule:
        f.write(f"{team}\n")

with open('/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/list_of_teams_odds.txt', 'w') as f:
    for team in list_of_teams_odds:
        f.write(f"{team}\n")

# Load and apply team name mappings
with open('/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/team_name_map.json', 'r') as f:
    team_name_map = json.load(f)

odds['home_team'] = odds['home_team'].replace(team_name_map)
odds['away_team'] = odds['away_team'].replace(team_name_map)
df_combined['home_team'] = df_combined['home_team'].replace(team_name_map)
df_combined['away_team'] = df_combined['away_team'].replace(team_name_map)



# Merge the dataframes
df_combined = df_combined.merge(odds, on=['home_team', 'away_team', 'Season'], how='left')

# create some additional columns which are needed
df_combined["home_goals"] = df_combined["score"].str.split('-').str[0].astype(float)
df_combined["away_goals"] = df_combined["score"].str.split('-').str[1].astype(float)




# safe the merged dataframe
df_combined.to_csv('/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/merged_data.csv', index=False)