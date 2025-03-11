import pandas as pd

# Load datasets
pl_merged_cleaned = pd.read_csv("/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_merged_dataset_cleaned.csv")
odds = pd.read_csv("/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/merged_odds.csv")
odds["Date"] = pd.to_datetime(odds["Date"], errors='coerce')

# Mapping dictionary
team_name_mapping = {
    'Man United': 'Manchester Utd',
    'Bournemouth': 'Bournemouth',
    'Fulham': 'Fulham',
    'Huddersfield': 'Huddersfield',
    'Newcastle': 'Newcastle Utd',
    'Watford': 'Watford',
    'Wolves': 'Wolves',
    'Arsenal': 'Arsenal',
    'Liverpool': 'Liverpool',
    'Southampton': 'Southampton',
    'Cardiff': 'Cardiff City',
    'Chelsea': 'Chelsea',
    'Everton': 'Everton',
    'Leicester': 'Leicester City',
    'Tottenham': 'Tottenham',
    'West Ham': 'West Ham',
    'Brighton': 'Brighton',
    'Burnley': 'Burnley',
    'Man City': 'Manchester City',
    'Crystal Palace': 'Crystal Palace',
    'Aston Villa': 'Aston Villa',
    'Norwich': 'Norwich City',
    'Sheffield United': 'Sheffield Utd',
    'West Brom': 'West Brom',
    'Leeds': 'Leeds United',
    'Brentford': 'Brentford',
    "Nott'm Forest": "Nott'ham Forest",
    'Luton': 'Luton Town',
    'Ipswich': 'Ipswich Town'
}

def update_team_names(df, column_name):
    """
    Replaces team names in the specified column of a DataFrame using the provided mapping dictionary.

    :param df: Pandas DataFrame
    :param column_name: Name of the column containing team names
    :return: DataFrame with updated team names
    """
    df[column_name] = df[column_name].replace(team_name_mapping)
    return df

# Update team names in the odds dataset
odds = update_team_names(odds, 'HomeTeam')
odds = update_team_names(odds, 'AwayTeam')

# Merge datasets on HomeTeam, AwayTeam, and Date
merged = pd.concat([pl_merged_cleaned, odds], axis=1, join='inner')
# delete colum HomeTeam, AwayTeam and Date
merged = merged.drop(columns=['HomeTeam', 'AwayTeam', 'Date'])

# Save merged dataset
output_path = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_merged_dataset_cleaned.csv"
merged.to_csv(output_path, index=False)
print(f"Combined data has been saved to: {output_path}")
