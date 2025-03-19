import pandas as pd
from project_specifics import team_name_mapping,MERGED_ODDS,CLEANED_MERGED, MERGED_OUTPUT

# Load datasets
pl_merged_cleaned = pd.read_csv(MERGED_OUTPUT)
print(pl_merged_cleaned.shape)
odds = pd.read_csv(MERGED_ODDS)
print(odds.shape)
odds["Date"] = pd.to_datetime(odds["Date"], errors='coerce')
print(odds.shape)


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

# Merge datasets on HomeTeam,  AwayTeam, and Date
merged = pd.concat([pl_merged_cleaned, odds], axis=1, join='inner')
# delete colum HomeTeam, AwayTeam and Date
merged = merged.drop(columns=['HomeTeam', 'AwayTeam', 'Date'])
print(merged.shape)
# Save merged dataset
output_path = CLEANED_MERGED
merged.to_csv(output_path, index=False)
print(f"Combined data has been saved to: {output_path}")
