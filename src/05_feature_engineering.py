import pandas as pd
import numpy as np

def compute_mean_stat_last_n_games(
    df,
    n,
    home_stat_col,
    away_stat_col,
    stat_name,
    home_team_col="home_team",
    away_team_col="away_team"
):
    """
    For each row in df, calculate:
      - The mean of 'home_stat_col' / 'away_stat_col' for the home team over its last n matches (regardless of venue).
      - The mean of 'home_stat_col' / 'away_stat_col' for the away team over its last n matches (regardless of venue).

    This mirrors the style of your reference code, but now uses 'home_team' / 'away_team'
    columns and separate home_stat_col / away_stat_col.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least:
          - 'home_team', 'away_team' (or custom, if you pass them in)
          - The two stat columns, e.g. 'Home_Expected_xG' and 'Away_Expected_xG'
    n : int
        Number of past matches to consider
    home_stat_col : str
        Column name containing the "home" version of the stat (e.g. 'Home_Expected_xG')
    away_stat_col : str
        Column name containing the "away" version of the stat (e.g. 'Away_Expected_xG')
    stat_name : str
        Will be used in naming the new output columns:
        'home_mean_{stat_name}_last_{n}' and 'away_mean_{stat_name}_last_{n}'
    home_team_col : str, default "home_team"
        Name of the column that tells us which team is at home
    away_team_col : str, default "away_team"
        Name of the column that tells us which team is away

    Returns
    -------
    df : pd.DataFrame
        Same DataFrame but with two new columns:
          - home_mean_{stat_name}_last_{n}
          - away_mean_{stat_name}_last_{n}
    """

    # Prepare names for the two new columns
    home_output_col = f"home_mean_{stat_name}_last_{n}"
    print(home_output_col)
    away_output_col = f"away_mean_{stat_name}_last_{n}"

    # Iterate through each match (row)
    for row_number in range(len(df)):
        # Identify the current row's home and away teams
        home_team = df.loc[row_number, home_team_col]
        away_team = df.loc[row_number, away_team_col]

        # --- 1) Get the last n matches for home_team ---
        home_team_matches = df[
            (
                (df[home_team_col] == home_team)  # Team was at home
                | (df[away_team_col] == home_team)  # Team was away
            )
            & (df.index < row_number)  # Matches must come before this match in the dataset
        ].tail(n)

        # Extract the stat from those matches
        home_team_values = []
        for i in range(len(home_team_matches)):
            # If the home_team was "home" in that prior match, use home_stat_col
            if home_team_matches.iloc[i][home_team_col] == home_team:
                home_team_values.append(home_team_matches.iloc[i][home_stat_col])
            else:
                # Otherwise it was "away" in that prior match, so use away_stat_col
                home_team_values.append(home_team_matches.iloc[i][away_stat_col])

        # --- 2) Get the last n matches for away_team ---
        away_team_matches = df[
            (
                (df[home_team_col] == away_team)
                | (df[away_team_col] == away_team)
            )
            & (df.index < row_number)
        ].tail(n)

        away_team_values = []
        for i in range(len(away_team_matches)):
            if away_team_matches.iloc[i][home_team_col] == away_team:
                away_team_values.append(away_team_matches.iloc[i][home_stat_col])
            else:
                away_team_values.append(away_team_matches.iloc[i][away_stat_col])

        # --- 3) Compute means and assign to new columns for current row ---
        if len(home_team_values) > 0:
            df.loc[row_number, home_output_col] = np.mean(home_team_values)
        else:
            df.loc[row_number, home_output_col] = np.nan

        if len(away_team_values) > 0:
            df.loc[row_number, away_output_col] = np.mean(away_team_values)
        else:
            df.loc[row_number, away_output_col] = np.nan

    return df
def get_match_result(score_str):
    # Split the string on the hyphen and remove extra spaces
    try:
        home_goals, away_goals = map(int, score_str.split("–"))
    except Exception as e:
        # if there's an error parsing, return NaN
        return np.nan

    if home_goals > away_goals:
        return 1   # Home win
    elif home_goals == away_goals:
        return 0   # Draw
    else:
        return -1  # Away win





pairs = [
    ["home_xg", "away_xg", "xg"],
    ["Home_Tackles_Tkl", "Away_Tackles_Tkl", "Tackles_Tkl"],
    ["Home_Tackles.1_TklW", "Away_Tackles.1_TklW", "Tackles_TklW"],
    ["Home_Tackles.2_Def 3rd", "Away_Tackles.2_Def 3rd", "Tackles_Def 3rd"],
    ["Home_Tackles.4_Att 3rd", "Away_Tackles.4_Att 3rd", "Tackles_Att 3rd"],
    ["Home_Blocks.1_Sh", "Away_Blocks.1_Sh", "Blocks_Sh"],
    ["Home_Total.3_TotDist", "Away_Total.3_TotDist", "Total_TotDist"],
    ["Home_Total.4_PrgDist", "Away_Total.4_PrgDist", "Total_PrgDist"],
    ["Home_xAG_nan", "Away_xAG_nan", "xAG_nan"],
    ["Home_xA_nan", "Away_xA_nan", "xA_nan"],
    ["Home_Touches.3_Mid 3rd", "Away_Touches.3_Mid 3rd", "Touches_Mid 3rd"],
    ["Home_Touches.4_Att 3rd", "Away_Touches.4_Att 3rd", "Touches_Att 3rd"],
    ["Home_Carries.2_PrgDist", "Away_Carries.2_PrgDist", "Carries_PrgDist"],
    ["Home_Standard_Gls", "Away_Standard_Gls", "Standard_Gls"],
    ["Home_Expected_xG", "Away_Expected_xG", "Expected_xG"],
    ["Home_Expected.1_npxG", "Away_Expected.1_npxG", "Expected_npxG"],
    ["Home_Expected.2_npxG/Sh", "Away_Expected.2_npxG/Sh", "Expected_npxG/Sh"],
    ["Home_Expected.3_G-xG", "Away_Expected.3_G-xG", "Expected_G-xG"],
    ["Home_Expected.4_np:G-xG", "Away_Expected.4_np:G-xG", "Expected_np:G-xG"],
]

df = pd.read_csv("/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_merged_dataset_cleaned.csv")
n = 3
# create features for every pair
for home_col, away_col, name in pairs:
    df = compute_mean_stat_last_n_games(
        df, n, home_col, away_col, name
    )
# drop the old columns


to_drop = [ 'Home_Tackles.1_TklW', 'Home_Tackles.2_Def 3rd',
       'Home_Tackles.4_Att 3rd', 'Home_Blocks.1_Sh', 'Away_Tackles_Tkl',
       'Away_Tackles.1_TklW', 'Away_Tackles.2_Def 3rd',
       'Away_Tackles.4_Att 3rd', 'Away_Blocks.1_Sh', 'Home_Total.3_TotDist',
       'Home_Total.4_PrgDist', 'Home_xAG_nan', 'Home_xA_nan',
       'Away_Total.3_TotDist', 'Away_Total.4_PrgDist', 'Away_xAG_nan',
       'Away_xA_nan', 'Home_Touches.3_Mid 3rd', 'Home_Touches.4_Att 3rd',
       'Home_Carries.2_PrgDist', 'Away_Touches.3_Mid 3rd',
       'Away_Touches.4_Att 3rd', 'Away_Carries.2_PrgDist', 'Home_Standard_Gls',
       'Home_Expected_xG', 'Home_Expected.1_npxG', 'Home_Expected.2_npxG/Sh',
       'Home_Expected.3_G-xG', 'Home_Expected.4_np:G-xG', 'Away_Standard_Gls',
       'Away_Expected_xG', 'Away_Expected.1_npxG', 'Away_Expected.2_npxG/Sh',
       'Away_Expected.3_G-xG', 'Away_Expected.4_np:G-xG']
df = df.drop(to_drop, axis=1)
df['match_result'] = df['score'].apply(get_match_result)
# save the final df 
df.to_csv("/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/PL_final_dataset.csv", index=False)
