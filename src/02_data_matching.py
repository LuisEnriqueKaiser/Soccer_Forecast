import pandas as pd

def transform_premier_league_data(input_csv: str) -> pd.DataFrame:
    """
    1) Read CSV with header=None so columns are [0, 1, 2, ...].
    2) Take the first row (row 0) and rename columns to "{idx}_{value_in_first_row}".
    3) Remove that first row from the data.
    4) Find the 'Match Report' column (name ends with '_Match Report').
    5) Find the 'Venue' or 'Home/Away' column (name ends with '_Venue' or '_Home/Away').
       Convert it to a categorical so that 'Home' sorts before 'Away'.
    6) Sort the DataFrame by [MatchReportCol, HomeAwayCol].
    7) Pair consecutive rows => first row is 'Home_', second row is 'Away_'.
    8) Return the resulting DataFrame.
    """

    # --- Step 1: Read the file with no header ---
    df_raw = pd.read_csv(input_csv, header=0)
    
    # If your file has trailing empty lines, remove them up front
    df_raw.dropna(how="all", inplace=True)

    # The first row (row 0) is used to rename columns
    row0 = df_raw.iloc[0]
    
    new_cols = []
    for col_idx in df_raw.columns:
        label = str(row0[col_idx]).strip()
        new_name = f"{col_idx}_{label}"
        new_cols.append(new_name)
        
    df_raw.columns = new_cols
    
    # --- Step 2: Drop that first row (it was only for naming) ---
    df = df_raw.iloc[1:].copy()
    df.dropna(how="all", inplace=True)  # drop any all-NaN rows
    df.reset_index(drop=True, inplace=True)
    
    # --- Step 3: Identify the 'Match Report' column and the 'Venue'/'Home/Away' column ---
    match_report_col = None
    home_away_col = None
    
    for c in df.columns:
        # e.g. "3_Match Report"
        if c.startswith("match_report"):
            match_report_col = c
        if c.startswith("venue"):
            home_away_col = c

    # If we found a home_away_col, let's ensure "Home" sorts before "Away".
    # We can do this by making it a categorical with the ordering we want:
    if home_away_col in df.columns:
        # We don't know exactly what the text is – let's assume "Home" and "Away".
        df[home_away_col] = df[home_away_col].astype(str).str.strip()
        cat_type = pd.CategoricalDtype(categories=["Home", "Away"], ordered=True)
        df[home_away_col] = df[home_away_col].astype(cat_type)
    
    # --- Step 4: Sort the DataFrame ---
    # We only sort by the columns we actually have
    sort_cols = []
    if match_report_col and match_report_col in df.columns:
        sort_cols.append(match_report_col)
    if home_away_col and home_away_col in df.columns:
        sort_cols.append(home_away_col)
    
    if sort_cols:
        df.sort_values(by=sort_cols, inplace=True)

    df.reset_index(drop=True, inplace=True)
    
    # --- Step 5: Pair consecutive rows (home, away) ---
    num_rows = len(df)
    if num_rows % 2 != 0:
        print(
            "Warning: Odd number of rows in the data after dropping the header "
            "and sorting. The last row cannot be paired as a match."
        )

    # We'll build a list of dicts, each dict = one match
    match_rows = []
    for i in range(0, num_rows, 2):
        home_data = df.iloc[i]
        if i + 1 < num_rows:
            away_data = df.iloc[i + 1]
        else:
            away_data = None  # no matching away row

        combined = {}
        for col in df.columns:
            combined[f"Home_{col}"] = home_data[col]
        if away_data is not None:
            for col in df.columns:
                combined[f"Away_{col}"] = away_data[col]

        match_rows.append(combined)
    
    df_final = pd.DataFrame(match_rows)
    return df_final


if __name__ == "__main__":
    # Example usage
    # shooting: 
    input_path = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/fbref/eng-premier_league_shooting_2018-2019_2019-2020_2020-2021_2021-2022_2022-2023_2023-2024_2024-2025.csv"
    df_result = transform_premier_league_data(input_path)
    df_result.rename(columns={"Home_opponent_nan": "Away_Team", "Away_opponent_nan": "Home_Team","Home_match_report_nan":"Match_report" }, inplace=True)
    # take subset
    columns_to_take = ["Match_report","Home_Standard_Gls", "Home_Expected_xG", "Home_Expected.1_npxG", 
    "Home_Expected.2_npxG/Sh", "Home_Expected.3_G-xG", "Home_Expected.4_np:G-xG",
     "Away_Standard_Gls", "Away_Expected_xG","Away_Expected.1_npxG", "Away_Expected.2_npxG/Sh", "Away_Expected.3_G-xG", "Away_Expected.4_np:G-xG"
    ]
    df_result = df_result[columns_to_take]
    df_result.to_csv("/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_shots.csv", index=False)

    # passing
    input_path = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/fbref/eng-premier_league_passing_2018-2019_2019-2020_2020-2021_2021-2022_2022-2023_2023-2024_2024-2025.csv"
    df_result = transform_premier_league_data(input_path)
    df_result.rename(columns={"Home_opponent_nan": "Away_Team", "Away_opponent_nan": "Home_Team","Home_match_report_nan":"Match_report" }, inplace=True)

    columns_to_take = ['Match_report','Home_Total.3_TotDist', 'Home_Total.4_PrgDist',
       'Home_xAG_nan','Home_xA_nan', 
       'Away_Total.3_TotDist','Away_Total.4_PrgDist', 'Away_xAG_nan', 'Away_xA_nan',
]
    df_result = df_result[columns_to_take]
    df_result.to_csv("/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_passing.csv", index=False)

    # posession
    input_path = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/fbref/eng-premier_league_possession_2018-2019_2019-2020_2020-2021_2021-2022_2022-2023_2023-2024_2024-2025.csv"
    df_result = transform_premier_league_data(input_path)
    df_result.rename(columns={"Home_opponent_nan": "Away_Team", "Away_opponent_nan": "Home_Team","Home_match_report_nan":"Match_report" }, inplace=True)

    columns_to_take = ['Match_report','Home_Touches.3_Mid 3rd',
       'Home_Touches.4_Att 3rd', 'Home_Carries.2_PrgDist',
       'Away_Touches.3_Mid 3rd', 'Away_Touches.4_Att 3rd', 'Away_Carries.2_PrgDist']

    df_result = df_result[columns_to_take]
    df_result.to_csv("/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_possession.csv", index=False)

    # defense 
    input_path = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/fbref/eng-premier_league_defense_2018-2019_2019-2020_2020-2021_2021-2022_2022-2023_2023-2024_2024-2025.csv"
    df_result = transform_premier_league_data(input_path)
    df_result.rename(columns={"Home_opponent_nan": "Away_Team", "Away_opponent_nan": "Home_Team","Home_match_report_nan":"Match_report" }, inplace=True)
    columns_to_take = ['Match_report', 'Home_Tackles_Tkl', 'Home_Tackles.1_TklW', 'Home_Tackles.2_Def 3rd', 'Home_Tackles.4_Att 3rd','Home_Blocks.1_Sh',
                       'Away_Tackles_Tkl', 'Away_Tackles.1_TklW', 'Away_Tackles.2_Def 3rd', 'Away_Tackles.4_Att 3rd','Away_Blocks.1_Sh'
                       ]
    df_result = df_result[columns_to_take]
    df_result.to_csv("/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_defense.csv", index=False)

    # schedule 
    input_path = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/fbref/eng-premier_league_schedule.csv"
    df_result = pd.read_csv(input_path)
    df_result.rename(columns={"match_report":"Match_report"}, inplace=True)
    df_result.to_csv("/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processed_schedule.csv", index=False)