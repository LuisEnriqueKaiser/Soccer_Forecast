#!/usr/bin/env python3

import pandas as pd
import numpy as np

###############################################################################
# Feature Engineering Functions
###############################################################################
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
    For each row in df, compute the mean of 'home_stat_col' / 'away_stat_col'
    for the last n matches for that team (time-aware).
    """
    home_output_col = f"home_mean_{stat_name}_last_{n}"
    away_output_col = f"away_mean_{stat_name}_last_{n}"
    # Debug print to confirm newly created columns
    # print(home_output_col, away_output_col)

    for row_number in range(len(df)):
        home_team = df.loc[row_number, home_team_col]
        away_team = df.loc[row_number, away_team_col]

        # Last n matches for home_team, chronologically before current row
        home_team_matches = df[
            ((df[home_team_col] == home_team) | (df[away_team_col] == home_team))
            & (df.index < row_number)
        ].tail(n)
        home_team_values = []
        for i in range(len(home_team_matches)):
            if home_team_matches.iloc[i][home_team_col] == home_team:
                home_team_values.append(home_team_matches.iloc[i][home_stat_col])
            else:
                home_team_values.append(home_team_matches.iloc[i][away_stat_col])

        # Last n matches for away_team
        away_team_matches = df[
            ((df[home_team_col] == away_team) | (df[away_team_col] == away_team))
            & (df.index < row_number)
        ].tail(n)
        away_team_values = []
        for i in range(len(away_team_matches)):
            if away_team_matches.iloc[i][home_team_col] == away_team:
                away_team_values.append(away_team_matches.iloc[i][home_stat_col])
            else:
                away_team_values.append(away_team_matches.iloc[i][away_stat_col])

        df.loc[row_number, home_output_col] = np.mean(home_team_values) if home_team_values else np.nan
        df.loc[row_number, away_output_col] = np.mean(away_team_values) if away_team_values else np.nan

    return df

def create_diff_and_ratio_features(df, home_col, away_col, prefix):
    """
    Create difference (home - away) and ratio (home / (away + epsilon)) features
    for each rolling stat. This helps capture relative strengths between teams.
    """
    diff_col = f"diff_{prefix}"
    ratio_col = f"ratio_{prefix}"
    epsilon = 1e-5

    df[diff_col] = df[home_col] - df[away_col]
    df[ratio_col] = df[home_col] / (df[away_col] + epsilon)
    return df

def get_match_result(score_str):
    """
    Convert a score string (e.g., '2–1') into:
      1 -> home win
      0 -> draw
     -1 -> away win
    """
    try:
        home_goals, away_goals = map(int, score_str.split("–"))
    except Exception:
        return np.nan
    if home_goals > away_goals:
        return 1
    elif home_goals == away_goals:
        return 0
    else:
        return -1

def drop_old_incomplete_rows(df, date_col, frac_threshold=0.5):
    """
    Drops rows where fraction of missing values >= frac_threshold AND date is in the past.
    """
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    now = pd.to_datetime("today")
    frac_missing = df.isna().sum(axis=1) / df.shape[1]
    cond_incomplete = frac_missing >= frac_threshold
    cond_in_past = df[date_col] < now
    df_clean = df[~(cond_incomplete & cond_in_past)].copy()
    df_clean.reset_index(drop=True, inplace=True)
    return df_clean

def impute_missing_with_columnmean_up_until_that_date(df):
    """
    Impute missing numeric values with the expanding mean from all PRIOR rows (shift(1)).
    Sort by date first. Some columns can remain NaN if there's no prior data.
    """
    df_imputed = df.copy()
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        cum_mean = df_imputed[col].expanding(min_periods=1).mean().shift(1)
        df_imputed[col] = df_imputed[col].fillna(cum_mean)
    return df_imputed

def prepare_data(df):
    """
    Convert date, sort by date, create home_win/away_win, match_result_cat.
    """
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['home_win'] = (df['match_result'] == 1).astype(int)
    df['away_win'] = (df['match_result'] == -1).astype(int)
    mapping = {-1: 0, 0: 1, 1: 2}
    df['match_result_cat'] = df['match_result'].map(mapping).astype('category')
    return df

def form_of_teams(df):
    """
    Rolling average form (previous 5 matches) for home/away.
    The average is computed from home_win or away_win flags, shifted by 1 so the current match isn't included.
    """
    df['home_form'] = df.groupby('home_team')['home_win'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
    )
    df['away_form'] = df.groupby('away_team')['away_win'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
    )
    return df

def compute_pi_rating_online(df):
    """
    Compute an online Pi rating for each team (pre-game rating).
    The idea is to adjust each team's cumulative score by a blend of match points + opponent rating.
    """
    teams = pd.unique(df[['home_team', 'away_team']].values.ravel('K'))
    cum_score = {team: 0.0 for team in teams}
    match_count = {team: 0 for team in teams}

    home_pi = []
    away_pi = []

    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        home_rating = cum_score[home] / match_count[home] if match_count[home] > 0 else 1.0
        away_rating = cum_score[away] / match_count[away] if match_count[away] > 0 else 1.0

        home_pi.append(home_rating)
        away_pi.append(away_rating)

        res = row['match_result']
        if res == 1:
            home_points = 3
            away_points = 0
        elif res == 0:
            home_points = 1
            away_points = 1
        elif res == -1:
            home_points = 0
            away_points = 3
        else:
            home_points = np.nan
            away_points = np.nan

        # Update cumulative scores
        cum_score[home] += (home_points + away_rating)
        cum_score[away] += (away_points + home_rating)
        match_count[home] += 1
        match_count[away] += 1

    df['home_pi_rating'] = home_pi
    df['away_pi_rating'] = away_pi
    return df

def implied_prob(df, home_odd, draw_odd, away_odd3):
    """
    Compute implied probabilities from odds. note that they have to add up to one 
    so i "clean" them in this way

    dynamic naming based on the name of the odd
    """
    home_col = f"implied_prob_home_{home_odd}"
    draw_col = f"implied_prob_draw_{draw_odd}"
    away_col = f"implied_prob_away_{away_odd3}"

    df[home_col] = 1/df[home_odd]
    df[draw_col] = 1/df[draw_odd]
    df[away_col] = 1/df[away_odd3]
    
    df[home_col] = df[home_col]/(df[home_col]+df[draw_col]+df[away_col])
    df[away_col] = df[draw_col]/(df[home_col]+df[draw_col]+df[away_col])
    df[away_col] = df[away_col]/(df[home_col]+df[draw_col]+df[away_col])
    return df

def give_integer_to_teams(df):
    """
    Give an integer to each team
    """
    teams = pd.unique(df[['home_team', 'away_team']].values.ravel('K'))
    team_dict = {team: i for i, team in enumerate(teams)}
    df['home_team_integer'] = df['home_team'].map(team_dict)
    df['away_team_integer'] = df['away_team'].map(team_dict)
    return df

###############################################################################
# Main script
###############################################################################
def main():
    # Paths (adjust as needed)
    INPUT_PATH = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_merged_dataset_cleaned.csv"
    TRAIN_OUTPUT_PATH = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/train_data.csv"
    TEST_OUTPUT_PATH = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/test_data.csv"

    # Read data
    df = pd.read_csv(INPUT_PATH)
    n = 3

    # 1) Rolling mean stats for each pair.
    #    Examples of stats: xG, tackles, passes, progressive carries, possession, etc.
    #    If you have other columns (shots_on_target, total_shots, etc.), add them here.
    pairs = [
        ["home_xg", "away_xg", "xg"],
        ["Home_Tackles_Tkl", "Away_Tackles_Tkl", "Tackles_Tkl"],
        ["Home_Touches.3_Mid 3rd", "Away_Touches.3_Mid 3rd", "Touches_Mid3rd"],
        ["Home_Touches.4_Att 3rd", "Away_Touches.4_Att 3rd", "Touches_Att3rd"],
        ["Home_Carries.2_PrgDist", "Away_Carries.2_PrgDist", "Carries_PrgDist"],
        ["Home_Standard_Gls", "Away_Standard_Gls", "Standard_Gls"],
        ["Home_Expected_xG", "Away_Expected_xG", "Expected_xG"],
        ["Home_Expected.1_npxG", "Away_Expected.1_npxG", "Expected_npxG"],
        # If your dataset has possession columns, for example:
        # ["Home_Possession", "Away_Possession", "Possession"],
    ]
    for home_col, away_col, name in pairs:
        df = compute_mean_stat_last_n_games(df, n, home_col, away_col, name)

    # 2) match_result from score, then Pi rating
    df['match_result'] = df['score'].apply(get_match_result)
    df = compute_pi_rating_online(df)

    # 2a) Create difference & ratio features from the rolling means
    #     (these help model the relative matchup between teams)

    # dynamic naming of the new rolling stats
    new_rolling_stats = [
        "mean_xg_last_3",
        "mean_Tackles_Tkl_last_3",
        "mean_Touches_Mid3rd_last_3",
        "mean_Touches_Att3rd_last_3",
        "mean_Carries_PrgDist_last_3",
        "mean_Standard_Gls_last_3",
        "mean_Expected_xG_last_3",
        "mean_Expected_npxG_last_3",
        # Add "mean_Possession_last_3" here if you included it above
    ]

    for stat_name in new_rolling_stats:
        home_col = f"home_{stat_name}"
        away_col = f"away_{stat_name}"
        create_diff_and_ratio_features(df, home_col, away_col, stat_name)

    # Also, difference in Pi rating can help indicate relative team strength
    df["diff_pi_rating"] = df["home_pi_rating"] - df["away_pi_rating"]


    # 3) Drop columns not needed.
    #    Many of your new features are stored as 'home_mean_...', 'away_mean_...', or
    #    'diff_...', 'ratio_...'. The lines below remove original raw columns.
    to_drop = [
        # Original raw columns you no longer need:
        'Home_Tackles_Tkl',
        'Home_Touches.3_Mid 3rd',
        'Home_Touches.4_Att 3rd',
        'Home_Carries.2_PrgDist',
        'Home_Standard_Gls',
        'Home_Expected_xG',
        'Home_Expected.1_npxG',
        'Away_Tackles_Tkl',
        'Away_Touches.3_Mid 3rd',
        'Away_Touches.4_Att 3rd',
        'Away_Carries.2_PrgDist',
        'Away_Standard_Gls',
        'Away_Expected_xG',
        'Away_Expected.1_npxG',
        'home_xg',
        'away_xg',
        'season',
        'Home_Expected.3_G-xG',
        'Away_Expected.3_G-xG','Away_Expected.2_npxG.Sh',
        'attendance','Home_Total.3_TotDist', "Away_Tackles.1_TklW", "Home_Tackles.1_TklW",
        'Home_Tackles.2_Def 3rd', 'Away_Tackles.4_Att 3rd', 'Home_Expected.2_npxG/Sh',
        'Home_xA_nan', 'Home_Expected.4_np:G-xG','Away_Total.3_TotDist','Away_xA_nan','Home_Total.4_PrgDist','Home_xAG_nan','away_win','Away_xAG_nan',
        'Away_Tackles.2_Def 3rd','Home_Blocks.1_Sh','Home_Blocks.1_Sh','Away_Expected.4_np:G-xG','Away_Total.4_PrgDist', 'Home_Tackles.4_Att 3rd','Away_Blocks.1_Sh', 'Away_Expected.2_npxG/Sh',
        # Potential columns with no further usage
        # 'Home_Possession', 'Away_Possession',
        # ... etc ...
    ]
    odds_columns = ["B365H","B365D",	"B365A"	,"PSH"	,"PSD"	"PSA","WHH","WHD","WHA"]

    df = implied_prob(df, "B365H", "B365D", "B365A")
    df = implied_prob(df, "PSH", "PSD", "PSA")
    df = implied_prob(df, "WHH", "WHD", "WHA")
    #df.drop(columns=odds_columns, inplace=True, errors='ignore')
    df = give_integer_to_teams(df)
    df.drop(columns=to_drop, inplace=True, errors='ignore')

    # 4) Prepare data & form
    df = prepare_data(df)
    df = form_of_teams(df)
    # 5) drop incomplete rows, date filter, and do time-aware imputations
    df = drop_old_incomplete_rows(df, date_col="date", frac_threshold=0.5)
    df = df[df["date"] < pd.to_datetime("today")]
    df = df.sort_values('date').reset_index(drop=True)
    df = impute_missing_with_columnmean_up_until_that_date(df)

    # 6) Time-based train/test split (90% / 10%)
    n_total = len(df)
    # split based on date
    df_train = df[df["date"] < pd.to_datetime("2024-01-01")]
    df_test = df[df["date"] >= pd.to_datetime("2024-01-01")]


    # 7) SAVE final train/test
    df_train.to_csv(TRAIN_OUTPUT_PATH, index=False)
    df_test.to_csv(TEST_OUTPUT_PATH, index=False)

    print(f"Saved training data to {TRAIN_OUTPUT_PATH}")
    print(f"Saved test data to {TEST_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
