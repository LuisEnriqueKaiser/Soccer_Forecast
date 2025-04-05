#!/usr/bin/env python3

import pandas as pd
import numpy as np
from project_specifics import CLEANED_MERGED, TRAIN_OUTPUT_PATH, TEST_OUTPUT_PATH, pairs, to_drop

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
      3 -> home win
      2 -> draw
      1 -> away win

    This is consistent with your usage:
       if home_goals > away_goals: return 3
       elif home_goals == away_goals: return 2
       else: return 1
    """
    try:
        home_goals, away_goals = map(int, score_str.split("–"))
    except Exception:
        return np.nan
    if home_goals > away_goals:
        return 3
    elif home_goals == away_goals:
        return 2
    else:
        return 1

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
    match_result uses (1=away,2=draw,3=home). We map them to match_result_cat the same way
    so 1->1, 2->2, 3->3.
    """
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['home_win'] = (df['match_result'] == 3).astype(int)
    df['away_win'] = (df['match_result'] == 1).astype(int)

    # Here we keep the same factor levels as the numeric code
    # 1 => away, 2 => draw, 3 => home
    mapping = {1:1, 2:2, 3:3}
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

        res = row['match_result']  # 3=home,2=draw,1=away
        if res == 3:
            home_points = 3
            away_points = 0
        elif res == 2:
            home_points = 1
            away_points = 1
        elif res == 1:
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
    Compute implied probabilities from odds. We normalize them so they sum to 1.
    """
    home_col = f"implied_prob_home_{home_odd}"
    draw_col = f"implied_prob_draw_{draw_odd}"
    away_col = f"implied_prob_away_{away_odd3}"

    df[home_col] = 1/df[home_odd]
    df[draw_col] = 1/df[draw_odd]
    df[away_col] = 1/df[away_odd3]
    
    denom = df[home_col] + df[draw_col] + df[away_col]
    df[home_col] = df[home_col] / denom
    df[draw_col] = df[draw_col] / denom
    df[away_col] = df[away_col] / denom
    return df

def give_integer_to_teams(df):
    """
    Give an integer to each team
    """
    teams = pd.unique(df[['home_team', 'away_team']].values.ravel('K'))
    team_dict = {team: i+1 for i, team in enumerate(teams)}
    df['home_team_integer'] = df['home_team'].map(team_dict)
    df['away_team_integer'] = df['away_team'].map(team_dict)
    return df

def add_points_from_previous_season(df):
    """
    For each team and season_number, compute total points in that season.
    Then, for each row (match), add how many points the home/away team
    earned in the prior season. If a team did not exist (i.e., newly promoted),
    they get 0 for previous season points.

    IMPORTANT: This code now uses your match_result encoding:
       3 => home win
       2 => draw
       1 => away win

    The function sums all matches from the prior season to get the team's total points,
    then merges them onto the next season's rows.
    """

    # 1) Build an auxiliary dataframe with each team's total points per season
    temp = df[['home_team', 'away_team', 'season_number', 'match_result']].copy()
    
    # Points for home side in each row
    # If home_result=3 => home gets 3 points, 2 => home gets 1, 1 => home gets 0
    temp['points_home'] = np.where(
        temp['match_result'] == 3, 3,
        np.where(temp['match_result'] == 2, 1,
                 np.where(temp['match_result'] == 1, 0, np.nan))
    )
    
    # Points for away side in each row
    # If match_result=1 => away gets 3 points, 2 => away gets 1, 3 => away gets 0
    temp['points_away'] = np.where(
        temp['match_result'] == 1, 3,
        np.where(temp['match_result'] == 2, 1,
                 np.where(temp['match_result'] == 3, 0, np.nan))
    )
    
    # Reshape: one "team" column
    df_home = temp[['home_team', 'season_number', 'points_home']].rename(
        columns={'home_team': 'team', 'points_home': 'points'}
    )
    df_away = temp[['away_team', 'season_number', 'points_away']].rename(
        columns={'away_team': 'team', 'points_away': 'points'}
    )
    df_team_season = pd.concat([df_home, df_away], ignore_index=True)
    
    # Sum points by (team, season_number)
    df_team_season_grouped = (
        df_team_season
        .groupby(['team','season_number'])['points']
        .sum()
        .reset_index()
        .rename(columns={'points':'total_points'})
    )
    
    # 2) Create a column referencing "next season" => shift points to next season
    df_team_season_grouped['season_number_next'] = df_team_season_grouped['season_number'] + 1
    
    # Rename to clarify these are prior-season points
    df_team_season_grouped.rename(columns={'total_points':'points_prev_season'}, inplace=True)
    
    # 3) Merge back for home teams
    df = df.merge(
        df_team_season_grouped[['team','season_number_next','points_prev_season']],
        how='left',
        left_on=['home_team','season_number'],
        right_on=['team','season_number_next']
    )
    df.rename(columns={'points_prev_season':'home_points_prev_season'}, inplace=True)
    df.drop(columns=['team','season_number_next'], inplace=True, errors='ignore')
    
    # 4) Merge back for away teams
    df = df.merge(
        df_team_season_grouped[['team','season_number_next','points_prev_season']],
        how='left',
        left_on=['away_team','season_number'],
        right_on=['team','season_number_next']
    )
    df.rename(columns={'points_prev_season':'away_points_prev_season'}, inplace=True)
    df.drop(columns=['team','season_number_next'], inplace=True, errors='ignore')
    
    # 5) If a team wasn't in the previous season, we fill that with 0
    df['home_points_prev_season'] = df['home_points_prev_season'].fillna(0)
    df['away_points_prev_season'] = df['away_points_prev_season'].fillna(0)

    return df

###############################################################################
# Main script
###############################################################################
def main():
    # Paths (adjust as needed)
    INPUT_PATH = CLEANED_MERGED

    # Read data
    df = pd.read_csv(INPUT_PATH)
    n = 3

    # (A) *** CREATE a 'season_number' column before add_points_from_previous_season ***
    # Because your league runs from August(Y) to May(Y+1), we do:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['season_number'] = df['date'].apply(lambda x: x.year + 1 if x.month >= 8 else x.year)
    # Make the numbering consecutive from 1..7 by subtracting 2017, for instance
    df['season_number'] = df['season_number'] - 2018

    # 1) Rolling mean stats for each pair (xG, tackles, etc.).
    for home_col, away_col, name in pairs:
        df = compute_mean_stat_last_n_games(df, n, home_col, away_col, name)

    # 2) match_result from 'score', then Pi rating
    df['match_result'] = df['score'].apply(get_match_result)
    df = compute_pi_rating_online(df)

    # 2a) Create difference & ratio features from the rolling means
    new_rolling_stats = [
        "mean_xg_last_3",
        "mean_Tackles_Tkl_last_3",
        "mean_Touches_Mid3rd_last_3",
        "mean_Touches_Att3rd_last_3",
        "mean_Carries_PrgDist_last_3",
        "mean_Standard_Gls_last_3",
        "mean_Expected_xG_last_3",
        "mean_Expected_npxG_last_3",
    ]
    for stat_name in new_rolling_stats:
        home_col = f"home_{stat_name}"
        away_col = f"away_{stat_name}"
        df = create_diff_and_ratio_features(df, home_col, away_col, stat_name)

    # Also, difference in Pi rating
    df["diff_pi_rating"] = df["home_pi_rating"] - df["away_pi_rating"]

    # 3) Add implied probabilities from odds
    try:
        df = implied_prob(df, "B365H", "B365D", "B365A")
        df = implied_prob(df, "PSH", "PSD", "PSA")
        df = implied_prob(df, "WHH", "WHD", "WHA")
    except:
        pass
    # 4) Give integer IDs to teams
    df = give_integer_to_teams(df)

    # 5) Add previous season points
    df = add_points_from_previous_season(df)

    # 6) Drop columns not needed
    df.drop(columns=to_drop, inplace=True, errors='ignore')

    # 7) Prepare data & form features
    df = prepare_data(df)
    df = form_of_teams(df)

    # 8) drop incomplete rows, date filter, time-aware imputations
    df = drop_old_incomplete_rows(df, date_col="date", frac_threshold=0.5)
    df = df[df["date"] < pd.to_datetime("today")]
    df = df.sort_values('date').reset_index(drop=True)
    df = impute_missing_with_columnmean_up_until_that_date(df)

    # drop columns not known on gameday (except the target)
    dropping = ["score", "home_win", "away_win", "match_result"]
    df = df.drop(columns=dropping, errors='ignore')

    # 9) Time-based train/test split (example: before/after 2024-07-01)
    df_train = df[df["date"] < pd.to_datetime("2024-10-01")]
    df_test = df[df["date"] >= pd.to_datetime("2024-10-01")]

    # 10) SAVE final train/test
    df_train.to_csv(TRAIN_OUTPUT_PATH, index=False)
    df_test.to_csv(TEST_OUTPUT_PATH, index=False)

    print(f"Saved training data to {TRAIN_OUTPUT_PATH}")
    print(f"Saved test data to {TEST_OUTPUT_PATH}")
    

if __name__ == "__main__":
    main()
