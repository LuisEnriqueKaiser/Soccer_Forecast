#!/usr/bin/env python3

import pandas as pd
import numpy as np
from project_specifics import CLEANED_MERGED, TRAIN_OUTPUT_PATH, TEST_OUTPUT_PATH, pairs_new, to_drop

###############################################################################
# Feature Engineering Functions
###############################################################################

def compute_mean_stat_last_n_games(
    df,
    n,
    home_stat_col,
    away_stat_col,
    stat_name,
    date_col="date",
    home_team_col="home_team",
    away_team_col="away_team"
):
    """
    For each match row, compute:
      home_mean_{stat_name}_last_{n}
      away_mean_{stat_name}_last_{n}

    This is the average of that team's last n matches (chronologically),
    ignoring whether they were home or away in those prior matches.

    Implementation notes:
      1) We ensure 'match_id' is a column in df so we can pivot/merge cleanly.
      2) We build a "long" DataFrame with one row for each team's involvement.
      3) We group by 'team', sorting by date, then do shift(1).rolling(n).mean().
      4) We pivot on match_id, yield separate columns for 'home' and 'away'.
      5) We rename & merge back into df.

    The final columns in df become:
      home_mean_{stat_name}_last_{n}
      away_mean_{stat_name}_last_{n}
    """

    # 1) Ensure we have a match_id column (0..N-1)
    #    If df already has match_id, no harm done. Otherwise create it.
    if "match_id" not in df.columns:
        df = df.reset_index(drop=True).reset_index(names="match_id")

    # 2) Build a long DataFrame
    long_rows = []
    for _, row in df.iterrows():
        mid = row["match_id"]
        long_rows.append({
            "match_id": mid,
            "team": row[home_team_col],
            "date": row[date_col],
            "stat": row[home_stat_col],
            "venue_for_roll": "home"
        })
        long_rows.append({
            "match_id": mid,
            "team": row[away_team_col],
            "date": row[date_col],
            "stat": row[away_stat_col],
            "venue_for_roll": "away"
        })

    long_df = pd.DataFrame(long_rows)

    # 3) Sort by (team, date)
    long_df.sort_values(["team", "date"], inplace=True)

    # 4) Compute rolling means with shift(1), ignoring the current row
    #    We use group_keys=False to avoid multi-index side effects
    long_df["rolling_mean"] = (
        long_df.groupby("team", group_keys=False)["stat"]
               .transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
    )

    # 5) Pivot so each match_id has a 'home' and 'away' rolling_mean
    pivoted = long_df.pivot(
        index="match_id",
        columns="venue_for_roll",
        values="rolling_mean"
    ).reset_index()  # so match_id is a normal column, not an index

    # pivoted now has columns: [match_id, home, away]
    home_out = f"home_mean_{stat_name}_last_{n}"
    away_out = f"away_mean_{stat_name}_last_{n}"
    pivoted.rename(columns={"home": home_out, "away": away_out}, inplace=True)

    # 6) Merge pivoted columns back to df on match_id
    df_rolled = pd.merge(df, pivoted, on="match_id", how="left")

    return df_rolled


def create_diff_and_ratio_features(df, home_col, away_col, prefix):
    """
    Create difference (home - away) and ratio (home / (away + epsilon)) features.
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
    cond_in_past = df[date_col] < now + pd.Timedelta(days=10)
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

    # If your 'date' is not guaranteed monotonic, we re-sort by date:
    df_imputed.sort_values("date", inplace=True)

    for col in numeric_cols:
        cum_mean = df_imputed[col].expanding(min_periods=1).mean().shift(1)
        df_imputed[col] = df_imputed[col].fillna(cum_mean)

    # After filling, restore the original order if needed:
    df_imputed.sort_values("match_id", inplace=True)

    return df_imputed


def prepare_data(df):
    """
    Convert date, sort by date, create home_win/away_win, match_result_cat.
    match_result uses (1=away,2=draw,3=home).
    """
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values("date", inplace=True)

    df['home_win'] = (df['match_result'] == 3).astype(int)
    df['away_win'] = (df['match_result'] == 1).astype(int)

    mapping = {1:1, 2:2, 3:3}
    df['match_result_cat'] = df['match_result'].map(mapping).astype('category')
    df.reset_index(drop=True, inplace=True)
    return df


def form_of_teams(df):
    """
    Rolling average form (previous 5 matches) for home/away.
    The average is computed from home_win or away_win flags, 
    shifted by 1 so the current match isn't included.
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
    The idea is to adjust each team's cumulative score by a blend
    of match points + opponent rating.
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

    df[home_col] = 1 / df[home_odd]
    df[draw_col] = 1 / df[draw_odd]
    df[away_col] = 1 / df[away_odd3]

    denom = df[home_col] + df[draw_col] + df[away_col]
    df[home_col] = df[home_col] / denom
    df[draw_col] = df[draw_col] / denom
    df[away_col] = df[away_col] / denom
    return df


def give_integer_to_teams(df):
    """
    Assign an integer ID to each team.
    """
    teams = pd.unique(df[['home_team', 'away_team']].values.ravel('K'))
    team_dict = {team: i + 1 for i, team in enumerate(teams)}
    df['home_team_integer'] = df['home_team'].map(team_dict)
    df['away_team_integer'] = df['away_team'].map(team_dict)
    return df


def add_points_from_previous_season(df):
    """
    For each team and season_number, compute total points in that season,
    then merge them as "previous season" points onto each row.
    """
    temp = df[['home_team', 'away_team', 'season_number', 'match_result']].copy()

    temp['points_home'] = np.where(
        temp['match_result'] == 3, 3,
        np.where(temp['match_result'] == 2, 1,
                 np.where(temp['match_result'] == 1, 0, np.nan))
    )
    temp['points_away'] = np.where(
        temp['match_result'] == 1, 3,
        np.where(temp['match_result'] == 2, 1,
                 np.where(temp['match_result'] == 3, 0, np.nan))
    )

    df_home = temp[['home_team', 'season_number', 'points_home']].rename(
        columns={'home_team': 'team', 'points_home': 'points'}
    )
    df_away = temp[['away_team', 'season_number', 'points_away']].rename(
        columns={'away_team': 'team', 'points_away': 'points'}
    )
    df_team_season = pd.concat([df_home, df_away], ignore_index=True)

    df_team_season_grouped = (
        df_team_season
        .groupby(['team', 'season_number'])['points']
        .sum()
        .reset_index()
        .rename(columns={'points': 'total_points'})
    )

    df_team_season_grouped['season_number_next'] = df_team_season_grouped['season_number'] + 1
    df_team_season_grouped.rename(columns={'total_points': 'points_prev_season'}, inplace=True)

    df = df.merge(
        df_team_season_grouped[['team', 'season_number_next', 'points_prev_season']],
        how='left',
        left_on=['home_team', 'season_number'],
        right_on=['team', 'season_number_next']
    )
    df.rename(columns={'points_prev_season': 'home_points_prev_season'}, inplace=True)
    df.drop(columns=['team', 'season_number_next'], inplace=True, errors='ignore')

    df = df.merge(
        df_team_season_grouped[['team', 'season_number_next', 'points_prev_season']],
        how='left',
        left_on=['away_team', 'season_number'],
        right_on=['team', 'season_number_next']
    )
    df.rename(columns={'points_prev_season': 'away_points_prev_season'}, inplace=True)
    df.drop(columns=['team', 'season_number_next'], inplace=True, errors='ignore')

    df['home_points_prev_season'] = df['home_points_prev_season'].fillna(0)
    df['away_points_prev_season'] = df['away_points_prev_season'].fillna(0)

    return df


###############################################################################
# Main script
###############################################################################
def main():
    # 1) Read data
    df = pd.read_csv(CLEANED_MERGED)
    columns = df.columns.tolist()
    # check if match rport is in the columns
    if "match_report" not in columns:
        raise ValueError("Match report not in columns")
    included_columns = [
        "date", "home_team", "away_team", "score", "venue", "score", "match_report"]
    not_included_columns = set(columns) - set(included_columns)
    print(f"Columns not included: {not_included_columns}")
    # 2) Drop columns not needed
    redundant_columns = [
        "date_defense", "date_passing", "date_possession", 
        "date_shooting", "date_shot_creation", "date_passing_types", 
        "date_misc", "date_keeper",

        "round", "round_passing", "round_possession", "round_shooting", 
        "round_shot_creation", "round_passing_types", "round_misc", 
        "round_keeper",

        "day_defense", "day_passing", "day_possession", "day_shooting", 
        "day_shot_creation", "day_passing_types", "day_misc", "day_keeper",

        "venue_defense", "venue_passing", "venue_possession", "venue_shooting", 
        "venue_shot_creation", "venue_passing_types", "venue_misc", "venue_keeper",

        "result_passing", "result_possession", "result_shooting", 
        "result_shot_creation", "result_passing_types", "result_misc", 
        "result_keeper",

        "GF_passing", "GF_possession", "GF_shooting", 
        "GF_shot_creation", "GF_passing_types", "GF_misc", "GF_keeper",

        "GA_passing", "GA_possession", "GA_shooting", 
        "GA_shot_creation", "GA_passing_types", "GA_misc", "GA_keeper",

        "opponent_passing", "opponent_possession", "opponent_shooting", 
        "opponent_shot_creation", "opponent_passing_types", "opponent_misc", 
        "opponent_keeper",

        "time_defense", "time_passing", "time_possession", "time_shooting", 
        "time_shot_creation", "time_passing_types", "time_misc", "time_keeper"
    ]
    df.drop(columns=redundant_columns, inplace=True, errors='ignore')

    # 3) Basic date sorting
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.sort_values("date", inplace=True)

    # 4) Add 'season_number' (optional, as you have in your data)
    #    If you want consecutive ints, do: (year + 1 if x.month >= 8 else year) - 2018
    if "season_number" not in df.columns:
        df["season_number"] = df["date"].apply(lambda x: x.year)

    # 5) Rolling means for each pair
    n = 3
    for home_col, away_col, name in pairs_new:
        print(f"Computing rolling mean for {name} ...")
        df = compute_mean_stat_last_n_games(
            df,
            n=n,
            home_stat_col=home_col,
            away_stat_col=away_col,
            stat_name=name,
            date_col="date",
            home_team_col="home_team",
            away_team_col="away_team",
        )

    # 6) match_result from 'score'
    df['match_result'] = df['score'].apply(get_match_result)

    # 7) Pi rating
    df = compute_pi_rating_online(df)

    df["diff_pi_rating"] = df["home_pi_rating"] - df["away_pi_rating"]

    # 9) Implied probabilities
    try:
        df = implied_prob(df, "B365H", "B365D", "B365A")
        df = implied_prob(df, "PSH", "PSD", "PSA")
        df = implied_prob(df, "WHH", "WHD", "WHA")
    except:
        pass

    # 10) Give integer IDs to teams
    df = give_integer_to_teams(df)

    # 11) Add previous season points
    df = add_points_from_previous_season(df)

    # 12) Drop columns not needed
    df.drop(columns=to_drop, inplace=True, errors='ignore')

    # 13) Prepare data & form features
    df = prepare_data(df)
    df = form_of_teams(df)

    # 14) drop incomplete rows, exclude future, time-aware imputation
    df = drop_old_incomplete_rows(df, date_col="date", frac_threshold=0.5)
    df = impute_missing_with_columnmean_up_until_that_date(df)

    dropping = ["score", "home_win", "away_win", "match_result"]
    df.drop(columns=dropping, inplace=True, errors='ignore')

    # drop not included columns
    df.drop(columns=not_included_columns, inplace=True, errors='ignore')
    


    # 16) Time-based split

    today = pd.to_datetime("today")
    df = df[df["date"] < today + pd.Timedelta(days=10)].copy()
    df_train = df[df["date"] < today].copy()
    df_test = df[df["date"] >= today].copy()

    # 17) Save
    df_train.to_csv(TRAIN_OUTPUT_PATH, index=False)
    df_test.to_csv(TEST_OUTPUT_PATH, index=False)
    df.to_csv("/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/combined.csv", index=False)
    print(f"Saved train -> {TRAIN_OUTPUT_PATH}")
    print(f"Saved test -> {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
