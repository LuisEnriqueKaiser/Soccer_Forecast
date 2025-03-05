import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds simple rolling average features for home and away goals.
    Computes a rolling mean (window=5) for both home and away goals grouped by team.
    """
    # Drop rows without goals (for rolling calculations)
    df_historical = df.dropna(subset=["home_goals", "away_goals"]).copy()
    
    # Compute rolling average for home goals grouped by home_team
    df_historical = df_historical.sort_values(by=["home_team", "date"]).reset_index(drop=True)
    df_historical["home_goals_rolling_mean"] = (
        df_historical.groupby("home_team")["home_goals"]
        .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    )
    
    # Compute rolling average for away goals grouped by away_team
    df_historical = df_historical.sort_values(by=["away_team", "date"]).reset_index(drop=True)
    df_historical["away_goals_rolling_mean"] = (
        df_historical.groupby("away_team")["away_goals"]
        .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    )
    
    # Merge the rolling features back into the original dataframe (by date, home_team, away_team)
    df_merged = pd.merge(
        df,
        df_historical[["home_team", "away_team", "date", "home_goals_rolling_mean", "away_goals_rolling_mean"]],
        on=["home_team", "away_team", "date"],
        how="left"
    )
    df_merged = df_merged.sort_values(by="date").reset_index(drop=True)
    return df_merged
