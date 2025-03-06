import pandas as pd
import numpy as np

def drop_old_incomplete_rows(df, date_col, frac_threshold=0.5):
    """
    Drops rows where:
      1) The fraction of missing values in that row is >= frac_threshold
      2) The date in that row is strictly in the past (compared to 'today')

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_col : str
        Name of the date column in df
    frac_threshold : float
        Fraction of columns that must be missing to consider the row as 'incomplete'.
        E.g. frac_threshold=0.5 => row is incomplete if >= 50% columns are NaN.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame
    """
    # Ensure the date column is of datetime type
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Current date (you could also fix a cutoff date if needed)
    now = pd.to_datetime("today")

    # Calculate fraction of missing values for each row
    frac_missing = df.isna().sum(axis=1) / df.shape[1]

    # Condition 1: fraction missing is >= threshold
    cond_incomplete = frac_missing >= frac_threshold
    # Condition 2: date is in the past
    cond_in_past = df[date_col] < now

    # Keep only rows that do NOT match (incomplete & in past)
    df_clean = df[~(cond_incomplete & cond_in_past)].copy()
    df_clean.reset_index(drop=True, inplace=True)
    return df_clean



df = pd.read_csv("/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_merged_dataset.csv")
df = drop_old_incomplete_rows(df, date_col="date", frac_threshold=0.5)
# save the cleaned dataset
df.to_csv("/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_merged_dataset_cleaned.csv", index=False)

