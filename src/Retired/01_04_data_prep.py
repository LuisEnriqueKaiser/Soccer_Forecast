import pandas as pd
import numpy as np
from project_specifics import MERGED_OUTPUT, CLEANED_MERGED

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

# Define function to assign season based on the month and year
def assign_season(date):
    year = date.year
    month = date.month
    if month >= 8:  # August to December
        return f"{year}-{year+1}"
    else:  # January to May (part of the previous season)
        return f"{year-1}-{year}"

def main():
    df = pd.read_csv(MERGED_OUTPUT)
    df = drop_old_incomplete_rows(df, date_col="date", frac_threshold=0.5)

    # Apply the function to assign season
    df['season'] = df['date'].apply(assign_season)
    # Number each unique season
    df['season_number'] = df['season'].astype('category').cat.codes + 1

    # save the cleaned dataset
    df.to_csv(CLEANED_MERGED, index=False)
    print(f"Final cleaned and merged data saved to {CLEANED_MERGED}")

# Allow the file to run standalone:
if __name__ == "__main__":
    main()