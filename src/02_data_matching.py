import os
import pandas as pd
import project_specifics as config

# Directly define the excluded columns.
EXCLUDED_COLUMNS = [
    'date', 'round', 'day', 'venue', 'result',
    'GF', 'GA', 'opponent', 'Err', 'time', 'match_report'
]


def normalize_col_name(col: str) -> str:
    """
    Normalize a new column name (e.g., "9_Match Report_nan") so we can match it
    against the original excluded names (e.g., "match_report").
    
    Steps:
    1) Remove leading digits and underscores (e.g., "9_").
    2) Convert to lowercase.
    3) Remove any occurrence of the literal string "nan".
    4) Remove all underscores and spaces.
       e.g., "9_Match Report_nan" -> "matchreport"
    """
    # Strip leading digits/underscores
    while col and (col[0].isdigit() or col[0] == '_'):
        col = col[1:]
    # Convert to lowercase
    col = col.lower()
    # Remove "nan"
    col = col.replace('nan', '')
    # Remove underscores and spaces
    col = col.replace('_', '').replace(' ', '')
    return col


def get_excluded_name(col: str) -> str | None:
    """
    Given a column name 'col' (e.g., "9_Match Report"), normalize it and
    check if it matches any name in EXCLUDED_COLUMNS (also normalized).
    
    If found, return the **original** excluded name from the list (e.g., "match_report").
    Otherwise, return None.
    """
    col_norm = normalize_col_name(col)
    for ex in EXCLUDED_COLUMNS:
        # Normalize the excluded name too, so "match_report" becomes "matchreport".
        ex_norm = ex.lower().replace(' ', '').replace('_', '')
        if col_norm == ex_norm:
            # Return the exact name from the user’s EXCLUDED_COLUMNS list
            return ex
    return None


def transform_premier_league_data(input_csv: str) -> pd.DataFrame:
    """
    1) Read CSV with header=0 so that the first row contains the names.
    2) Rename columns as "{idx}_{value_in_first_row}" but handle empty/NaN gracefully.
    3) Remove that first row (used for naming).
    4) Identify the 'Match Report' and 'Venue'/'Home/Away' columns by name search.
    5) Convert the home/away column to a categorical type with "Home" < "Away".
    6) Sort by (Match_report, Home/Away).
    7) Pair consecutive rows (home, away) -> single combined record.
    8) Prefix most columns with Home_/Away_, except those in EXCLUDED_COLUMNS (once, unprefixed).
    9) Return the resulting DataFrame.
    """
    # --- Step 1: Read CSV and drop fully empty rows ---
    df_raw = pd.read_csv(input_csv, header=0)
    df_raw.dropna(how="all", inplace=True)

    # --- Step 2: Build new column names from the first row, avoiding "_nan" ---
    row0 = df_raw.iloc[0].fillna('')  # Convert real NaNs to empty strings
    new_cols = []
    for old_col in df_raw.columns:
        label = str(row0[old_col]).strip()
        if label.lower() == 'nan':
            label = ''
        new_name = f"{old_col}_{label}".rstrip('_')  # Remove trailing underscore if label is empty
        new_cols.append(new_name)

    df_raw.columns = new_cols

    # --- Step 3: Remove the first row (used for naming) and reindex ---
    df = df_raw.iloc[1:].copy()
    df.dropna(how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- Step 4: Identify 'Match Report' and 'Venue' columns by searching their names ---
    match_report_col = None
    home_away_col = None
    for c in df.columns:
        c_lc = c.lower()
        if "match report" in c_lc:
            match_report_col = c
        if "venue" in c_lc or "home/away" in c_lc:
            home_away_col = c

    # --- Step 5: Convert the home/away column to a categorical (Home < Away) ---
    if home_away_col in df.columns:
        df[home_away_col] = df[home_away_col].astype(str).str.strip()
        cat_type = pd.CategoricalDtype(categories=["Home", "Away"], ordered=True)
        df[home_away_col] = df[home_away_col].astype(cat_type)

    # --- Step 6: Sort by match_report, then home_away (if found) ---
    sort_cols = []
    if match_report_col in df.columns:
        sort_cols.append(match_report_col)
    if home_away_col in df.columns:
        sort_cols.append(home_away_col)
    if sort_cols:
        df.sort_values(by=sort_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- Step 7: Pair consecutive rows (Home, Away) into single-row matches ---
    num_rows = len(df)
    if num_rows % 2 != 0:
        print("Warning: Odd number of rows. The last row can't be paired.")

    match_rows = []
    for i in range(0, num_rows, 2):
        home_data = df.iloc[i]
        away_data = df.iloc[i + 1] if (i + 1 < num_rows) else None

        combined = {}
        if away_data is None:
            # No away row (odd row out) -> store only home data
            for col in df.columns:
                excluded_match = get_excluded_name(col)
                if excluded_match:
                    combined[excluded_match] = home_data[col]
                else:
                    combined[f"Home_{col}"] = home_data[col]
        else:
            # We have both a Home and Away row
            for col in df.columns:
                excluded_match = get_excluded_name(col)
                if excluded_match:
                    # For excluded columns, store them once, unprefixed, from Home row
                    combined[excluded_match] = home_data[col]
                else:
                    # Otherwise, prefix with Home_ and Away_
                    combined[f"Home_{col}"] = home_data[col]
                    combined[f"Away_{col}"] = away_data[col]

        match_rows.append(combined)

    df_final = pd.DataFrame(match_rows)
    return df_final


def process_and_save(input_file: str, output_file: str):
    """
    - If "schedule" appears in input_file, copy CSV as-is (no transformation).
    - Otherwise, transform the data via transform_premier_league_data().
    - After transformation, rename columns (if config.COLUMN_RENAME exists).
    - Finally, save to output_file.
    """
    # If it's a schedule file, just copy it unmodified
    if "schedule" in input_file.lower():
        df = pd.read_csv(input_file, header=0)
        df.to_csv(output_file, index=False)
        return

    # Transform the CSV into single-row-per-match
    df = transform_premier_league_data(input_file)

    # Attempt to rename columns from config
    try:
        df.rename(columns=config.COLUMN_RENAME, inplace=True)
    except:
        pass

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the result
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to: {output_file}")


if __name__ == "__main__":
    # Example usage for multiple files:

    # Process Shooting Data
    process_and_save(
        input_file=config.SHOOTING_FILE,
        output_file=config.SHOOTING_OUTPUT
    )

    # Process Passing Data
    process_and_save(
        input_file=config.PASSING_FILE,
        output_file=config.PASSING_OUTPUT
    )

    # Process Possession Data
    process_and_save(
        input_file=config.POSSESSION_FILE,
        output_file=config.POSSESSION_OUTPUT
    )

    # Process Defense Data
    process_and_save(
        input_file=config.DEFENSE_FILE,
        output_file=config.DEFENSE_OUTPUT
    )

    # Process Schedule Data
    process_and_save(
        input_file=config.SCHEDULE_FILE,
        output_file=config.SCHEDULE_OUTPUT
    )

    process_and_save(
        input_file=config.SHOT_CREATION_FILE,
        output_file=config.SHOT_CREATION_OUTPUT
    )

    process_and_save(
        input_file=config.PASSING_TYPES_FILE,
        output_file=config.PASSING_TYPES_OUTPUT
    )

    process_and_save(
        input_file=config.MISC_FILE,
        output_file=config.MISC_OUTPUT
    )

    process_and_save(
        input_file=config.KEEPER_FILE,
        output_file=config.KEEPER_OUTPUT
    )
    print("Done with all datasets.")