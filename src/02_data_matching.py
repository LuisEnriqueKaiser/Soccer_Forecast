import pandas as pd
import project_specifics as config
import os

def transform_premier_league_data(input_csv: str) -> pd.DataFrame:
    """
    1) Read CSV with header=0 so that the first row contains the names.
    2) Rename columns as "{idx}_{value_in_first_row}".
    3) Remove the first row (which was used for naming).
    4) Identify the 'Match Report' column and the 'Venue' (or 'Home/Away') column.
    5) For the venue column, convert it to a categorical type with "Home" sorting before "Away".
    6) Sort the DataFrame by the identified columns.
    7) Pair consecutive rows (first row as Home, second row as Away) and combine them into one record.
    8) Return the resulting DataFrame.
    """
    # --- Step 1: Read the file and remove any all-NaN rows (which might be trailing empty lines) ---
    df_raw = pd.read_csv(input_csv, header=0)
    df_raw.dropna(how="all", inplace=True)

    # --- Step 2: Use the first row to build new column names ---
    row0 = df_raw.iloc[0]
    new_cols = []
    for col_idx in df_raw.columns:
        label = str(row0[col_idx]).strip()
        new_name = f"{col_idx}_{label}"
        new_cols.append(new_name)
    df_raw.columns = new_cols

    # --- Step 3: Drop the first row (it was only used for header naming) ---
    df = df_raw.iloc[1:].copy()
    df.dropna(how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- Step 4: Identify the 'Match Report' and 'Venue'/'Home/Away' columns ---
    match_report_col = None
    home_away_col = None
    for c in df.columns:
        lc = c.lower()
        if "match report" in lc:
            match_report_col = c
        if "venue" in lc or "home/away" in lc:
            home_away_col = c

    # --- Step 5: If the home/away column exists, convert it to a categorical with order Home < Away ---
    if home_away_col and home_away_col in df.columns:
        df[home_away_col] = df[home_away_col].astype(str).str.strip()
        cat_type = pd.CategoricalDtype(categories=["Home", "Away"], ordered=True)
        df[home_away_col] = df[home_away_col].astype(cat_type)

    # --- Step 6: Sort the DataFrame using available sort columns (Match_report then Home/Away) ---
    sort_cols = []
    if match_report_col and match_report_col in df.columns:
        sort_cols.append(match_report_col)
    if home_away_col and home_away_col in df.columns:
        sort_cols.append(home_away_col)
    if sort_cols:
        df.sort_values(by=sort_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- Step 7: Pair consecutive rows to yield a single row per match ---
    num_rows = len(df)
    if num_rows % 2 != 0:
        print(
            "Warning: Odd number of rows in the data after dropping header and sorting. "
            "The last row cannot be paired as a match."
        )

    match_rows = []
    for i in range(0, num_rows, 2):
        home_data = df.iloc[i]
        if i + 1 < num_rows:
            away_data = df.iloc[i + 1]
        else:
            # If we have an odd row out, skip it or handle it as needed
            away_data = None

        combined = {}
        for col in df.columns:
            combined[f"Home_{col}"] = home_data[col]
        if away_data is not None:
            for col in df.columns:
                combined[f"Away_{col}"] = away_data[col]

        match_rows.append(combined)

    df_final = pd.DataFrame(match_rows)
    return df_final


def process_and_save(input_file: str, output_file: str, columns: list):
    """
    Transforms the input CSV file, applies renaming as specified in config.COLUMN_RENAME,
    takes a subset of columns, and writes the result to the output file.
    """
    # skip if it is the schedule file
    if "schedule" in input_file.lower():
        df = pd.read_csv(input_file, header = 0)
        df.to_csv(output_file, index=False)
        # end the function
        return
    
    df = transform_premier_league_data(input_file)
    # Rename columns as per the config rules
    try:
        df.rename(columns=config.COLUMN_RENAME, inplace=True)
    except:
        pass
    # Take the subset of desired columns
    df = df[columns]
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to: {output_file}")


if __name__ == "__main__":
    # Process Shooting Data
    process_and_save(
        input_file=config.SHOOTING_FILE,
        output_file=config.SHOOTING_OUTPUT,
        columns=config.SHOOTING_COLUMNS
    )

    # Process Passing Data
    process_and_save(
        input_file=config.PASSING_FILE,
        output_file=config.PASSING_OUTPUT,
        columns=config.PASSING_COLUMNS
    )

    # Process Possession Data
    process_and_save(
        input_file=config.POSSESSION_FILE,
        output_file=config.POSSESSION_OUTPUT,
        columns=config.POSSESSION_COLUMNS
    )

    # Process Defense Data
    process_and_save(
        input_file=config.DEFENSE_FILE,
        output_file=config.DEFENSE_OUTPUT,
        columns=config.DEFENSE_COLUMNS
    )
    print("Done with the main datasets.")
    # Process Schedule data the same way (so we also get a single row per match)
    process_and_save(
        input_file=config.SCHEDULE_FILE,
        output_file=config.SCHEDULE_OUTPUT,
        columns=config.SCHEDULE_COLUMNS
    )
