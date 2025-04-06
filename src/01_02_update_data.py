import pandas as pd

def combine_and_filter_csv(old_path: str, new_path: str) -> pd.DataFrame:
    """
    Reads two CSV files (old_path and new_path), filters rows in the old DataFrame
    where the 'Date' is older than the oldest date in the new DataFrame, then
    concatenates the new data onto the filtered old data.

    :param old_path: Path to the old CSV file
    :param new_path: Path to the new CSV file
    :return: A single DataFrame containing the filtered old data and the new data
    """

    # Load old and new CSVs
    df_old = pd.read_csv(old_path)
    df_new = pd.read_csv(new_path)

    # Convert Date columns to datetime if needed
    df_old['date'] = pd.to_datetime(df_old['date'])
    df_new['date'] = pd.to_datetime(df_new['date'])

    # Determine the oldest date in the new DataFrame
    oldest_new_date = df_new['date'].min()
    print(f"Oldest date in new DataFrame: {oldest_new_date}")
    # Filter old DataFrame to only include rows with Date >= oldest_new_date
    df_old_filtered = df_old[df_old['date'] < oldest_new_date]

    # Concatenate the old (filtered) and new DataFrames
    combined_df = pd.concat([df_old_filtered, df_new])
    # sort by date
    combined_df = combined_df.sort_values(by='date')
    return combined_df

if __name__ == "__main__":
    # List of file paths
    file_paths = [
    "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processed_schedule_update.csv",
    "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processed_schedule.csv",
    "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_defense_update.csv",
    "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_defense.csv",
    "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_keeper_update.csv",
    "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_keeper.csv",
    "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_misc_update.csv",
    "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_misc.csv",
    "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_passing_types_update.csv",
    "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_passing_types.csv",
    "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_passing_update.csv",
    "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_passing.csv",
    "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_possession_update.csv",
    "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_possession.csv",
    "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_shot_creation_update.csv",
    "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_shot_creation.csv",
    "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_shots_update.csv",
    "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_shots.csv"
    ]


    # Create (output_path, old_path, new_path) triples for each pair of old/new files you want to combine.
    # You can change the output filenames as you see fit.
    # Below is just one example of pairing them (defense, goal_shot_creation, keeper, etc.).


    pairs = [
    (file_paths[1], file_paths[0], file_paths[1]),
    (file_paths[3], file_paths[2], file_paths[3]),
    (file_paths[5], file_paths[4], file_paths[5]),
    (file_paths[7], file_paths[6], file_paths[7]),
    (file_paths[9], file_paths[8], file_paths[9]),
    (file_paths[11], file_paths[10], file_paths[11]),
    (file_paths[13], file_paths[12], file_paths[13]),
    (file_paths[15], file_paths[14], file_paths[15]),
    (file_paths[17], file_paths[16], file_paths[17]),
    ]

    # Process each pair
    for output_csv, new_csv_path, old_csv_path in pairs:
        combined_df = combine_and_filter_csv(old_csv_path, new_csv_path)
        combined_df.to_csv(output_csv, index=False)
        print(f"Combined data saved to: {output_csv}")
