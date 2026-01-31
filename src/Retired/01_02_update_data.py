import os

import pandas as pd
import project_specifics as config

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




    # Create (output_path, old_path, new_path) triples for each pair of old/new files you want to combine.
    # You can change the output filenames as you see fit.
    # Below is just one example of pairing them (defense, goal_shot_creation, keeper, etc.).

def main():
    pairs = []
    filepath = config.filepath_data
    addition_path = config.addition_path
    filepath = os.path.join(filepath,  addition_path)
    path_to_search_in = os.path.join(filepath, addition_path)

    # loop thorugh the fbref directory
    for filename in os.listdir(config.filepath_data + addition_path):
        # if in the file there is no "only", then proceed
        print("here")
        if "only" not in filename and filename.endswith(".csv"):
            basefile = os.path.join(filepath,  filename)
            # search for the update file, it is the basefile with "_only_current.csv instead of .csv
            updatefile = basefile.replace(".csv", "_only_current.csv")
            # check if the update file exists
            print(updatefile)
            if os.path.exists(updatefile):
                # create the output file, it is the basefile with "_updated.csv instead of .csv
                outputfile = basefile.replace(".csv", "_updated.csv")
                # add the pair to the list
                pairs.append((outputfile, updatefile, basefile))

    print(pairs)
    # Process each pair
    for output_csv, new_csv_path, old_csv_path in pairs:
        combined_df = combine_and_filter_csv(old_csv_path, new_csv_path)
        combined_df.to_csv(output_csv, index=False)
        print(f"Combined data saved to: {output_csv}")


# run the main script

if __name__ == "__main__":
    # List of file paths
    main()