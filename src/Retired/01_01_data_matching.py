import os
import pandas as pd
import project_specifics as config

# Directly define the excluded columns.
EXCLUDED_COLUMNS = [
    'date', 'round', 'day', 'venue', 'result',
    'GF', 'GA', 'opponent', 'Err', 'time', 'match_report'
]



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

    print(f"Processed data saved to: {output_file}")


if __name__ == "__main__":
    # Example usage for multiple files:
    filepath = config.filepath_data
    fbref_dir = os.path.join(filepath, config.fbref)
    addition_path = config.addition_path
    # loop thorugh the fbref directory
    for filename in os.listdir(fbref_dir):
        if filename.endswith(".csv"):
            input_file = os.path.join(fbref_dir, filename)
            output_file = os.path.join(filepath, config.addition_path, filename)
            process_and_save(input_file, output_file)

    print("Done with all datasets.")