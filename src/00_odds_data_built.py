'''
Description: This script reads in the odds data from the CSV files and combines them into a single DataFrame.
This data is later merged with the game level advanced stats. The final DataFrame is saved to a CSV file.
'''


import pandas as pd
from project_specifics import csv_files, base_cols, bookmaker_odds_cols, output_path_00






# Combine them into a single list of columns to keep
keep_cols = base_cols + bookmaker_odds_cols

df_list = []

for file_path in csv_files:
    try:
        df = pd.read_csv(file_path)
        
        # We only want the intersection of the columns that exist
        existing_cols = df.columns.intersection(keep_cols)
        
        # Keep only those columns
        df = df[existing_cols]
        
        # Append to our list
        df_list.append(df)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Concatenate all DataFrames (row-bind)
final_df = pd.concat(df_list, ignore_index=True)

# Drop columns with missing values
final_df = final_df.dropna(axis=1, how='any')

# Write out the final DataFrame
final_df.to_csv(output_path_00, index=False)

print(f"Combined data has been saved to: {output_path_00}")
