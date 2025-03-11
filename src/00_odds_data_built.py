import pandas as pd

# Define file path
filepath = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/odds/"

# List of input file paths
csv_files = [
    filepath + "1819.csv",
    filepath + "1920.csv",
    filepath + "2021.csv",
    filepath + "2122.csv",
    filepath + "2223.csv",
    filepath + "2324.csv",
    filepath + "2425.csv"
]

# The columns we definitely need
base_cols = [
    "Date",       # Match date
    "HomeTeam",   # Home team name
    "AwayTeam"    # Away team name
]

# 1X2 match odds for major bookmakers + aggregated columns (pre-closing):
bookmaker_odds_cols = [
    "B365H", "B365D", "B365A",  # Bet365
    "BFH", "BFD", "BFA",  # Betfair
    "BSH", "BSD", "BSA",  # Blue Square
    "BWH", "BWD", "BWA",  # Bet&Win
    "GBH", "GBD", "GBA",  # Gamebookers
    "IWH", "IWD", "IWA",  # Interwetten
    "LBH", "LBD", "LBA",  # Ladbrokes
    "PSH", "PSD", "PSA",  # Pinnacle
    "SOH", "SOD", "SOA",  # Sporting Odds
    "SBH", "SBD", "SBA",  # Sportingbet
    "SJH", "SJD", "SJA",  # Stan James
    "SYH", "SYD", "SYA",  # Stanleybet
    "VCH", "VCD", "VCA",  # VC Bet
    "WHH", "WHD", "WHA",  # William Hill
    "Bb1X2", "BbMxH", "BbAvH", "BbMxD", "BbAvD", "BbMxA", "BbAvA",  # BetBrain Aggregated Odds
    "MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA",  # Market Max/Average Odds
    "BFEH", "BFED", "BFEA"  # Betfair Exchange Odds
]

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
output_path = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/merged_odds.csv"
final_df.to_csv(output_path, index=False)

print(f"Combined data has been saved to: {output_path}")
