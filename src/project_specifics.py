'''
this file serves as an overview of all projectwide variables and constants which can be changed by the zser to adapt the project to their needs
'''
import os



# script 00_odds_data_built.py
filepath_00 = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/odds/"
output_path_00 = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/merged_odds.csv"

csv_files = [
    filepath_00 + "1819.csv",
    filepath_00 + "1920.csv",
    filepath_00 + "2021.csv",
    filepath_00 + "2122.csv",
    filepath_00 + "2223.csv",
    filepath_00 + "2324.csv",
    filepath_00 + "2425.csv"
]


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


# script 01_download_data.py
# Define leagues, seasons, and advanced stats

leagues = [
    "ENG-Premier League",  # Premier League
]
seasons = ["2018-2019","2019-2020" , "2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]

advanced_stats = [
        'keeper', 'shooting', 'passing', 'passing_types', 
        'goal_shot_creation', 'defense', 'possession', 'misc'
]
    
# File path where CSV files will be saved
filepath_01 = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/fbref/"
    


# script 02_data_matching.py
# Base directory for the project
BASE_DIR = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast"

# Directories for data input and output
DATA_DIR = os.path.join(BASE_DIR, "data")
FBREF_DIR = os.path.join(DATA_DIR, "fbref")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Input file paths
SHOOTING_FILE = os.path.join(
    FBREF_DIR,
    "eng-premier_league_shooting_2018-2019_2019-2020_2020-2021_2021-2022_2022-2023_2023-2024_2024-2025.csv"
)
PASSING_FILE = os.path.join(
    FBREF_DIR,
    "eng-premier_league_passing_2018-2019_2019-2020_2020-2021_2021-2022_2022-2023_2023-2024_2024-2025.csv"
)
POSSESSION_FILE = os.path.join(
    FBREF_DIR,
    "eng-premier_league_possession_2018-2019_2019-2020_2020-2021_2021-2022_2022-2023_2023-2024_2024-2025.csv"
)
DEFENSE_FILE = os.path.join(
    FBREF_DIR,
    "eng-premier_league_defense_2018-2019_2019-2020_2020-2021_2021-2022_2022-2023_2023-2024_2024-2025.csv"
)
SCHEDULE_FILE = os.path.join(FBREF_DIR, "eng-premier_league_schedule.csv")

# Output file paths
SHOOTING_OUTPUT = os.path.join(PROCESSED_DIR, "PL_processes_shots.csv")
PASSING_OUTPUT = os.path.join(PROCESSED_DIR, "PL_processes_passing.csv")
POSSESSION_OUTPUT = os.path.join(PROCESSED_DIR, "PL_processes_possession.csv")
DEFENSE_OUTPUT = os.path.join(PROCESSED_DIR, "PL_processes_defense.csv")
SCHEDULE_OUTPUT = os.path.join(PROCESSED_DIR, "PL_processed_schedule.csv")

# Column renaming: these keys (as produced by transform) are renamed for consistency.
COLUMN_RENAME = {
    "Home_opponent_nan": "Away_Team",
    "Away_opponent_nan": "Home_Team",
    "Home_match_report_nan": "Match_report"
}

# Columns to take after transformation for each dataset

SHOOTING_COLUMNS = [
    "Match_report",
    "Home_Standard_Gls", "Home_Expected_xG", "Home_Expected.1_npxG",
    "Home_Expected.2_npxG/Sh", "Home_Expected.3_G-xG", "Home_Expected.4_np:G-xG",
    "Away_Standard_Gls", "Away_Expected_xG", "Away_Expected.1_npxG",
    "Away_Expected.2_npxG/Sh", "Away_Expected.3_G-xG", "Away_Expected.4_np:G-xG"
]

PASSING_COLUMNS = [
    "Match_report", "Home_Total.3_TotDist", "Home_Total.4_PrgDist",
    "Home_xAG_nan", "Home_xA_nan",
    "Away_Total.3_TotDist", "Away_Total.4_PrgDist", "Away_xAG_nan", "Away_xA_nan"
]

POSSESSION_COLUMNS = [
    "Match_report", "Home_Touches.3_Mid 3rd", "Home_Touches.4_Att 3rd",
    "Home_Carries.2_PrgDist", "Away_Touches.3_Mid 3rd", "Away_Touches.4_Att 3rd",
    "Away_Carries.2_PrgDist"
]

DEFENSE_COLUMNS = [
    "Match_report", "Home_Tackles_Tkl", "Home_Tackles.1_TklW", "Home_Tackles.2_Def 3rd",
    "Home_Tackles.4_Att 3rd", "Home_Blocks.1_Sh",
    "Away_Tackles_Tkl", "Away_Tackles.1_TklW", "Away_Tackles.2_Def 3rd",
    "Away_Tackles.4_Att 3rd", "Away_Blocks.1_Sh"
]

# schedule columns are week day	date	time	home_team	home_xg	score	away_xg	away_team	attendance	venue	referee	Match_report
SCHEDULE_COLUMNS = [
    "Match_report", "week", "day", "date", "time", "home_team", "home_xg", "score",
    "away_xg", "away_team", "attendance", "venue", "referee"]

# script 03_data_merge.py
# this script also relies on the defined variables in 02_data_matching.py

# Merged dataset output path
MERGED_OUTPUT = os.path.join(PROCESSED_DIR, "PL_merged_dataset.csv")


# script 04_00_data_prep.py
CLEANED_MERGED = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_merged_dataset_cleaned.csv"
MERGED_ODDS = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/merged_odds.csv"


# script 04_01_odds_merge.py
# Mapping dictionary
team_name_mapping = {
    'Man United': 'Manchester Utd',
    'Bournemouth': 'Bournemouth',
    'Fulham': 'Fulham',
    'Huddersfield': 'Huddersfield',
    'Newcastle': 'Newcastle Utd',
    'Watford': 'Watford',
    'Wolves': 'Wolves',
    'Arsenal': 'Arsenal',
    'Liverpool': 'Liverpool',
    'Southampton': 'Southampton',
    'Cardiff': 'Cardiff City',
    'Chelsea': 'Chelsea',
    'Everton': 'Everton',
    'Leicester': 'Leicester City',
    'Tottenham': 'Tottenham',
    'West Ham': 'West Ham',
    'Brighton': 'Brighton',
    'Burnley': 'Burnley',
    'Man City': 'Manchester City',
    'Crystal Palace': 'Crystal Palace',
    'Aston Villa': 'Aston Villa',
    'Norwich': 'Norwich City',
    'Sheffield United': 'Sheffield Utd',
    'West Brom': 'West Brom',
    'Leeds': 'Leeds United',
    'Brentford': 'Brentford',
    "Nott'm Forest": "Nott'ham Forest",
    'Luton': 'Luton Town',
    'Ipswich': 'Ipswich Town'
}



# scrip 05_00_feature_engineering.py
# Paths (adjust as needed)
TRAIN_OUTPUT_PATH = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/train_data.csv"
TEST_OUTPUT_PATH = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/test_data.csv"

pairs = [
        ["home_xg", "away_xg", "xg"],
        ["Home_Tackles_Tkl", "Away_Tackles_Tkl", "Tackles_Tkl"],
        ["Home_Touches.3_Mid 3rd", "Away_Touches.3_Mid 3rd", "Touches_Mid3rd"],
        ["Home_Touches.4_Att 3rd", "Away_Touches.4_Att 3rd", "Touches_Att3rd"],
        ["Home_Carries.2_PrgDist", "Away_Carries.2_PrgDist", "Carries_PrgDist"],
        ["Home_Standard_Gls", "Away_Standard_Gls", "Standard_Gls"],
        ["Home_Expected_xG", "Away_Expected_xG", "Expected_xG"],
        ["Home_Expected.1_npxG", "Away_Expected.1_npxG", "Expected_npxG"],
        # Add more pairs if needed
    ]


to_drop = [
        'Home_Tackles_Tkl', 'Home_Touches.3_Mid 3rd', 'Home_Touches.4_Att 3rd',
        'Home_Carries.2_PrgDist', 'Home_Standard_Gls', 'Home_Expected_xG',
        'Home_Expected.1_npxG', 'Away_Tackles_Tkl', 'Away_Touches.3_Mid 3rd',
        'Away_Touches.4_Att 3rd', 'Away_Carries.2_PrgDist', 'Away_Standard_Gls',
        'Away_Expected_xG', 'Away_Expected.1_npxG', 'home_xg', 'away_xg',
        'season','Home_Expected.3_G-xG','Away_Expected.3_G-xG','Away_Expected.2_npxG.Sh',
        'attendance','Home_Total.3_TotDist', "Away_Tackles.1_TklW", "Home_Tackles.1_TklW",
        'Home_Tackles.2_Def 3rd', 'Away_Tackles.4_Att 3rd', 'Home_Expected.2_npxG/Sh',
        'Home_xA_nan', 'Home_Expected.4_np:G-xG','Away_Total.3_TotDist','Away_xA_nan',
        'Home_Total.4_PrgDist','Home_xAG_nan','away_win','Away_xAG_nan','Away_Tackles.2_Def 3rd',
        'Home_Blocks.1_Sh','Away_Expected.4_np:G-xG','Away_Total.4_PrgDist','Home_Tackles.4_Att 3rd',
        'Away_Blocks.1_Sh','Away_Expected.2_npxG/Sh'
    ]


# script 05_01_pca.py
  # File paths: adjust as needed




# script 06_00_model.py
TABLES_DIR = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/results/tables"
FIGURES_DIR = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/results/figures"
RESULTS_FREQ = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/test_data_predictions_freq.csv"



# script 08 plots comparison 
RESULTS_FOLDER = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/"
OUTPUT_FOLDER  = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/results/comparison_models"