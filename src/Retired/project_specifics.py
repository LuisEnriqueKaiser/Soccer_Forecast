'''
    this file serves as an overview of all projectwide variables and constants which can be changed by the zser to adapt the project to their needs
'''
import os



# script 00_odds_data_built.py
filepath_00 = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/odds/"
output_path_00 = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/merged_odds.csv"
filepath_data = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/"
fbref = "fbref"
addition_path = "processed"
oods = "odds"

france =  "Uber"
spain = "PD"
germany = "BL"
italy = "SA"
england = "PL"

relevant_odds = [
    "B365H", "B365D", "B365A",
    "BWH", "BWD", "BWA",
    "GBH", "GBD", "GBA",
    "IWH", "IWD", "IWA"
]

relevant_columns = ["Date", "HomeTeam", "AwayTeam", "Season"]

# deleting the first row
# building based on match_report
# make a home and away prefix for each column 

# Note: first build home and away team goals based on score columns
schedule = {"week": "week","date": "date", "home_team": "home_team", "away_team": "away_team", "season": "season",
           "home_team_goals": "home_team_goals", "away_team_goals": "away_team_goals", "match_report": "match_report",
           "home_xg": "home_xg", "away_xg": "away_xg"}



defense_data = {"Int": "int", "Blocks": "blocks", "Clr": "clr", "Err": "err", "season": "season"}

passing = {"Total": "pas_tot", "Total.1" : "pas_att", "Total.4": "pgrssv_dist", "Total.3": "tot_dist", "Ast": "ast",
           "xAG": "exp_assists_g", "xA": "exp_assists"}

passing_type ={"Pass Types.5": "crosses"}

shots = {"Standard.2":"SoT", "Expected": "xg","Standard.3":"SoT_over_goals", 
         "Expected.1": "npxg", "Expected.2": "npxg_per_shot"}



csv_files = [
    filepath_00 + "1819.csv",
    filepath_00 + "1920.csv",
    filepath_00 + "2021.csv",
    filepath_00 + "2122.csv",
    filepath_00 + "2223.csv",
    filepath_00 + "2324.csv",
    filepath_00 + "2425.csv",
    filepath_data + "2526.csv",
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
    "ENG-Premier League",  # Premier League#
    "ESP-La Liga",  # La Liga
    "ITA-Serie A",  # Serie A
    "GER-Bundesliga",  # Bundesliga
    "FRA-Ligue 1",  # Ligue 1
]
seasons = ["2005-2006", "2006-2007", "2007-2008",
    "2008-2009", "2009-2010", "2010-2011",
    "2011-2012", "2012-2013", "2013-2014",
    "2014-2015", "2015-2016", "2016-2017",
    "2017-2018",    "2018-2019","2019-2020" , "2020-2021",
    "2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]


advanced_stats = ["schedule","shooting","passing", "goal_shot_creation"
"possession", "defense", "passing_types"]
    



# File path where CSV files will be saved
filepath_01 = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/fbref/"

EXCLUDED_COLUMNS = ["date", "round", "day", "venue", "result", "GF", "GA", "opponent", "Err", "time", "match_report"]


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

SHOT_CREATION_FILE = os.path.join(
    FBREF_DIR,
    "eng-premier_league_goal_shot_creation_2018-2019_2019-2020_2020-2021_2021-2022_2022-2023_2023-2024_2024-2025.csv"
)
PASSING_TYPES_FILE = os.path.join(
    FBREF_DIR,
    "eng-premier_league_passing_types_2018-2019_2019-2020_2020-2021_2021-2022_2022-2023_2023-2024_2024-2025.csv"
)
MISC_FILE = os.path.join(
    FBREF_DIR,
    "eng-premier_league_misc_2018-2019_2019-2020_2020-2021_2021-2022_2022-2023_2023-2024_2024-2025.csv"
)

KEEPER_FILE = os.path.join(
    FBREF_DIR,
    "eng-premier_league_keeper_2018-2019_2019-2020_2020-2021_2021-2022_2022-2023_2023-2024_2024-2025.csv"
)

# input directory for the update files 

# Input file paths
SHOOTING_FILE_UPDATE = os.path.join(
    FBREF_DIR,
    "eng-premier_league_shooting_2024-2025.csv"
)
PASSING_FILE_UPDATE = os.path.join(
    FBREF_DIR,
    "eng-premier_league_passing_2024-2025.csv"
)
POSSESSION_FILE_UPDATE = os.path.join(
    FBREF_DIR,
    "eng-premier_league_possession_2024-2025.csv"
)
DEFENSE_FILE_UPDATE = os.path.join(
    FBREF_DIR,
    "eng-premier_league_defense_2024-2025.csv"
)
SCHEDULE_FILE_UPDATE = os.path.join(FBREF_DIR, "eng-premier_league_schedule_updated.csv")

SHOT_CREATION_FILE_UPDATE = os.path.join(
    FBREF_DIR,
    "eng-premier_league_goal_shot_creation_2024-2025.csv"
)
PASSING_TYPES_FILE_UPDATE = os.path.join(
    FBREF_DIR,
    "eng-premier_league_passing_types_2024-2025.csv"
)
MISC_FILE_UPDATE = os.path.join(
    FBREF_DIR,
    "eng-premier_league_misc_2024-2025.csv"
)

KEEPER_FILE_UPDATE = os.path.join(
    FBREF_DIR,
    "eng-premier_league_keeper_2024-2025.csv"
)











# Output file paths
SHOOTING_OUTPUT = os.path.join(PROCESSED_DIR, "PL_processes_shots.csv")
SHOT_CREATION_OUTPUT = os.path.join(PROCESSED_DIR, "PL_processes_shot_creation.csv")
PASSING_OUTPUT = os.path.join(PROCESSED_DIR, "PL_processes_passing.csv")
PASSING_TYPES_OUTPUT = os.path.join(PROCESSED_DIR, "PL_processes_passing_types.csv")
POSSESSION_OUTPUT = os.path.join(PROCESSED_DIR, "PL_processes_possession.csv")
DEFENSE_OUTPUT = os.path.join(PROCESSED_DIR, "PL_processes_defense.csv")
SCHEDULE_OUTPUT = os.path.join(PROCESSED_DIR, "PL_processed_schedule.csv")
MISC_OUTPUT = os.path.join(PROCESSED_DIR, "PL_processes_misc.csv")
KEEPER_OUTPUT = os.path.join(PROCESSED_DIR, "PL_processes_keeper.csv")

# Output file paths for the update files
SHOOTING_OUTPUT_UPDATE = os.path.join(PROCESSED_DIR, "PL_processes_shots_update.csv")
SHOT_CREATION_OUTPUT_UPDATE = os.path.join(PROCESSED_DIR, "PL_processes_shot_creation_update.csv")
PASSING_OUTPUT_UPDATE = os.path.join(PROCESSED_DIR, "PL_processes_passing_update.csv")
PASSING_TYPES_OUTPUT_UPDATE = os.path.join(PROCESSED_DIR, "PL_processes_passing_types_update.csv")
POSSESSION_OUTPUT_UPDATE = os.path.join(PROCESSED_DIR, "PL_processes_possession_update.csv")
DEFENSE_OUTPUT_UPDATE = os.path.join(PROCESSED_DIR, "PL_processes_defense_update.csv")
SCHEDULE_OUTPUT_UPDATE = os.path.join(PROCESSED_DIR, "PL_processed_schedule_update.csv")
MISC_OUTPUT_UPDATE = os.path.join(PROCESSED_DIR, "PL_processes_misc_update.csv")
KEEPER_OUTPUT_UPDATE = os.path.join(PROCESSED_DIR, "PL_processes_keeper_update.csv")


# Column renaming: these keys (as produced by transform) are renamed for consistency.
COLUMN_RENAME = {
    "Home_opponent_nan": "Away_Team",
    "Away_opponent_nan": "Home_Team",
    "Home_match_report_nan": "Match_report"
}


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
# Variant → Canonical team name
team_name_map = {
    # --- France ------------------------------------------------------------
    "Paris S-G": "Paris Saint‑Germain",
    "Paris SG": "Paris Saint‑Germain",
    "Lille": "Lille OSC",
    "Marseille": "Olympique de Marseille",
    "Nancy": "AS Nancy Lorraine",
    "Nantes": "FC Nantes",
    "Nice": "OGC Nice",
    "Saint-Étienne": "AS Saint‑Étienne",
    "St Etienne": "AS Saint‑Étienne",
    "Sochaux": "FC Sochaux‑Montbéliard",
    "Strasbourg": "RC Strasbourg Alsace",
    "Le Mans": "Le Mans FC",
    "Troyes": "ES Troyes AC",
    "Toulouse": "Toulouse FC",
    "Monaco": "AS Monaco",
    "Bordeaux": "FC Girondins de Bordeaux",
    "Lens": "RC Lens",
    "Ajaccio": "AC Ajaccio",
    "Gazélec Ajaccio": "Gazélec Ajaccio",
    "Ajaccio GFCO": "Gazélec Ajaccio",
    "Metz": "FC Metz",
    "Lyon": "Olympique Lyonnais",
    "Rennes": "Stade Rennais FC",
    "Auxerre": "AJ Auxerre",
    "Sedan": "CS Sedan Ardennes",
    "Lorient": "FC Lorient",
    "Valenciennes": "Valenciennes FC",
    "Caen": "SM Caen",
    "Le Havre": "Le Havre AC",
    "Grenoble": "Grenoble Foot 38",
    "Montpellier": "Montpellier HSC",
    "Boulogne": "US Boulogne",
    "Arles-Avignon": "AC Arles‑Avignon",
    "Arles": "AC Arles‑Avignon",
    "Brest": "Stade Brestois 29",
    "Dijon": "Dijon FCO",
    "Evian": "Evian Thonon Gaillard FC",
    "Evian Thonon Gaillard": "Evian Thonon Gaillard FC",
    "Reims": "Stade de Reims",
    "Bastia": "SC Bastia",
    "Guingamp": "En Avant Guingamp",
    "Angers": "Angers SCO",
    "Amiens": "Amiens SC",
    "Nîmes": "Nîmes Olympique",
    "Nimes": "Nîmes Olympique",
    "Clermont Foot": "Clermont Foot 63",
    "Clermont": "Clermont Foot 63",

    # --- Italy -------------------------------------------------------------
    "Fiorentina": "ACF Fiorentina",
    "Livorno": "AS Livorno Calcio",
    "Ascoli": "Ascoli Calcio 1898",
    "Inter": "Inter Milan",
    "Juventus": "Juventus FC",
    "Lazio": "SS Lazio",
    "Parma": "Parma Calcio 1913",
    "Reggina": "Reggina 1914",
    "Siena": "AC Siena",
    "Udinese": "Udinese Calcio",
    "Milan": "AC Milan",
    "Palermo": "Palermo FC",
    "Treviso": "Treviso FBC 1993",
    "Sampdoria": "UC Sampdoria",
    "Roma": "AS Roma",
    "Messina": "ACR Messina",
    "Cagliari": "Cagliari Calcio",
    "Empoli": "Empoli FC",
    "Chievo": "AC Chievo Verona",
    "Lecce": "US Lecce",
    "Torino": "Torino FC",
    "Atalanta": "Atalanta BC",
    "Catania": "Calcio Catania",
    "Genoa": "Genoa CFC",
    "Napoli": "SSC Napoli",
    "Bologna": "Bologna FC 1909",
    "Bari": "SSC Bari",
    "Cesena": "AC Cesena",
    "Brescia": "Brescia Calcio",
    "Novara": "Novara FC",
    "Pescara": "Delfino Pescara 1936",
    "Hellas Verona": "Hellas Verona FC",
    "Verona": "Hellas Verona FC",
    "Sassuolo": "US Sassuolo Calcio",
    "Frosinone": "Frosinone Calcio",
    "Carpi": "Carpi FC 1909",
    "Crotone": "FC Crotone",
    "Benevento": "Benevento Calcio",
    "SPAL": "SPAL",
    "Spal": "SPAL",
    "Spezia": "Spezia Calcio",
    "Salernitana": "US Salernitana 1919",
    "Venezia": "Venezia FC",
    "Monza": "AC Monza",
    "Cremonese": "US Cremonese",
    "Como": "Como 1907",

    # --- Spain -------------------------------------------------------------
    "Alavés": "Deportivo Alavés",
    "Alaves": "Deportivo Alavés",
    "Athletic Club": "Athletic Bilbao",
    "Ath Bilbao": "Athletic Bilbao",
    "Valencia": "Valencia CF",
    "Atlético Madrid": "Atlético Madrid",
    "Ath Madrid": "Atlético Madrid",
    "Celta Vigo": "Celta de Vigo",
    "Celta": "Celta de Vigo",
    "Cádiz": "Cádiz CF",
    "Cadiz": "Cádiz CF",
    "Espanyol": "RCD Espanyol",
    "Espanol": "RCD Espanyol",
    "Mallorca": "RCD Mallorca",
    "Osasuna": "CA Osasuna",
    "Sevilla": "Sevilla FC",
    "La Coruña": "Deportivo La Coruña",
    "La Coruna": "Deportivo La Coruña",
    "Real Madrid": "Real Madrid",
    "Betis": "Real Betis",
    "Barcelona": "FC Barcelona",
    "Getafe": "Getafe CF",
    "Málaga": "Málaga CF",
    "Malaga": "Málaga CF",
    "Racing Sant": "Racing Santander",
    "Santander": "Racing Santander",
    "Real Sociedad": "Real Sociedad",
    "Sociedad": "Real Sociedad",
    "Villarreal": "Villarreal CF",
    "Zaragoza": "Real Zaragoza",
    "Recreativo": "Recreativo de Huelva",
    "Gimnàstic": "Gimnàstic de Tarragona",
    "Gimnastic": "Gimnàstic de Tarragona",
    "Levante": "Levante UD",
    "Real Murcia": "Real Murcia",
    "Murcia": "Real Murcia",
    "Almería": "UD Almería",
    "Almeria": "UD Almería",
    "Valladolid": "Real Valladolid",
    "Numancia": "CD Numancia",
    "Sporting Gijón": "Sporting de Gijón",
    "Sp Gijon": "Sporting de Gijón",
    "Tenerife": "CD Tenerife",
    "Xerez": "Xerez CD",
    "Hércules": "Hércules CF",
    "Hercules": "Hércules CF",
    "Granada": "Granada CF",
    "Rayo Vallecano": "Rayo Vallecano",
    "Vallecano": "Rayo Vallecano",
    "Elche": "Elche CF",
    "Eibar": "SD Eibar",
    "Córdoba": "Córdoba CF",
    "Cordoba": "Córdoba CF",
    "Las Palmas": "UD Las Palmas",
    "Leganés": "CD Leganés",
    "Leganes": "CD Leganés",
    "Girona": "Girona FC",
    "Huesca": "SD Huesca",

    # --- Germany -----------------------------------------------------------
    "Bayern Munich": "FC Bayern Munich",
    "Hamburger SV": "Hamburger SV",
    "Hamburg": "Hamburger SV",
    "Hannover 96": "Hannover 96",
    "Hannover": "Hannover 96",
    "Köln": "1. FC Köln",
    "FC Koln": "1. FC Köln",
    "Koln": "1. FC Köln",
    "MSV Duisburg": "MSV Duisburg",
    "Duisburg": "MSV Duisburg",
    "Werder Bremen": "SV Werder Bremen",
    "Wolfsburg": "VfL Wolfsburg",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "Eint Frankfurt": "Eintracht Frankfurt",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "Schalke 04": "FC Schalke 04",
    "Nürnberg": "1. FC Nürnberg",
    "Nurnberg": "1. FC Nürnberg",
    "Leverkusen": "Bayer 04 Leverkusen",
    "Kaiserslautern": "1. FC Kaiserslautern",
    "Dortmund": "Borussia Dortmund",
    "M'gladbach": "Borussia Mönchengladbach",
    "Gladbach": "Borussia Mönchengladbach",
    "Arminia": "Arminia Bielefeld",
    "Bielefeld": "Arminia Bielefeld",
    "Hertha BSC": "Hertha BSC",
    "Hertha": "Hertha BSC",
    "Mainz 05": "1. FSV Mainz 05",
    "Mainz": "1. FSV Mainz 05",
    "Stuttgart": "VfB Stuttgart",
    "Energie Cottbus": "Energie Cottbus",
    "Cottbus": "Energie Cottbus",
    "AA Aachen": "Alemannia Aachen",
    "Aachen": "Alemannia Aachen",
    "Bochum": "VfL Bochum",
    "Karlsruher": "Karlsruher SC",
    "Karlsruhe": "Karlsruher SC",
    "Hansa Rostock": "Hansa Rostock",
    "Hoffenheim": "TSG 1899 Hoffenheim",
    "Freiburg": "SC Freiburg",
    "St. Pauli": "FC St. Pauli",
    "St Pauli": "FC St. Pauli",
    "Augsburg": "FC Augsburg",
    "Greuther Fürth": "Greuther Fürth",
    "Greuther Furth": "Greuther Fürth",
    "Düsseldorf": "Fortuna Düsseldorf",
    "Dusseldorf": "Fortuna Düsseldorf",
    "Fortuna Dusseldorf": "Fortuna Düsseldorf",
    "BTSV": "Eintracht Braunschweig",
    "Braunschweig": "Eintracht Braunschweig",
    "Paderborn 07": "SC Paderborn 07",
    "Paderborn": "SC Paderborn 07",
    "Darmstadt 98": "Darmstadt 98",
    "Darmstadt": "Darmstadt 98",
    "Ingolstadt 04": "FC Ingolstadt 04",
    "Ingolstadt": "FC Ingolstadt 04",
    "RB Leipzig": "RB Leipzig",
    "Holstein Kiel": "Holstein Kiel",
    "Union Berlin": "Union Berlin",
    "Heidenheim": "1. FC Heidenheim",
    "Elversberg": "SV Elversberg",

    # --- England -----------------------------------------------------------
    "Aston Villa": "Aston Villa",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Manchester City": "Manchester City",
    "Man City": "Manchester City",
    "Middlesbrough": "Middlesbrough",
    "Portsmouth": "Portsmouth",
    "Sunderland": "Sunderland",
    "West Ham": "West Ham United",
    "Arsenal": "Arsenal",
    "Wigan Athletic": "Wigan Athletic",
    "Wigan": "Wigan Athletic",
    "West Brom": "West Bromwich Albion",
    "West Bromwich Albion": "West Bromwich Albion",
    "Tottenham": "Tottenham Hotspur",
    "Tottenham Hotspur": "Tottenham Hotspur",
    "Newcastle Utd": "Newcastle United",
    "Newcastle": "Newcastle United",
    "Manchester Utd": "Manchester United",
    "Man United": "Manchester United",
    "Manchester United": "Manchester United",
    "Charlton Ath": "Charlton Athletic",
    "Charlton": "Charlton Athletic",
    "Blackburn": "Blackburn Rovers",
    "Birmingham City": "Birmingham City",
    "Birmingham": "Birmingham City",
    "Liverpool": "Liverpool",
    "Bolton": "Bolton Wanderers",
    "Bolton Wanderers": "Bolton Wanderers",
    "Chelsea": "Chelsea",
    "Sheffield Utd": "Sheffield United",
    "Sheffield United": "Sheffield United",
    "Reading": "Reading",
    "Watford": "Watford",
    "Derby County": "Derby County",
    "Derby": "Derby County",
    "Hull City": "Hull City",
    "Hull": "Hull City",
    "Stoke City": "Stoke City",
    "Stoke": "Stoke City",
    "Wolves": "Wolverhampton Wanderers",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
    "Burnley": "Burnley",
    "Blackpool": "Blackpool",
    "QPR": "Queens Park Rangers",
    "Queens Park Rangers": "Queens Park Rangers",
    "Swansea City": "Swansea City",
    "Swansea": "Swansea City",
    "Norwich City": "Norwich City",
    "Norwich": "Norwich City",
    "Southampton": "Southampton",
    "Crystal Palace": "Crystal Palace",
    "Cardiff City": "Cardiff City",
    "Cardiff": "Cardiff City",
    "Leicester City": "Leicester City",
    "Leicester": "Leicester City",
    "Bournemouth": "Bournemouth",
    "Brighton": "Brighton & Hove Albion",
    "Brighton & Hove Albion": "Brighton & Hove Albion",
    "Huddersfield": "Huddersfield Town",
    "Huddersfield Town": "Huddersfield Town",
    "Leeds United": "Leeds United",
    "Leeds": "Leeds United",
    "Brentford": "Brentford",
    "Nottingham Forest": "Nottingham Forest",
    "Nott'ham Forest": "Nottingham Forest",
    "Nott'm Forest": "Nottingham Forest",
    "Luton Town": "Luton Town",
    "Ipswich Town": "Ipswich Town",
    "Ipswich": "Ipswich Town",

    # --- Misc --------------------------------------------------------------
    "nan": None
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

pairs_new = [
    # 1/3
    ["Home_1/3", "Away_1/3", "OneThird"],
    
    # Aerial Duels
    ["Home_Aerial Duels_Won", "Away_Aerial Duels_Won", "AerialDuels_Won"],
    ["Home_Aerial Duels.1_Lost", "Away_Aerial Duels.1_Lost", "AerialDuels_1_Lost"],
    ["Home_Aerial Duels.2_Won%", "Away_Aerial Duels.2_Won%", "AerialDuels_2_WonPct"],
    
    # Assists, Key Passes, etc.
    ["Home_Ast", "Away_Ast", "Ast"],
    ["Home_KP", "Away_KP", "KP"],
    ["Home_xA", "Away_xA", "xA"],
    ["Home_xAG", "Away_xAG", "xAG"],
    
    # Att
    ["Home_Att", "Away_Att", "Att"],
    
    # Blocks
    ["Home_Blocks_Blocks", "Away_Blocks_Blocks", "Blocks_Blocks"],
    ["Home_Blocks.1_Sh", "Away_Blocks.1_Sh", "Blocks_1_Sh"],
    ["Home_Blocks.2_Pass", "Away_Blocks.2_Pass", "Blocks_2_Pass"],
    
    # Carries
    ["Home_Carries_Carries", "Away_Carries_Carries", "Carries_Carries"],
    ["Home_Carries.1_TotDist", "Away_Carries.1_TotDist", "Carries_1_TotDist"],
    ["Home_Carries.2_PrgDist", "Away_Carries.2_PrgDist", "Carries_2_PrgDist"],
    ["Home_Carries.3_PrgC", "Away_Carries.3_PrgC", "Carries_3_PrgC"],
    ["Home_Carries.4_1/3", "Away_Carries.4_1/3", "Carries_4_OneThird"],
    ["Home_Carries.5_CPA", "Away_Carries.5_CPA", "Carries_5_CPA"],
    ["Home_Carries.6_Mis", "Away_Carries.6_Mis", "Carries_6_Mis"],
    ["Home_Carries.7_Dis", "Away_Carries.7_Dis", "Carries_7_Dis"],

    # Challenges
    ["Home_Challenges_Tkl", "Away_Challenges_Tkl", "Challenges_Tkl"],
    ["Home_Challenges.1_Att", "Away_Challenges.1_Att", "Challenges_1_Att"],
    ["Home_Challenges.2_Tkl%", "Away_Challenges.2_Tkl%", "Challenges_2_TklPct"],
    ["Home_Challenges.3_Lost", "Away_Challenges.3_Lost", "Challenges_3_Lost"],

    # Corner Kicks
    ["Home_Corner Kicks_In", "Away_Corner Kicks_In", "CornerKicks_In"],
    ["Home_Corner Kicks.1_Out", "Away_Corner Kicks.1_Out", "CornerKicks_1_Out"],
    ["Home_Corner Kicks.2_Str", "Away_Corner Kicks.2_Str", "CornerKicks_2_Str"],
    
    # Crosses
    ["Home_Crosses_Opp", "Away_Crosses_Opp", "Crosses_Opp"],
    ["Home_Crosses.1_Stp", "Away_Crosses.1_Stp", "Crosses_1_Stp"],
    ["Home_Crosses.2_Stp%", "Away_Crosses.2_Stp%", "Crosses_2_StpPct"],

    # CrsPA, ProgPasses
    ["Home_CrsPA", "Away_CrsPA", "CrsPA"],
    ["Home_PrgP", "Away_PrgP", "PrgP"],

    # Goal Kicks
    ["Home_Goal Kicks_Att", "Away_Goal Kicks_Att", "GoalKicks_Att"],
    ["Home_Goal Kicks.1_Launch%", "Away_Goal Kicks.1_Launch%", "GoalKicks_1_LaunchPct"],
    ["Home_Goal Kicks.2_AvgLen", "Away_Goal Kicks.2_AvgLen", "GoalKicks_2_AvgLen"],

    # Int, Tkl+Int, Clearances
    ["Home_Int", "Away_Int", "Int"],
    ["Home_Tkl+Int", "Away_Tkl+Int", "TklPlusInt"],
    ["Home_Clr", "Away_Clr", "Clr"],

    # Launched
    ["Home_Launched_Cmp", "Away_Launched_Cmp", "Launched_Cmp"],
    ["Home_Launched.1_Att", "Away_Launched.1_Att", "Launched_1_Att"],
    ["Home_Launched.2_Cmp%", "Away_Launched.2_Cmp%", "Launched_2_CmpPct"],
    
    # Long
    ["Home_Long_Cmp", "Away_Long_Cmp", "Long_Cmp"],
    ["Home_Long.1_Att", "Away_Long.1_Att", "Long_1_Att"],
    ["Home_Long.2_Cmp%", "Away_Long.2_Cmp%", "Long_2_CmpPct"],
    
    # Medium
    ["Home_Medium_Cmp", "Away_Medium_Cmp", "Medium_Cmp"],
    ["Home_Medium.1_Att", "Away_Medium.1_Att", "Medium_1_Att"],
    ["Home_Medium.2_Cmp%", "Away_Medium.2_Cmp%", "Medium_2_CmpPct"],

    # Outcomes
    ["Home_Outcomes_Cmp", "Away_Outcomes_Cmp", "Outcomes_Cmp"],
    ["Home_Outcomes.1_Off", "Away_Outcomes.1_Off", "Outcomes_1_Off"],
    ["Home_Outcomes.2_Blocks", "Away_Outcomes.2_Blocks", "Outcomes_2_Blocks"],

    # Pass Types
    ["Home_Pass Types_Live", "Away_Pass Types_Live", "PassTypes_Live"],
    ["Home_Pass Types.1_Dead", "Away_Pass Types.1_Dead", "PassTypes_1_Dead"],
    ["Home_Pass Types.2_FK", "Away_Pass Types.2_FK", "PassTypes_2_FK"],
    ["Home_Pass Types.3_TB", "Away_Pass Types.3_TB", "PassTypes_3_TB"],
    ["Home_Pass Types.4_Sw", "Away_Pass Types.4_Sw", "PassTypes_4_Sw"],
    ["Home_Pass Types.5_Crs", "Away_Pass Types.5_Crs", "PassTypes_5_Crs"],
    ["Home_Pass Types.6_TI", "Away_Pass Types.6_TI", "PassTypes_6_TI"],
    ["Home_Pass Types.7_CK", "Away_Pass Types.7_CK", "PassTypes_7_CK"],

    # Passes (GK) / Thr / Launch%
    ["Home_Passes_Att (GK)", "Away_Passes_Att (GK)", "Passes_AttGK"],
    ["Home_Passes.1_Thr", "Away_Passes.1_Thr", "Passes_1_Thr"],
    ["Home_Passes.2_Launch%", "Away_Passes.2_Launch%", "Passes_2_LaunchPct"],
    ["Home_Passes.3_AvgLen", "Away_Passes.3_AvgLen", "Passes_3_AvgLen"],

    # Penalty Kicks
    ["Home_Penalty Kicks_PKatt", "Away_Penalty Kicks_PKatt", "PenaltyKicks_PKatt"],
    ["Home_Penalty Kicks.1_PKA", "Away_Penalty Kicks.1_PKA", "PenaltyKicks_1_PKA"],
    ["Home_Penalty Kicks.2_PKsv", "Away_Penalty Kicks.2_PKsv", "PenaltyKicks_2_PKsv"],
    ["Home_Penalty Kicks.3_PKm", "Away_Penalty Kicks.3_PKm", "PenaltyKicks_3_PKm"],

    # Performance
    ["Home_Performance_CrdY", "Away_Performance_CrdY", "Performance_CrdY"],
    ["Home_Performance.1_CrdR", "Away_Performance.1_CrdR", "Performance_1_CrdR"],
    ["Home_Performance.2_2CrdY", "Away_Performance.2_2CrdY", "Performance_2_2CrdY"],
    ["Home_Performance.3_Fls", "Away_Performance.3_Fls", "Performance_3_Fls"],
    ["Home_Performance.4_Fld", "Away_Performance.4_Fld", "Performance_4_Fld"],
    ["Home_Performance.5_Off", "Away_Performance.5_Off", "Performance_5_Off"],
    ["Home_Performance.6_Crs", "Away_Performance.6_Crs", "Performance_6_Crs"],
    ["Home_Performance.7_Int", "Away_Performance.7_Int", "Performance_7_Int"],
    ["Home_Performance.8_TklW", "Away_Performance.8_TklW", "Performance_8_TklW"],
    ["Home_Performance.9_PKwon", "Away_Performance.9_PKwon", "Performance_9_PKwon"],
    ["Home_Performance.10_PKcon", "Away_Performance.10_PKcon", "Performance_10_PKcon"],
    ["Home_Performance.11_OG", "Away_Performance.11_OG", "Performance_11_OG"],
    ["Home_Performance.12_Recov", "Away_Performance.12_Recov", "Performance_12_Recov"],

    # Possession
    ["Home_Poss", "Away_Poss", "Poss"],

    # PPA
    ["Home_PPA", "Away_PPA", "PPA"],

    # SCA Types
    ["Home_SCA Types_SCA", "Away_SCA Types_SCA", "SCATypes_SCA"],
    ["Home_SCA Types.1_PassLive", "Away_SCA Types.1_PassLive", "SCATypes_1_PassLive"],
    ["Home_SCA Types.2_PassDead", "Away_SCA Types.2_PassDead", "SCATypes_2_PassDead"],
    ["Home_SCA Types.3_TO", "Away_SCA Types.3_TO", "SCATypes_3_TO"],
    ["Home_SCA Types.4_Sh", "Away_SCA Types.4_Sh", "SCATypes_4_Sh"],
    ["Home_SCA Types.5_Fld", "Away_SCA Types.5_Fld", "SCATypes_5_Fld"],
    ["Home_SCA Types.6_Def", "Away_SCA Types.6_Def", "SCATypes_6_Def"],

    # Short
    ["Home_Short_Cmp", "Away_Short_Cmp", "Short_Cmp"],
    ["Home_Short.1_Att", "Away_Short.1_Att", "Short_1_Att"],
    ["Home_Short.2_Cmp%", "Away_Short.2_Cmp%", "Short_2_CmpPct"],

    # Standard
    ["Home_Standard_Gls", "Away_Standard_Gls", "Standard_Gls"],
    ["Home_Standard.1_Sh", "Away_Standard.1_Sh", "Standard_1_Sh"],
    ["Home_Standard.2_SoT", "Away_Standard.2_SoT", "Standard_2_SoT"],
    ["Home_Standard.3_SoT%", "Away_Standard.3_SoT%", "Standard_3_SoTPct"],
    ["Home_Standard.4_G/Sh", "Away_Standard.4_G/Sh", "Standard_4_GPerSh"],
    ["Home_Standard.5_G/SoT", "Away_Standard.5_G/SoT", "Standard_5_GPerSoT"],
    ["Home_Standard.6_Dist", "Away_Standard.6_Dist", "Standard_6_Dist"],
    ["Home_Standard.7_FK", "Away_Standard.7_FK", "Standard_7_FK"],
    ["Home_Standard.8_PK", "Away_Standard.8_PK", "Standard_8_PK"],
    ["Home_Standard.9_PKatt", "Away_Standard.9_PKatt", "Standard_9_PKatt"],

    # Sweeper
    ["Home_Sweeper_#OPA", "Away_Sweeper_#OPA", "Sweeper_NumOPA"],
    ["Home_Sweeper.1_AvgDist", "Away_Sweeper.1_AvgDist", "Sweeper_1_AvgDist"],

    # Tackles
    ["Home_Tackles_Tkl", "Away_Tackles_Tkl", "Tackles_Tkl"],
    ["Home_Tackles.1_TklW", "Away_Tackles.1_TklW", "Tackles_1_TklW"],
    ["Home_Tackles.2_Def 3rd", "Away_Tackles.2_Def 3rd", "Tackles_2_Def3rd"],
    ["Home_Tackles.3_Mid 3rd", "Away_Tackles.3_Mid 3rd", "Tackles_3_Mid3rd"],
    ["Home_Tackles.4_Att 3rd", "Away_Tackles.4_Att 3rd", "Tackles_4_Att3rd"],

    # Take-Ons
    ["Home_Take-Ons_Att", "Away_Take-Ons_Att", "TakeOns_Att"],
    ["Home_Take-Ons.1_Succ", "Away_Take-Ons.1_Succ", "TakeOns_1_Succ"],
    ["Home_Take-Ons.2_Succ%", "Away_Take-Ons.2_Succ%", "TakeOns_2_SuccPct"],
    ["Home_Take-Ons.3_Tkld", "Away_Take-Ons.3_Tkld", "TakeOns_3_Tkld"],
    ["Home_Take-Ons.4_Tkld%", "Away_Take-Ons.4_Tkld%", "TakeOns_4_TkldPct"],

    # Total
    ["Home_Total_Cmp", "Away_Total_Cmp", "Total_Cmp"],
    ["Home_Total.1_Att", "Away_Total.1_Att", "Total_1_Att"],
    ["Home_Total.2_Cmp%", "Away_Total.2_Cmp%", "Total_2_CmpPct"],
    ["Home_Total.3_TotDist", "Away_Total.3_TotDist", "Total_3_TotDist"],
    ["Home_Total.4_PrgDist", "Away_Total.4_PrgDist", "Total_4_PrgDist"],

    # Touches
    ["Home_Touches_Touches", "Away_Touches_Touches", "Touches_Touches"],
    ["Home_Touches.1_Def Pen", "Away_Touches.1_Def Pen", "Touches_DefPen"],
    ["Home_Touches.2_Def 3rd", "Away_Touches.2_Def 3rd", "Touches_Def3rd"],
    ["Home_Touches.3_Mid 3rd", "Away_Touches.3_Mid 3rd", "Touches_Mid3rd"],
    ["Home_Touches.4_Att 3rd", "Away_Touches.4_Att 3rd", "Touches_Att3rd"],
    ["Home_Touches.5_Att Pen", "Away_Touches.5_Att Pen", "Touches_AttPen"],
    ["Home_Touches.6_Live", "Away_Touches.6_Live", "Touches_Live"],

    # Receiving
    ["Home_Receiving_Rec", "Away_Receiving_Rec", "Receiving_Rec"],
    ["Home_Receiving.1_PrgR", "Away_Receiving.1_PrgR", "Receiving_PrgR"],

    # xG (the "home_xg"/"away_xg" columns from your original core data)
    ["home_xg", "away_xg", "xg"],

    # Expected_xG (and sub-metrics)
    ["Home_Expected_xG", "Away_Expected_xG", "Expected_xG"],
    ["Home_Expected.1_npxG", "Away_Expected.1_npxG", "Expected_1_npxG"],
    ["Home_Expected.2_npxG/Sh", "Away_Expected.2_npxG/Sh", "Expected_2_npxGPerSh"],
    ["Home_Expected.3_G-xG", "Away_Expected.3_G-xG", "Expected_3_GminusxG"],
    ["Home_Expected.4_np:G-xG", "Away_Expected.4_np:G-xG", "Expected_4_npGminusxG"],
]
