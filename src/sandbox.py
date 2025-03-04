import soccerdata as sd
import pandas as pd

def build_dataset_for_leagues(start_year=2014, end_year=2023):
    """
    Builds datasets for Bundesliga, Premier League, and La Liga for the given time range
    (inclusive), merging schedule info with available match stats from FBref.

    One CSV file per league:
        - bundesliga_YY_YY.csv
        - premier_league_YY_YY.csv
        - la_liga_YY_YY.csv
    """

    # Map our chosen leagues to their names in FBref
    leagues = {
        "bundesliga": "GER-Bundesliga",       # German Bundesliga
        "premier_league": "ENG-Premier League",   # English Premier League
        "la_liga": "ESP-La Liga",         # Spanish La Liga
    }

    # Create a list of seasons in the format 'YY'
    seasons = [f'{str(year)[-2:]}' for year in range(start_year, end_year + 1)]

    for league_name, fbref_league in leagues.items():
        # Initialize the FBref scraper for the specific league and seasons
        fbref = sd.FBref(leagues=[fbref_league], seasons=seasons)

        # Read the match schedules and results
        df_matches = fbref.read_team_match_stats(stat_type='schedule')

        # Save to CSV
        csv_filename = f"{league_name}_{start_year % 100}_{end_year % 100}.csv"
        df_matches.to_csv(csv_filename, index=False)
        print(f"Saved {csv_filename}")

if __name__ == "__main__":
    build_dataset_for_leagues(2014, 2023)
