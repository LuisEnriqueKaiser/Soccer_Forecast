import os
import soccerdata as sd
import pandas as pd

def save_schedule_for_league(league, seasons, filepath, current_season_active=False):
    """
    Reads and saves the schedule for a given league and list of seasons.
    If the file already exists and the current season is not active, it skips the download.
    
    Parameters:
        league (str): The league name.
        seasons (list of str): List of season strings.
        filepath (str): Directory to save the CSV file.
        current_season_active (bool): If True, update file even if it exists; if False, skip download.
    """
    filename = f"{league}_schedule.csv".replace(" ", "_").lower()
    full_path = os.path.join(filepath, filename)
    
    # Check if file exists and if current season is not active
    if os.path.exists(full_path) and not current_season_active:
        print(f"File {full_path} already exists and is not current season. Skipping download.")
        return
    
    fbref = sd.FBref(leagues=league, seasons=seasons)
    df_schedule = fbref.read_schedule()
    df_schedule.to_csv(full_path, index=False)
    print(f"Saved schedule: {full_path}")

def save_advanced_stats(leagues, seasons, advanced_stats, filepath, current_season_active=False):
    """
    Reads and saves advanced team match stats for each league and stat type.
    If the file for a given league/stat combination already exists and the current season is not active,
    the download is skipped.
    
    Parameters:
        leagues (list of str): List of league names.
        seasons (list of str): List of season strings.
        advanced_stats (list of str): List of advanced stat types.
        filepath (str): Directory to save the CSV files.
        current_season_active (bool): If True, update file even if it exists; if False, skip download.
    """
    for league in leagues:
        fbref = sd.FBref(leagues=league, seasons=seasons)
        for stat in advanced_stats:
            seasons_str = "_".join(seasons)
            filename = f"{league}_{stat}_{seasons_str}.csv".replace(" ", "_").lower()
            full_path = os.path.join(filepath, filename)
            
            # Check if file exists and if current season is not active
            if os.path.exists(full_path) and not current_season_active:
                print(f"File {full_path} already exists and is not current season. Skipping download.")
                continue
            
            print(f"Reading {stat} stats in {league} for seasons {seasons}")
            try:
                df_stats = fbref.read_team_match_stats(stat_type=stat)
                df_stats.to_csv(full_path, index=False)
                print(f"Saved advanced stats: {full_path}")
            except Exception as e:
                print(f"Error reading {stat} stats in {league}: {e}")

def main():
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
    filepath = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/fbref/"
    
    # Set this flag to True if you want to update (download) current season files even if they already exist.
    # For historical (non-current) data, leave it as False so that files are skipped if already downloaded.
    current_season_active = False  # Change to True if the current season is in progress.
    
    # Save schedule CSVs for specific leagues (here ENG and ESP)
    for league in leagues:
        save_schedule_for_league(league, seasons, filepath, current_season_active)
    
    # Save advanced stats for all leagues in the list
    save_advanced_stats(leagues, seasons, advanced_stats, filepath, current_season_active)

if __name__ == "__main__":
    main()
