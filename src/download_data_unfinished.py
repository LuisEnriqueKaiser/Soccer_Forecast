#!/usr/bin/env python3
import soccerdata as sd
import pandas as pd
import os

# Define the leagues, seasons, and advanced stats of interest
leagues = [
    "ENG-Premier League",  # Premier League
    "ESP-La Liga",         # La Liga
    "GER-Bundesliga"       # Bundesliga
]
seasons = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]
advanced_stats = [
    'keeper', 'shooting', 'passing', 'passing_types',
    'goal_shot_creation', 'defense', 'possession', 'misc'
]

# Define the file path where CSVs will be saved.
filepath = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/fbref/"

def file_already_saved(filepath, league, stat, seasons):
    """
    Build a filename from league, stat, and seasons.
    Return a tuple (exists, full_path), where exists is True if the file exists.
    """
    # Build a filename; note that the seasons list will be represented as a string.
    filename = f"{league}_{stat}_{seasons}.csv".replace(" ", "_").lower()
    full_path = os.path.join(filepath, filename)
    return os.path.exists(full_path), full_path

# ---------------------------
# Save Schedule CSVs
# ---------------------------
for league in ["ENG-Premier League", "ESP-La Liga"]:
    fbref = sd.FBref(leagues=league, seasons=seasons)
    df_schedule = fbref.read_schedule()
    # Rename match_report to match_id if necessary
    if "match_report" in df_schedule.columns and "match_id" not in df_schedule.columns:
        df_schedule.rename(columns={"match_report": "match_id"}, inplace=True)
    schedule_filename = f"{league}_schedule_{seasons}.csv".replace(" ", "_").lower()
    schedule_path = os.path.join(filepath, schedule_filename)
    if os.path.exists(schedule_path):
        print(f"[INFO] Schedule file {schedule_path} already exists; skipping.")
    else:
        df_schedule.to_csv(schedule_path, index=False)
        print(f"[INFO] Saved schedule for {league} to {schedule_path}")

# ---------------------------
# Read Advanced Stats into a List
# ---------------------------
advanced_stats_data = []
for league in leagues:
    fbref = sd.FBref(leagues=league, seasons=seasons)
    for stat in advanced_stats:
        print(f"[INFO] Reading {stat} stats in {league} in {seasons}")
        try:
            df_stat = fbref.read_team_match_stats(stat_type=stat)
            advanced_stats_data.append((league, stat, df_stat))
        except Exception as e:
            print(f"[ERROR] Error reading {stat} stats in {league}: {e}")

# ---------------------------
# Save Advanced Stats CSVs (with file existence check)
# ---------------------------
for league, stat, df in advanced_stats_data:
    # Build filename for this league and stat
    exists, full_path = file_already_saved(filepath, league, stat, seasons)
    if exists:
        print(f"[INFO] File {full_path} already exists; skipping saving for {league} {stat}.")
    else:
        df.to_csv(full_path, index=False)
        print(f"[INFO] Saved {full_path}")
