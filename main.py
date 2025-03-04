import os
import yaml
import glob
import pandas as pd

from src.data_ingestion import download_and_merge_fbref, clear_cache, get_upcoming_odds
from src.data_cleaning import clean_match_data
from src.feature_engineering import build_features
from src.model import train_model
from src.evaluate import calculate_value_bets, backtest

def main():
    # Load configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    data_dir = config["data_dir"]
    fbref_leagues = config.get("fbref_leagues", [])
    fbref_seasons = config.get("fbref_seasons", [])
    
    if not fbref_leagues or not fbref_seasons:
        print("No FBref leagues or seasons specified in config. Nothing to download.")
        return
    
    # For each league, download and merge FBref data from all specified seasons.
    for league in fbref_leagues:
        print(f"--- Processing {league} ---")
        df_merged = download_and_merge_fbref(league, fbref_seasons, data_dir)
        if df_merged.empty:
            print(f"No data merged for {league}.")
            continue
        
        # Save the merged data per league (already done in data_ingestion, but we can print info)
        processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        merge_game_path = os.path.join(processed_dir, f"{league.replace(' ', '_')}_merged_fbref.csv")
        df_merged.to_csv(merge_game_path, index=False)
        print(f"Merged FBref data for {league} saved to {merge_game_path} with {df_merged.shape[0]} rows and {df_merged.shape[1]} columns.")
        
        # Clean the merged data
        df_clean = clean_match_data(df_merged)
        # Build features
        df_features = build_features(df_clean)
        # Train model (only for historical matches that have result info)
        model, label_encoder = train_model(df_features)
        # Evaluate
        df_values = calculate_value_bets(df_features, model, label_encoder)
        df_backtest = backtest(df_values)
        backtest_file = os.path.join(processed_dir, f"{league.replace(' ', '_')}_backtest_results.csv")
        df_backtest.to_csv(backtest_file, index=False)
        print(f"Historical backtest results saved to {backtest_file}")
        
        # OPTIONAL: Forecasting upcoming matches using the same FBref data (if applicable)
        # For this simplified version, forecasting is not included.
    
    # PART 9: Create a download log and merge all CSV datasets from data_dir
    log_path = os.path.join(processed_dir, "download_log.csv")
    merge_path = os.path.join(processed_dir, "all_merged_data.csv")
    create_download_log_and_merge(data_dir, log_path, merge_path)
    
def create_download_log_and_merge(data_dir: str, log_path: str, merge_path: str):
    csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    log_entries = []
    all_dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            num_rows = df.shape[0]
            log_entries.append({"file": f, "num_rows": num_rows, "status": "success"})
            all_dfs.append(df)
        except Exception as e:
            log_entries.append({"file": f, "num_rows": 0, "status": f"failed: {e}"})
    log_df = pd.DataFrame(log_entries)
    log_df.to_csv(log_path, index=False)
    print(f"Download log saved to {log_path}")
    if all_dfs:
        merged_df = pd.concat(all_dfs, sort=True, ignore_index=True)
        merged_df.to_csv(merge_path, index=False)
        print(f"Merged data from all CSVs saved to {merge_path}")
    else:
        print("No CSV files found to merge.")

if __name__ == "__main__":
    main()
