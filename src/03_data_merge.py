import pandas as pd

def merge_datasets():
    # Load each CSV into a DataFrame
    schedule_df   = pd.read_csv("/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processed_schedule.csv")
    defense_df    = pd.read_csv("/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_defense.csv")
    passing_df    = pd.read_csv("/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_passing.csv")
    possession_df = pd.read_csv("/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_possession.csv")
    shots_df      = pd.read_csv("/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_processes_shots.csv")
    
    # Merge datasets based on the "Match_report" column.
    # Using an outer join ensures that if one file is missing a match, you'll still keep the row.
    merged_df = schedule_df.merge(defense_df, on="Match_report", how="left")
    merged_df = merged_df.merge(passing_df, on="Match_report", how="left")
    merged_df = merged_df.merge(possession_df, on="Match_report", how="left")
    merged_df = merged_df.merge(shots_df, on="Match_report", how="left")
    
    return merged_df

if __name__ == "__main__":
    merged_data = merge_datasets()
    
    # Uncomment below to save the merged dataframe if needed.
    merged_data.to_csv("/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/PL_merged_dataset.csv", index=False)
