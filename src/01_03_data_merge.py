import pandas as pd
from project_specifics import (
    SCHEDULE_OUTPUT,
    DEFENSE_OUTPUT,
    PASSING_OUTPUT,
    POSSESSION_OUTPUT,
    SHOOTING_OUTPUT,
    SHOT_CREATION_OUTPUT,
    PASSING_TYPES_OUTPUT,
    MISC_OUTPUT,
    KEEPER_OUTPUT,
    MERGED_OUTPUT
)


def merge_datasets():
    # Read the schedule dataframe
    schedule_df = pd.read_csv(SCHEDULE_OUTPUT)
    # Ensure the merge column is named 'match_report' (uncomment if needed)
    # schedule_df.rename(columns={'Match_report': 'match_report'}, inplace=True)
    print("schedule_df shape:", schedule_df.shape)

    # Function to load and check stats dataframes
    def load_and_check_stats(path, df_name):
        df = pd.read_csv(path).set_index("match_report")
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            print(f"Warning: {df_name} has {duplicates} duplicate match_report entries. Aggregating or dropping duplicates.")
            df = df[~df.index.duplicated(keep='first')]
        print(f"{df_name} shape after processing:", df.shape)
        return df

    # Load each stats dataframe, check for duplicates, and set index
    defense_df = load_and_check_stats(DEFENSE_OUTPUT, "defense_df")
    passing_df = load_and_check_stats(PASSING_OUTPUT, "passing_df")
    possession_df = load_and_check_stats(POSSESSION_OUTPUT, "possession_df")
    shots_df = load_and_check_stats(SHOOTING_OUTPUT, "shots_df")
    shot_creation_df = load_and_check_stats(SHOT_CREATION_OUTPUT, "shot_creation_df")
    passing_types_df = load_and_check_stats(PASSING_TYPES_OUTPUT, "passing_types_df")
    misc_df = load_and_check_stats(MISC_OUTPUT, "misc_df")
    keeper_df = load_and_check_stats(KEEPER_OUTPUT, "keeper_df")

    # Merge all stats dataframes onto the schedule dataframe
    merged_df = schedule_df.copy()
    stats_dfs = [
        (defense_df, "defense"),
        (passing_df, "passing"),
        (possession_df, "possession"),
        (shots_df, "shooting"),
        (shot_creation_df, "shot_creation"),
        (passing_types_df, "passing_types"),
        (misc_df, "misc"),
        (keeper_df, "keeper")
    ]

    for df, suffix in stats_dfs:
        # Avoid column name conflicts by using suffixes
        merged_df = merged_df.merge(
            df,
            left_on='match_report',
            right_index=True,
            how='left',
            suffixes=('', f'_{suffix}')
        )

    # Save the merged dataframe
    merged_df.to_csv(MERGED_OUTPUT, index=False)
    print(f"Merged dataset saved to: {MERGED_OUTPUT}")
    print("merged_df shape:", merged_df.shape)

if __name__ == "__main__":
    merge_datasets()