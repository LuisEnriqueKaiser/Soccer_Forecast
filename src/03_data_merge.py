import pandas as pd
from project_specifics import (
    SCHEDULE_OUTPUT,
    DEFENSE_OUTPUT,
    PASSING_OUTPUT,
    POSSESSION_OUTPUT,
    SHOOTING_OUTPUT,
    MERGED_OUTPUT
)

def unify_to_one_line_per_match(df_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Example of collapsing multiple lines per match into exactly one line.
    Adjust as needed:
      - If you see that each match_report has 2 lines (home/away), or partial lines
        for sub-stats, group them by match_report and sum/mean, or pivot them.
      - Return the collapsed DataFrame so that each match_report is unique.
    """
    # Example: if you can just group-and-sum numeric columns
    #    (If your real data needs pivoting or more logic, do so here)
    group_cols = [c for c in df_stats.columns if c != "Match_report"]
    # Summarize or pivot. As a trivial example, sum numeric columns:
    df_collapsed = df_stats.groupby("Match_report", as_index=False)[group_cols].sum()
    return df_collapsed

def merge_datasets():
    # 1) Read schedule.  Rename match_report -> Match_report if needed
    schedule_df = pd.read_csv(SCHEDULE_OUTPUT)
    schedule_df.rename(columns={"match_report": "Match_report"}, inplace=True)

    # 2) Read & unify each stats DataFrame so it has exactly 1 row per match
    defense_df    = unify_to_one_line_per_match(pd.read_csv(DEFENSE_OUTPUT))
    passing_df    = unify_to_one_line_per_match(pd.read_csv(PASSING_OUTPUT))
    possession_df = unify_to_one_line_per_match(pd.read_csv(POSSESSION_OUTPUT))
    shots_df      = unify_to_one_line_per_match(pd.read_csv(SHOOTING_OUTPUT))

    # 3) Merge everything onto the schedule by "Match_report"
    #    use left or outer join depending on which matches you want to keep
    merged_df = schedule_df.merge(defense_df, on="Match_report", how="left")
    merged_df = merged_df.merge(passing_df, on="Match_report", how="left")
    merged_df = merged_df.merge(possession_df, on="Match_report", how="left")
    merged_df = merged_df.merge(shots_df, on="Match_report", how="left")

    merged_df.to_csv(MERGED_OUTPUT, index=False)
    print(f"Merged dataset saved to: {MERGED_OUTPUT}")

if __name__ == "__main__":
    merge_datasets()
