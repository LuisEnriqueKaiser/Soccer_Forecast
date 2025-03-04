import pandas as pd

def clean_match_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the FBref merged game data.
    - If a 'score' column exists, parses it to create 'home_goals' and 'away_goals'.
    - Creates a 'result' column from 'score' (if possible) or uses an existing 'FTR' column.
    - Prints unique results for debugging.
    """
    # If a score column exists, attempt to parse it.
    if "score" in df.columns:
        def parse_score(s):
            try:
                parts = s.split("-")
                if len(parts) == 2:
                    return int(parts[0].strip()), int(parts[1].strip())
            except:
                return None, None
            return None, None
        # Only fill in if home_goals and away_goals are missing.
        if "home_goals" not in df.columns or "away_goals" not in df.columns:
            parsed = df["score"].apply(parse_score)
            df["home_goals"] = parsed.apply(lambda x: x[0] if x[0] is not None else pd.NA)
            df["away_goals"] = parsed.apply(lambda x: x[1] if x[1] is not None else pd.NA)
    
    # If result column is missing, try to create it.
    if "result" not in df.columns:
        if "FTR" in df.columns:
            df["result"] = df["FTR"]
        elif "home_goals" in df.columns and "away_goals" in df.columns:
            df["result"] = df.apply(
                lambda row: 'H' if row["home_goals"] > row["away_goals"]
                else ('A' if row["home_goals"] < row["away_goals"] else 'D'), axis=1
            )
        else:
            print("Warning: No result information available.")
    print("Unique results in data after cleaning:", df["result"].unique())
    return df
