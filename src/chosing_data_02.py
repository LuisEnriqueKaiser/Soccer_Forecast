import pandas as pd


# ----------------------------
# Helpers
# ----------------------------
def _read_fbref_csv(path: str) -> pd.DataFrame:
    """
    FBref CSV exports often contain repeated header rows and an initial junk row.
    This keeps rows where venue is Home/Away and match_report looks valid.
    """
    df = pd.read_csv(path)

    # Drop first row (often a duplicated header-ish row in your files)
    if len(df) > 0:
        df = df.iloc[1:].copy()

    # Keep only real match rows
    if "venue" in df.columns:
        df = df[df["venue"].isin(["Home", "Away"])].copy()

    if "match_report" in df.columns:
        df = df[df["match_report"].notna()].copy()
        df = df[df["match_report"] != "Match Report"].copy()

    return df


def _ensure_unique_key(df: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
    """
    Enforce exactly one row per key combo. Keeps the first occurrence deterministically.
    """
    df = df.drop_duplicates(subset=keys, keep="first").copy()
    return df


# ----------------------------
# Schedule
# ----------------------------
def build_schedule_all_years(leagues: list[str]) -> pd.DataFrame:
    frames = []

    for league in leagues:
        schedule_path = (
            f"/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/fbref/"
            f"{league.lower()}_schedule_2015-2016_2016-2017_2017-2018_2018-2019_2019-2020_"
            f"2020-2021_2021-2022_2022-2023_2023-2024_2024-2025.csv"
        )

        df = pd.read_csv(schedule_path)

        # Derive goals from score (safe when score is NaN)
        score = df.get("score")
        if score is not None:
            split = score.astype(str).str.split("–", expand=True)
            if split.shape[1] >= 2:
                df["home_team_goals"] = split[0]
                df["away_team_goals"] = split[1]
            else:
                df["home_team_goals"] = pd.NA
                df["away_team_goals"] = pd.NA
        else:
            df["home_team_goals"] = pd.NA
            df["away_team_goals"] = pd.NA

        df["league"] = league

        # Restrict to needed columns
        keep = [
            "week",
            "date",
            "home_team",
            "away_team",
            "home_team_goals",
            "away_team_goals",
            "match_report",
            "home_xg",
            "away_xg",
            "league",
        ]
        df = df[[c for c in keep if c in df.columns]].copy()

        # Clean obvious junk rows
        df = df[df["match_report"].notna()].copy()
        df = df[df["match_report"] != "Match Report"].copy()

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    # NOTE: schedule files currently span multiple seasons but you are not attaching season.
    # This script keeps your current keying strategy and prevents row explosions by enforcing
    # uniqueness in advanced tables (league + season + match_report) and using those keys there.
    # If you later add season to schedule, also enforce uniqueness on that composite key.
    out = out.drop_duplicates(subset=["league", "match_report"], keep="first").copy()
    return out


# ----------------------------
# Generic "Home/Away -> one row" builder
# ----------------------------
def build_venue_merged(
    *,
    leagues: list[str],
    years: list[str],
    stat_slug: str,                 # e.g. "defense", "passing", "shooting", "passing_types"
    stat_cols_map: dict[str, str],  # source column -> output base name (without _home/_away)
    require_cols: list[str] = None, # optional: extra required columns in the source
) -> pd.DataFrame:
    frames = []
    require_cols = require_cols or []

    for league in leagues:
        for year in years:
            path = (
                f"/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/fbref/"
                f"{league.lower()}_{stat_slug}_{year}.csv"
            )

            df = _read_fbref_csv(path)
            df["season"] = year
            df["league"] = league

            # Ensure all columns exist
            for col in stat_cols_map.keys():
                if col not in df.columns:
                    df[col] = pd.NA
            for col in require_cols:
                if col not in df.columns:
                    df[col] = pd.NA

            home = df[df["venue"] == "Home"].copy()
            away = df[df["venue"] == "Away"].copy()

            key = ["match_report", "league", "season"]

            # Subset for merge
            home_keep = key + list(stat_cols_map.keys())
            away_keep = key + list(stat_cols_map.keys())

            home = home[home_keep].copy()
            away = away[away_keep].copy()

            merged = home.merge(
                away,
                on=key,
                how="inner",
                suffixes=("_home", "_away"),
            )

            # Rename to target names
            rename_home = {f"{k}_home": f"{v}_home" for k, v in stat_cols_map.items()}
            rename_away = {f"{k}_away": f"{v}_away" for k, v in stat_cols_map.items()}
            merged = merged.rename(columns={**rename_home, **rename_away})

            # Enforce 1 row per match key (critical to prevent row explosion later)
            merged = _ensure_unique_key(merged, key, f"{stat_slug}:{league}:{year}")

            frames.append(merged)

    out = pd.concat(frames, ignore_index=True)
    out = _ensure_unique_key(out, ["match_report", "league", "season"], f"{stat_slug}:ALL")
    return out


# ----------------------------
# Wrappers for your stats
# ----------------------------
def build_defense_merged(leagues: list[str], years: list[str], defense_data_cols: dict[str, str]) -> pd.DataFrame:
    return build_venue_merged(
        leagues=leagues,
        years=years,
        stat_slug="defense",
        stat_cols_map=defense_data_cols,
    )


def build_passing_merged(leagues: list[str], years: list[str], passing: dict[str, str]) -> pd.DataFrame:
    return build_venue_merged(
        leagues=leagues,
        years=years,
        stat_slug="passing",
        stat_cols_map=passing,
    )


def build_passing_types_merged(leagues: list[str], years: list[str], passing_type: dict[str, str]) -> pd.DataFrame:
    return build_venue_merged(
        leagues=leagues,
        years=years,
        stat_slug="passing_types",
        stat_cols_map=passing_type,
    )


def build_shooting_merged(leagues: list[str], years: list[str], shots: dict[str, str]) -> pd.DataFrame:
    return build_venue_merged(
        leagues=leagues,
        years=years,
        stat_slug="shooting",
        stat_cols_map=shots,
    )


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    leagues = [
        "eng-premier_league",
        "esp-la_liga",
        "ita-serie_a",
        "ger-bundesliga",
        "fra-ligue_1",
    ]

    years = [
        "2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020",
        "2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025",
        "2025-2026",
    ]

    # Build schedule
    all_years_schedule = build_schedule_all_years(leagues)
    all_years_schedule.to_pickle(
        "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/schedule_all_years.pkl"
    )

    # Defense
    defense_data_cols = {"Int": "int", "Blocks": "blocks", "Clr": "clr", "Err": "err", "season": "season"}
    all_leagues_defense = build_defense_merged(leagues, years, defense_data_cols)
    all_leagues_defense.to_pickle(
        "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/defense_data_merged.pkl"
    )

    # Passing
    passing = {
        "Total": "pas_tot",
        "Total.1": "pas_att",
        "Total.4": "pgrssv_dist",
        "Total.3": "tot_dist",
        "Ast": "ast",
        "xAG": "exp_assists_g",
        "xA": "exp_assists",
    }
    passing_all_leagues = build_passing_merged(leagues, years, passing)
    passing_all_leagues.to_pickle(
        "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/passing_data_merged.pkl"
    )

    # Passing types
    passing_type = {"Pass Types.5": "crosses"}
    passing_types_all_leagues = build_passing_types_merged(leagues, years, passing_type)
    passing_types_all_leagues.to_pickle(
        "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/passing_types_data_merged.pkl"
    )

    # Shooting
    shots = {
        "Standard.2": "SoT",
        "Expected": "xg",
        "Standard.3": "SoT_over_goals",
        "Expected.1": "npxg",
        "Expected.2": "npxg_per_shot",
    }
    shots_all_leagues = build_shooting_merged(leagues, years, shots)
    shots_all_leagues.to_pickle(
        "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/shooting_data_merged.pkl"
    )
