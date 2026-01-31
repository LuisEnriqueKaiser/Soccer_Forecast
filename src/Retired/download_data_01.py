import os
import time
import soccerdata as sd
from project_specifics import leagues, seasons, filepath_01, advanced_stats

# FBref: ~10 requests / minute → ~1 request every 6 seconds
RATE_LIMIT = 6  # seconds between top-level calls
MIN_SEASON = "2015-2016"  # lower bound for the DB


def league_to_slug(league: str) -> str:
    """ENG-Premier League -> eng-premier_league"""
    return league.replace(" ", "_").lower()


def split_seasons(seasons_all, min_season: str):
    """
    Take the global seasons list from project_specifics and:

    - keep only seasons >= min_season
    - use the last one as the current season
    - everything before is treated as historical
     """
    filtered = [s for s in seasons_all if s >= min_season]
    if not filtered:
        raise ValueError(
            f"No seasons >= {min_season} in project_specifics.seasons; "
            f"got: {seasons_all}"
        )

    current_season = filtered[-1]
    historical = filtered[:-1]  # may be empty if you only have one season
    return historical, current_season


def save_schedule_block(
    league: str,
    seasons_block: list[str],
    out_path: str,
    overwrite: bool,
    no_cache: bool,
):
    """
    Generic helper:
    - If out_path exists and overwrite=False → do nothing.
    - Else instantiate FBref for given league + seasons_block,
      read schedule and write CSV.
    """
    if not seasons_block:
        return

    if os.path.exists(out_path) and not overwrite:
        print(f"[SKIP] {out_path} already exists (overwrite=False).")
        return

    print(f"[FBREF] {league} – seasons {seasons_block}")
    fbref = sd.FBref(leagues=league, seasons=seasons_block, no_cache=no_cache)
    fbref.rate_limit = RATE_LIMIT

    df_schedule = fbref.read_schedule()
    df_schedule.to_csv(out_path, index=False)
    print(f"[OK] Saved schedule → {out_path}")

    # Spacing out top-level calls; FBref itself also rate-limits internal requests
    time.sleep(RATE_LIMIT)

def save_shooting_block(
    league: str,
    seasons_block: list[str],
    out_path: str,
    overwrite: bool,
    no_cache: bool,
):  
    
    if not seasons_block:
        return
    for season in seasons_block:
        # replace all seasons, built like             hist_suffix = "_".join(historical_seasons)
        # with single season
        season_path = out_path.replace("_".join(seasons_block), season)
        #print(f"[INFO] Processing shooting data for {league} season {season}...")
        if os.path.exists(season_path) and not overwrite:
            print(f"[SKIP] {season_path} already exists (overwrite=False).")
            continue
        else: 


            fbref = sd.FBref(leagues=league, seasons=[season], no_cache=no_cache)
            fbref.rate_limit = RATE_LIMIT  
            try:
                df_shooting = fbref.read_team_match_stats(stat_type="shooting", opponent_stats=True)
            except: 
                # wait and try again 
                print(f"[WARN] Retrying shooting for {league} seasons {seasons_block} after error...")
                time.sleep(RATE_LIMIT + 10)
                try: 
                    print(f"[FBREF] {league} – seasons {seasons_block} (shooting) - 2nd attempt")
                    df_shooting = fbref.read_team_match_stats(stat_type="shooting", opponent_stats=True)
                except: 
                    print(f"[ERROR] Second attempt failed for shooting for {league} seasons {seasons_block}. Skipping...")
                    time.sleep(RATE_LIMIT)
                    try: 
                        print(f"[FBREF] {league} – seasons {seasons_block} (shooting) - 3rd attempt")
                        df_shooting = fbref.read_team_match_stats(stat_type="shooting", opponent_stats=True)
                    except:
                        print(f"[ERROR] Third attempt failed for shooting for {league} seasons {seasons_block}. Skipping...")
            # if it exists, save
            if df_shooting is not None and not df_shooting.empty:
                df_shooting.to_csv(season_path, index=False)

    time.sleep(RATE_LIMIT)

def save_passing_block(
    league: str,
    seasons_block: list[str],
    out_path: str,
    overwrite: bool,
    no_cache: bool,
):  
    if not seasons_block:
        return
    for season in seasons_block:
        # replace all seasons, built like             hist_suffix = "_".join(historical_seasons)
        # with single season
        season_path = out_path.replace("_".join(seasons_block), season)
        print(f"[INFO] Processing passing data for {league} season {season}...")
        if os.path.exists(season_path) and not overwrite:
            print(f"[SKIP] {season_path} already exists (overwrite=False).")
            continue
        else: 
            fbref = sd.FBref(leagues=league, seasons=[season], no_cache=no_cache)
            fbref.rate_limit = RATE_LIMIT  
            try:
                df_passing = fbref.read_team_match_stats(stat_type="passing", opponent_stats=True)
            except: 
                # wait and try again 
                print(f"[WARN] Retrying passing for {league} seasons {seasons_block} after error...")
                time.sleep(RATE_LIMIT + 10)
                try: 
                    print(f"[FBREF] {league} – seasons {seasons_block} (passing) - 2nd attempt")
                    df_passing = fbref.read_team_match_stats(stat_type="passing", opponent_stats=True)
                except: 
                    print(f"[ERROR] Second attempt failed for passing for {league} seasons {seasons_block}. Skipping...")
                    time.sleep(RATE_LIMIT)
                    try: 
                        print(f"[FBREF] {league} – seasons {seasons_block} (passing) - 3rd attempt")
                        df_passing = fbref.read_team_match_stats(stat_type="passing", opponent_stats=True)
                    except:
                        print(f"[ERROR] Third attempt failed for passing for {league} seasons {seasons_block}. Skipping...")
            # if it exists, save
            if df_passing is not None and not df_passing.empty:
                df_passing.to_csv(season_path, index=False)

    time.sleep(RATE_LIMIT)

def save_goal_shot_creation_block(
    league: str,
    seasons_block: list[str],
    out_path: str,
    overwrite: bool,
    no_cache: bool,
):  
    if not seasons_block:
        return
    for season in seasons_block:
        # replace all seasons, built like             hist_suffix = "_".join(historical_seasons)
        # with single season
        season_path = out_path.replace("_".join(seasons_block), season)
        print(f"[INFO] Processing goal_shot_creation data for {league} season {season}...")
        if os.path.exists(season_path) and not overwrite:
            print(f"[SKIP] {season_path} already exists (overwrite=False).")
            continue
        else: 
            fbref = sd.FBref(leagues=league, seasons=[season], no_cache=no_cache)
            fbref.rate_limit = RATE_LIMIT  
            try:
                df_goal_shot_creation = fbref.read_team_match_stats(stat_type="goal_shot_creation", opponent_stats=True)
            except: 
                # wait and try again 
                print(f"[WARN] Retrying goal_shot_creation for {league} seasons {seasons_block} after error...")
                time.sleep(RATE_LIMIT + 10)
                try: 
                    print(f"[FBREF] {league} – seasons {seasons_block} (goal_shot_creation) - 2nd attempt")
                    df_goal_shot_creation = fbref.read_team_match_stats(stat_type="goal_shot_creation", opponent_stats=True)
                except: 
                    print(f"[ERROR] Second attempt failed for goal_shot_creation for {league} seasons {seasons_block}. Skipping...")
                    time.sleep(RATE_LIMIT)
                    try: 
                        print(f"[FBREF] {league} – seasons {seasons_block} (goal_shot_creation) - 3rd attempt")
                        df_goal_shot_creation = fbref.read_team_match_stats(stat_type="goal_shot_creation", opponent_stats=True)
                    except:
                        print(f"[ERROR] Third attempt failed for goal_shot_creation for {league} seasons {seasons_block}. Skipping...")
            # if it exists, save
            if df_goal_shot_creation is not None and not df_goal_shot_creation.empty:
                df_goal_shot_creation.to_csv(season_path, index=False)

    time.sleep(RATE_LIMIT)

def save_possession_block(
    league: str,
    seasons_block: list[str],
    out_path: str,
    overwrite: bool,
    no_cache: bool,
):  
    if not seasons_block:
        return
    for season in seasons_block:
        # replace all seasons, built like             hist_suffix = "_".join(historical_seasons)
        # with single season
        season_path = out_path.replace("_".join(seasons_block), season)
        print(f"[INFO] Processing possession data for {league} season {season}...")
        if os.path.exists(season_path) and not overwrite:
            print(f"[SKIP] {season_path} already exists (overwrite=False).")
            continue
        else: 
            fbref = sd.FBref(leagues=league, seasons=[season], no_cache=no_cache)
            fbref.rate_limit = RATE_LIMIT  
            try:
                df_possession = fbref.read_team_match_stats(stat_type="possession", opponent_stats=True)
            except: 
                # wait and try again 
                print(f"[WARN] Retrying possession for {league} seasons {seasons_block} after error...")
                time.sleep(RATE_LIMIT + 10)
                try: 
                    print(f"[FBREF] {league} – seasons {seasons_block} (possession) - 2nd attempt")
                    df_possession = fbref.read_team_match_stats(stat_type="possession", opponent_stats=True)
                except: 
                    print(f"[ERROR] Second attempt failed for possession for {league} seasons {seasons_block}. Skipping...")
                    time.sleep(RATE_LIMIT)
                    try: 
                        print(f"[FBREF] {league} – seasons {seasons_block} (possession) - 3rd attempt")
                        df_possession = fbref.read_team_match_stats(stat_type="possession", opponent_stats=True)
                    except:
                        print(f"[ERROR] Third attempt failed for possession for {league} seasons {seasons_block}. Skipping...")
            # if it exists, save
            if df_possession is not None and not df_possession.empty:
                df_possession.to_csv(season_path, index=False)

    time.sleep(RATE_LIMIT)

def save_defense_block(
    league: str,
    seasons_block: list[str],
    out_path: str,
    overwrite: bool,
    no_cache: bool,
):
    if not seasons_block:
        return
    for season in seasons_block:
        # replace all seasons, built like             hist_suffix = "_".join(historical_seasons)
        # with single season
        season_path = out_path.replace("_".join(seasons_block), season)
        print(f"[INFO] Processing defense data for {league} season {season}...")
        if os.path.exists(season_path) and not overwrite:
            print(f"[SKIP] {season_path} already exists (overwrite=False).")
            continue
        else: 
            fbref = sd.FBref(leagues=league, seasons=[season], no_cache=no_cache)
            fbref.rate_limit = RATE_LIMIT  
            try:
                df_defense = fbref.read_team_match_stats(stat_type="defense", opponent_stats=True)
            except: 
                # wait and try again 
                print(f"[WARN] Retrying defense for {league} seasons {seasons_block} after error...")
                time.sleep(RATE_LIMIT + 10)
                try: 
                    print(f"[FBREF] {league} – seasons {seasons_block} (defense) - 2nd attempt")
                    df_defense = fbref.read_team_match_stats(stat_type="defense", opponent_stats=True)
                except: 
                    print(f"[ERROR] Second attempt failed for defense for {league} seasons {seasons_block}. Skipping...")
                    time.sleep(RATE_LIMIT)
                    try: 
                        print(f"[FBREF] {league} – seasons {seasons_block} (defense) - 3rd attempt")
                        df_defense = fbref.read_team_match_stats(stat_type="defense", opponent_stats=True)
                    except:
                        print(f"[ERROR] Third attempt failed for defense for {league} seasons {seasons_block}. Skipping...")
            # if it exists, save
            if df_defense is not None and not df_defense.empty:
                df_defense.to_csv(season_path, index=False)

    time.sleep(RATE_LIMIT)


def save_passing_types_block(
    league: str,
    seasons_block: list[str],
    out_path: str,
    overwrite: bool,
    no_cache: bool,
):
    if not seasons_block:
        return
    for season in seasons_block:
        # replace all seasons, built like             hist_suffix = "_".join(historical_seasons)
        # with single season
        season_path = out_path.replace("_".join(seasons_block), season)
        print(f"[INFO] Processing passing_types data for {league} season {season}...")
        if os.path.exists(season_path) and not overwrite:
            print(f"[SKIP] {season_path} already exists (overwrite=False).")
            continue
        else: 
            fbref = sd.FBref(leagues=league, seasons=[season], no_cache=no_cache)
            fbref.rate_limit = RATE_LIMIT  
            try:
                df_passing_types = fbref.read_team_match_stats(stat_type="passing_types", opponent_stats=True)
            except: 
                # wait and try again 
                print(f"[WARN] Retrying passing_types for {league} seasons {seasons_block} after error...")
                time.sleep(RATE_LIMIT + 10)
                try: 
                    print(f"[FBREF] {league} – seasons {seasons_block} (passing_types) - 2nd attempt")
                    df_passing_types = fbref.read_team_match_stats(stat_type="passing_types", opponent_stats=True)
                except: 
                    print(f"[ERROR] Second attempt failed for passing_types for {league} seasons {seasons_block}. Skipping...")
                    time.sleep(RATE_LIMIT)
                    try: 
                        print(f"[FBREF] {league} – seasons {seasons_block} (passing_types) - 3rd attempt")
                        df_passing_types = fbref.read_team_match_stats(stat_type="passing_types", opponent_stats=True)
                    except:
                        print(f"[ERROR] Third attempt failed for passing_types for {league} seasons {seasons_block}. Skipping...")
            # if it exists, save
            if df_passing_types is not None and not df_passing_types.empty:
                df_passing_types.to_csv(season_path, index=False)

    time.sleep(RATE_LIMIT)

def main():
    # Use your global seasons list, but only from 2015-2016 upward
    historical_seasons, current_season = split_seasons(seasons, MIN_SEASON)

    for league in leagues:
        slug = league_to_slug(league)

        # ------------------------------------------------------------------
        # 1) Historical schedule: 2015-2016 up to the season before current
        #    - Written once
        #    - Skipped on later runs if the file is already there
        # ------------------------------------------------------------------
        if historical_seasons:
            hist_suffix = "_".join(historical_seasons)
            hist_filename = f"{slug}_schedule_{hist_suffix}.csv"
            hist_path = os.path.join(filepath_01, hist_filename)

            save_schedule_block(
                league=league,
                seasons_block=historical_seasons,
                out_path=hist_path,
                overwrite=False,   # never re-write once done
                no_cache=False,    # FBref uses its own HTML cache for old seasons
            )
            hist_filename = f"{slug}_shooting_{hist_suffix}.csv"
            hist_path = os.path.join(filepath_01, hist_filename)
            if "shooting" in advanced_stats:
                time.sleep(RATE_LIMIT)
                save_shooting_block(
                    league=league,
                    seasons_block=historical_seasons,
                    out_path=hist_path,
                    overwrite=False,   # never re-write once done
                    no_cache=False,    # FBref uses its own HTML cache for old seasons
                )
            hist_filename = f"{slug}_passing_{hist_suffix}.csv"
            hist_path = os.path.join(filepath_01, hist_filename)

            if "passing" in advanced_stats:
                time.sleep(RATE_LIMIT)
                save_passing_block(
                    league=league,
                    seasons_block=historical_seasons,
                    out_path=hist_path,
                    overwrite=False,   # never re-write once done
                    no_cache=False,    # FBref uses its own HTML cache for old seasons
                )
            
            hist_filename = f"{slug}_goal_shot_creation_{hist_suffix}.csv"
            hist_path = os.path.join(filepath_01, hist_filename)
            if "goal_shot_creation" in advanced_stats:
                time.sleep(RATE_LIMIT)
                save_goal_shot_creation_block(
                    league=league,
                    seasons_block=historical_seasons,
                    out_path=hist_path,
                    overwrite=False,   # never re-write once done
                    no_cache=False,    # FBref uses its own HTML cache for old seasons
                )
            
            hist_filename = f"{slug}_possession_{hist_suffix}.csv"
            hist_path = os.path.join(filepath_01, hist_filename)
            if "possession" in advanced_stats:
                time.sleep(RATE_LIMIT)
                save_possession_block(
                    league=league,
                    seasons_block=historical_seasons,
                    out_path=hist_path,
                    overwrite=False,   # never re-write once done
                    no_cache=False,    # FBref uses its own HTML cache for old seasons
                )
            hist_filename = f"{slug}_defense_{hist_suffix}.csv"
            hist_path = os.path.join(filepath_01, hist_filename)
            if "defense" in advanced_stats:
                time.sleep(RATE_LIMIT)
                save_defense_block(
                    league=league,
                    seasons_block=historical_seasons,
                    out_path=hist_path,
                    overwrite=False,   # never re-write once done
                    no_cache=False,    # FBref uses its own HTML cache for old seasons
                )
            hist_filename = f"{slug}_passing_types_{hist_suffix}.csv"
            hist_path = os.path.join(filepath_01, hist_filename)
            if "passing_types" in advanced_stats:
                time.sleep(RATE_LIMIT)
                save_passing_types_block(
                    league=league,
                    seasons_block=historical_seasons,
                    out_path=hist_path,
                    overwrite=False,   # never re-write once done
                    no_cache=False,    # FBref uses its own HTML cache for old seasons
                )

            

        # ------------------------------------------------------------------
        # 2) Current season schedule:
        #    - Always overwritten
        #    - no_cache=True to force fresh HTML for an ongoing season
        # ------------------------------------------------------------------
        curr_filename = f"{slug}_schedule_{current_season}.csv"
        curr_path = os.path.join(filepath_01, curr_filename)

        save_schedule_block(
            league=league,
            seasons_block=[current_season],
            out_path=curr_path,
            overwrite=True,    # always re-write current season
            no_cache=True,     # FBref bypasses cache for up-to-date data
        )

        curr_filename = f"{slug}_shooting_{current_season}.csv"
        curr_path = os.path.join(filepath_01, curr_filename)
        if "shooting" in advanced_stats:
            time.sleep(RATE_LIMIT)
            save_shooting_block(
                league=league,
                seasons_block=[current_season],
                out_path=curr_path,
                overwrite=True,    # always re-write current season
                no_cache=True,     # FBref bypasses cache for up-to-date data
            )
        curr_filename = f"{slug}_passing_{current_season}.csv"
        curr_path = os.path.join(filepath_01, curr_filename)
        if "passing" in advanced_stats:
            time.sleep(RATE_LIMIT)
            save_passing_block(
                league=league,
                seasons_block=[current_season],
                out_path=curr_path,
                overwrite=True,    # always re-write current season
                no_cache=True,     # FBref bypasses cache for up-to-date data
            )
        curr_filename = f"{slug}_goal_shot_creation_{current_season}.csv"
        curr_path = os.path.join(filepath_01, curr_filename)
        if "goal_shot_creation" in advanced_stats:
            time.sleep(RATE_LIMIT)
            save_goal_shot_creation_block(
                league=league,
                seasons_block=[current_season],
                out_path=curr_path,
                overwrite=True,    # always re-write current season
                no_cache=True,     # FBref bypasses cache for up-to-date data
            )
        curr_filename = f"{slug}_possession_{current_season}.csv"
        curr_path = os.path.join(filepath_01, curr_filename)
        if "possession" in advanced_stats:  
            time.sleep(RATE_LIMIT)
            save_possession_block(
                league=league,
                seasons_block=[current_season],
                out_path=curr_path,
                overwrite=True,    # always re-write current season
                no_cache=True,     # FBref bypasses cache for up-to-date data
            )
        curr_filename = f"{slug}_defense_{current_season}.csv"
        curr_path = os.path.join(filepath_01, curr_filename)
        if "defense" in advanced_stats:
            time.sleep(RATE_LIMIT)
            save_defense_block(
                league=league,
                seasons_block=[current_season],
                out_path=curr_path,
                overwrite=True,    # always re-write current season
                no_cache=True,     # FBref bypasses cache for up-to-date data
            )
        curr_filename = f"{slug}_passing_types_{current_season}.csv"
        curr_path = os.path.join(filepath_01, curr_filename)
        if "passing_types" in advanced_stats:
            time.sleep(RATE_LIMIT)
            save_passing_types_block(
                league=league,
                seasons_block=[current_season],
                out_path=curr_path,
                overwrite=True,    # always re-write current season
                no_cache=True,     # FBref bypasses cache for up-to-date data
            )
            

if __name__ == "__main__":
    main()
