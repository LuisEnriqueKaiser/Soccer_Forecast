import os
import pandas as pd
import soccerdata as sd

def download_fbref_season(league: str, season: str) -> pd.DataFrame:
    """
    Downloads schedule data from FBref for a single season of a given league.
    Returns a DataFrame with columns such as date, home_team, away_team, score, etc.
    """
    fb = sd.FBref(leagues=[league], seasons=[season])
    df = fb.read_schedule()
    df["league"] = league
    df["season"] = season
    return df

def download_and_merge_fbref(league: str, seasons: list, data_dir: str) -> pd.DataFrame:
    """
    Downloads schedule data from FBref for multiple seasons of a given league,
    merges them into a single DataFrame, converts the date column to datetime,
    and saves the merged file to data_dir/fbref/<league>_all_seasons.csv.
    Returns the merged DataFrame.
    """
    merged_dfs = []
    for season in seasons:
        print(f"Downloading FBref data for {league}, {season}...")
        df_season = download_fbref_season(league, season)
        if not df_season.empty:
            merged_dfs.append(df_season)
    
    if not merged_dfs:
        print(f"No FBref data downloaded for {league} with seasons {seasons}.")
        return pd.DataFrame()
    
    df_all = pd.concat(merged_dfs, ignore_index=True)
    if "date" in df_all.columns:
        df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce", dayfirst=True)
    fbref_dir = os.path.join(data_dir, "fbref")
    os.makedirs(fbref_dir, exist_ok=True)
    out_path = os.path.join(fbref_dir, f"{league.replace(' ', '_')}_all_seasons.csv")
    df_all.to_csv(out_path, index=False)
    print(f"Merged FBref data for {league} saved to {out_path} with {df_all.shape[0]} rows.")
    return df_all

def clear_cache(cache_dir: str = None):
    """
    Clears the SoccerData MatchHistory cache.
    (Not used in this FBref-only version.)
    """
    import shutil
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), "soccerdata", "data", "MatchHistory")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cache cleared at {cache_dir}")
    else:
        print(f"Cache path {cache_dir} does not exist.")

def get_upcoming_odds(league_url: str, output_path: str) -> pd.DataFrame:
    """
    (Optional) Scrapes upcoming match odds from a bookmaker site.
    Returns an empty DataFrame if an error occurs.
    """
    import requests
    from bs4 import BeautifulSoup
    try:
        resp = requests.get(league_url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"Error fetching upcoming odds page: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "html.parser")
    matches = []
    for row in soup.select("div.match-row"):
        try:
            home_team = row.select_one("span.home-team").get_text(strip=True)
            away_team = row.select_one("span.away-team").get_text(strip=True)
            match_date = row.select_one("span.match-date").get_text(strip=True)
            home_odds = row.select_one("span.home-odds").get_text(strip=True)
            draw_odds = row.select_one("span.draw-odds").get_text(strip=True)
            away_odds = row.select_one("span.away-odds").get_text(strip=True)
            matches.append({
                "home_team": home_team,
                "away_team": away_team,
                "match_date": match_date,
                "b365_home_odds": float(home_odds),
                "b365_draw_odds": float(draw_odds),
                "b365_away_odds": float(away_odds),
            })
        except Exception as inner_e:
            print(f"Error parsing an odds row: {inner_e}")
            continue
    df_upcoming = pd.DataFrame(matches)
    if not df_upcoming.empty:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_upcoming.to_csv(output_path, index=False)
        print(f"Upcoming odds saved to {output_path}")
    return df_upcoming
