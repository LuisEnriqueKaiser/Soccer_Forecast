import argparse
import json
import os
import time
from typing import Iterable

import pandas as pd
import requests

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "data", "nba", "processed")
DEFAULT_RAW_DIR = os.path.join(BASE_DIR, "data", "nba", "raw")

NBA_STATS_BASE = "https://stats.nba.com/stats"

NBA_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _nba_get(endpoint: str, params: dict, sleep_s: float = 0.6) -> dict:
    url = f"{NBA_STATS_BASE}/{endpoint}"
    resp = requests.get(url, params=params, headers=NBA_HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(sleep_s)
    return resp.json()


def _resultset_to_df(payload: dict, resultset_name: str) -> pd.DataFrame:
    for rs in payload.get("resultSets", []):
        if rs.get("name") == resultset_name:
            return pd.DataFrame(rs.get("rowSet", []), columns=rs.get("headers", []))
    raise KeyError(f"resultSet '{resultset_name}' not found")


def _fetch_league_team_gamelog(season: str, season_type: str, sleep_s: float) -> pd.DataFrame:
    params = {
        "LeagueID": "00",
        "PlayerOrTeam": "T",
        "Season": season,
        "SeasonType": season_type,
        "Sorter": "DATE",
        "Direction": "ASC",
    }
    payload = _nba_get("leaguegamelog", params, sleep_s=sleep_s)
    df = _resultset_to_df(payload, "LeagueGameLog")
    if df.empty:
        return df

    df.columns = [c.lower() for c in df.columns]
    df = df.rename(
        columns={
            "game_date": "game_date",
            "game_id": "game_id",
            "team_id": "team_id",
            "team_name": "team_name",
            "team_abbreviation": "team_abbr",
            "matchup": "matchup",
        }
    )
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


def _fetch_boxscore_advanced(game_id: str, sleep_s: float) -> pd.DataFrame:
    params = {
        "GameID": game_id,
        "StartPeriod": 0,
        "EndPeriod": 10,
        "StartRange": 0,
        "EndRange": 28800,
        "RangeType": 0,
    }
    payload = _nba_get("boxscoreadvancedv2", params, sleep_s=sleep_s)
    df = _resultset_to_df(payload, "TeamStats")
    if df.empty:
        return df

    df["GAME_ID"] = game_id
    return df


def _parse_matchup(matchup: str) -> tuple[str | None, str | None]:
    if not isinstance(matchup, str):
        return None, None
    if " vs. " in matchup:
        parts = matchup.split(" vs. ")
        if len(parts) == 2:
            return "home", parts[1]
    if " @ " in matchup:
        parts = matchup.split(" @ ")
        if len(parts) == 2:
            return "away", parts[1]
    return None, None


def _coerce_numeric(df: pd.DataFrame, exclude: Iterable[str]) -> pd.DataFrame:
    for col in df.columns:
        if col in exclude:
            continue
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def _clean_team_advanced(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    df.columns = [c.strip().lower() for c in df.columns]

    if "min" in df.columns:
        def _min_to_float(val):
            if isinstance(val, str) and ":" in val:
                mins, secs = val.split(":")
                try:
                    return float(mins) + float(secs) / 60.0
                except ValueError:
                    return pd.NA
            return val

        df["min"] = df["min"].apply(_min_to_float)

    exclude = {
        "game_id",
        "team_id",
        "team_abbreviation",
        "team_city",
        "team_name",
        "min",
        "game_date",
        "matchup",
        "home_away",
        "opponent_abbr",
        "season",
        "season_type",
    }
    df = _coerce_numeric(df, exclude=exclude)

    return df


def build_team_advanced_dataset(
    season: str,
    season_type: str = "Regular Season",
    sleep_s: float = 0.6,
    raw_dir: str | None = None,
) -> pd.DataFrame:
    game_log = _fetch_league_team_gamelog(season, season_type, sleep_s)
    if game_log.empty:
        return game_log

    if raw_dir:
        _ensure_dir(raw_dir)
        raw_path = os.path.join(raw_dir, f"leaguegamelog_{season}_{season_type}.csv")
        game_log.to_csv(raw_path, index=False)

    unique_game_ids = sorted(game_log["game_id"].unique())
    advanced_rows = []

    for idx, game_id in enumerate(unique_game_ids, start=1):
        df_game = _fetch_boxscore_advanced(game_id, sleep_s)
        if df_game is None or df_game.empty:
            continue
        advanced_rows.append(df_game)
        if idx % 100 == 0:
            print(f"[INFO] Fetched {idx}/{len(unique_game_ids)} games")

    if not advanced_rows:
        return pd.DataFrame()

    df_adv = pd.concat(advanced_rows, ignore_index=True)
    df_adv.columns = [c.lower() for c in df_adv.columns]

    merge_cols = [
        "game_id",
        "team_id",
        "team_abbreviation",
        "team_name",
        "matchup",
        "game_date",
    ]
    for col in merge_cols:
        if col not in game_log.columns:
            game_log[col] = pd.NA

    df = df_adv.merge(game_log[merge_cols], on=["game_id", "team_id"], how="left")

    home_away, opponent = zip(*df["matchup"].apply(_parse_matchup))
    df["home_away"] = home_away
    df["opponent_abbr"] = opponent
    df["season"] = season
    df["season_type"] = season_type

    df = _clean_team_advanced(df)

    return df


def save_team_advanced_dataset(
    season: str,
    season_type: str = "Regular Season",
    out_dir: str = DEFAULT_OUT_DIR,
    sleep_s: float = 0.6,
    overwrite: bool = False,
    raw_dir: str | None = None,
) -> str:
    _ensure_dir(out_dir)
    safe_season_type = season_type.lower().replace(" ", "_")
    out_path = os.path.join(out_dir, f"nba_team_advanced_{season}_{safe_season_type}.csv")

    if os.path.exists(out_path) and not overwrite:
        print(f"[SKIP] {out_path} already exists (overwrite=False).")
        return out_path

    df = build_team_advanced_dataset(
        season, season_type, sleep_s=sleep_s, raw_dir=raw_dir
    )
    if df.empty:
        raise RuntimeError("No data returned from NBA stats API.")

    df.to_csv(out_path, index=False)
    print(f"[OK] Saved {len(df)} rows -> {out_path}")
    return out_path


def _season_str(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _current_season_start_year(today: pd.Timestamp) -> int:
    # NBA season starts in the fall; treat July as season rollover.
    return today.year if today.month >= 7 else today.year - 1


def _last_n_plus_current_seasons(n: int, today: pd.Timestamp | None = None) -> list[str]:
    if today is None:
        today = pd.Timestamp.today()
    current_start = _current_season_start_year(today)
    start_years = list(range(current_start - n, current_start + 1))
    return [_season_str(y) for y in start_years]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and clean NBA team advanced stats per game.")
    parser.add_argument(
        "--season",
        help="NBA season string like 2024-25 (if omitted, uses last 10 + current)",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        help="List of NBA seasons like 2019-20 2020-21 ...",
    )
    parser.add_argument(
        "--season-type",
        default="Regular Season",
        help="Regular Season or Playoffs",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Output directory for cleaned CSV",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.6,
        help="Seconds to sleep between NBA stats API requests",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw league game log CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = DEFAULT_RAW_DIR if args.save_raw else None
    _ensure_dir(args.out_dir)
    if raw_dir:
        _ensure_dir(raw_dir)

    if args.seasons:
        seasons = args.seasons
    elif args.season:
        seasons = [args.season]
    else:
        seasons = _last_n_plus_current_seasons(10)

    out_paths = []
    for season in seasons:
        out_path = save_team_advanced_dataset(
            season=season,
            season_type=args.season_type,
            out_dir=args.out_dir,
            sleep_s=args.sleep,
            overwrite=args.overwrite,
            raw_dir=raw_dir,
        )
        out_paths.append(out_path)

    meta = {
        "seasons": seasons,
        "season_type": args.season_type,
        "out_paths": out_paths,
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }
    if len(out_paths) == 1:
        meta_path = out_paths[0].replace(".csv", ".meta.json")
    else:
        meta_path = os.path.join(args.out_dir, "nba_team_advanced_batch.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if raw_dir:
        print(f"[OK] Raw league game log saved to {raw_dir}")


if __name__ == "__main__":
    main()
