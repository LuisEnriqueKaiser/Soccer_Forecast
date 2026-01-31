from __future__ import annotations

from datetime import datetime, timezone
import json
from urllib.request import urlopen
from pathlib import Path

import pandas as pd
import streamlit as st


DATA_DIR = Path("data")
LOG_PATH = DATA_DIR / "bet_requests.csv"
SUGGESTED_PATH = DATA_DIR / "suggested_bets.csv"
SCHEDULE_DIR = DATA_DIR / "fbref"
MAX_BOOKMAKERS = 10
OUTCOMES = ("home", "draw", "away")
NBA_SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
TOP5_LEAGUES = {
    "eng-premier_league": {
        "label": "Premier League",
        "path": SCHEDULE_DIR / "eng-premier_league_schedule_2025-2026.csv",
    },
    "esp-la_liga": {
        "label": "La Liga",
        "path": SCHEDULE_DIR / "esp-la_liga_schedule_2025-2026.csv",
    },
    "ger-bundesliga": {
        "label": "Bundesliga",
        "path": SCHEDULE_DIR / "ger-bundesliga_schedule_2025-2026.csv",
    },
    "ita-serie_a": {
        "label": "Serie A",
        "path": SCHEDULE_DIR / "ita-serie_a_schedule_2025-2026.csv",
    },
    "fra-ligue_1": {
        "label": "Ligue 1",
        "path": SCHEDULE_DIR / "fra-ligue_1_schedule_2025-2026.csv",
    },
}
GERMAN_BOOKMAKERS = [
    "Tipico",
    "bwin",
    "bet365",
    "Betano",
    "Interwetten",
    "Ladbrokes",
    "Unibet",
    "Bet-at-home",
    "Pinnacle",
    "Custom",
]
LOG_COLUMNS = [
    "timestamp_utc",
    "match",
    "market",
    "selected_outcome",
    "tax_rate",
    "bankroll",
    "max_bet_cap",
    "kelly_fraction",
    "polymarket_home_odds",
    "polymarket_draw_odds",
    "polymarket_away_odds",
    "polymarket_implied_home",
    "polymarket_implied_draw",
    "polymarket_implied_away",
    "polymarket_implied_selected",
    "avg_bookmaker_implied_home",
    "avg_bookmaker_implied_draw",
    "avg_bookmaker_implied_away",
    "avg_bookmaker_implied_selected",
    "outcome",
    "bookmaker_name",
    "bookmaker_home_odds",
    "bookmaker_draw_odds",
    "bookmaker_away_odds",
    "bookmaker_overround",
    "bookmaker_implied_home",
    "bookmaker_implied_draw",
    "bookmaker_implied_away",
    "bookmaker_implied_selected",
    "bookmaker_odds_selected",
    "effective_odds",
    "fair_odds_polymarket",
    "fair_odds_avg_bookmaker",
    "edge_vs_polymarket",
    "edge_vs_avg_bookmaker",
    "edge",
    "mispricing_source",
    "is_mispriced",
    "kelly_full",
    "kelly_fractional",
    "recommended_stake",
    "suggested_bet",
]
SUGGESTED_COLUMNS = [
    "saved_at_utc",
    "match",
    "market",
    "outcome",
    "bookmaker_name",
    "bookmaker_odds_selected",
    "effective_odds",
    "edge_vs_polymarket",
    "edge_vs_avg_bookmaker",
    "edge",
    "mispricing_source",
    "recommended_stake",
    "stake_actual",
    "placed",
    "placed_at_utc",
    "result",
    "settled_at_utc",
    "pnl",
    "tax_rate",
    "bankroll",
    "max_bet_cap",
    "kelly_fraction",
    "polymarket_implied_selected",
    "avg_bookmaker_implied_selected",
]


def initialize_log() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(LOG_PATH, index=False)
        return

    # Lightweight schema migration so older logs do not break appends.
    history = pd.read_csv(LOG_PATH)
    missing = [col for col in LOG_COLUMNS if col not in history.columns]
    if missing:
        for col in missing:
            history[col] = pd.NA
        history = history.reindex(columns=LOG_COLUMNS)
        history.to_csv(LOG_PATH, index=False)


def initialize_suggested_log() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SUGGESTED_PATH.exists():
        pd.DataFrame(columns=SUGGESTED_COLUMNS).to_csv(SUGGESTED_PATH, index=False)
        return

    history = pd.read_csv(SUGGESTED_PATH)
    missing = [col for col in SUGGESTED_COLUMNS if col not in history.columns]
    if missing:
        for col in missing:
            history[col] = pd.NA
        history = history.reindex(columns=SUGGESTED_COLUMNS)
        history.to_csv(SUGGESTED_PATH, index=False)


def normalize_implied_probs(
    odds: dict[str, float], outcomes: tuple[str, ...] = OUTCOMES
) -> tuple[dict[str, float], float]:
    implied_raw = {outcome: 1 / odds[outcome] for outcome in outcomes}
    overround = sum(implied_raw.values())
    if overround <= 0:
        raise ValueError("Implied probability sum must be positive.")
    implied_norm = {outcome: implied_raw[outcome] / overround for outcome in outcomes}
    return implied_norm, overround


def normalize_probabilities(
    probs: dict[str, float], outcomes: tuple[str, ...] = OUTCOMES
) -> tuple[dict[str, float], float]:
    total = sum(probs[outcome] for outcome in outcomes)
    if total <= 0:
        raise ValueError("Probability sum must be positive.")
    normalized = {outcome: probs[outcome] / total for outcome in outcomes}
    return normalized, total


def compute_fractional_kelly_stake(
    probability: float,
    effective_odds: float,
    bankroll: float,
    max_bet_cap: float,
    kelly_fraction: float,
) -> tuple[float, float, float]:
    """
    Compute full Kelly fraction, applied fractional Kelly, and a capped stake.
    """
    b = effective_odds - 1.0
    q = 1.0 - probability
    if b <= 0:
        return 0.0, 0.0, 0.0

    kelly_full = (b * probability - q) / b
    kelly_full = max(kelly_full, 0.0)
    kelly_fractional = kelly_full * kelly_fraction
    stake = bankroll * kelly_fractional
    stake_capped = min(stake, max_bet_cap)
    return kelly_full, kelly_fractional, stake_capped


def calculate_value_edges(
    polymarket_probs: dict[str, float] | None,
    use_polymarket: bool,
    tax_rate: float,
    bookmakers: list[dict[str, float]],
    bankroll: float,
    max_bet_cap: float,
    kelly_fraction: float,
    outcomes: tuple[str, ...] = OUTCOMES,
) -> tuple[pd.DataFrame, dict[str, float] | None, dict[str, float]]:
    polymarket_overround = (
        sum(polymarket_probs.values()) if use_polymarket and polymarket_probs else 0.0
    )
    rows: list[dict[str, float | str | bool]] = []
    bookmaker_data: list[dict[str, object]] = []

    for bookmaker in bookmakers:
        odds_home = bookmaker["home_odds"]
        odds_draw = bookmaker.get("draw_odds")
        odds_away = bookmaker["away_odds"]
        odds_map = {"home": odds_home, "away": odds_away}
        if "draw" in outcomes:
            odds_map["draw"] = float(odds_draw)
        implied_norm, overround = normalize_implied_probs(odds_map, outcomes)
        bookmaker_data.append(
            {
                "name": bookmaker["name"],
                "odds_home": odds_home,
                "odds_draw": odds_draw,
                "odds_away": odds_away,
                "odds_map": odds_map,
                "implied_norm": implied_norm,
                "overround": overround,
            }
        )

    if bookmaker_data:
        avg_probs = {
            outcome: float(
                pd.Series(
                    [bd["implied_norm"][outcome] for bd in bookmaker_data]  # type: ignore[index]
                ).mean()
            )
            for outcome in outcomes
        }
    else:
        avg_probs = {outcome: 0.0 for outcome in outcomes}

    for bd in bookmaker_data:
        name = str(bd["name"])
        odds_home = float(bd["odds_home"])
        odds_draw = float(bd["odds_draw"]) if bd["odds_draw"] not in (None, pd.NA) else pd.NA
        odds_away = float(bd["odds_away"])
        odds_map = bd["odds_map"]  # type: ignore[assignment]
        implied_norm = bd["implied_norm"]  # type: ignore[assignment]
        overround = float(bd["overround"])

        for outcome in outcomes:
            odds_selected = float(odds_map[outcome])
            effective_odds = odds_selected * (1 - tax_rate)
            polymarket_implied_selected = (
                polymarket_probs[outcome] if use_polymarket and polymarket_probs else pd.NA
            )
            avg_implied_selected = avg_probs[outcome]
            fair_odds_polymarket = (
                1 / polymarket_implied_selected
                if use_polymarket and polymarket_probs
                else pd.NA
            )
            fair_odds_avg = 1 / avg_implied_selected if avg_implied_selected > 0 else pd.NA
            bookmaker_implied_selected = float(implied_norm[outcome])
            edge_vs_polymarket = (
                (effective_odds / fair_odds_polymarket) - 1
                if use_polymarket and polymarket_probs
                else pd.NA
            )
            edge_vs_avg = (
                (effective_odds / fair_odds_avg) - 1 if avg_implied_selected > 0 else pd.NA
            )
            edge_vs_avg_val = float(edge_vs_avg) if not pd.isna(edge_vs_avg) else None

            edge_candidates: list[float] = []
            if use_polymarket and not pd.isna(edge_vs_polymarket):
                edge_candidates.append(float(edge_vs_polymarket))
            if edge_vs_avg_val is not None:
                edge_candidates.append(edge_vs_avg_val)
            edge = max(edge_candidates) if edge_candidates else 0.0

            pm_positive = (
                (not pd.isna(edge_vs_polymarket)) and (float(edge_vs_polymarket) > 0)
                if use_polymarket
                else False
            )
            avg_positive = (edge_vs_avg_val is not None) and (edge_vs_avg_val > 0)
            if pm_positive and avg_positive:
                mispricing_source = "Polymarket & Market Average"
            elif pm_positive:
                mispricing_source = "Polymarket"
            elif avg_positive:
                mispricing_source = "Market Average"
            else:
                mispricing_source = "None"

            use_avg_reference = (
                not use_polymarket
                or edge_vs_avg_val is None
                or (
                    not pd.isna(edge_vs_polymarket)
                    and edge_vs_avg_val > float(edge_vs_polymarket)
                )
            )
            reference_probability = (
                avg_implied_selected
                if use_avg_reference
                else float(polymarket_implied_selected)
            )

            kelly_full, kelly_fractional, stake_capped = compute_fractional_kelly_stake(
                probability=reference_probability,
                effective_odds=effective_odds,
                bankroll=bankroll,
                max_bet_cap=max_bet_cap,
                kelly_fraction=kelly_fraction,
            )
            rows.append(
                {
                    "outcome": outcome,
                    "selected_outcome": outcome,
                    "bookmaker_name": name,
                    "bookmaker_home_odds": odds_home,
                    "bookmaker_draw_odds": odds_draw,
                    "bookmaker_away_odds": odds_away,
                    "bookmaker_overround": overround,
                    "bookmaker_implied_home": implied_norm.get("home", pd.NA),
                    "bookmaker_implied_draw": implied_norm.get("draw", pd.NA),
                    "bookmaker_implied_away": implied_norm.get("away", pd.NA),
                    "bookmaker_implied_selected": bookmaker_implied_selected,
                    "bookmaker_odds_selected": odds_selected,
                    "effective_odds": effective_odds,
                    "polymarket_implied_home": (
                        polymarket_probs.get("home", pd.NA)
                        if use_polymarket and polymarket_probs
                        else pd.NA
                    ),
                    "polymarket_implied_draw": (
                        polymarket_probs.get("draw", pd.NA)
                        if use_polymarket and polymarket_probs
                        else pd.NA
                    ),
                    "polymarket_implied_away": (
                        polymarket_probs.get("away", pd.NA)
                        if use_polymarket and polymarket_probs
                        else pd.NA
                    ),
                    "polymarket_implied_selected": polymarket_implied_selected,
                    "avg_bookmaker_implied_selected": avg_implied_selected,
                    "fair_odds_polymarket": fair_odds_polymarket,
                    "fair_odds_avg_bookmaker": fair_odds_avg,
                    "edge_vs_polymarket": edge_vs_polymarket,
                    "edge_vs_avg_bookmaker": edge_vs_avg,
                    "edge": edge,
                    "mispricing_source": mispricing_source,
                    "is_mispriced": edge > 0,
                    "kelly_full": kelly_full,
                    "kelly_fractional": kelly_fractional,
                    "recommended_stake": stake_capped,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["suggested_bet"] = df["edge"] > 0
        df["avg_bookmaker_implied_selected"] = df["outcome"].map(avg_probs)

    df["polymarket_overround"] = polymarket_overround

    return df.sort_values("edge", ascending=False), polymarket_probs, avg_probs


def append_log(
    match: str,
    market: str,
    bankroll: float,
    max_bet_cap: float,
    kelly_fraction: float,
    polymarket_probs_input: dict[str, float],
    polymarket_probs: dict[str, float] | None,
    avg_bookmaker_probs: dict[str, float],
    tax_rate: float,
    results: pd.DataFrame,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    polymarket_selected = (
        results["outcome"].map(polymarket_probs)
        if polymarket_probs
        else pd.Series([pd.NA] * len(results))
    )
    avg_bookmaker_selected = results["outcome"].map(avg_bookmaker_probs)
    log_rows = results.assign(
        timestamp_utc=timestamp,
        match=match,
        market=market,
        tax_rate=tax_rate,
        bankroll=bankroll,
        max_bet_cap=max_bet_cap,
        kelly_fraction=kelly_fraction,
        polymarket_home_odds=polymarket_probs_input["home"],
        polymarket_draw_odds=polymarket_probs_input.get("draw", pd.NA),
        polymarket_away_odds=polymarket_probs_input["away"],
        polymarket_implied_home=(
            polymarket_probs["home"] if polymarket_probs else pd.NA
        ),
        polymarket_implied_draw=(
            polymarket_probs.get("draw", pd.NA) if polymarket_probs else pd.NA
        ),
        polymarket_implied_away=(
            polymarket_probs["away"] if polymarket_probs else pd.NA
        ),
        polymarket_implied_selected=polymarket_selected if polymarket_probs else pd.NA,
        avg_bookmaker_implied_home=avg_bookmaker_probs.get("home", pd.NA),
        avg_bookmaker_implied_draw=avg_bookmaker_probs.get("draw", pd.NA),
        avg_bookmaker_implied_away=avg_bookmaker_probs.get("away", pd.NA),
        avg_bookmaker_implied_selected=avg_bookmaker_selected,
    )
    # Ensure selected_outcome is populated per row for backward compatibility.
    log_rows["selected_outcome"] = log_rows["outcome"]
    log_rows = log_rows.reindex(columns=LOG_COLUMNS)
    log_rows.to_csv(LOG_PATH, mode="a", index=False, header=False)


def append_suggested_bets(suggested_rows: pd.DataFrame) -> int:
    if suggested_rows.empty:
        return 0
    saved_at = datetime.now(timezone.utc).isoformat()
    to_save = suggested_rows.assign(
        saved_at_utc=saved_at,
        stake_actual=pd.to_numeric(suggested_rows["recommended_stake"], errors="coerce"),
        placed=False,
        placed_at_utc=pd.NA,
        result="pending",
        settled_at_utc=pd.NA,
        pnl=pd.NA,
    )
    to_save = to_save.reindex(columns=SUGGESTED_COLUMNS)
    to_save.to_csv(SUGGESTED_PATH, mode="a", index=False, header=False)
    return int(len(to_save))


def clear_saved_requests() -> None:
    pd.DataFrame(columns=LOG_COLUMNS).to_csv(LOG_PATH, index=False)


@st.cache_data(show_spinner=False)
def load_league_schedule(league_key: str) -> pd.DataFrame:
    meta = TOP5_LEAGUES[league_key]
    path = meta["path"]
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "league_key",
                "league_label",
                "date",
                "time",
                "home_team",
                "away_team",
                "match",
                "match_label",
            ]
        )

    df = pd.read_csv(path)
    if df.empty:
        return df

    df = df.copy()
    df["league_key"] = league_key
    df["league_label"] = meta["label"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    today_utc = datetime.now(timezone.utc).date()
    df = df[df["date"].dt.date >= today_utc]
    df["match"] = df["home_team"] + " vs " + df["away_team"]
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    df["time_str"] = df["time"].fillna("").astype(str)
    df["match_label"] = (
        df["date_str"]
        + " "
        + df["time_str"]
        + " • "
        + df["home_team"]
        + " vs "
        + df["away_team"]
    )
    return df


@st.cache_data(show_spinner=False)
def load_nba_schedule() -> pd.DataFrame:
    try:
        with urlopen(NBA_SCHEDULE_URL, timeout=10) as resp:
            data = json.load(resp)
    except Exception:
        return pd.DataFrame(
            columns=[
                "date",
                "time",
                "home_team",
                "away_team",
                "match",
                "match_label",
            ]
        )

    game_dates = data.get("leagueSchedule", {}).get("gameDates", [])
    rows = []
    for entry in game_dates:
        for game in entry.get("games", []):
            game_time = game.get("gameDateTimeUTC")
            home = game.get("homeTeam", {})
            away = game.get("awayTeam", {})
            home_name = f"{home.get('teamCity', '')} {home.get('teamName', '')}".strip()
            away_name = f"{away.get('teamCity', '')} {away.get('teamName', '')}".strip()
            rows.append(
                {
                    "date": game_time,
                    "time": "",
                    "home_team": home_name,
                    "away_team": away_name,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "time",
                "home_team",
                "away_team",
                "match",
                "match_label",
            ]
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    today_utc = datetime.now(timezone.utc)
    df = df[df["date"] >= today_utc]
    df["match"] = df["home_team"] + " vs " + df["away_team"]
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    df["time_str"] = df["date"].dt.strftime("%H:%M")
    df["match_label"] = (
        df["date_str"] + " " + df["time_str"] + " • " + df["home_team"] + " vs " + df["away_team"]
    )
    return df


def upcoming_matches_for_league(league_key: str, limit: int = 10) -> pd.DataFrame:
    schedule = load_league_schedule(league_key)
    if schedule.empty:
        return schedule
    schedule = schedule.sort_values(["date", "time"], ascending=[True, True])
    return schedule.head(limit)


def delete_rows(path: Path, columns: list[str], indices: list[int]) -> int:
    if not indices:
        return 0
    df = pd.read_csv(path)
    if df.empty:
        return 0
    remaining = df.drop(index=indices, errors="ignore")
    remaining = remaining.reindex(columns=columns)
    remaining.to_csv(path, index=False)
    return int(len(df) - len(remaining))


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def compute_trade_pnl(row: pd.Series) -> float | None:
    result = str(row.get("result", "pending")).lower()
    stake_val = row.get("stake_actual", row.get("recommended_stake", 0.0))
    stake = float(stake_val or 0.0)
    eff_odds = float(row.get("effective_odds", 0.0) or 0.0)
    if stake <= 0 or eff_odds <= 1:
        return 0.0
    if result == "won":
        return stake * (eff_odds - 1.0)
    if result == "lost":
        return -stake
    return None


def _row_stake(row: pd.Series) -> float:
    stake_val = row.get("stake_actual", row.get("recommended_stake", 0.0))
    if pd.isna(stake_val):
        stake_val = row.get("recommended_stake", 0.0)
    return float(stake_val or 0.0)


def update_suggested_bet_status(
    index: int,
    *,
    placed: bool | None = None,
    result: str | None = None,
) -> tuple[bool, str | None]:
    df = pd.read_csv(SUGGESTED_PATH)
    if df.empty or index not in df.index:
        return False, "Suggested bets file is empty or the row no longer exists."

    df["placed"] = df["placed"].map(_coerce_bool)
    df["stake_used"] = df.apply(_row_stake, axis=1)

    row = df.loc[index]
    prior_placed = bool(row.get("placed", False))
    match_name = str(row.get("match", ""))
    stake_here = float(row.get("stake_used", 0.0))
    max_bet_cap = float(row.get("max_bet_cap", 0.0) or 0.0)
    per_event_cap = 2.0 * max_bet_cap if max_bet_cap > 0 else 0.0

    if placed and not prior_placed:
        placed_same_match = df[
            (df.index != index) & (df["placed"]) & (df["match"].astype(str) == match_name)
        ]
        current_exposure = float(placed_same_match["stake_used"].sum())
        if per_event_cap > 0 and (current_exposure + stake_here) > per_event_cap + 1e-9:
            return (
                False,
                (
                    f"Per-event cap exceeded for '{match_name}'. "
                    f"Cap={per_event_cap:.2f}, current placed={current_exposure:.2f}, "
                    f"this stake={stake_here:.2f}."
                ),
            )

    now_utc = datetime.now(timezone.utc).isoformat()
    if placed is not None:
        df.at[index, "placed"] = bool(placed)
        if placed:
            if pd.isna(df.at[index, "placed_at_utc"]):
                df.at[index, "placed_at_utc"] = now_utc
        else:
            df.at[index, "placed_at_utc"] = pd.NA

    if result is not None:
        df.at[index, "result"] = result
        if result in {"won", "lost"}:
            df.at[index, "settled_at_utc"] = now_utc
        else:
            df.at[index, "settled_at_utc"] = pd.NA

    pnl_value = compute_trade_pnl(df.loc[index])
    df.at[index, "pnl"] = pnl_value if pnl_value is not None else pd.NA

    df = df.reindex(columns=SUGGESTED_COLUMNS)
    df.to_csv(SUGGESTED_PATH, index=False)
    return True, None


def update_suggested_bet_stake(index: int, stake_actual: float) -> bool:
    df = pd.read_csv(SUGGESTED_PATH)
    if df.empty or index not in df.index:
        return False
    df.at[index, "stake_actual"] = float(stake_actual)
    pnl_value = compute_trade_pnl(df.loc[index])
    df.at[index, "pnl"] = pnl_value if pnl_value is not None else pd.NA
    df = df.reindex(columns=SUGGESTED_COLUMNS)
    df.to_csv(SUGGESTED_PATH, index=False)
    return True


def portfolio_metrics(df: pd.DataFrame, starting_bankroll: float) -> dict[str, float]:
    if df.empty:
        return {
            "settled_pnl": 0.0,
            "bankroll_current": starting_bankroll,
            "total_return_pct": 0.0,
            "avg_trade_return_pct": 0.0,
            "settled_count": 0.0,
        }

    if "placed" in df.columns:
        df = df.copy()
        df["placed"] = df["placed"].map(_coerce_bool)
        placed_mask = df["placed"]
    else:
        placed_mask = pd.Series([True] * len(df), index=df.index)

    result_lower = df["result"].astype(str).str.lower()
    settled_mask = result_lower.isin(["won", "lost"]) & placed_mask
    settled = df[settled_mask].copy()
    if settled.empty:
        return {
            "settled_pnl": 0.0,
            "bankroll_current": starting_bankroll,
            "total_return_pct": 0.0,
            "avg_trade_return_pct": 0.0,
            "settled_count": 0.0,
        }

    settled["stake"] = pd.to_numeric(settled["recommended_stake"], errors="coerce").fillna(0.0)
    if "stake_actual" in settled.columns:
        settled["stake"] = pd.to_numeric(settled["stake_actual"], errors="coerce").fillna(
            settled["stake"]
        )
    settled["pnl_val"] = pd.to_numeric(settled["pnl"], errors="coerce").fillna(0.0)
    settled_pnl = float(settled["pnl_val"].sum())
    bankroll_current = float(starting_bankroll + settled_pnl)
    total_return_pct = float(settled_pnl / starting_bankroll) if starting_bankroll > 0 else 0.0
    per_trade_return = settled["pnl_val"] / settled["stake"].replace(0, pd.NA)
    avg_trade_return_pct = float(per_trade_return.dropna().mean()) if not per_trade_return.dropna().empty else 0.0

    return {
        "settled_pnl": settled_pnl,
        "bankroll_current": bankroll_current,
        "total_return_pct": total_return_pct,
        "avg_trade_return_pct": avg_trade_return_pct,
        "settled_count": float(len(settled)),
    }


def risk_metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "avg_stake_placed": 0.0,
            "std_stake_placed": 0.0,
            "risk_at_stake_open": 0.0,
            "open_bets_count": 0.0,
            "max_event_exposure": 0.0,
            "avg_event_exposure": 0.0,
        }

    work = df.copy()
    work["placed"] = work["placed"].map(_coerce_bool)
    work["stake_used"] = work.apply(_row_stake, axis=1)
    result_lower = work["result"].astype(str).str.lower()
    settled_mask = result_lower.isin(["won", "lost"])
    placed_mask = work["placed"]

    placed_bets = work[placed_mask]
    if placed_bets.empty:
        return {
            "avg_stake_placed": 0.0,
            "std_stake_placed": 0.0,
            "risk_at_stake_open": 0.0,
            "open_bets_count": 0.0,
            "max_event_exposure": 0.0,
            "avg_event_exposure": 0.0,
        }

    avg_stake_placed = float(placed_bets["stake_used"].mean())
    std_stake_placed = float(placed_bets["stake_used"].std(ddof=0) or 0.0)

    open_mask = placed_mask & (~settled_mask)
    open_bets = work[open_mask]
    risk_at_stake_open = float(open_bets["stake_used"].sum())
    open_bets_count = float(len(open_bets))

    exposure_by_event = placed_bets.groupby("match")["stake_used"].sum()
    max_event_exposure = float(exposure_by_event.max() or 0.0)
    avg_event_exposure = float(exposure_by_event.mean() or 0.0)

    return {
        "avg_stake_placed": avg_stake_placed,
        "std_stake_placed": std_stake_placed,
        "risk_at_stake_open": risk_at_stake_open,
        "open_bets_count": open_bets_count,
        "max_event_exposure": max_event_exposure,
        "avg_event_exposure": avg_event_exposure,
    }


def bulk_rows_template(
    outcomes: tuple[str, ...],
    rows: int,
    bookmaker_names: list[str],
    matches: list[str] | None = None,
) -> pd.DataFrame:
    match_list = matches or ["" for _ in range(rows)]
    if len(match_list) < rows:
        match_list = match_list + [""] * (rows - len(match_list))

    base: dict[str, list[object]] = {
        "match": match_list[:rows],
        "pm_home": [0.0 for _ in range(rows)],
        "pm_away": [0.0 for _ in range(rows)],
    }
    if "draw" in outcomes:
        base["pm_draw"] = [0.0 for _ in range(rows)]

    for idx, name in enumerate(bookmaker_names):
        base[f"bk{idx+1}_name"] = [name for _ in range(rows)]
        base[f"bk{idx+1}_home"] = [0.0 for _ in range(rows)]
        base[f"bk{idx+1}_away"] = [0.0 for _ in range(rows)]
        if "draw" in outcomes:
            base[f"bk{idx+1}_draw"] = [0.0 for _ in range(rows)]

    return pd.DataFrame(base)


def main() -> None:
    st.set_page_config(page_title="Odds Mispricing Finder", layout="wide")
    st.title("Odds Mispricing Finder")
    st.markdown(
        "Enter full 1X2 odds (home/draw/away) for Polymarket and bookmakers. "
        "We normalize implied probabilities to sum to 1 per market, apply tax, "
        "compute edges for all outcomes, and save results to `data/bet_requests.csv`."
    )

    initialize_log()
    initialize_suggested_log()

    with st.sidebar:
        st.header("Sport")
        sport = st.radio(
            "Select sport",
            options=["Soccer", "NBA"],
            horizontal=True,
        )
        st.header("Leagues")
        upcoming_matches_list: list[str] = []
        if sport == "Soccer":
            league_keys = list(TOP5_LEAGUES.keys())
            league_labels = {k: TOP5_LEAGUES[k]["label"] for k in league_keys}
            selected_league = st.radio(
                "League ribbon",
                options=league_keys,
                format_func=lambda k: league_labels[k],
                horizontal=True,
            )
            upcoming = upcoming_matches_for_league(selected_league, limit=10)
            if upcoming.empty:
                st.caption("No upcoming matches found for this league.")
            else:
                match_options = {
                    row["match_label"]: row["match"] for _, row in upcoming.iterrows()
                }
                selected_match_label = st.selectbox(
                    "Upcoming matches (next 10)",
                    options=list(match_options.keys()),
                )
                st.session_state["match_input"] = match_options[selected_match_label]
                upcoming_matches_list = list(match_options.values())
        else:
            upcoming = load_nba_schedule().sort_values("date").head(10)
            if upcoming.empty:
                st.caption("No upcoming NBA matches found.")
            else:
                match_options = {
                    row["match_label"]: row["match"] for _, row in upcoming.iterrows()
                }
                selected_match_label = st.selectbox(
                    "Upcoming NBA matches (next 10)",
                    options=list(match_options.keys()),
                )
                st.session_state["match_input"] = match_options[selected_match_label]
                upcoming_matches_list = list(match_options.values())

        st.header("Match Details")
        match = st.text_input(
            "Match",
            placeholder="Team A vs Team B",
            key="match_input",
        )
        market = st.text_input("Market name", placeholder="Match odds (1X2)")
        tax_rate = st.number_input(
            "Tax rate (decimal)",
            min_value=0.0,
            max_value=0.99,
            value=0.05,
            step=0.01,
            format="%.2f",
            help="Enter 0.05 for 5% tax on winnings.",
        )
        st.header("Staking")
        bankroll = st.number_input(
            "Betting capacity / bankroll",
            min_value=0.0,
            value=3000.0,
            step=10.0,
            format="%.2f",
            help="Total bankroll available for staking recommendations.",
        )
        max_bet_cap = st.number_input(
            "Max bet cap",
            min_value=0.0,
            value=100.0,
            step=5.0,
            format="%.2f",
            help="Hard cap per suggested bet.",
        )
        kelly_fraction = st.number_input(
            "Kelly fraction",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            format="%.2f",
            help="0.50 means half-Kelly staking.",
        )
        st.header("Market Type")
        two_way_market = st.checkbox(
            "Two-way market (no draw)",
            value=(sport == "NBA"),
            help="Use this for NBA or other sports without draws.",
        )
        outcomes = ("home", "away") if two_way_market else OUTCOMES

        st.header("Bookmakers (apply with Update)")
        default_bookmakers = ["Tipico", "bwin", "bet365"]
        if "pending_bookmakers" not in st.session_state:
            st.session_state["pending_bookmakers"] = default_bookmakers
        if "applied_bookmakers" not in st.session_state:
            st.session_state["applied_bookmakers"] = default_bookmakers

        pending_bookmakers = st.multiselect(
            "Select bookmakers",
            options=GERMAN_BOOKMAKERS,
            default=st.session_state["pending_bookmakers"],
        )
        if st.button("Update", key="apply_bookmakers"):
            st.session_state["pending_bookmakers"] = pending_bookmakers
            st.session_state["applied_bookmakers"] = pending_bookmakers

        applied_bookmakers = st.session_state.get("applied_bookmakers", default_bookmakers)
        if not applied_bookmakers:
            applied_bookmakers = ["Bookmaker 1"]

    st.subheader("Polymarket Probabilities")
    use_polymarket = st.checkbox(
        "Use Polymarket probabilities (if available)",
        value=True,
        help="Disable this when Polymarket has no market for the match.",
    )
    pm_cols = st.columns(3 if "draw" in outcomes else 2)
    with pm_cols[0]:
        pm_home = st.number_input(
            "Polymarket home probability",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.01,
            format="%.2f",
            key="pm_home",
            disabled=not use_polymarket,
        )
    if "draw" in outcomes:
        with pm_cols[1]:
            pm_draw = st.number_input(
                "Polymarket draw probability",
                min_value=0.0,
                max_value=1.0,
                value=0.27,
                step=0.01,
                format="%.2f",
                key="pm_draw",
                disabled=not use_polymarket,
            )
        with pm_cols[2]:
            pm_away = st.number_input(
                "Polymarket away probability",
                min_value=0.0,
                max_value=1.0,
                value=0.28,
                step=0.01,
                format="%.2f",
                key="pm_away",
                disabled=not use_polymarket,
            )
    else:
        pm_draw = 0.0
        with pm_cols[1]:
            pm_away = st.number_input(
                "Polymarket away probability",
                min_value=0.0,
                max_value=1.0,
                value=0.55,
                step=0.01,
                format="%.2f",
                key="pm_away",
                disabled=not use_polymarket,
            )
    polymarket_probs_input = {"home": pm_home, "draw": pm_draw, "away": pm_away}
    if use_polymarket:
        polymarket_probs, polymarket_sum = normalize_probabilities(
            polymarket_probs_input, outcomes
        )
        if abs(polymarket_sum - 1.0) > 0.02:
            st.warning(
                "Polymarket probabilities do not sum to 1 "
                f"(sum={polymarket_sum:.3f}). We normalize them to 1 internally."
            )
        else:
            st.caption(
                "Polymarket probabilities are close to summing to 1 "
                f"(sum={polymarket_sum:.3f})."
            )
    else:
        polymarket_probs = None
        st.info("Polymarket disabled for this match. Using Market Average only.")

    st.subheader("Bookmaker Odds")
    bookmaker_entries = []
    cols = st.columns(2)
    bookmaker_count = min(len(applied_bookmakers), MAX_BOOKMAKERS)
    for idx in range(bookmaker_count):
        with cols[idx % 2]:
            name = applied_bookmakers[idx]
            st.markdown(f"**{name}**")

            home_odds = st.number_input(
                f"{name} home odds",
                min_value=1.01,
                value=2.3,
                step=0.01,
                format="%.2f",
                key=f"home_odds_{idx}",
            )
            if "draw" in outcomes:
                draw_odds = st.number_input(
                    f"{name} draw odds",
                    min_value=1.01,
                    value=3.5,
                    step=0.01,
                    format="%.2f",
                    key=f"draw_odds_{idx}",
                )
            else:
                draw_odds = pd.NA
            away_odds = st.number_input(
                f"{name} away odds",
                min_value=1.01,
                value=3.0,
                step=0.01,
                format="%.2f",
                key=f"away_odds_{idx}",
            )
            bookmaker_entries.append(
                {
                    "name": name,
                    "home_odds": home_odds,
                    "draw_odds": draw_odds,
                    "away_odds": away_odds,
                }
            )

    st.subheader("Bulk Odds Entry (up to 10 matches)")
    st.caption(
        "Paste odds for up to 10 matches. One row = one match. "
        "Columns are created for each bookmaker you selected."
    )
    bulk_enabled = st.toggle("Enable bulk entry", value=False)
    if bulk_enabled:
        if (
            "bulk_rows" not in st.session_state
            or st.session_state.get("bulk_outcomes") != outcomes
            or st.session_state.get("bulk_bookmaker_names") != applied_bookmakers
        ):
            st.session_state["bulk_rows"] = bulk_rows_template(
                outcomes,
                rows=min(10, max(1, len(upcoming_matches_list) or 10)),
                bookmaker_names=applied_bookmakers,
                matches=upcoming_matches_list,
            )
            st.session_state["bulk_outcomes"] = outcomes
            st.session_state["bulk_bookmaker_names"] = applied_bookmakers
        bulk_df = st.data_editor(
            st.session_state["bulk_rows"],
            num_rows="fixed",
            use_container_width=True,
        )
        save_bulk_suggested = st.checkbox(
            "Save suggested bets from bulk",
            value=False,
        )
        if st.button("Run bulk analysis"):
            summaries = []
            for _, row in bulk_df.iterrows():
                match_name = str(row.get("match", "")).strip()
                if not match_name:
                    continue

                pm_probs_input = {
                    "home": float(row.get("pm_home", 0.0)),
                    "away": float(row.get("pm_away", 0.0)),
                }
                if "draw" in outcomes:
                    pm_probs_input["draw"] = float(row.get("pm_draw", 0.0))
                else:
                    pm_probs_input["draw"] = 0.0

                if use_polymarket:
                    pm_probs, _ = normalize_probabilities(pm_probs_input, outcomes)
                else:
                    pm_probs = None

                bookmaker_rows = []
                for idx in range(bookmaker_count):
                    name = str(row.get(f"bk{idx+1}_name", f"Bookmaker {idx+1}")).strip()
                    home_val = float(row.get(f"bk{idx+1}_home", 0.0) or 0.0)
                    away_val = float(row.get(f"bk{idx+1}_away", 0.0) or 0.0)
                    if "draw" in outcomes:
                        draw_val = float(row.get(f"bk{idx+1}_draw", 0.0) or 0.0)
                    else:
                        draw_val = pd.NA
                    if not name or home_val <= 0 or away_val <= 0:
                        continue
                    if "draw" in outcomes and (draw_val is pd.NA or draw_val <= 0):
                        continue
                    bookmaker_rows.append(
                        {
                            "name": name,
                            "home_odds": home_val,
                            "draw_odds": draw_val,
                            "away_odds": away_val,
                        }
                    )
                if not bookmaker_rows:
                    continue

                results_bulk, _, avg_probs_bulk = calculate_value_edges(
                    pm_probs,
                    use_polymarket,
                    tax_rate,
                    bookmaker_rows,
                    bankroll,
                    max_bet_cap,
                    kelly_fraction,
                    outcomes,
                )
                if results_bulk.empty:
                    continue

                append_log(
                    match_name,
                    market,
                    bankroll,
                    max_bet_cap,
                    kelly_fraction,
                    pm_probs_input,
                    pm_probs,
                    avg_probs_bulk,
                    tax_rate,
                    results_bulk,
                )

                suggested_bulk = results_bulk[results_bulk["suggested_bet"]]
                if save_bulk_suggested and not suggested_bulk.empty:
                    suggested_bulk = suggested_bulk.assign(
                        match=match_name,
                        market=market,
                        tax_rate=tax_rate,
                        bankroll=bankroll,
                        max_bet_cap=max_bet_cap,
                        kelly_fraction=kelly_fraction,
                    )
                    append_suggested_bets(
                        suggested_bulk[
                            [
                                "match",
                                "market",
                                "outcome",
                                "bookmaker_name",
                                "bookmaker_odds_selected",
                                "effective_odds",
                                "edge_vs_polymarket",
                                "edge_vs_avg_bookmaker",
                                "edge",
                                "mispricing_source",
                                "recommended_stake",
                                "tax_rate",
                                "bankroll",
                                "max_bet_cap",
                                "kelly_fraction",
                                "polymarket_implied_selected",
                                "avg_bookmaker_implied_selected",
                            ]
                        ]
                    )

                best = results_bulk.iloc[0]
                summaries.append(
                    {
                        "match": match_name,
                        "outcome": best["outcome"],
                        "bookmaker": best["bookmaker_name"],
                        "edge": best["edge"],
                        "stake": best["recommended_stake"],
                    }
                )

            if summaries:
                st.success("Bulk analysis complete.")
                st.dataframe(pd.DataFrame(summaries), use_container_width=True)
            else:
                st.warning("No valid rows found in bulk input.")

    # Persist results across reruns so follow-up actions (like saving suggested bets)
    # still have access to the latest analysis.
    if "last_results" not in st.session_state:
        st.session_state["last_results"] = None
        st.session_state["last_polymarket_probs"] = None
        st.session_state["last_avg_bookmaker_probs"] = None
        st.session_state["last_context"] = None

    analyze_clicked = st.button("Analyze and save")
    if analyze_clicked:
        try:
            results, polymarket_probs, avg_bookmaker_probs = calculate_value_edges(
                polymarket_probs,
                use_polymarket,
                tax_rate,
                bookmaker_entries,
                bankroll,
                max_bet_cap,
                kelly_fraction,
                outcomes,
            )
        except ValueError as exc:
            st.error(f"Invalid odds: {exc}")
            return

        if results.empty:
            st.warning("Add at least one bookmaker.")
        else:
            append_log(
                match,
                market,
                bankroll,
                max_bet_cap,
                kelly_fraction,
                polymarket_probs_input,
                polymarket_probs,
                avg_bookmaker_probs,
                tax_rate,
                results,
            )
            st.session_state["last_results"] = results
            st.session_state["last_polymarket_probs"] = polymarket_probs
            st.session_state["last_avg_bookmaker_probs"] = avg_bookmaker_probs
            st.session_state["last_context"] = {
                "match": match,
                "market": market,
                "tax_rate": tax_rate,
                "bankroll": bankroll,
                "max_bet_cap": max_bet_cap,
                "kelly_fraction": kelly_fraction,
            }
            st.success("Analysis saved to data/bet_requests.csv")

    results = st.session_state.get("last_results")
    polymarket_probs = st.session_state.get("last_polymarket_probs")
    avg_bookmaker_probs = st.session_state.get("last_avg_bookmaker_probs")
    last_context = st.session_state.get("last_context")

    if isinstance(results, pd.DataFrame) and avg_bookmaker_probs:
        st.subheader("Market Metrics")
        pm_cols = st.columns(3)
        if polymarket_probs:
            pm_cols[0].metric("Polymarket home", f"{polymarket_probs['home']:.2%}")
            if "draw" in outcomes:
                pm_cols[1].metric("Polymarket draw", f"{polymarket_probs['draw']:.2%}")
                pm_cols[2].metric("Polymarket away", f"{polymarket_probs['away']:.2%}")
            else:
                pm_cols[1].metric("Polymarket away", f"{polymarket_probs['away']:.2%}")
                pm_cols[2].metric("Polymarket draw", "N/A")
        else:
            pm_cols[0].metric("Polymarket home", "N/A")
            pm_cols[1].metric("Polymarket draw", "N/A")
            pm_cols[2].metric("Polymarket away", "N/A")
        avg_cols = st.columns(3)
        avg_cols[0].metric("Avg bookmaker home", f"{avg_bookmaker_probs['home']:.2%}")
        if "draw" in outcomes:
            avg_cols[1].metric("Avg bookmaker draw", f"{avg_bookmaker_probs['draw']:.2%}")
            avg_cols[2].metric("Avg bookmaker away", f"{avg_bookmaker_probs['away']:.2%}")
        else:
            avg_cols[1].metric("Avg bookmaker draw", "N/A")
            avg_cols[2].metric("Avg bookmaker away", f"{avg_bookmaker_probs['away']:.2%}")

        st.subheader("Mispriced Odds Results")
        show_results_table = st.toggle("Show full results table", value=False)
        if show_results_table:
            st.dataframe(
                results.style.format(
                    {
                        "recommended_stake": "{:.2f}",
                        "bookmaker_home_odds": "{:.2f}",
                        "bookmaker_draw_odds": "{:.2f}",
                        "bookmaker_away_odds": "{:.2f}",
                        "effective_odds": "{:.2f}",
                        "bookmaker_overround": "{:.2f}",
                        "polymarket_implied_home": "{:.2%}",
                        "polymarket_implied_draw": "{:.2%}",
                        "polymarket_implied_away": "{:.2%}",
                        "bookmaker_implied_home": "{:.2%}",
                        "bookmaker_implied_draw": "{:.2%}",
                        "bookmaker_implied_away": "{:.2%}",
                        "polymarket_implied_selected": "{:.2%}",
                        "bookmaker_implied_selected": "{:.2%}",
                        "bookmaker_odds_selected": "{:.2f}",
                        "fair_odds_polymarket": "{:.2f}",
                        "fair_odds_avg_bookmaker": "{:.2f}",
                        "edge_vs_polymarket": "{:.2%}",
                        "edge_vs_avg_bookmaker": "{:.2%}",
                        "kelly_full": "{:.2%}",
                        "kelly_fractional": "{:.2%}",
                        "edge": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )
        suggested = results[results["suggested_bet"]]
        label_map = {"home": "Home win", "draw": "Draw", "away": "Away win"}
        suggested_reset = suggested.reset_index(drop=True).copy()
        if not suggested_reset.empty:
            suggested_reset["outcome_label"] = suggested_reset["outcome"].map(label_map)
        st.session_state["last_suggested_reset"] = suggested_reset

        if suggested["is_mispriced"].any():
            st.info("Profitable bets found (edge > 0):")
            for _, row in suggested.iterrows():
                outcome_label = label_map[row["outcome"]]
                source_label = (
                    "Market Average" if not use_polymarket else row["mispricing_source"]
                )
                st.write(
                    f"- **{outcome_label}** at **{row['bookmaker_name']}** "
                    f"odds **{row['bookmaker_odds_selected']:.2f}** "
                    f"(edge {row['edge']:.2%}, source {source_label}, "
                    f"stake {row['recommended_stake']:.2f})"
                )

            suggested_display = suggested[
                [
                    "outcome",
                    "bookmaker_name",
                    "bookmaker_odds_selected",
                    "effective_odds",
                    "polymarket_implied_selected",
                    "avg_bookmaker_implied_selected",
                    "edge_vs_polymarket",
                    "edge_vs_avg_bookmaker",
                    "edge",
                    "mispricing_source",
                    "recommended_stake",
                ]
            ].copy()
            suggested_display["outcome_label"] = suggested_display["outcome"].map(label_map)
            suggested_display = suggested_display.drop(columns=["outcome"])
            show_suggested_table = st.toggle("Show suggested bets table", value=False)
            if show_suggested_table:
                st.dataframe(
                    suggested_display.style.format(
                        {
                            "bookmaker_odds_selected": "{:.2f}",
                            "effective_odds": "{:.2f}",
                            "polymarket_implied_selected": "{:.2%}",
                            "avg_bookmaker_implied_selected": "{:.2%}",
                            "edge_vs_polymarket": "{:.2%}",
                            "edge_vs_avg_bookmaker": "{:.2%}",
                            "edge": "{:.2%}",
                            "recommended_stake": "{:.2f}",
                        }
                    ),
                    use_container_width=True,
                )
        else:
            if not use_polymarket:
                st.info(
                    "No profitable bets after tax. Showing top 3 closest lines "
                    "vs Market Average."
                )
                top_lines = results.sort_values("edge", ascending=False).head(3)
                for _, row in top_lines.iterrows():
                    outcome_label = label_map[row["outcome"]]
                    st.write(
                        f"- **{outcome_label}** at **{row['bookmaker_name']}** "
                        f"odds **{row['bookmaker_odds_selected']:.2f}** "
                        f"(edge {row['edge']:.2%}, source Market Average)"
                    )
            else:
                st.info("No mispriced odds found after tax.")

        st.subheader("Save Suggested Bets")
        last_suggested_reset = st.session_state.get("last_suggested_reset")
        if isinstance(last_suggested_reset, pd.DataFrame) and not last_suggested_reset.empty:
            selection_options = {
                f"{idx + 1}. {row['outcome_label']} @ {row['bookmaker_name']} "
                f"(edge {row['edge']:.2%}, source {row['mispricing_source']}, "
                f"stake {row['recommended_stake']:.2f})": idx
                for idx, row in last_suggested_reset.iterrows()
            }
            selected_labels = st.multiselect(
                "Select suggested bets to save for later execution",
                options=list(selection_options.keys()),
                key="save_suggested_multiselect",
            )
        else:
            selection_options = {}
            selected_labels = st.multiselect(
                "Select suggested bets to save for later execution",
                options=[],
                key="save_suggested_multiselect",
            )

        save_disabled = not selection_options
        if st.button(
            "Save selected suggested bets",
            key="save_suggested_button",
            disabled=save_disabled,
        ):
            if not last_context:
                st.error("No analysis context available. Click 'Analyze and save' first.")
                return
            selected_indices = [selection_options[label] for label in selected_labels]
            selected_df = last_suggested_reset.iloc[selected_indices].copy()
            selected_df = selected_df.assign(**last_context)
            saved_count = append_suggested_bets(
                selected_df[
                    [
                        "match",
                        "market",
                        "outcome",
                        "bookmaker_name",
                        "bookmaker_odds_selected",
                        "effective_odds",
                        "edge_vs_polymarket",
                        "edge_vs_avg_bookmaker",
                        "edge",
                        "mispricing_source",
                        "recommended_stake",
                        "tax_rate",
                        "bankroll",
                        "max_bet_cap",
                        "kelly_fraction",
                        "polymarket_implied_selected",
                        "avg_bookmaker_implied_selected",
                    ]
                ]
            )
            if saved_count:
                st.success(f"Saved {saved_count} suggested bet(s) to data/suggested_bets.csv")
            else:
                st.warning("No suggested bets were selected.")

    st.divider()
    st.subheader("Saved Requests")
    if st.button("Delete saved requests"):
        clear_saved_requests()
        st.success("Saved requests cleared.")
    history = pd.read_csv(LOG_PATH)
    if history.empty:
        st.caption("No saved bets yet.")
    else:
        request_options = {
            (
                f"{idx}. {row.get('timestamp_utc', '')} | {row.get('match', '')} | "
                f"{row.get('market', '')} | {row.get('outcome', '')} @ {row.get('bookmaker_name', '')} "
                f"(edge {row.get('edge', 0):.2%})"
            ): int(idx)
            for idx, row in history.iterrows()
        }
        selected_request_labels = st.multiselect(
            "Select saved requests to delete",
            options=list(request_options.keys()),
        )
        if st.button("Delete selected saved requests"):
            selected_indices = [request_options[label] for label in selected_request_labels]
            deleted = delete_rows(LOG_PATH, LOG_COLUMNS, selected_indices)
            if deleted:
                st.success(f"Deleted {deleted} saved request(s).")
                st.rerun()
            else:
                st.warning("No saved requests were selected.")
        show_requests = st.toggle("Show saved requests table", value=False)
        if show_requests:
            st.dataframe(history, use_container_width=True)

    st.divider()
    st.subheader("Saved Suggested Bets")
    suggested_history = pd.read_csv(SUGGESTED_PATH)
    if suggested_history.empty:
        st.caption("No suggested bets saved yet.")
    else:
        suggested_options = {
            (
                f"{idx}. {row.get('saved_at_utc', '')} | {row.get('match', '')} | "
                f"{row.get('outcome', '')} @ {row.get('bookmaker_name', '')} "
                f"(edge {row.get('edge', 0):.2%}, stake {row.get('recommended_stake', 0):.2f})"
            ): int(idx)
            for idx, row in suggested_history.iterrows()
        }
        selected_suggested_labels = st.multiselect(
            "Select saved suggested bets to delete",
            options=list(suggested_options.keys()),
        )
        if st.button("Delete selected suggested bets"):
            selected_indices = [suggested_options[label] for label in selected_suggested_labels]
            deleted = delete_rows(SUGGESTED_PATH, SUGGESTED_COLUMNS, selected_indices)
            if deleted:
                st.success(f"Deleted {deleted} saved suggested bet(s).")
                st.rerun()
            else:
                st.warning("No suggested bets were selected.")
        show_saved_suggested = st.toggle("Show saved suggested bets table", value=False)
        if show_saved_suggested:
            st.dataframe(suggested_history, use_container_width=True)

    st.divider()
    st.subheader("Execution & Portfolio")
    suggested_exec = pd.read_csv(SUGGESTED_PATH)
    if suggested_exec.empty:
        st.caption("No suggested bets to execute yet.")
    else:
        suggested_exec["placed"] = suggested_exec["placed"].map(_coerce_bool)
        suggested_exec["stake_used"] = suggested_exec.apply(_row_stake, axis=1)
        suggested_exec["max_bet_cap"] = pd.to_numeric(
            suggested_exec.get("max_bet_cap", 0.0), errors="coerce"
        ).fillna(0.0)
        suggested_exec["per_event_cap"] = 2.0 * suggested_exec["max_bet_cap"]
        placed_exposure = (
            suggested_exec[suggested_exec["placed"]]
            .groupby("match")["stake_used"]
            .sum()
            .to_dict()
        )

        bankroll_candidates = pd.to_numeric(
            suggested_exec.get("bankroll", pd.Series(dtype=float)), errors="coerce"
        ).dropna()
        default_bankroll = (
            float(bankroll_candidates.iloc[-1]) if not bankroll_candidates.empty else 1000.0
        )
        starting_bankroll = st.number_input(
            "Starting bankroll for portfolio tracking",
            min_value=0.0,
            value=default_bankroll,
            step=10.0,
            format="%.2f",
            help="Portfolio metrics use this starting bankroll plus settled PnL.",
        )

        label_map = {"home": "Home win", "draw": "Draw", "away": "Away win"}
        pending_bets = suggested_exec[~suggested_exec["placed"]]
        executed_bets = suggested_exec[suggested_exec["placed"]]

        st.markdown("**Pending Saved Bets (Not Yet Executed)**")
        if pending_bets.empty:
            st.caption("No pending bets.")
        else:
            for idx, row in pending_bets.iterrows():
                outcome_label = label_map.get(str(row.get("outcome", "")), str(row.get("outcome", "")))
                header = (
                    f"{idx}. {row.get('match', '')} • {outcome_label} @ {row.get('bookmaker_name', '')} "
                    f"(edge {float(row.get('edge', 0.0)):.2%})"
                )
                with st.expander(header, expanded=False):
                    stake_actual_raw = row.get("stake_actual", row.get("recommended_stake", 0.0))
                    if pd.isna(stake_actual_raw):
                        stake_actual_raw = row.get("recommended_stake", 0.0)
                    stake_actual_val = float(stake_actual_raw or 0.0)
                    match_name = str(row.get("match", ""))
                    cap = float(row.get("per_event_cap", 0.0) or 0.0)
                    current_exposure = float(placed_exposure.get(match_name, 0.0))
                    remaining = cap - current_exposure if cap > 0 else float("inf")

                    cap_cols = st.columns(3)
                    cap_cols[0].metric("Per-event cap", f"{cap:.2f}" if cap > 0 else "—")
                    cap_cols[1].metric("Placed exposure", f"{current_exposure:.2f}")
                    cap_cols[2].metric("Remaining cap", f"{remaining:.2f}" if cap > 0 else "—")

                    info_cols = st.columns(3)
                    info_cols[0].metric(
                        "Recommended stake", f"{float(row.get('recommended_stake', 0.0)):.2f}"
                    )
                    info_cols[1].metric("Actual stake", f"{stake_actual_val:.2f}")
                    info_cols[2].metric("Effective odds", f"{float(row.get('effective_odds', 0.0)):.2f}")

                    stake_cols = st.columns(2)
                    new_stake = stake_cols[0].number_input(
                        "Edit actual stake",
                        min_value=0.0,
                        value=stake_actual_val,
                        step=1.0,
                        format="%.2f",
                        key=f"stake_actual_pending_{idx}",
                    )
                    if stake_cols[1].button("Save stake", key=f"save_stake_pending_{idx}"):
                        update_suggested_bet_stake(idx, float(new_stake))
                        st.rerun()

                    if st.button("Bet placed (lock in)", key=f"place_pending_{idx}"):
                        success, message = update_suggested_bet_status(idx, placed=True)
                        if success:
                            st.rerun()
                        else:
                            st.error(message or "Could not place bet.")

                    if st.button("Delete pending bet", key=f"delete_pending_{idx}"):
                        deleted = delete_rows(SUGGESTED_PATH, SUGGESTED_COLUMNS, [int(idx)])
                        if deleted:
                            st.success("Pending bet deleted.")
                            st.rerun()
                        else:
                            st.error("Could not delete pending bet.")

        st.markdown("**Executed Bets (Eligible for P&L)**")
        if executed_bets.empty:
            st.caption("No executed bets yet.")
        else:
            for idx, row in executed_bets.iterrows():
                outcome_label = label_map.get(str(row.get("outcome", "")), str(row.get("outcome", "")))
                header = (
                    f"{idx}. {row.get('match', '')} • {outcome_label} @ {row.get('bookmaker_name', '')} "
                    f"(edge {float(row.get('edge', 0.0)):.2%})"
                )
                with st.expander(header, expanded=False):
                    stake_actual_raw = row.get("stake_actual", row.get("recommended_stake", 0.0))
                    if pd.isna(stake_actual_raw):
                        stake_actual_raw = row.get("recommended_stake", 0.0)
                    stake_actual_val = float(stake_actual_raw or 0.0)
                    match_name = str(row.get("match", ""))
                    cap = float(row.get("per_event_cap", 0.0) or 0.0)
                    current_exposure = float(placed_exposure.get(match_name, 0.0))

                    cap_cols = st.columns(3)
                    cap_cols[0].metric("Per-event cap", f"{cap:.2f}" if cap > 0 else "—")
                    cap_cols[1].metric("Placed exposure", f"{current_exposure:.2f}")
                    cap_cols[2].metric("Result", str(row.get("result", "pending")))

                    info_cols = st.columns(3)
                    info_cols[0].metric(
                        "Recommended stake", f"{float(row.get('recommended_stake', 0.0)):.2f}"
                    )
                    info_cols[1].metric("Actual stake", f"{stake_actual_val:.2f}")
                    info_cols[2].metric("Effective odds", f"{float(row.get('effective_odds', 0.0)):.2f}")

                    stake_cols = st.columns(2)
                    new_stake = stake_cols[0].number_input(
                        "Edit actual stake",
                        min_value=0.0,
                        value=stake_actual_val,
                        step=1.0,
                        format="%.2f",
                        key=f"stake_actual_exec_{idx}",
                    )
                    if stake_cols[1].button("Save stake", key=f"save_stake_exec_{idx}"):
                        update_suggested_bet_stake(idx, float(new_stake))
                        st.rerun()

                    btn_cols = st.columns(3)
                    if btn_cols[0].button("Won", key=f"won_{idx}"):
                        success, message = update_suggested_bet_status(idx, placed=True, result="won")
                        if success:
                            st.rerun()
                        else:
                            st.error(message or "Could not mark as won.")
                    if btn_cols[1].button("Lost", key=f"lost_{idx}"):
                        success, message = update_suggested_bet_status(idx, placed=True, result="lost")
                        if success:
                            st.rerun()
                        else:
                            st.error(message or "Could not mark as lost.")
                    if btn_cols[2].button("Reset to pending", key=f"reset_{idx}"):
                        success, message = update_suggested_bet_status(idx, placed=False, result="pending")
                        if success:
                            st.rerun()
                        else:
                            st.error(message or "Could not reset bet.")

        suggested_exec = pd.read_csv(SUGGESTED_PATH)
        suggested_exec["placed"] = suggested_exec["placed"].map(_coerce_bool)
        suggested_exec["pnl"] = pd.to_numeric(suggested_exec["pnl"], errors="coerce")
        metrics = portfolio_metrics(suggested_exec, starting_bankroll)
        metric_cols = st.columns(4)
        metric_cols[0].metric("Settled PnL", f"{metrics['settled_pnl']:.2f}")
        metric_cols[1].metric("Current bankroll", f"{metrics['bankroll_current']:.2f}")
        metric_cols[2].metric("Total return", f"{metrics['total_return_pct']:.2%}")
        metric_cols[3].metric("Avg trade return", f"{metrics['avg_trade_return_pct']:.2%}")

        risk = risk_metrics(suggested_exec)
        risk_cols = st.columns(6)
        risk_cols[0].metric("Avg stake placed", f"{risk['avg_stake_placed']:.2f}")
        risk_cols[1].metric("Stake std dev", f"{risk['std_stake_placed']:.2f}")
        risk_cols[2].metric("Risk at stake (open)", f"{risk['risk_at_stake_open']:.2f}")
        risk_cols[3].metric("Open bets", f"{risk['open_bets_count']:.0f}")
        risk_cols[4].metric("Max event exposure", f"{risk['max_event_exposure']:.2f}")
        risk_cols[5].metric("Avg event exposure", f"{risk['avg_event_exposure']:.2f}")

        show_exec_table = st.toggle("Show execution table", value=False)
        if show_exec_table:
            st.dataframe(suggested_exec, use_container_width=True)


if __name__ == "__main__":
    main()
