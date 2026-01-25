from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st


DATA_DIR = Path("data")
LOG_PATH = DATA_DIR / "bet_requests.csv"
MAX_BOOKMAKERS = 10


def initialize_log() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        pd.DataFrame(
            columns=[
                "request_id",
                "timestamp_utc",
                "match",
                "market",
                "polymarket_odds",
                "tax_rate",
                "bookmaker_name",
                "bookmaker_odds",
                "effective_odds",
                "polymarket_implied_prob",
                "bookmaker_implied_prob",
                "fair_odds",
                "edge",
                "is_mispriced",
                "suggested_bet",
            ]
        ).to_csv(LOG_PATH, index=False)


def calculate_value_edges(
    polymarket_odds: float,
    tax_rate: float,
    bookmakers: list[dict[str, float]],
) -> pd.DataFrame:
    polymarket_implied = 1 / polymarket_odds
    fair_odds = 1 / polymarket_implied
    rows: list[dict[str, float | str | bool]] = []

    for bookmaker in bookmakers:
        odds = bookmaker["odds"]
        effective_odds = odds * (1 - tax_rate)
        bookmaker_implied = 1 / odds
        edge = (effective_odds / fair_odds) - 1
        rows.append(
            {
                "bookmaker_name": bookmaker["name"],
                "bookmaker_odds": odds,
                "effective_odds": effective_odds,
                "bookmaker_implied_prob": bookmaker_implied,
                "polymarket_implied_prob": polymarket_implied,
                "fair_odds": fair_odds,
                "edge": edge,
                "is_mispriced": edge > 0,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        best_edge = df["edge"].max()
        df["suggested_bet"] = df["edge"] == best_edge
    return df.sort_values("edge", ascending=False)


def append_log(
    match: str,
    market: str,
    polymarket_odds: float,
    tax_rate: float,
    results: pd.DataFrame,
    request_id: str,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    log_rows = results.assign(
        request_id=request_id,
        timestamp_utc=timestamp,
        match=match,
        market=market,
        polymarket_odds=polymarket_odds,
        tax_rate=tax_rate,
    )
    ordered = [
        "request_id",
        "timestamp_utc",
        "match",
        "market",
        "polymarket_odds",
        "tax_rate",
        "bookmaker_name",
        "bookmaker_odds",
        "effective_odds",
        "polymarket_implied_prob",
        "bookmaker_implied_prob",
        "fair_odds",
        "edge",
        "is_mispriced",
        "suggested_bet",
    ]
    log_rows[ordered].to_csv(LOG_PATH, mode="a", index=False, header=False)


def main() -> None:
    st.set_page_config(page_title="Odds Mispricing Finder", layout="wide")
    st.title("Odds Mispricing Finder")
    st.markdown(
        "Enter Polymarket odds, add bookmaker lines, and estimate mispriced odds "
        "after tax. The results are saved locally to `data/bet_requests.csv`."
    )

    initialize_log()

    with st.sidebar:
        st.header("Match Details")
        match = st.text_input("Match", placeholder="Team A vs Team B")
        market = st.text_input("Market", placeholder="Home win / Draw / Away win")
        polymarket_odds = st.number_input(
            "Polymarket decimal odds",
            min_value=1.01,
            value=2.0,
            step=0.01,
            format="%.2f",
        )
        tax_rate = st.number_input(
            "Tax rate (decimal)",
            min_value=0.0,
            max_value=0.99,
            value=0.05,
            step=0.01,
            format="%.2f",
            help="Enter 0.05 for 5% tax on winnings.",
        )
        bookmaker_count = st.slider("Number of bookmakers", 1, MAX_BOOKMAKERS, 3)

    st.subheader("Bookmaker Odds")
    bookmaker_entries = []
    cols = st.columns(2)
    for idx in range(bookmaker_count):
        with cols[idx % 2]:
            st.markdown(f"**Bookmaker {idx + 1}**")
            name = st.text_input(
                f"Name {idx + 1}",
                value=f"Bookmaker {idx + 1}",
                key=f"name_{idx}",
            )
            odds = st.number_input(
                f"Decimal odds {idx + 1}",
                min_value=1.01,
                value=2.2,
                step=0.01,
                format="%.2f",
                key=f"odds_{idx}",
            )
            bookmaker_entries.append({"name": name, "odds": odds})

    if st.button("Analyze and save"):
        if not match.strip():
            st.warning("Please enter a match name.")
            st.stop()
        if not market.strip():
            st.warning("Please enter a market name.")
            st.stop()
        results = calculate_value_edges(polymarket_odds, tax_rate, bookmaker_entries)
        if results.empty:
            st.warning("Add at least one bookmaker.")
        else:
            request_id = uuid4().hex[:8]
            append_log(match, market, polymarket_odds, tax_rate, results, request_id)
            st.success("Analysis saved to data/bet_requests.csv")
            st.subheader("Mispriced Odds Results")
            st.dataframe(
                results.style.format(
                    {
                        "bookmaker_odds": "{:.2f}",
                        "effective_odds": "{:.2f}",
                        "polymarket_implied_prob": "{:.2%}",
                        "bookmaker_implied_prob": "{:.2%}",
                        "fair_odds": "{:.2f}",
                        "edge": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )
            suggested = results[results["suggested_bet"]]
            if suggested["is_mispriced"].any():
                best = suggested.iloc[0]
                st.info(
                    f"Suggested bet: **{best['bookmaker_name']}** "
                    f"at **{best['bookmaker_odds']:.2f}** "
                    f"(edge {best['edge']:.2%})."
                )
            else:
                st.info("No mispriced odds found after tax.")

    st.divider()
    st.subheader("Saved Requests")
    history = pd.read_csv(LOG_PATH)
    if history.empty:
        st.caption("No saved bets yet.")
    else:
        history = history.sort_values("timestamp_utc", ascending=False)
        request_ids = history["request_id"].unique().tolist()
        selected_request = st.selectbox(
            "View a saved request",
            options=request_ids,
            format_func=lambda value: f"Request {value}",
        )
        st.dataframe(
            history[history["request_id"] == selected_request],
            use_container_width=True,
        )
        st.download_button(
            "Download bet history CSV",
            data=history.to_csv(index=False),
            file_name="bet_requests.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
