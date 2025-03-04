import pandas as pd

def calculate_value_bets(df: pd.DataFrame, model, label_encoder, stake=1.0):
    """
    Uses the trained model to predict outcome probabilities and calculates value bets.
    Assumes the dataframe has 'home_goals_rolling_mean' and 'away_goals_rolling_mean' features.
    Returns a DataFrame with predicted probabilities and computed value.
    """
    features = ["home_goals_rolling_mean", "away_goals_rolling_mean"]
    df_predict = df.dropna(subset=features).copy()
    
    probs = model.predict_proba(df_predict[features])
    
    idx_H = label_encoder.transform(["H"])[0]
    idx_D = label_encoder.transform(["D"])[0]
    idx_A = label_encoder.transform(["A"])[0]
    
    df_predict["prob_home"] = probs[:, idx_H]
    df_predict["prob_draw"] = probs[:, idx_D]
    df_predict["prob_away"] = probs[:, idx_A]
    
    df_predict["model_odds_home"] = 1.0 / df_predict["prob_home"]
    df_predict["model_odds_draw"] = 1.0 / df_predict["prob_draw"]
    df_predict["model_odds_away"] = 1.0 / df_predict["prob_away"]
    
    # Use bookmaker odds if present, else assume 0
    for col in ["b365_home_odds", "b365_draw_odds", "b365_away_odds"]:
        if col not in df_predict.columns:
            df_predict[col] = 0
    
    df_predict["value_home"] = df_predict["prob_home"] * df_predict["b365_home_odds"] - 1
    df_predict["value_draw"] = df_predict["prob_draw"] * df_predict["b365_draw_odds"] - 1
    df_predict["value_away"] = df_predict["prob_away"] * df_predict["b365_away_odds"] - 1

    def get_best_bet(row):
        bets = {"H": row["value_home"], "D": row["value_draw"], "A": row["value_away"]}
        best = max(bets, key=bets.get)
        return best if bets[best] > 0 else None

    df_predict["best_bet"] = df_predict.apply(get_best_bet, axis=1)
    return df_predict

def backtest(df: pd.DataFrame, stake=1.0):
    """
    Evaluates the strategy on historical data.
    Only considers rows with a 'result' column.
    """
    df_test = df.dropna(subset=["result"]).copy()
    
    def get_return(row):
        if row["best_bet"] is None:
            return 0.0
        if row["best_bet"] == "H" and row["result"] == "H":
            return (row.get("b365_home_odds", 0) * stake) - stake
        elif row["best_bet"] == "D" and row["result"] == "D":
            return (row.get("b365_draw_odds", 0) * stake) - stake
        elif row["best_bet"] == "A" and row["result"] == "A":
            return (row.get("b365_away_odds", 0) * stake) - stake
        else:
            return -stake
    
    df_test["profit"] = df_test.apply(get_return, axis=1)
    total_profit = df_test["profit"].sum()
    n_bets = df_test["best_bet"].notnull().sum()
    roi = total_profit / (n_bets * stake) if n_bets > 0 else 0.0
    
    print(f"Backtest results over {len(df_test)} matches:")
    print(f" - Bets placed: {n_bets}")
    print(f" - Total profit: {total_profit:.2f}")
    print(f" - ROI: {roi:.2%}")
    
    return df_test
