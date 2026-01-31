#!/usr/bin/env python3
"""Hyper‑parameter tuning for a combined Pi‑rating + multinomial‑logit model
--------------------------------------------------------------------------
This script
1. Loads a soccer results file (merged_data.csv style)
2. Builds a scikit‑learn **Pipeline** consisting of
      PiTransformer  →  StandardScaler  →  LogisticRegression
   where PiTransformer exposes the four rating hyper‑parameters
   (lambda, gamma, b, c).
3. Optimises **all seven** hyper‑parameters jointly using **Optuna** with
   walk‑forward (TimeSeriesSplit) cross‑validation and log‑loss scoring.
4. Retrains the best pipeline on the *entire* history and
   – saves the fitted model (joblib)
   – appends probability columns to the dataframe and writes them out.

Usage
-----
    python pi_logit_hypersearch.py \
        --data   /path/to/merged_data.csv \
        --trials 200 \
        --seed   42

Outputs (same directory as the input file unless --outdir is given)
-------------------------------------------------------------------
    ├─ merged_data_with_pi_predictions_tuned.csv
    └─ pi_logit_best_model.joblib

Dependencies
------------
    pip install pandas numpy scikit-learn optuna joblib
"""

from __future__ import annotations

import argparse
import joblib
import optuna
import numpy as np
import pandas as pd
from copy import deepcopy
from math import log10
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
#  Pi‑rating transformer  (stateful!)
# ---------------------------------------------------------------------------
class PiTransformer(BaseEstimator, TransformerMixin):
    """Generate three Pi‑rating features per match.

    Hyper‑parameters
    ----------------
    lam   : learning rate (λ)             – float > 0
    gamma : home↔away bleed‑through (γ)   – 0 ≤ γ ≤ 1
    b, c  : parameters of rating→goal mapping (see original script)
    """

    def __init__(self, lam: float = 0.035, gamma: float = 0.7,
                 b: float = 10.0, c: float = 3.0):
        self.lam = lam; self.gamma = gamma; self.b = b; self.c = c

    # ---------------- internal helpers ------------------------------------
    def _transform_rating(self, r: float) -> float:
        val = self.b ** (abs(r) / self.c) - 1
        return -val if r < 0 else val

    def _expected_diff(self, h, a, ratings):
        # make sure both teams exist
        if h not in ratings:
            ratings[h] = {"home": 0.0, "away": 0.0}
        if a not in ratings:
            ratings[a] = {"home": 0.0, "away": 0.0}
        return self._transform_rating(ratings[h]["home"]) - \
            self._transform_rating(ratings[a]["away"])
    # ---------------- sklearn API -----------------------------------------
    def fit(self, X: pd.DataFrame, y=None):
        """Learn team ratings from the *chronologically sorted* matches X."""
        X_sorted = X.sort_values("date")
        self._ratings_: dict[str, dict[str, float]] = {}

        def ensure(team: str):
            if team not in self._ratings_:
                self._ratings_[team] = {"home": 0.0, "away": 0.0}
            return self._ratings_[team]

        for _, row in X_sorted.iterrows():
            h, a = row["home_team"], row["away_team"]
            exp_diff  = self._expected_diff(h, a, self._ratings_)
            score_diff = row["goals_home"] - row["goals_away"]
            w_err = self.c * log10(1 + abs(score_diff - exp_diff))
            sign = 1 if exp_diff < score_diff else -1
            dh = sign * w_err * self.lam
            da = -dh
            # update with bleed‑through
            ensure(h)["home"] += dh
            ensure(h)["away"] += dh * self.gamma
            ensure(a)["away"] += da
            ensure(a)["home"] += da * self.gamma
        return self

    def transform(self, X: pd.DataFrame):
        """Add three Pi features (using *frozen* ratings)."""
        h_rating = []; a_rating = []; exp_gd = []
        for h, a in zip(X["home_team"], X["away_team"]):
            h_rating.append(self._ratings_.get(h, {}).get("home", 0.0))
            a_rating.append(self._ratings_.get(a, {}).get("away", 0.0))
            exp_gd.append(self._expected_diff(h, a, self._ratings_))
        return pd.DataFrame({
            "pi_home_rating": h_rating,
            "pi_away_rating": a_rating,
            "pi_exp_gd":      exp_gd,
        }, index=X.index)

# ---------------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """Return the Pi→Scaler→Logit pipeline with *placeholder* params."""
    return Pipeline([
        ("pi", PiTransformer()),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression( solver="saga",      # supports L1/elastic
                                   max_iter=3000,
                                   n_jobs=1)),
    ])


def make_labels(df: pd.DataFrame) -> pd.Series:
    """Encode outcome as 0‑home‑win / 1‑draw / 2‑away‑win."""
    y = np.where(df["goals_home"] > df["goals_away"], 0,
         np.where(df["goals_home"] == df["goals_away"], 1, 2))
    return pd.Series(y, index=df.index)

# ---------------------------------------------------------------------------
#  Optuna objective
# ---------------------------------------------------------------------------

def objective(trial, X: pd.DataFrame, y: pd.Series, tscv: TimeSeriesSplit):
    pipe = build_pipeline()

    # ---------- suggest Pi parameters -------------------------------------
    pipe.set_params(
        pi__lam   = trial.suggest_float("lam",   1e-3, 1e-1, log=True),
        pi__gamma = trial.suggest_float("gamma", 0.4, 0.95),
        pi__b     = trial.suggest_int("b", 6, 14),
        pi__c     = trial.suggest_float("c", 1.5, 4.5),
    )
    # ---------- suggest logit parameters ----------------------------------
    penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
    C       = trial.suggest_float("C", 1e-3, 1e3, log=True)
    l1_ratio = (trial.suggest_float("l1_ratio", 0.1, 0.9)
                if penalty == "elasticnet" else None)
    pipe.set_params(lr__penalty=penalty, lr__C=C, lr__l1_ratio=l1_ratio)

    # ---------------- walk‑forward CV -------------------------------------
    losses = []
    for tr_idx, te_idx in tscv.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)
        losses.append(log_loss(y_te, p))
    return float(np.mean(losses))

# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tune Pi + Logit model")
    default_csv = Path(
        "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/processed/merged_data_with_pi.csv")
    parser.add_argument("--data", type=Path, default=default_csv,
                        help=f"CSV with merged match data (default: {default_csv})")
    parser.add_argument("--trials", type=int, default=200,
                        help="Optuna trials (default 200)")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--outdir", type=Path, default=None,
                        help="Directory for outputs (defaults to data dir)")
    args = parser.parse_args()

    outdir = args.outdir or args.data.parent
    outdir.mkdir(parents=True, exist_ok=True)

    # --------------- load & prep data -------------------------------------
    df = pd.read_csv(args.data, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    y  = make_labels(df)

    # --------------- set up CV & study ------------------------------------
    tscv = TimeSeriesSplit(n_splits=6)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(lambda tr: objective(tr, df, y, tscv),
                   n_trials=args.trials, show_progress_bar=True)

    print("Best CV log‑loss:", study.best_value)
    print("Best params:\n", study.best_params)

    # --------------- rebuild & fit best pipeline --------------------------
    best_pipe = build_pipeline()
    best_pipe.set_params(
        pi__lam   = study.best_params["lam"],
        pi__gamma = study.best_params["gamma"],
        pi__b     = study.best_params["b"],
        pi__c     = study.best_params["c"],
        lr__penalty = study.best_params["penalty"],
        lr__C       = study.best_params["C"],
        lr__l1_ratio = study.best_params.get("l1_ratio"),
    )
    best_pipe.fit(df, y)

    # --------------- predictions & persistence ---------------------------
    proba = best_pipe.predict_proba(df)
    df[["pred_home_prob", "pred_draw_prob", "pred_away_prob"]] = proba

    pred_path  = outdir / "merged_data_with_pi_predictions_tuned.csv"
    model_path = outdir / "pi_logit_best_model.joblib"
    df.to_csv(pred_path, index=False)
    joblib.dump(best_pipe, model_path)

    print("\n✅ Done. Results written to:")
    print("   ", pred_path.resolve())
    print("   ", model_path.resolve())

if __name__ == "__main__":
    main()
