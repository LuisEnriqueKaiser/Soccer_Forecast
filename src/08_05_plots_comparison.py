#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from project_specifics import RESULTS_FOLDER, OUTPUT_FOLDER

def main():
    # -------------------------------------------------------------------------
    # 1. PATHS AND CONFIG
    # -------------------------------------------------------------------------
    results_folder = RESULTS_FOLDER
    output_folder  = OUTPUT_FOLDER
    os.makedirs(output_folder, exist_ok=True)

    # Use a built-in Matplotlib style; if you get an error, just remove or change it
    plt.style.use("ggplot")

    # You can still tweak some general rcParams if desired
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10

    # -------------------------------------------------------------------------
    # 2. READ THE CSV FILES
    # -------------------------------------------------------------------------
    df1 = pd.read_csv(os.path.join(results_folder, 'results_bayes_base.csv'))
    df2 = pd.read_csv(os.path.join(results_folder, 'results_bayes_season.csv'))
    df3 = pd.read_csv(os.path.join(results_folder, 'test_data_predictions_freq.csv'))
    df4 = pd.read_csv(os.path.join(results_folder, 'test_data.csv'))

    # -------------------------------------------------------------------------
    # 3. MERGE THE DATAFRAMES
    # -------------------------------------------------------------------------
    merged_df = pd.concat([df1, df2, df3, df4], axis=1, join='outer')
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    # take subset where the gameweek is smaller than 25
    merged_df = merged_df[merged_df['week'] < 25]
    merged_df.reset_index(inplace=True)

    # -------------------------------------------------------------------------
    # 4. KEEP ONLY RELEVANT COLUMNS
    # -------------------------------------------------------------------------
    columns_to_keep = [
        "home_team", "away_team", "date",
        "prob_away_mean_base_bayes", "prob_away_lower_base_bayes", "prob_away_upper_base_bayes",
        "prob_draw_mean_base_bayes", "prob_draw_lower_base_bayes", "prob_draw_upper_base_bayes",
        "prob_home_mean_base_bayes", "prob_home_lower_base_bayes", "prob_home_upper_base_bayes",
        "predicted_category_base_bayes", "true_category",
        "prob_away_mean_bayes_adv", "prob_away_lower_bayes_adv", "prob_away_upper_bayes_adv",
        "prob_draw_mean_bayes_adv", "prob_draw_lower_bayes_adv", "prob_draw_upper_bayes_adv",
        "prob_home_mean_bayes_adv", "prob_home_lower_bayes_adv", "prob_home_upper_bayes_adv",
        "predicted_category_bayes_adv", "predicted_label_bayes_adv",
        "rf_prediction", "rf_prediction_proba_home_win", "rf_prediction_proba_draw", "rf_prediction_proba_away_win",
        "logit_prediction", "logit_prediction_proba_home_win", "logit_prediction_proba_draw", "logit_prediction_proba_away_win",
        "week", "season_number",
        "B365H", "B365D", "B365A", "PSH", "PSD", "PSA", "WHH", "WHD", "WHA",
        "match_result",
        "implied_prob_home_B365H", "implied_prob_draw_B365D", "implied_prob_away_B365A",
        "implied_prob_home_PSH",  "implied_prob_draw_PSD",  "implied_prob_away_PSA",
        "implied_prob_home_WHH",  "implied_prob_draw_WHD",  "implied_prob_away_WHA",
        "home_team_integer", "away_team_integer", "match_result_cat"
    ]
    columns_to_keep = [c for c in columns_to_keep if c in merged_df.columns]
    merged_df = merged_df[columns_to_keep]

    # -------------------------------------------------------------------------
    # 5. COMPUTE MEAN IMPLIED PROBABILITIES (HOME, DRAW, AWAY)
    # -------------------------------------------------------------------------
    merged_df["mean_implied_prob_home"] = merged_df[
        ["implied_prob_home_B365H", "implied_prob_home_PSH", "implied_prob_home_WHH"]
    ].mean(axis=1)

    merged_df["mean_implied_prob_draw"] = merged_df[
        ["implied_prob_draw_B365D", "implied_prob_draw_PSD", "implied_prob_draw_WHD"]
    ].mean(axis=1)

    merged_df["mean_implied_prob_away"] = merged_df[
        ["implied_prob_away_B365A", "implied_prob_away_PSA", "implied_prob_away_WHA"]
    ].mean(axis=1)
    # save the merged dataframe
    folder_merged = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/"
    merged_df.to_csv(os.path.join(folder_merged, 'merged_data.csv'), index=False)
    # -------------------------------------------------------------------------
    # 6. CALCULATE ABSOLUTE DIFFERENCES VS. MEAN IMPLIED FOR EACH MODEL
    # -------------------------------------------------------------------------
    # Bayes Base
    merged_df["abs_diff_home_base_bayes"] = (
        merged_df["mean_implied_prob_home"] - merged_df["prob_home_mean_base_bayes"]
    ).abs()
    merged_df["abs_diff_draw_base_bayes"] = (
        merged_df["mean_implied_prob_draw"] - merged_df["prob_draw_mean_base_bayes"]
    ).abs()
    merged_df["abs_diff_away_base_bayes"] = (
        merged_df["mean_implied_prob_away"] - merged_df["prob_away_mean_base_bayes"]
    ).abs()

    # Bayes Advanced
    merged_df["abs_diff_home_bayes_adv"] = (
        merged_df["mean_implied_prob_home"] - merged_df["prob_home_mean_bayes_adv"]
    ).abs()
    merged_df["abs_diff_draw_bayes_adv"] = (
        merged_df["mean_implied_prob_draw"] - merged_df["prob_draw_mean_bayes_adv"]
    ).abs()
    merged_df["abs_diff_away_bayes_adv"] = (
        merged_df["mean_implied_prob_away"] - merged_df["prob_away_mean_bayes_adv"]
    ).abs()

    # Random Forest
    merged_df["abs_diff_home_rf"] = (
        merged_df["mean_implied_prob_home"] - merged_df["rf_prediction_proba_home_win"]
    ).abs()
    merged_df["abs_diff_draw_rf"] = (
        merged_df["mean_implied_prob_draw"] - merged_df["rf_prediction_proba_draw"]
    ).abs()
    merged_df["abs_diff_away_rf"] = (
        merged_df["mean_implied_prob_away"] - merged_df["rf_prediction_proba_away_win"]
    ).abs()

    # Logistic Regression
    merged_df["abs_diff_home_logit"] = (
        merged_df["mean_implied_prob_home"] - merged_df["logit_prediction_proba_home_win"]
    ).abs()
    merged_df["abs_diff_draw_logit"] = (
        merged_df["mean_implied_prob_draw"] - merged_df["logit_prediction_proba_draw"]
    ).abs()
    merged_df["abs_diff_away_logit"] = (
        merged_df["mean_implied_prob_away"] - merged_df["logit_prediction_proba_away_win"]
    ).abs()

    # -------------------------------------------------------------------------
    # 7. WEEKLY AVERAGES
    # -------------------------------------------------------------------------
    # Ensure we pass a LIST of columns, not a tuple
    if "week" in merged_df.columns:
        columns_for_groupby = [
            "abs_diff_home_base_bayes", "abs_diff_draw_base_bayes", "abs_diff_away_base_bayes",
            "abs_diff_home_bayes_adv",  "abs_diff_draw_bayes_adv",  "abs_diff_away_bayes_adv",
            "abs_diff_home_rf",         "abs_diff_draw_rf",         "abs_diff_away_rf",
            "abs_diff_home_logit",      "abs_diff_draw_logit",      "abs_diff_away_logit"
        ]
        weekly_avg_diff = (
            merged_df
            .groupby("week")[columns_for_groupby]  # <--- LIST not tuple
            .mean()
            .dropna()
        )
    else:
        weekly_avg_diff = None

    # -------------------------------------------------------------------------
    # 8. ADV. BAYES PROBS + CREDIBLE INTERVALS vs. MEAN IMPLIED
    #    We'll limit the plot to ~10 game days for clarity
    # -------------------------------------------------------------------------
    df_plot = merged_df.copy()

    if "date" in df_plot.columns:
        df_plot["date"] = pd.to_datetime(df_plot["date"], errors="coerce")
        df_plot.sort_values("date", inplace=True)

    # Limit to first 10 unique weeks (if available), else first ~40 rows
    if "week" in df_plot.columns and not df_plot["week"].isna().all():
        unique_weeks = sorted(df_plot["week"].dropna().unique())
        subset_weeks = unique_weeks[:10]  # first 10 weeks
        df_plot = df_plot[df_plot["week"].isin(subset_weeks)].copy()
    else:
        df_plot = df_plot.head(40).copy()

    df_plot.reset_index(drop=True, inplace=True)

    if "date" in df_plot.columns and df_plot["date"].notna().any():
        x_vals = df_plot["date"]
        x_label = "Date"
    else:
        x_vals = df_plot.index
        x_label = "Match Index"

    def plot_advanced_bayes_vs_implied(df_sub, outcome, out_prefix):
        lo_col = f"prob_{outcome}_lower_bayes_adv"
        mn_col = f"prob_{outcome}_mean_bayes_adv"
        up_col = f"prob_{outcome}_upper_bayes_adv"
        implied_col = f"mean_implied_prob_{outcome}"

        needed = [lo_col, mn_col, up_col, implied_col]
        if not all(c in df_sub.columns for c in needed):
            print(f"Skipping {outcome} – columns missing.")
            return

        plt.figure(figsize=(9, 5))

        # Fill Bayesian credible interval
        plt.fill_between(
            x_vals,
            df_sub[lo_col],
            df_sub[up_col],
            color="lightsalmon",
            alpha=0.4,
            label='Bayes Credible Interval'
        )

        # Bayes mean line
        plt.plot(
            x_vals,
            df_sub[mn_col],
            color="red",
            linewidth=2.5,
            marker='o',
            markersize=5,
            label='Bayes Adv Mean'
        )

        # Implied line
        plt.plot(
            x_vals,
            df_sub[implied_col],
            color="blue",
            linewidth=2.0,
            marker='s',
            markersize=5,
            linestyle='--',
            label='Mean Implied Probability'
        )

        plt.title(f"{outcome.title()} Probability: Adv. Bayes vs. Implied (Subset)")
        plt.xlabel(x_label)
        plt.ylabel("Probability")
        plt.legend(loc='best')
        plt.grid(True)

        out_file = os.path.join(output_folder, f"{out_prefix}_{outcome}_bayes_vs_implied_subset.png")
        plt.savefig(out_file, bbox_inches='tight')
        print(f"Saved figure: {out_file}")
        plt.close()

    for outcome in ["home", "draw", "away"]:
        plot_advanced_bayes_vs_implied(df_plot, outcome, "adv_bayes")

    # -------------------------------------------------------------------------
    # 9. EXTRA INSIGHT: WEEKLY ABS DIFF PLOTS + BAR CHARTS
    # -------------------------------------------------------------------------
    if weekly_avg_diff is not None and not weekly_avg_diff.empty:
        model_cols_map = {
            "Home": [
                "abs_diff_home_base_bayes", "abs_diff_home_bayes_adv",
                "abs_diff_home_rf", "abs_diff_home_logit"
            ],
            "Draw": [
                "abs_diff_draw_base_bayes", "abs_diff_draw_bayes_adv",
                "abs_diff_draw_rf", "abs_diff_draw_logit"
            ],
            "Away": [
                "abs_diff_away_base_bayes", "abs_diff_away_bayes_adv",
                "abs_diff_away_rf", "abs_diff_away_logit"
            ],
        }

        line_colors = {
            "abs_diff_home_base_bayes":  "tab:blue",
            "abs_diff_home_bayes_adv":   "tab:orange",
            "abs_diff_home_rf":          "tab:green",
            "abs_diff_home_logit":       "tab:red",

            "abs_diff_draw_base_bayes":  "tab:blue",
            "abs_diff_draw_bayes_adv":   "tab:orange",
            "abs_diff_draw_rf":          "tab:green",
            "abs_diff_draw_logit":       "tab:red",

            "abs_diff_away_base_bayes":  "tab:blue",
            "abs_diff_away_bayes_adv":   "tab:orange",
            "abs_diff_away_rf":          "tab:green",
            "abs_diff_away_logit":       "tab:red",
        }

        # (A) Weekly line plots
        for outcome_label, col_list in model_cols_map.items():
            plt.figure(figsize=(10, 5))
            for col in col_list:
                lbl = col.replace("abs_diff_", "").replace("_", " ").title()
                plt.plot(
                    weekly_avg_diff.index,
                    weekly_avg_diff[col],
                    label=lbl,
                    linewidth=2.0,
                    color=line_colors.get(col, None)
                )
            plt.title(f"Weekly Avg Abs Difference - {outcome_label} Predictions")
            plt.xlabel("Game Week")
            plt.ylabel("Avg Abs Difference")
            plt.legend()
            plt.grid(True)
            out_file = os.path.join(output_folder, f"weekly_avg_absdiff_{outcome_label.lower()}.png")
            plt.savefig(out_file, bbox_inches='tight')
            print(f"Saved figure: {out_file}")
            plt.close()

        # (B) Overall bar chart
        def compute_mean_abs_diff_metrics(outcome):
            base_col  = f"abs_diff_{outcome}_base_bayes"
            adv_col   = f"abs_diff_{outcome}_bayes_adv"
            rf_col    = f"abs_diff_{outcome}_rf"
            logit_col = f"abs_diff_{outcome}_logit"
            return {
                "Base Bayes":   round(weekly_avg_diff[base_col].mean(), 4),
                "Bayes Adv":    round(weekly_avg_diff[adv_col].mean(), 4),
                "Random Forest":round(weekly_avg_diff[rf_col].mean(), 4),
                "Logit":        round(weekly_avg_diff[logit_col].mean(), 4)
            }

        color_map = {
            "Base Bayes":    "tab:blue",
            "Bayes Adv":     "tab:orange",
            "Random Forest": "tab:green",
            "Logit":         "tab:red"
        }

        for outcome in ["home", "draw", "away"]:
            data = compute_mean_abs_diff_metrics(outcome)
            labels = list(data.keys())
            values = list(data.values())
            colors = [color_map.get(k, "gray") for k in labels]

            plt.figure(figsize=(7, 5))
            x_positions = np.arange(len(labels))
            plt.bar(x_positions, values, color=colors)
            plt.title(f"Mean Absolute Difference - {outcome.title()} Predictions")
            plt.ylabel("Mean Abs Difference")
            plt.ylim(0, max(values) + 0.02)
            plt.xticks(x_positions, labels, rotation=45)
            plt.grid(True, axis="y")

            out_file = os.path.join(output_folder, f"bar_overall_mad_{outcome}.png")
            plt.savefig(out_file, bbox_inches='tight')
            print(f"Saved figure: {out_file}")
            plt.close()

    # -------------------------------------------------------------------------
    # 10. OPTIONAL EXTRA: SCATTER PLOTS (Bayes Adv vs. Implied)
    # -------------------------------------------------------------------------
    for outcome in ["home", "draw", "away"]:
        bayes_col   = f"prob_{outcome}_mean_bayes_adv"
        implied_col = f"mean_implied_prob_{outcome}"
        if bayes_col not in merged_df.columns or implied_col not in merged_df.columns:
            continue

        valid_df = merged_df.dropna(subset=[bayes_col, implied_col])
        if valid_df.empty:
            continue

        plt.figure(figsize=(6, 5))
        plt.scatter(valid_df[bayes_col], valid_df[implied_col], alpha=0.4, color="purple", edgecolor="k")
        plt.title(f"Advanced Bayes vs. Implied - {outcome.title()}")
        plt.xlabel(f"Bayes Adv Probability ({outcome.title()})")
        plt.ylabel(f"Implied Probability ({outcome.title()})")
        plt.grid(True)
        out_file = os.path.join(output_folder, f"scatter_bayes_vs_implied_{outcome}.png")
        plt.savefig(out_file, bbox_inches='tight')
        print(f"Saved figure: {out_file}")
        plt.close()

    print("\nAll done! Plots have been saved in:", output_folder)

if __name__ == "__main__":
    main()
