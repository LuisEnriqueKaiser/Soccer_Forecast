import os
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Paths: Adjust to match your setup
# ------------------------------------------------------------------------------
DATA_PATH = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/merged_data.csv"
OUTPUT_DIR = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/results/credible_intervalls"
SUBSET_SIZE = 20  # how many matches to plot

def main():
    # Make sure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Read data
    df = pd.read_csv(DATA_PATH)
    
    # Filter for weeks 10–20 (as an example); adjust if needed
    df = df[(df["week"] >= 10) & (df["week"] <= 20)]
    
    # Just take the first SUBSET_SIZE matches
    df_subset = df.iloc[:SUBSET_SIZE].reset_index(drop=True)

    # Create a figure with three vertical subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 1) HOME probabilities
    # Fill the 95% Bayesian CI
    axes[0].fill_between(
        df_subset.index, 
        df_subset["prob_home_lower_bayes_adv"], 
        df_subset["prob_home_upper_bayes_adv"], 
        alpha=0.2, 
        label="Home 95% CI"
    )
    # Bookmaker's implied probability
    axes[0].scatter(
        df_subset.index, 
        df_subset["mean_implied_prob_home"], 
        color="blue", 
        label="Home Bookie Prob"
    )
    # Bayesian advanced model's mean (point prediction)
    axes[0].scatter(
        df_subset.index,
        df_subset["prob_home_mean_bayes_adv"],
        color="orange",
        marker="x",
        label="Home Bayesian Pred."
    )
    axes[0].set_ylabel("Probability")
    axes[0].set_ylim(0, 1)
    axes[0].set_title(f"Home Probability (Subset of {SUBSET_SIZE} Matches)")
    axes[0].legend()

    # 2) DRAW probabilities
    axes[1].fill_between(
        df_subset.index, 
        df_subset["prob_draw_lower_bayes_adv"], 
        df_subset["prob_draw_upper_bayes_adv"], 
        alpha=0.2, 
        label="Draw 95% CI"
    )
    axes[1].scatter(
        df_subset.index, 
        df_subset["mean_implied_prob_draw"], 
        color="blue", 
        label="Draw Bookie Prob"
    )
    axes[1].scatter(
        df_subset.index,
        df_subset["prob_draw_mean_bayes_adv"],
        color="orange",
        marker="x",
        label="Draw Bayesian Pred."
    )
    axes[1].set_ylabel("Probability")
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"Draw Probability (Subset of {SUBSET_SIZE} Matches)")
    axes[1].legend()

    # 3) AWAY probabilities
    axes[2].fill_between(
        df_subset.index, 
        df_subset["prob_away_lower_bayes_adv"], 
        df_subset["prob_away_upper_bayes_adv"], 
        alpha=0.2, 
        label="Away 95% CI"
    )
    axes[2].scatter(
        df_subset.index, 
        df_subset["mean_implied_prob_away"], 
        color="blue", 
        label="Away Bookie Prob"
    )
    axes[2].scatter(
        df_subset.index,
        df_subset["prob_away_mean_bayes_adv"],
        color="orange",
        marker="x",
        label="Away Bayesian Pred."
    )
    axes[2].set_xlabel("Match Index in Subset")
    axes[2].set_ylabel("Probability")
    axes[2].set_ylim(0, 1)
    axes[2].set_title(f"Away Probability (Subset of {SUBSET_SIZE} Matches)")
    axes[2].legend()

    # Improve layout
    plt.tight_layout()

    # Save and/or show
    out_file = os.path.join(OUTPUT_DIR, f"subset_{SUBSET_SIZE}_matches_probabilities.png")
    plt.savefig(out_file, dpi=300)


if __name__ == "__main__":
    main()
