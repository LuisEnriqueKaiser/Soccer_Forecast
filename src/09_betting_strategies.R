###############################################################################
# Example R script: Compare two simplistic betting strategies
# using Bayesian credible intervals vs. Random Forest predictions,
# with a dynamic 10% stake per triggered outcome,
# AND a plot of total capital over time for both strategies.
###############################################################################
rm(list=ls())

library(dplyr)
library(stargazer)
library(ggplot2)

# 1) Read data
DATA_PATH <- "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/merged_data.csv"
df <- read.csv(DATA_PATH, stringsAsFactors = FALSE)

# Drop rows with missing values (if any)
df <- df[complete.cases(df),]

# 2) Check columns
required_cols <- c(
  # Bayesian lower bounds
  "prob_home_lower_bayes_adv", "prob_draw_lower_bayes_adv", "prob_away_lower_bayes_adv",
  # Implied probabilities
  "mean_implied_prob_home", "mean_implied_prob_draw", "mean_implied_prob_away",
  # RF predicted probabilities
  "rf_prediction_proba_home_win", "rf_prediction_proba_draw", "rf_prediction_proba_away_win",
  # Odds columns
  "B365H", "B365D", "B365A",
  # Actual match result (1=Away Win, 2=Draw, 3=Home Win)
  "match_result_cat"
)
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop("Missing columns in df: ", paste(missing_cols, collapse=", "))
}

# 3) Create bet flags
df <- df %>%
  mutate(
    # Bayesian strategy flags
    bet_home_bayes = (mean_implied_prob_home < prob_home_lower_bayes_adv),
    bet_draw_bayes = (mean_implied_prob_draw < prob_draw_lower_bayes_adv),
    bet_away_bayes = (mean_implied_prob_away < prob_away_lower_bayes_adv),
    
    # Random Forest strategy flags
    bet_home_rf = (rf_prediction_proba_home_win > mean_implied_prob_home),
    bet_draw_rf = (rf_prediction_proba_draw     > mean_implied_prob_draw),
    bet_away_rf = (rf_prediction_proba_away_win > mean_implied_prob_away)
  )

# 4) A helper function to simulate a strategy with dynamic capital
#    - fraction=0.1 => invests 10% of capital for each triggered outcome.
#    - Returns: 
#       - 'capital_history': capital after each match
#       - 'final_capital': capital after last match
#       - 'num_bets': total number of bets placed
simulate_dynamic <- function(df, 
                             bet_home_flag, bet_draw_flag, bet_away_flag, 
                             odds_home, odds_draw, odds_away,
                             result_col = "match_result_cat",
                             initial_capital = 100, fraction = 0.1) {
  
  capital <- initial_capital
  num_bets <- 0
  n <- nrow(df)
  capital_history <- numeric(n)  # store capital after each match
  
  for (i in seq_len(n)) {
    # capital at start of match i
    current_capital <- capital
    
    # 1) Home Bet
    if (df[[bet_home_flag]][i]) {
      stake <- fraction * current_capital
      num_bets <- num_bets + 1
      # If actual result = 3 => home win
      if (df[[result_col]][i] == 3) {
        # Win => gain stake*(odds - 1)
        capital <- capital + stake * (df[[odds_home]][i] - 1)
      } else {
        # Lose => -stake
        capital <- capital - stake
      }
    }
    
    # 2) Draw Bet
    if (df[[bet_draw_flag]][i]) {
      stake <- fraction * current_capital
      num_bets <- num_bets + 1
      if (df[[result_col]][i] == 2) {
        capital <- capital + stake * (df[[odds_draw]][i] - 1)
      } else {
        capital <- capital - stake
      }
    }
    
    # 3) Away Bet
    if (df[[bet_away_flag]][i]) {
      stake <- fraction * current_capital
      num_bets <- num_bets + 1
      if (df[[result_col]][i] == 1) {
        capital <- capital + stake * (df[[odds_away]][i] - 1)
      } else {
        capital <- capital - stake
      }
    }
    
    # capital after match i
    capital_history[i] <- capital
  }
  
  list(capital_history = capital_history,
       final_capital   = capital,
       num_bets        = num_bets)
}


# 5) Run the simulation for each strategy
# 5a) Bayesian
bayes_result <- simulate_dynamic(
  df,
  bet_home_flag = "bet_home_bayes",
  bet_draw_flag = "bet_draw_bayes",
  bet_away_flag = "bet_away_bayes",
  odds_home     = "B365H",
  odds_draw     = "B365D",
  odds_away     = "B365A",
  result_col    = "match_result_cat",
  initial_capital = 100,
  fraction = 0.1
)
final_capital_bayes <- bayes_result$final_capital
num_bets_bayes      <- bayes_result$num_bets
profit_bayes        <- final_capital_bayes - 100
roi_bayes           <- profit_bayes / 100

# 5b) Random Forest
rf_result <- simulate_dynamic(
  df,
  bet_home_flag = "bet_home_rf",
  bet_draw_flag = "bet_draw_rf",
  bet_away_flag = "bet_away_rf",
  odds_home     = "B365H",
  odds_draw     = "B365D",
  odds_away     = "B365A",
  result_col    = "match_result_cat",
  initial_capital = 100,
  fraction = 0.05
)
final_capital_rf <- rf_result$final_capital
num_bets_rf      <- rf_result$num_bets
profit_rf        <- final_capital_rf - 100
roi_rf           <- profit_rf / 100


# 6) Summarize results in a data.frame
results_table <- data.frame(
  Strategy         = c("Bayesian CI", "Random Forest"),
  Number_of_Bets   = c(num_bets_bayes, num_bets_rf),
  Final_Capital    = c(final_capital_bayes, final_capital_rf),
  Profit           = c(profit_bayes, profit_rf),
  ROI              = c(roi_bayes, roi_rf)
)

# 7) Print results to console
print(results_table)

# 8) Save results as a LaTeX table
output_file <- "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/results/betting_strategies_summary.tex"
if (file.exists(output_file)) {
  file.remove(output_file)
}

sink(output_file)
stargazer(
  results_table,
  summary = FALSE,
  rownames = FALSE,
  title = "Comparison of Bayesian CI vs Random Forest Betting (10% Stake)",
  label = "tab:betting_comparison",
  float = TRUE,
  align = TRUE,
  digits = 3,
  out = output_file
)
sink()

cat("LaTeX table saved to:", output_file, "\n")

# 9) Plot the total payoff (capital) over time for both strategies
#    We'll create a combined data frame from the capital_history vectors.
df_plot <- data.frame(
  match_index = rep(seq_len(nrow(df)), 2),
  capital = c(bayes_result$capital_history, rf_result$capital_history),
  strategy = rep(c("Bayesian CI", "Random Forest"), each = nrow(df))
)

p <- ggplot(df_plot, aes(x = match_index, y = capital, color = strategy)) +
  geom_line() +
  labs(title = "Capital Over Time",
       x = "Match Index",
       y = "Capital",
       color = "Strategy") +
  theme_minimal()

# Save the plot
plot_path <- "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/results/capital_over_time.png"
ggsave(plot_path, p, width = 10, height = 6, dpi = 300)

cat("Plot of capital over time saved to:", plot_path, "\n")

