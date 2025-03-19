###############################################################################
# Compare Four Betting Strategies (with 10% tax per bet):
# 1) Bayesian (Dynamic Stake of 10% + 10% tax each bet)
# 2) Random Forest (Dynamic Stake of 10% + 10% tax each bet)
# 3) Bayesian (Fixed Stake, e.g. 10 units + 10% tax each bet)
# 4) Random Forest (Fixed Stake, e.g. 10 units + 10% tax each bet)
#
# Each strategy tracks:
#   - capital over time
#   - number of bets, wins, losses
#   - average odds
#   - final capital, profit, ROI
#
# We then plot total capital over time for all four strategies.
# 
# PLEASE NOTE:
#   The 10% tax is subtracted from capital immediately for each bet placed.
###############################################################################

rm(list=ls())

library(dplyr)
library(stargazer)
library(ggplot2)

# 1) Read data
DATA_PATH <- "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/merged_data.csv"
df <- read.csv(DATA_PATH, stringsAsFactors = FALSE)

# Drop rows with missing values
df <- df[complete.cases(df),]

###############################################################################
# 2) Single-bookmaker implied probabilities (Bet365)
###############################################################################
df <- df %>%
  mutate(
    implied_prob_home = 1 / B365H,
    implied_prob_draw = 1 / B365D,
    implied_prob_away = 1 / B365A
  )

# 3) Check that needed columns exist
required_cols <- c(
  # Bayesian lower bounds
  "prob_home_lower_bayes_adv", "prob_draw_lower_bayes_adv", "prob_away_lower_bayes_adv",
  
  # Random Forest predicted probabilities
  "rf_prediction_proba_home_win", "rf_prediction_proba_draw", "rf_prediction_proba_away_win",
  
  # Bookmaker odds
  "B365H", "B365D", "B365A",
  
  # Actual match result (1=Away Win, 2=Draw, 3=Home Win)
  "match_result_cat",
  
  # Single-bookmaker implied probabilities
  "implied_prob_home", "implied_prob_draw", "implied_prob_away"
)

missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop("Missing columns in df: ", paste(missing_cols, collapse=", "))
}

###############################################################################
# 4) Define bet flags for Bayesian & Random Forest
###############################################################################
THRESHOLD_RF <- 0.2  # Adjust to tune how often the RF bets

df <- df %>%
  mutate(
    # Bayesian strategy flags
    bet_home_bayes = (implied_prob_home < prob_home_lower_bayes_adv),
    bet_draw_bayes = (implied_prob_draw < prob_draw_lower_bayes_adv),
    bet_away_bayes = (implied_prob_away < prob_away_lower_bayes_adv),
    
    # Random Forest strategy flags
    bet_home_rf  = ((rf_prediction_proba_home_win - implied_prob_home) > THRESHOLD_RF),
    bet_draw_rf  = ((rf_prediction_proba_draw     - implied_prob_draw) > THRESHOLD_RF),
    bet_away_rf  = ((rf_prediction_proba_away_win - implied_prob_away) > THRESHOLD_RF)
  )

###############################################################################
# 5) Simulation functions (with 10% tax on each stake)
###############################################################################
# We'll interpret "10 percent taxes per bet" as 10% of the stake is subtracted 
# from the capital immediately every time a bet is placed (win or lose).

TAX_RATE <- 0.1

# (A) Dynamic stake simulation with 10% bet tax
simulate_dynamic <- function(df, 
                             bet_home_flag, bet_draw_flag, bet_away_flag, 
                             odds_home, odds_draw, odds_away,
                             result_col       = "match_result_cat",
                             initial_capital  = 100, 
                             fraction         = 0.2,
                             tax_rate         = 0.1) {
  
  capital <- initial_capital
  n <- nrow(df)
  
  # Tracking
  capital_history <- numeric(n)
  num_bets        <- 0
  num_wins        <- 0
  sum_of_odds     <- 0
  
  for (i in seq_len(n)) {
    current_capital <- capital
    
    # 1) Home Bet
    if (df[[bet_home_flag]][i]) {
      stake <- fraction * current_capital
      tax   <- stake * tax_rate       # 10% tax on the stake
      num_bets <- num_bets + 1
      placed_odds <- df[[odds_home]][i]
      sum_of_odds <- sum_of_odds + placed_odds
      
      # Immediately pay the tax
      capital <- capital - tax
      
      # If correct outcome, add profit
      if (df[[result_col]][i] == 3) {  # home win
        capital <- capital + stake * (placed_odds - 1)
        num_wins <- num_wins + 1
      } else {
        # If lose, subtract stake from capital
        capital <- capital - stake
      }
    }
    
    # 2) Draw Bet
    if (df[[bet_draw_flag]][i]) {
      stake <- fraction * current_capital
      tax   <- stake * tax_rate
      num_bets <- num_bets + 1
      placed_odds <- df[[odds_draw]][i]
      sum_of_odds <- sum_of_odds + placed_odds
      
      capital <- capital - tax
      
      if (df[[result_col]][i] == 2) { # draw
        capital <- capital + stake * (placed_odds - 1)
        num_wins <- num_wins + 1
      } else {
        capital <- capital - stake
      }
    }
    
    # 3) Away Bet
    if (df[[bet_away_flag]][i]) {
      stake <- fraction * current_capital
      tax   <- stake * tax_rate
      num_bets <- num_bets + 1
      placed_odds <- df[[odds_away]][i]
      sum_of_odds <- sum_of_odds + placed_odds
      
      capital <- capital - tax
      
      if (df[[result_col]][i] == 1) { # away win
        capital <- capital + stake * (placed_odds - 1)
        num_wins <- num_wins + 1
      } else {
        capital <- capital - stake
      }
    }
    
    # Track capital after this match
    capital_history[i] <- capital
  }
  
  num_losses <- num_bets - num_wins
  avg_odds   <- ifelse(num_bets > 0, sum_of_odds / num_bets, NA)
  
  list(
    capital_history = capital_history,
    final_capital   = capital,
    num_bets        = num_bets,
    num_wins        = num_wins,
    num_losses      = num_losses,
    avg_odds        = avg_odds
  )
}

# (B) Fixed stake simulation with 10% bet tax
simulate_fixed_stake <- function(df, 
                                 bet_home_flag, bet_draw_flag, bet_away_flag,
                                 odds_home, odds_draw, odds_away,
                                 result_col       = "match_result_cat",
                                 initial_capital  = 100, 
                                 stake_size       = 10,
                                 tax_rate         = 0.1) {
  
  capital <- initial_capital
  n <- nrow(df)
  
  capital_history <- numeric(n)
  num_bets        <- 0
  num_wins        <- 0
  sum_of_odds     <- 0
  
  for (i in seq_len(n)) {
    # 1) Home Bet
    if (df[[bet_home_flag]][i]) {
      tax <- stake_size * tax_rate
      num_bets <- num_bets + 1
      placed_odds <- df[[odds_home]][i]
      sum_of_odds <- sum_of_odds + placed_odds
      
      # Pay tax on stake
      capital <- capital - tax
      
      if (df[[result_col]][i] == 3) { # home win
        capital <- capital + stake_size * (placed_odds - 1)
        num_wins <- num_wins + 1
      } else {
        capital <- capital - stake_size
      }
    }
    
    # 2) Draw Bet
    if (df[[bet_draw_flag]][i]) {
      tax <- stake_size * tax_rate
      num_bets <- num_bets + 1
      placed_odds <- df[[odds_draw]][i]
      sum_of_odds <- sum_of_odds + placed_odds
      
      capital <- capital - tax
      
      if (df[[result_col]][i] == 2) {
        capital <- capital + stake_size * (placed_odds - 1)
        num_wins <- num_wins + 1
      } else {
        capital <- capital - stake_size
      }
    }
    
    # 3) Away Bet
    if (df[[bet_away_flag]][i]) {
      tax <- stake_size * tax_rate
      num_bets <- num_bets + 1
      placed_odds <- df[[odds_away]][i]
      sum_of_odds <- sum_of_odds + placed_odds
      
      capital <- capital - tax
      
      if (df[[result_col]][i] == 1) { # away win
        capital <- capital + stake_size * (placed_odds - 1)
        num_wins <- num_wins + 1
      } else {
        capital <- capital - stake_size
      }
    }
    
    capital_history[i] <- capital
  }
  
  num_losses <- num_bets - num_wins
  avg_odds   <- ifelse(num_bets > 0, sum_of_odds / num_bets, NA)
  
  list(
    capital_history = capital_history,
    final_capital   = capital,
    num_bets        = num_bets,
    num_wins        = num_wins,
    num_losses      = num_losses,
    avg_odds        = avg_odds
  )
}

###############################################################################
# 6) Run the four strategies (now with 10% tax each bet)
###############################################################################

# (A) Bayesian Dynamic
bayes_dynamic <- simulate_dynamic(
  df,
  bet_home_flag = "bet_home_bayes",
  bet_draw_flag = "bet_draw_bayes",
  bet_away_flag = "bet_away_bayes",
  odds_home     = "B365H",
  odds_draw     = "B365D",
  odds_away     = "B365A",
  result_col    = "match_result_cat",
  initial_capital = 100,
  fraction = 0.02,       # dynamic stake fraction
  tax_rate = 0.1        # 10% tax
)

# (B) Random Forest Dynamic
rf_dynamic <- simulate_dynamic(
  df,
  bet_home_flag = "bet_home_rf",
  bet_draw_flag = "bet_draw_rf",
  bet_away_flag = "bet_away_rf",
  odds_home     = "B365H",
  odds_draw     = "B365D",
  odds_away     = "B365A",
  result_col    = "match_result_cat",
  initial_capital = 100,
  fraction = 0.02,
  tax_rate = 0.1
)

# (C) Bayesian Fixed
bayes_fixed <- simulate_fixed_stake(
  df,
  bet_home_flag = "bet_home_bayes",
  bet_draw_flag = "bet_draw_bayes",
  bet_away_flag = "bet_away_bayes",
  odds_home     = "B365H",
  odds_draw     = "B365D",
  odds_away     = "B365A",
  result_col    = "match_result_cat",
  initial_capital = 100,
  stake_size = 0.5,
  tax_rate = 0.1
)

# (D) Random Forest Fixed
rf_fixed <- simulate_fixed_stake(
  df,
  bet_home_flag = "bet_home_rf",
  bet_draw_flag = "bet_draw_rf",
  bet_away_flag = "bet_away_rf",
  odds_home     = "B365H",
  odds_draw     = "B365D",
  odds_away     = "B365A",
  result_col    = "match_result_cat",
  initial_capital = 100,
  stake_size = 0.5,
  tax_rate = 0.1
)

###############################################################################
# 7) Summarize Results
###############################################################################
compute_profit <- function(final_capital, initial_capital=100) {
  profit <- final_capital - initial_capital
  roi    <- profit / initial_capital
  list(profit=profit, roi=roi)
}

bayes_dynamic_stats <- compute_profit(bayes_dynamic$final_capital, 100)
rf_dynamic_stats    <- compute_profit(rf_dynamic$final_capital, 100)
bayes_fixed_stats   <- compute_profit(bayes_fixed$final_capital, 100)
rf_fixed_stats      <- compute_profit(rf_fixed$final_capital, 100)

results_table <- data.frame(
  Strategy      = c("Bayesian (Dynamic, w. Tax)",
                    "RF (Dynamic, w. Tax)",
                    "Bayesian (Fixed, w. Tax)",
                    "RF (Fixed, w. Tax)"),
  Num_Bets      = c(bayes_dynamic$num_bets,
                    rf_dynamic$num_bets,
                    bayes_fixed$num_bets,
                    rf_fixed$num_bets),
  Wins          = c(bayes_dynamic$num_wins,
                    rf_dynamic$num_wins,
                    bayes_fixed$num_wins,
                    rf_fixed$num_wins),
  Losses        = c(bayes_dynamic$num_losses,
                    rf_dynamic$num_losses,
                    bayes_fixed$num_losses,
                    rf_fixed$num_losses),
  Avg_Odds      = c(bayes_dynamic$avg_odds,
                    rf_dynamic$avg_odds,
                    bayes_fixed$avg_odds,
                    rf_fixed$avg_odds),
  Final_Capital = c(bayes_dynamic$final_capital,
                    rf_dynamic$final_capital,
                    bayes_fixed$final_capital,
                    rf_fixed$final_capital),
  Profit        = c(bayes_dynamic_stats$profit,
                    rf_dynamic_stats$profit,
                    bayes_fixed_stats$profit,
                    rf_fixed_stats$profit),
  ROI           = c(bayes_dynamic_stats$roi,
                    rf_dynamic_stats$roi,
                    bayes_fixed_stats$roi,
                    rf_fixed_stats$roi)
)

print(results_table)

###############################################################################
# 8) (Optional) Save Results as LaTeX Table
###############################################################################
output_file <- "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/results/betting_strategies_summary.tex"
if (file.exists(output_file)) file.remove(output_file)

sink(output_file)
stargazer(
  results_table,
  summary  = FALSE,
  rownames = FALSE,
  title    = "Comparison of 4 Betting Strategies with 10% Bet Tax",
  label    = "tab:betting_comparison",
  float    = TRUE,
  align    = TRUE,
  digits   = 3,
  out      = output_file
)
sink()

cat("LaTeX table saved to:", output_file, "\n")

###############################################################################
# 9) Plot total capital over time for all four strategies
###############################################################################
df_plot <- data.frame(
  match_index = rep(seq_len(nrow(df)), 4),
  capital     = c(
    bayes_dynamic$capital_history,
    rf_dynamic$capital_history,
    bayes_fixed$capital_history,
    rf_fixed$capital_history
  ),
  strategy    = rep(c("Bayes Dyn (Tax)", "RF Dyn (Tax)",
                      "Bayes Fixed (Tax)", "RF Fixed (Tax)"), 
                    each = nrow(df))
)

p <- ggplot(df_plot, aes(x = match_index, y = capital, color = strategy)) +
  geom_line() +
  labs(title = "Capital Over Time (4 Strategies, 10% Tax per Bet)",
       x     = "Match Index",
       y     = "Capital",
       color = "Strategy") +
  theme_minimal()

plot_path <- "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/results/capital_over_time.png"
ggsave(plot_path, p, width = 10, height = 6, dpi = 300)

cat("Plot of capital over time saved to:", plot_path, "\n")
