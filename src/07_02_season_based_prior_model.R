###############################################################################
# 07_02_season_based_prior_model_updated.R
#
# Bayesian ordinal model with an explicit home-advantage parameter
# (Season-based prior, but the very first season is zero-centered)
# Now including min-max normalization of the performance measure.
#
# This version includes:
#   - The interactive if-statement for re-running or loading the model (from 07).
#   - LaTeX table generation with stargazer.
#   - Plotting, including a final team-strength plot that only shows teams that
#     actually played in the final (current) season.
#   - Model code is unchanged from the original 07_02 file.
#
# ADDED:
#   1) Single traceplot for team_strength[1,1], beta[1], c[1], c[2].
#   2) Season-based team strength posterior plot excluding season 7.
###############################################################################

rm(list = ls())
library(rstan)
library(dplyr)
library(ggplot2)
library(stargazer)

# Use multiple cores for faster sampling and auto-write compiled models to disk
options(mc.cores = parallel::detectCores())

# -----------------------------------------------------------------------------
# 1) Stan Model Code -- UNCHANGED
# -----------------------------------------------------------------------------
stan_model_code <- "
data {
  int<lower=1> N;                  // number of matches
  int<lower=2> K;                  // number of outcome categories (3)
  int<lower=1> T;                  // number of teams
  int<lower=1> S;                  // number of seasons
  int<lower=1,upper=T> home_team[N]; 
  int<lower=1,upper=T> away_team[N];
  int<lower=1,upper=S> season_id[N];  
  int<lower=1,upper=K> y[N];       // observed outcome

  int<lower=0> P;                  // number of additional predictors
  matrix[N, P] X;                  // predictor matrix

  // Performance measure for each (season, team), shape: S x T, normalized to [-1, +1]
  matrix[S, T] performance;
}

parameters {
  // Team strengths, row=s (season), col=t (team)
  matrix[S, T] team_strength;
  
  // Standard deviation for how tightly we trust performance each season
  real<lower=0> sigma_season;
  
  // Std dev for the very first season prior
  real<lower=0> sigma_team_init;
  
  // Ordered logistic cutpoints
  ordered[K-1] c;
  
  // Regression coefficients
  vector[P] beta;
  
  // Home advantage
  real home_adv;
}

model {
  // 1) First season prior
  for (t in 1:T) {
    team_strength[1, t] ~ normal(0, sigma_team_init);
  }

  // 2) For each subsequent season s=2..S
  for (s in 2:S) {
    for (t in 1:T) {
      team_strength[s, t] ~ normal(performance[s, t], sigma_season);
    }
  }

  // 3) Other priors
  sigma_season ~ normal(0, 2);
  sigma_team_init ~ normal(0, 2);
  c ~ normal(0, 5);
  beta ~ normal(0, 5);
  home_adv ~ normal(0, 1);

  // 4) Likelihood
  for (n in 1:N) {
    real eta = (team_strength[season_id[n], home_team[n]] + home_adv)
               - team_strength[season_id[n], away_team[n]]
               + dot_product(X[n], beta);
    y[n] ~ ordered_logistic(eta, c);
  }
}

generated quantities {
  // Store linear predictor for each match
  vector[N] eta;
  for (n in 1:N) {
    eta[n] = (team_strength[season_id[n], home_team[n]] + home_adv)
             - team_strength[season_id[n], away_team[n]]
             + dot_product(X[n], beta);
  }
}
"

# -----------------------------------------------------------------------------
# 2) TRAINING DATA PREPARATION
# -----------------------------------------------------------------------------
data_raw <- read.csv(
  "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/train_data.csv", 
  stringsAsFactors = FALSE
)

# Convert match_result: -1=Away Win, 0=Draw, 1=Home Win --> 1,2,3 (for Stan)
data_raw <- data_raw %>%
  mutate(
    match_result_cat = as.integer(
      factor(match_result, levels = c(-1, 0, 1), labels = c(1, 2, 3))
    )
  )

# Keep only relevant columns
principal_component_names <- c("home_team", "away_team", "match_result_cat",
                               "PC1", "PC2", "PC3", "PC4", "PC5", 
                               "PC6", "PC7", "PC8", "PC9")
data_raw <- data_raw[, (names(data_raw) %in% principal_component_names) |
                       names(data_raw) %in% c("season_number",
                                              "home_points_prev_season",
                                              "away_points_prev_season")]

# Drop rows with missing predictor values
predictor_cols <- setdiff(
  names(data_raw),
  c("match_result", "match_result_cat", "home_team", "away_team", 
    "season_number", "home_points_prev_season", "away_points_prev_season")
)
data_raw <- data_raw[complete.cases(data_raw[, predictor_cols]), ]

# -----------------------------------------------------------------------------
# 3) READ TEST DATA & COMBINE SEASONS/TEAMS
# -----------------------------------------------------------------------------
test_file <- "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/test_data.csv"
test_data <- read.csv(test_file, stringsAsFactors = FALSE)

# 3A) Combine training & test data to unify season indexing
all_data_seasons <- dplyr::bind_rows(
  data_raw %>% dplyr::select(season_number),
  test_data %>% dplyr::select(season_number)
)
unique_seasons <- sort(unique(all_data_seasons$season_number))
S_num <- length(unique_seasons)
season_index <- setNames(seq_len(S_num), unique_seasons)

# Overwrite training data's season_id
data_raw$season_id <- season_index[data_raw$season_number]

# 3B) Combine training & test data to unify team indexing
all_teams <- unique(c(data_raw$home_team, data_raw$away_team,
                      test_data$home_team,  test_data$away_team))
teams <- sort(all_teams)
T_num <- length(teams)
team_index <- setNames(seq_len(T_num), teams)

# Overwrite training data's team_idx
data_raw$home_team_idx <- team_index[data_raw$home_team]
data_raw$away_team_idx <- team_index[data_raw$away_team]

# -----------------------------------------------------------------------------
# 3C) BUILD THE PERFORMANCE MATRIX (from previous season points, scaled to [-1, +1])
# -----------------------------------------------------------------------------
test_data_tmp <- test_data %>%
  select(
    season_number, home_team, away_team,
    home_points_prev_season, away_points_prev_season
  ) %>%
  mutate(
    home_team_idx = team_index[home_team],
    away_team_idx = team_index[away_team]
  )

df_home_train <- data.frame(
  team = data_raw$home_team,
  season_number = data_raw$season_number,
  perf = data_raw$home_points_prev_season
)
df_away_train <- data.frame(
  team = data_raw$away_team,
  season_number = data_raw$season_number,
  perf = data_raw$away_points_prev_season
)
df_home_test <- data.frame(
  team = test_data_tmp$home_team,
  season_number = test_data_tmp$season_number,
  perf = test_data_tmp$home_points_prev_season
)
df_away_test <- data.frame(
  team = test_data_tmp$away_team,
  season_number = test_data_tmp$season_number,
  perf = test_data_tmp$away_points_prev_season
)

df_perf_combined <- dplyr::bind_rows(
  df_home_train, df_away_train,
  df_home_test,  df_away_test
)

# Summarize by (team, season_number)
df_perf_combined <- df_perf_combined %>%
  group_by(team, season_number) %>%
  summarise(performance_value = mean(perf, na.rm = TRUE)) %>%
  ungroup()

# Min-max scale to [-1, +1]
minVal <- min(df_perf_combined$performance_value, na.rm = TRUE)
maxVal <- max(df_perf_combined$performance_value, na.rm = TRUE)
rangeVal <- maxVal - minVal
if (rangeVal == 0) {
  df_perf_combined$performance_value <- 0
} else {
  df_perf_combined$performance_value <- -1 + 2 * (
    (df_perf_combined$performance_value - minVal) / rangeVal
  )
}

# Build matrix[S_num, T_num]
performance_mat <- matrix(0, nrow = S_num, ncol = T_num)
for (i in seq_len(nrow(df_perf_combined))) {
  s_idx <- season_index[ df_perf_combined$season_number[i] ]
  t_idx <- team_index[ df_perf_combined$team[i] ]
  performance_mat[s_idx, t_idx] <- df_perf_combined$performance_value[i]
}

# -----------------------------------------------------------------------------
# 4) PREPARE TRAINING STAN DATA
# -----------------------------------------------------------------------------
X <- as.matrix(data_raw[, predictor_cols])
P <- ncol(X)

stan_data <- list(
  N          = nrow(data_raw),
  K          = 3,
  T          = T_num,
  S          = S_num,
  home_team  = data_raw$home_team_idx,
  away_team  = data_raw$away_team_idx,
  y          = data_raw$match_result_cat,
  P          = P,
  X          = X,
  season_id  = data_raw$season_id,
  performance = performance_mat
)

# -----------------------------------------------------------------------------
# 5) MODEL FITTING WITH STAN -- IF-STATEMENT & SAVING
# -----------------------------------------------------------------------------
output_dir <- "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/results/bayes_season_prior"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}
fit_file <- file.path(output_dir, "stan_fit_season_prior.rds")

cat("Do you want to rerun the MCMC sampling? (yes/no): ")
user_input <- tolower(readline())

if (user_input == "yes") {
  cat("Compiling and running MCMC sampling...\n")
  
  stan_model <- stan_model(model_code = stan_model_code)
  fit <- sampling(
    stan_model,
    data    = stan_data,
    iter    = 5000,
    warmup  = 1000,
    chains  = 4,
    seed    = 123,
    control = list(adapt_delta = 0.95),
    refresh = 100
  )
  
  saveRDS(fit, fit_file)
  cat("Stan model fit saved to:", fit_file, "\n")
  
} else {
  if (file.exists(fit_file)) {
    cat("Loading saved Stan model fit...\n")
    fit <- readRDS(fit_file)
    cat("Stan model fit loaded successfully!\n")
  } else {
    stop("No saved Stan model fit found! You need to run MCMC at least once.")
  }
}

print(fit, pars = c("team_strength", "home_adv", "c",
                    "sigma_season", "sigma_team_init", "beta"))

# -----------------------------------------------------------------------------
# 6) POSTERIOR DIAGNOSTICS / TEAM STRENGTHS
# -----------------------------------------------------------------------------
posterior_samples <- rstan::extract(fit)

###############################################################################
# 6A) NEW: Single Combined Traceplot for:
#    1) team_strength[1,1]  (Team=1, Season=1)
#    2) beta[1]
#    3) cutpoints c[1] and c[2]
###############################################################################
library(bayesplot)  # for mcmc_trace
posterior_array <- as.array(fit)

# Choose any subset you like; here's an example for:
# team_strength[1,1], beta[1], c[1], c[2]
params_for_trace <- c("team_strength[1,1]", "beta[1]", "c[1]", "c[2]")

trace_plot <- mcmc_trace(posterior_array, pars = params_for_trace) +
  ggplot2::theme_minimal()

ggsave(
  filename = file.path(output_dir, "traceplot_combined.png"),
  plot     = trace_plot,
  width    = 10,
  height   = 7,
  dpi      = 300
)
cat("Saved combined traceplot to 'traceplot_combined.png'\n")

###############################################################################
# 6B) Posterior Means for Final Season + Plot
###############################################################################
final_season_idx    <- S_num
final_season_number <- unique_seasons[final_season_idx]
team_strength_means <- colMeans(posterior_samples$team_strength[, final_season_idx, ])

team_strength_df <- data.frame(
  team     = teams,
  strength = team_strength_means
)

teams_in_final_season <- unique(c(
  data_raw %>% filter(season_number == final_season_number) %>% pull(home_team),
  data_raw %>% filter(season_number == final_season_number) %>% pull(away_team),
  test_data %>% filter(season_number == final_season_number) %>% pull(home_team),
  test_data %>% filter(season_number == final_season_number) %>% pull(away_team)
))

team_strength_df <- team_strength_df %>%
  filter(team %in% teams_in_final_season)

ggplot(team_strength_df, aes(x = reorder(team, strength), y = strength)) +
  geom_point() +
  coord_flip() +
  labs(x = "Team", y = "Estimated Strength (Final Season)", 
       title = paste("Posterior Mean Estimates - Season", final_season_number))

###############################################################################
# 6C) Example: Posterior Distribution for home_adv
###############################################################################
home_adv_mean <- mean(posterior_samples$home_adv)
home_adv_ci   <- quantile(posterior_samples$home_adv, probs = c(0.025, 0.975))
cat("Home advantage mean:", home_adv_mean, 
    "95% CI:", home_adv_ci[1], "to", home_adv_ci[2], "\n")

df_home_adv <- data.frame(home_adv = posterior_samples$home_adv)
ggplot(df_home_adv, aes(x = home_adv)) +
  geom_histogram(bins = 50, alpha = 0.7) +
  geom_vline(xintercept = mean(df_home_adv$home_adv), linetype = "dashed") +
  labs(title = "Posterior Distribution of Home Advantage", x = "home_adv")

###############################################################################
# 6D) NEW: Plot Season-Based Team Strength (for a chosen team) Excluding Season 7
###############################################################################
team_idx_to_plot <- 1  # for example, team #1
n_draws <- dim(posterior_samples$team_strength)[1]
S       <- dim(posterior_samples$team_strength)[2]

df_team_season <- data.frame(
  draw   = rep(1:n_draws, times = S),
  season = rep(1:S, each = n_draws),
  value  = as.vector(posterior_samples$team_strength[ , , team_idx_to_plot])
)

# Exclude season 7
df_team_season <- df_team_season %>% filter(season != 7)

p_excl_season7 <- ggplot(df_team_season, aes(x = value)) +
  geom_histogram(bins = 40, alpha = 0.7) +
  facet_wrap(~ season, scales = "free_y") +
  geom_vline(xintercept = 0, linetype = "dashed") +  # optional reference line
  labs(
    title = paste("Team", team_idx_to_plot, ": Posterior Strength (Excl. Season 7)"),
    x = "Team Strength",
    y = "Count"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(output_dir, "team_strength_excl_season7.png"),
  plot     = p_excl_season7,
  width    = 10,
  height   = 6,
  dpi      = 300
)
cat("Saved team-strength distribution excluding season 7.\n")

# -----------------------------------------------------------------------------
# 6E) Generate LaTeX Summary Table via stargazer
# -----------------------------------------------------------------------------
output_file <- file.path(output_dir, "Bayesian_Season_Prior_Model_Results.tex")

if (file.exists(output_file)) {
  file.remove(output_file)
}

summary_fit <- summary(fit)$summary

params_subset <- c("home_adv", "sigma_season", "sigma_team_init", "c[1]", "c[2]")
subset_fit <- summary_fit[params_subset, ]
subset_fit <- subset_fit[, c("mean", "se_mean", "n_eff", "Rhat")]

subset_fit <- data.frame(Parameter = rownames(subset_fit), subset_fit, row.names = NULL)
colnames(subset_fit) <- c("Parameter", "Mean", "SE Mean", "Effective Sample Size (N_eff)", "Rhat")

sink(output_file)
stargazer(subset_fit, summary = FALSE, float = TRUE, align = TRUE,
          title = "Summary MCMC Estimation (Season-based Prior Model)",
          label = "tab:mcmc_summary_season",
          out = output_file)
sink()

cat("LaTeX table saved in:", output_file, "\n")

# -----------------------------------------------------------------------------
# 7) TRAINING SET POSTERIOR PREDICTIVE CHECK
# -----------------------------------------------------------------------------
posterior_thinned <- lapply(posterior_samples, function(x) {
  if (is.matrix(x) || is.data.frame(x)) {
    x[seq(1, nrow(x), by = 10), , drop = FALSE]
  } else if (is.vector(x)) {
    x[seq(1, length(x), by = 10)]
  } else {
    x
  }
})

S_draws <- dim(posterior_thinned$c)[1]
N       <- dim(posterior_thinned$eta)[2]

post_probs <- array(0, dim = c(N, 3, S_draws))
for (s in 1:S_draws) {
  c1 <- posterior_thinned$c[s,1]
  c2 <- posterior_thinned$c[s,2]
  for (n in 1:N) {
    eta_n <- posterior_thinned$eta[s,n]
    p1 <- plogis(c1 - eta_n)
    p2 <- plogis(c2 - eta_n)
    post_probs[n,1,s] <- p1
    post_probs[n,2,s] <- p2 - p1
    post_probs[n,3,s] <- 1 - p2
  }
}

mean_probs <- apply(post_probs, c(1,2), mean)
predicted_cat <- apply(mean_probs, 1, which.max)
true_cat <- data_raw$match_result_cat
accuracy <- mean(predicted_cat == true_cat, na.rm = TRUE)
cat("In-sample accuracy (train):", accuracy, "\n")

# -----------------------------------------------------------------------------
# 8) TEST SET PREDICTIONS
# -----------------------------------------------------------------------------
test_data <- test_data %>%
  mutate(
    season_id = season_index[season_number],
    home_team_idx = team_index[home_team],
    away_team_idx = team_index[away_team],
    match_result_cat = as.integer(
      factor(match_result, levels = c(-1,0,1), labels = c(1,2,3))
    )
  )

predictor_cols_test <- c("PC1","PC2","PC3","PC4","PC5","PC6","PC7")
X_test <- as.matrix(test_data[, predictor_cols_test])
N_test <- nrow(test_data)

posterior      <- rstan::extract(fit)
S_full         <- dim(posterior$c)[1]
post_probs_test <- array(0, dim = c(N_test, 3, S_full))

for (s in 1:S_full) {
  c1 <- posterior$c[s,1]
  c2 <- posterior$c[s,2]
  ts <- posterior$team_strength[s, , ]
  beta_s <- posterior$beta[s, ]
  home_adv_s <- posterior$home_adv[s]
  
  for (n in 1:N_test) {
    eta_test <- (ts[test_data$season_id[n], test_data$home_team_idx[n]] + home_adv_s) -
      ts[test_data$season_id[n], test_data$away_team_idx[n]] +
      sum(X_test[n, ] * beta_s)
    p1 <- plogis(c1 - eta_test)
    p2 <- plogis(c2 - eta_test)
    post_probs_test[n, 1, s] <- p1
    post_probs_test[n, 2, s] <- p2 - p1
    post_probs_test[n, 3, s] <- 1 - p2
  }
}

mean_probs_test     <- apply(post_probs_test, c(1,2), mean)
predicted_cat_test  <- apply(mean_probs_test, 1, which.max)
true_cat_test       <- test_data$match_result_cat

accuracy_test <- mean(predicted_cat_test == true_cat_test, na.rm = TRUE)
cat("Out-of-sample (test) accuracy:", accuracy_test, "\n")

conf_matrix_test <- table(Actual = true_cat_test, Predicted = predicted_cat_test)
cat("Out-of-sample (test) confusion matrix:\n")
print(conf_matrix_test)

conf_df_test <- as.data.frame(conf_matrix_test)
colnames(conf_df_test) <- c("Actual", "Predicted", "Freq")
conf_df_test$Actual <- factor(conf_df_test$Actual, levels = c(1,2,3),
                              labels = c("Away Win", "Draw", "Home Win"))
conf_df_test$Predicted <- factor(conf_df_test$Predicted, levels = c(1,2,3),
                                 labels = c("Away Win", "Draw", "Home Win"))

ggplot(conf_df_test, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "grey70") +
  geom_text(aes(label = Freq), color = "black") +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(x="Predicted Result", y="Actual Result",
       title="Out-of-sample Confusion Matrix") +
  theme_minimal()

###############################################################################
# 9) ADD CREDIBLE INTERVALS & CREATE RESULTS DATAFRAME
###############################################################################
# Here we change the quantiles from 25% and 75% to 5% and 95% for a 90% credible interval.
p1_quants <- apply(post_probs_test[, 1, ], 1, quantile, probs = c(0.025, 0.975))
p2_quants <- apply(post_probs_test[, 2, ], 1, quantile, probs = c(0.05, 0.975))
p3_quants <- apply(post_probs_test[, 3, ], 1, quantile, probs = c(0.05, 0.975))

results <- data.frame(
  home_team = test_data$home_team,
  away_team = test_data$away_team,
  date = test_data$date,
  Match_report = test_data$Match_report,
  prob_away_mean_bayes_adv  = mean_probs_test[, 1],
  prob_away_lower_bayes_adv = p1_quants[1, ],
  prob_away_upper_bayes_adv = p1_quants[2, ],
  prob_draw_mean_bayes_adv  = mean_probs_test[, 2],
  prob_draw_lower_bayes_adv = p2_quants[1, ],
  prob_draw_upper_bayes_adv = p2_quants[2, ],
  prob_home_mean_bayes_adv  = mean_probs_test[, 3],
  prob_home_lower_bayes_adv = p3_quants[1, ],
  prob_home_upper_bayes_adv = p3_quants[2, ],
  predicted_category_bayes_adv = predicted_cat_test,
  predicted_label_bayes_adv = factor(predicted_cat_test,
                                     levels = c(1, 2, 3),
                                     labels = c("Away Win", "Draw", "Home Win")),
  true_category = true_cat_test
)

write.csv(results, "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/results_bayes_season.csv", row.names = FALSE)
cat("Saved 'results_bayes_season.csv' with mean probabilities and 90% credible intervals.\n")

###############################################################################
# End of 07_02_season_based_prior_model_updated.R
###############################################################################
