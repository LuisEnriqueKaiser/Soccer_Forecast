###############################################################################
# 07_02_season_based_prior_model_corrected.R
#
# Bayesian ordinal model with an explicit home-advantage parameter
# (Season-based prior). 
# This corrected script fixes:
#   1) Inconsistency in predictor columns (train vs test).
#   2) Attempting to select home_team / away_team after dropping them.
#   3) Retains the home_team/away_team string columns, so we can label plots.
###############################################################################

rm(list = ls())
library(rstan)
library(dplyr)
library(ggplot2)
library(stargazer)
library(bayesplot)

options(mc.cores = parallel::detectCores())

# -----------------------------------------------------------------------------
# 1) Stan Model Code
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
  sigma_season ~ normal(0, 5);
  sigma_team_init ~ normal(0, 5);
  c ~ normal(0, 5);
  beta ~ normal(0, 3);
  home_adv ~ normal(0, 4);

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

# Keep the columns we need, including home_team/away_team for labeling.
# *IMPORTANT*: Include 'home_team' and 'away_team' so we don't break the code
# that merges them later.
principal_component_names <- c(
  "home_team", "away_team",        # keep names for labeling final-season plot
  "home_team_idx", "away_team_idx",
  "season_id", 
  "match_result_cat",
  # PC columns
  "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9",
  # Points from previous season
  "home_points_prev_season",
  "away_points_prev_season"
)

# Overwrite indices from your custom integer columns
data_raw$home_team_idx <- as.integer(data_raw$home_team_integer)
data_raw$away_team_idx <- as.integer(data_raw$away_team_integer)
data_raw$season_id     <- as.integer(data_raw$season_number)

# Subset to only these columns
data_raw <- data_raw[ , names(data_raw) %in% principal_component_names]

# We'll define the predictor columns *explicitly* as the PC columns only:
predictor_cols <- c("PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9")

# Drop rows with missing predictor values
data_raw <- data_raw[complete.cases(data_raw[, predictor_cols]), ]

# -----------------------------------------------------------------------------
# 3) TEST DATA & DETERMINE S, T
# -----------------------------------------------------------------------------
test_file <- "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/test_data.csv"
test_data <- read.csv(test_file, stringsAsFactors = FALSE)
test_data$season_id     <- as.integer(test_data$season_number)
test_data$home_team_idx <- as.integer(test_data$home_team_integer)
test_data$away_team_idx <- as.integer(test_data$away_team_integer)

S_num <- max(c(data_raw$season_id,  test_data$season_id))
T_num <- max(c(data_raw$home_team_idx, data_raw$away_team_idx,
               test_data$home_team_idx, test_data$away_team_idx))

# 3C) BUILD PERFORMANCE MATRIX: (S_num x T_num)
df_home_train <- data.frame(
  season_id = data_raw$season_id,
  team_idx  = data_raw$home_team_idx,
  perf      = data_raw$home_points_prev_season
)
df_away_train <- data.frame(
  season_id = data_raw$season_id,
  team_idx  = data_raw$away_team_idx,
  perf      = data_raw$away_points_prev_season
)
df_home_test <- data.frame(
  season_id = test_data$season_id,
  team_idx  = test_data$home_team_idx,
  perf      = test_data$home_points_prev_season
)
df_away_test <- data.frame(
  season_id = test_data$season_id,
  team_idx  = test_data$away_team_idx,
  perf      = test_data$away_points_prev_season
)

df_perf_combined <- bind_rows(
  df_home_train, df_away_train,
  df_home_test, df_away_test
)

df_perf_combined <- df_perf_combined %>%
  group_by(season_id, team_idx) %>%
  summarise(performance_value = mean(perf, na.rm = TRUE)) %>%
  ungroup()

# Min–max scale to [-1, +1]
minVal  <- min(df_perf_combined$performance_value, na.rm = TRUE)
maxVal  <- max(df_perf_combined$performance_value, na.rm = TRUE)
rangeVal <- maxVal - minVal

if (rangeVal == 0) {
  df_perf_combined$performance_value <- 0
} else {
  df_perf_combined$performance_value <- -1 + 2 * (
    (df_perf_combined$performance_value - minVal) / rangeVal
  )
}

performance_mat <- matrix(0, nrow = S_num, ncol = T_num)
for (i in seq_len(nrow(df_perf_combined))) {
  s_idx <- df_perf_combined$season_id[i]
  t_idx <- df_perf_combined$team_idx[i]
  performance_mat[s_idx, t_idx] <- df_perf_combined$performance_value[i]
}

# -----------------------------------------------------------------------------
# 4) PREPARE TRAINING STAN DATA
# -----------------------------------------------------------------------------
X <- as.matrix(data_raw[, predictor_cols])  # 9 columns
P <- ncol(X)

stan_data <- list(
  N           = nrow(data_raw),
  K           = 3,
  T           = T_num,
  S           = S_num,
  home_team   = data_raw$home_team_idx,
  away_team   = data_raw$away_team_idx,
  y           = data_raw$match_result_cat,
  P           = P,
  X           = X,
  season_id   = data_raw$season_id,
  performance = performance_mat
)

# -----------------------------------------------------------------------------
# 5) MODEL FITTING
# -----------------------------------------------------------------------------
output_dir <- "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/results/bayes_season_prior"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}
fit_file <- file.path(output_dir, "stan_fit_season_prior.rds")

cat("Do you want to rerun the MCMC sampling? (yes/no): ")
user_input <- tolower(readline())
#yes

if (user_input == "yes") {
  cat("Compiling and running MCMC sampling...\n")
  stan_model <- stan_model(model_code = stan_model_code)
  fit <- sampling(
    stan_model,
    data    = stan_data,
    iter    = 1000,
    warmup  = 200,
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

# 6A) Traceplot for a few parameters
posterior_array <- as.array(fit)
params_for_trace <- c("team_strength[1,1]", "beta[1]", "c[1]", "c[2]")
trace_plot <- mcmc_trace(posterior_array, pars = params_for_trace) + ggplot2::theme_minimal()

ggsave(
  filename = file.path(output_dir, "traceplot_combined.png"),
  plot     = trace_plot,
  width    = 10,
  height   = 7,
  dpi      = 300
)
cat("Saved combined traceplot to 'traceplot_combined.png'\n")

# 6B) Posterior Means for Final Season + Plot (with Team Names)
final_season_idx <- S_num
team_strength_means <- colMeans(posterior_samples$team_strength[, final_season_idx, ])

df_strength_final <- data.frame(
  team_idx = seq_len(T_num),
  strength = team_strength_means
)

# Restrict to teams that appear in final season
teams_in_final <- unique(c(
  data_raw  %>% filter(season_id == final_season_idx) %>% 
    select(home_team_idx, away_team_idx) %>% unlist(),
  test_data %>% filter(season_id == final_season_idx) %>%
    select(home_team_idx, away_team_idx) %>% unlist()
))

df_strength_final <- df_strength_final %>%
  filter(team_idx %in% teams_in_final)

# Now we can safely build a team-names data frame, because we *kept* home_team/away_team
team_names_df <- bind_rows(
  data_raw %>% select(team = home_team, idx = home_team_idx),
  data_raw %>% select(team = away_team, idx = away_team_idx)
) %>% distinct() %>% arrange(idx)

df_strength_final <- merge(
  df_strength_final,
  team_names_df,
  by.x = "team_idx", by.y = "idx", all.x = TRUE
)

# Quick plot of final-season strengths
ggplot(df_strength_final, aes(x = reorder(team, strength), y = strength)) +
  geom_point() +
  coord_flip() +
  labs(
    x     = "Team",
    y     = "Estimated Strength (Final Season)",
    title = paste("Posterior Mean Estimates - Season", final_season_idx)
  )

# 6C) Posterior distribution for home_adv
home_adv_samples <- posterior_samples$home_adv
home_adv_mean <- mean(home_adv_samples)
home_adv_ci   <- quantile(home_adv_samples, probs = c(0.025, 0.975))
cat("Home advantage mean:", home_adv_mean,
    "95% CI:", home_adv_ci[1], "to", home_adv_ci[2], "\n")

df_home_adv <- data.frame(home_adv = home_adv_samples)
ggplot(df_home_adv, aes(x = home_adv)) +
  geom_histogram(bins = 50, alpha = 0.7) +
  geom_vline(xintercept = home_adv_mean, linetype = "dashed") +
  labs(title = "Posterior Distribution of Home Advantage", x = "home_adv")

# 6D) Example: Plot Season-Based Strength (excluding final season)
team_idx_to_plot <- 1
n_draws <- dim(posterior_samples$team_strength)[1]
S       <- dim(posterior_samples$team_strength)[2]
df_team_season <- data.frame(
  draw   = rep(seq_len(n_draws), times = S),
  season = rep(seq_len(S), each = n_draws),
  value  = as.vector(posterior_samples$team_strength[ , , team_idx_to_plot])
)
df_team_season <- df_team_season %>% filter(season != S_num)

ggplot(df_team_season, aes(x = value)) +
  geom_histogram(bins = 40, alpha = 0.7) +
  facet_wrap(~ season, scales = "free_y") +
  labs(
    title = paste("Team", team_idx_to_plot, ": Posterior Strength (Excl. Season", S_num, ")"),
    x = "Team Strength", y = "Count"
  ) +
  theme_minimal()

# 6E) LaTeX Summary
summary_fit <- summary(fit)$summary
params_subset <- c("home_adv", "sigma_season", "sigma_team_init", "c[1]", "c[2]")
subset_fit <- summary_fit[params_subset, c("mean", "se_mean", "n_eff", "Rhat")]
subset_fit <- data.frame(Parameter = rownames(subset_fit), subset_fit, row.names = NULL)
colnames(subset_fit) <- c("Parameter", "Mean", "SE Mean", "Effective Sample Size (N_eff)", "Rhat")

output_file <- file.path(output_dir, "Bayesian_Season_Prior_Model_Results.tex")
if (file.exists(output_file)) {
  file.remove(output_file)
}
sink(output_file)
stargazer(
  subset_fit,
  summary = FALSE, float = TRUE, align = TRUE,
  title = "Summary MCMC Estimation (Season-based Prior Model)",
  label = "tab:mcmc_summary_season",
  out   = output_file
)
sink()
cat("LaTeX table saved in:", output_file, "\n")

# -----------------------------------------------------------------------------
# 7) IN-SAMPLE POSTERIOR PREDICTIVE CHECK
# -----------------------------------------------------------------------------
posterior_thinned <- lapply(posterior_samples, function(x) {
  if (is.matrix(x) || is.data.frame(x)) {
    # Thin by factor of 10
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

for (s in seq_len(S_draws)) {
  c1 <- posterior_thinned$c[s,1]
  c2 <- posterior_thinned$c[s,2]
  
  for (n in seq_len(N)) {
    eta_n <- posterior_thinned$eta[s,n]
    p1 <- plogis(c1 - eta_n)
    p2 <- plogis(c2 - eta_n)
    post_probs[n,1,s] <- p1
    post_probs[n,2,s] <- p2 - p1
    post_probs[n,3,s] <- 1 - p2
  }
}

mean_probs    <- apply(post_probs, c(1,2), mean)
predicted_cat <- apply(mean_probs, 1, which.max)
true_cat      <- data_raw$match_result_cat
accuracy      <- mean(predicted_cat == true_cat, na.rm = TRUE)
cat("In-sample accuracy (train):", accuracy, "\n")

# -----------------------------------------------------------------------------
# 8) TEST SET PREDICTIONS
# -----------------------------------------------------------------------------
# The test set must use the *same* predictor columns as training (9 PCs)


predictor_cols_test <- predictor_cols
X_test <- as.matrix(test_data[, predictor_cols_test])  # 9 columns
N_test <- nrow(test_data)

posterior <- rstan::extract(fit)
S_full    <- dim(posterior$c)[1]
post_probs_test <- array(0, dim = c(N_test, 3, S_full))

for (s in seq_len(S_full)) {
  c1 <- posterior$c[s,1]
  c2 <- posterior$c[s,2]
  ts <- posterior$team_strength[s, , ]
  beta_s <- posterior$beta[s, ]
  home_adv_s <- posterior$home_adv[s]
  
  for (n in seq_len(N_test)) {
    eta_test <- (ts[test_data$season_id[n], test_data$home_team_idx[n]] + home_adv_s) -
      ts[test_data$season_id[n], test_data$away_team_idx[n]] +
      sum(X_test[n, ] * beta_s)
    p1 <- plogis(c1 - eta_test)
    p2 <- plogis(c2 - eta_test)
    post_probs_test[n,1,s] <- p1
    post_probs_test[n,2,s] <- p2 - p1
    post_probs_test[n,3,s] <- 1 - p2
  }
}

mean_probs_test    <- apply(post_probs_test, c(1,2), mean)
predicted_cat_test <- apply(mean_probs_test, 1, which.max)
true_cat_test      <- test_data$match_result_cat

accuracy_test <- mean(predicted_cat_test == true_cat_test, na.rm = TRUE)
cat("Out-of-sample (test) accuracy:", accuracy_test, "\n")

conf_matrix_test <- table(Actual = true_cat_test, Predicted = predicted_cat_test)
cat("Out-of-sample (test) confusion matrix:\n")
print(conf_matrix_test)


# Optionally plot the confusion matrix
conf_df_test <- as.data.frame(conf_matrix_test)
colnames(conf_df_test) <- c("Actual", "Predicted", "Freq")
conf_df_test$Actual    <- factor(conf_df_test$Actual,    levels = c(1,2,3),
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

# 9) Add credible intervals & Save Results
p1_quants <- apply(post_probs_test[, 1, ], 1, quantile, probs = c(0.4, 0.6))
p2_quants <- apply(post_probs_test[, 2, ], 1, quantile, probs = c(0.4, 0.6))
p3_quants <- apply(post_probs_test[, 3, ], 1, quantile, probs = c(0.4, 0.6))

results <- data.frame(
  home_team_idx = test_data$home_team_idx,
  away_team_idx = test_data$away_team_idx,
  date          = test_data$date,
  Match_report  = test_data$Match_report,
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
  predicted_label_bayes_adv = factor(
    predicted_cat_test, 
    levels = c(1, 2, 3),
    labels = c("Away Win", "Draw", "Home Win")
  ),
  true_category = true_cat_test
)

out_csv <- "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/results/results_bayes_season.csv"
write.csv(results, out_csv, row.names = FALSE)
cat("Saved results with mean probabilities and 95% CIs to:", out_csv, "\n")

###############################################################################
# End of script
###############################################################################
