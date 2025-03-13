###############################################################################
# Bayesian ordinal model with an explicit home-advantage parameter
# (Season-based prior, but the very first season is zero-centered)
# Now including min-max normalization of the performance measure.
###############################################################################

rm(list = ls())
library(rstan)
library(dplyr)
library(ggplot2)

# Use multiple cores for faster sampling and auto-write compiled models to disk
options(mc.cores = parallel::detectCores())

# -----------------------------------------------------------------------------
# 1) Stan Model Code (First season prior = Normal(0, sigma_team_init), subsequent = Normal(performance, sigma_season))
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
  // 1) First season prior: team_strength[1,t] ~ Normal(0, sigma_team_init)
  for (t in 1:T) {
    team_strength[1, t] ~ normal(0, sigma_team_init);
  }

  // 2) For each subsequent season s=2..S:
  //    team_strength[s,t] ~ Normal(performance[s,t], sigma_season)
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

  // 4) Likelihood: ordered logistic
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
# We'll combine train & test so we have all seasons covered
test_data_tmp <- test_data %>%
  select(
    season_number, home_team, away_team,
    home_points_prev_season, away_points_prev_season
  ) %>%
  mutate(
    home_team_idx = team_index[home_team],
    away_team_idx = team_index[away_team]
  )

# Create a data frame of (team, season_number, previous_season_points) from both TRAIN and TEST
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

df_perf_combined <- dplyr::bind_rows(df_home_train, df_away_train,
                                     df_home_test, df_away_test)

# 3C.i) Summarize by (team, season_number)
df_perf_combined <- df_perf_combined %>%
  group_by(team, season_number) %>%
  summarise(performance_value = mean(perf, na.rm = TRUE)) %>%
  ungroup()

# 3C.ii) Min-max scale the performance_value to [-1, +1]
minVal <- min(df_perf_combined$performance_value, na.rm = TRUE)
maxVal <- max(df_perf_combined$performance_value, na.rm = TRUE)
rangeVal <- maxVal - minVal
if (rangeVal == 0) {
  # fallback: if all performance is identical, set them all to 0
  df_perf_combined$performance_value <- 0
} else {
  df_perf_combined$performance_value <- -1 + 2 * (
    (df_perf_combined$performance_value - minVal) / rangeVal
  )
}

# 3C.iii) Build the matrix[S_num, T_num]
performance_mat <- matrix(0, nrow = S_num, ncol = T_num)  # default 0
for (i in seq_len(nrow(df_perf_combined))) {
  s_idx <- season_index[ df_perf_combined$season_number[i] ]
  t_idx <- team_index[ df_perf_combined$team[i] ]
  performance_mat[s_idx, t_idx] <- df_perf_combined$performance_value[i]
}
# For s=1, these values won't be used in the prior anyway.

# -----------------------------------------------------------------------------
# 4) PREPARE TRAINING STAN DATA
# -----------------------------------------------------------------------------
X <- as.matrix(data_raw[, predictor_cols])
P <- ncol(X)

stan_data <- list(
  N          = nrow(data_raw),
  K          = 3,  # 3 outcome categories (Away Win, Draw, Home Win)
  T          = T_num,
  S          = S_num,
  home_team  = data_raw$home_team_idx,
  away_team  = data_raw$away_team_idx,
  y          = data_raw$match_result_cat,
  P          = P,
  X          = X,
  season_id  = data_raw$season_id,
  performance = performance_mat   # used for s >= 2 in the Stan model
)

# -----------------------------------------------------------------------------
# 5) MODEL FITTING WITH STAN
# -----------------------------------------------------------------------------
stan_model <- stan_model(model_code = stan_model_code)

fit <- sampling(
  stan_model,
  data    = stan_data,
  iter    = 500,    # total iterations per chain
  warmup  = 100,     # burn-in
  chains  = 4,       
  seed    = 123,     
  control = list(adapt_delta = 0.95),
  refresh = 100
)

print(fit, pars = c("team_strength", "home_adv", "c",
                    "sigma_season", "sigma_team_init", "beta"))

# -----------------------------------------------------------------------------
# 6) POSTERIOR DIAGNOSTICS / TEAM STRENGTHS
# -----------------------------------------------------------------------------
posterior_samples <- rstan::extract(fit)

# Traceplots for a few parameters
traceplot(fit, pars = c("home_adv", "sigma_season", "sigma_team_init"))

# Compute mean team strengths for the final season
final_season_idx <- S_num
team_strength_means <- colMeans(posterior_samples$team_strength[, final_season_idx, ])
team_strength_df <- data.frame(
  team = teams,
  strength = team_strength_means
)

# Quick visual
ggplot(team_strength_df, aes(x = reorder(team, strength), y = strength)) +
  geom_point() +
  coord_flip() +
  labs(x = "Team", y = "Estimated Strength (Final Season)", 
       title = "Posterior Mean Estimates (First-Season = 0 Prior)")

# Check the posterior for home_adv
home_adv_mean <- mean(posterior_samples$home_adv)
home_adv_ci <- quantile(posterior_samples$home_adv, probs = c(0.025, 0.975))
cat("Home advantage mean:", home_adv_mean, 
    "95% CI:", home_adv_ci[1], "to", home_adv_ci[2], "\n")

# Posterior distribution plot for home_adv
df_home_adv <- data.frame(home_adv = posterior_samples$home_adv)
ggplot(df_home_adv, aes(x = home_adv)) +
  geom_histogram(bins = 50, alpha = 0.7) +
  geom_vline(xintercept = mean(df_home_adv$home_adv), linetype = "dashed") +
  labs(title = "Posterior Distribution of Home Advantage", x = "home_adv")

# Posterior distribution of sigma_season
df_sigma <- data.frame(sigma_season = posterior_samples$sigma_season)
ggplot(df_sigma, aes(x = sigma_season)) +
  geom_histogram(bins = 50, alpha = 0.7) +
  geom_vline(xintercept = mean(df_sigma$sigma_season), linetype = "dashed") +
  labs(title = "Posterior Distribution of sigma_season", x = "sigma_season")

# -----------------------------------------------------------------------------
# 7) OPTIONAL: TRAINING SET POSTERIOR PREDICTIVE CHECK
# -----------------------------------------------------------------------------
posterior_thinned <- lapply(posterior_samples, function(x) {
  # for big samples, thinning can help
  if (is.matrix(x) || is.data.frame(x)) {
    x[seq(1, nrow(x), by = 10), , drop = FALSE]
  } else if (is.vector(x)) {
    x[seq(1, length(x), by = 10)]
  } else {
    x
  }
})

S_draws <- dim(posterior_thinned$c)[1]     
N <- dim(posterior_thinned$eta)[2]

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

predictor_cols <- c("PC1","PC2","PC3","PC4","PC5","PC6","PC7")
X_test <- as.matrix(test_data[, predictor_cols])
N_test <- nrow(test_data)

posterior <- rstan::extract(fit)
S_full <- dim(posterior$c)[1]
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
    post_probs_test[n, 1, s] <- p1         # away win
    post_probs_test[n, 2, s] <- p2 - p1    # draw
    post_probs_test[n, 3, s] <- 1 - p2     # home win
  }
}

mean_probs_test <- apply(post_probs_test, c(1,2), mean)
predicted_cat_test <- apply(mean_probs_test, 1, which.max)
true_cat_test <- test_data$match_result_cat

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
p1_quants <- apply(post_probs_test[, 1, ], 1, quantile, probs = c(0.25, 0.75))
p2_quants <- apply(post_probs_test[, 2, ], 1, quantile, probs = c(0.25, 0.75))
p3_quants <- apply(post_probs_test[, 3, ], 1, quantile, probs = c(0.25, 0.75))

results <- data.frame(
  home_team = test_data$home_team,
  away_team = test_data$away_team,
  date = test_data$date,
  matchreport = test_data$Match_report,
  prob_away_mean  = mean_probs_test[, 1],
  prob_away_lower = p1_quants[1, ],
  prob_away_upper = p1_quants[2, ],
  prob_draw_mean  = mean_probs_test[, 2],
  prob_draw_lower = p2_quants[1, ],
  prob_draw_upper = p2_quants[2, ],
  prob_home_mean  = mean_probs_test[, 3],
  prob_home_lower = p3_quants[1, ],
  prob_home_upper = p3_quants[2, ],
  predicted_category = predicted_cat_test,
  predicted_label = factor(predicted_cat_test,
                           levels = c(1, 2, 3),
                           labels = c("Away Win", "Draw", "Home Win")),
  true_category = true_cat_test
)

write.csv(results, "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/results_bayes_season.csv", row.names = FALSE)
cat("Saved 'results_bayes_season.csv' with mean probabilities and 50% credible intervals.\n")
