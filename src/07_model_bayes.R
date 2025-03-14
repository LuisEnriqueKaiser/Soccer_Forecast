###############################################################################
# Bayesian ordinal model with an explicit home-advantage parameter
# using Stan and including the code to produce credible intervals
# for per-match outcome probabilities.
###############################################################################

rm(list = ls())
library(rstan)
library(dplyr)
library(ggplot2)

# Use multiple cores for faster sampling and auto-write compiled models to disk
options(mc.cores = parallel::detectCores())

# -----------------------------------------------------------------------------
# 1) Stan Model Code
# -----------------------------------------------------------------------------
stan_model_code <- "
data {
  int<lower=1> N;                 // number of matches
  int<lower=2> K;                 // number of outcome categories (3 for away win, draw, home win)
  int<lower=1> T;                 // number of teams
  int<lower=1,upper=T> home_team[N]; 
  int<lower=1,upper=T> away_team[N];
  int<lower=1,upper=K> y[N];      // observed outcome as 1..K

  int<lower=0> P;                 // number of additional predictors
  matrix[N, P] X;                 // predictor matrix (match-specific covariates)
}
parameters {
  // Team-level random effects
  vector[T] team_strength;        // latent strength for each team
  real home_adv;                  // home-field advantage parameter
  
  // Ordered logistic cutpoints
  ordered[K-1] c;                

  // Hyperparameter for team_strength prior
  real<lower=0> sigma_team;      

  // Coefficients for additional predictors
  vector[P] beta;
}
model {
  // Priors
  sigma_team ~ normal(0, 3);
  team_strength ~ normal(0, sigma_team);

  // Weakly informative prior for home-field advantage
  home_adv ~ normal(0, 3);

  c ~ normal(0, 4);
  beta ~ normal(0, 3);

  // Likelihood: for each match, the latent score 
  //   (home team's strength + home_adv) - (away team's strength) + X*beta
  for (n in 1:N) {
    real eta = (team_strength[home_team[n]] + home_adv)
               - team_strength[away_team[n]]
               + dot_product(X[n], beta);
    y[n] ~ ordered_logistic(eta, c);
  }
}
generated quantities {
  // We store the latent score 'eta' for each match to facilitate posterior predictions
  vector[N] eta;
  for (n in 1:N) {
    eta[n] = (team_strength[home_team[n]] + home_adv)
             - team_strength[away_team[n]]
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
data_raw <- data_raw[, (names(data_raw) %in% principal_component_names)]

# Team indices
teams <- sort(unique(c(data_raw$home_team, data_raw$away_team)))
T_num <- length(teams)
team_index <- setNames(seq_len(T_num), teams)

data_raw <- data_raw %>%
  mutate(
    home_team_idx = team_index[home_team],
    away_team_idx = team_index[away_team]
  )

# Identify predictor columns (excluding outcome/team/indices)
predictor_cols <- setdiff(
  names(data_raw),
  c("match_result", "match_result_cat", 
    "home_team", "away_team",
    "home_team_idx", "away_team_idx")
)

# Drop rows with missing predictor values
data_raw <- data_raw[complete.cases(data_raw[, predictor_cols]), ]

# Build matrix X
X <- as.matrix(data_raw[, predictor_cols])
P <- ncol(X)

# Stan data list
stan_data <- list(
  N = nrow(data_raw),
  K = 3,  # 3 outcome categories
  T = T_num,
  home_team = data_raw$home_team_idx,
  away_team = data_raw$away_team_idx,
  y = data_raw$match_result_cat,
  P = P,
  X = X
)

# -----------------------------------------------------------------------------
# 3) MODEL FITTING WITH STAN
# -----------------------------------------------------------------------------
# Define file paths
output_dir <- "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/results/Bayes_base_results"
fit_file <- file.path(output_dir, "stan_fit.rds")

# Ask the user if they want to rerun MCMC or load saved model
cat("Do you want to rerun the MCMC sampling? (yes/no): ")
user_input <- tolower(readline())

if (user_input == "yes") {
  cat("Running MCMC sampling...\n")
  
  # Compile the Stan model
  stan_model <- stan_model(model_code = stan_model_code)
  
  # Run the MCMC sampling
  fit <- sampling(
    stan_model,
    data    = stan_data,
    iter    = 5000,    # total iterations per chain
    warmup  = 1000,     # burn-in
    chains  = 4,       
    seed    = 123,     
    control = list(adapt_delta = 0.95),
    refresh = 100
  )
  
  # Save the fitted model
  saveRDS(fit, fit_file)
  cat("Stan model fit saved to:", fit_file, "\n")
  
} else {
  # Load the saved model if the user chooses not to rerun MCMC
  if (file.exists(fit_file)) {
    cat("Loading saved Stan model fit...\n")
    fit <- readRDS(fit_file)
    cat("Stan model fit loaded successfully!\n")
  } else {
    stop("No saved Stan model fit found! You need to run MCMC at least once.")
  }
}

# Print key parameter summaries
print(fit, pars = c("team_strength", "home_adv", "c", "sigma_team", "beta"))

output_dir <- "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/results/Bayes_base_results"
fit_file <- file.path(output_dir, "stan_fit.rds")
saveRDS(fit, fit_file)

summary_fit <- summary(fit)$summary

# Load necessary libraries
library(stargazer)

# Define output directory
output_dir <- "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/results/Bayes_base_results"
output_file <- file.path(output_dir, "Bayesian_Model_Results.tex")

# Delete old file if it exists
if (file.exists(output_file)) {
  file.remove(output_file)
}

# Extract summary statistics from the Stan model fit
summary_fit <- summary(fit)$summary

# Select a subset of key parameters
params_subset <- c("home_adv", "sigma_team", "c[1]", "c[2]", "beta[1]")  # Add more if needed
subset_fit <- summary_fit[params_subset, ]

# Restrict to desired columns: mean, se_mean, n_eff, Rhat
subset_fit <- subset_fit[, c("mean", "se_mean", "n_eff", "Rhat")]

# Convert row names (parameter names) into a column
subset_fit <- data.frame(Parameter = rownames(subset_fit), subset_fit, row.names = NULL)
colnames(subset_fit) <- c("Parameter", "Mean", "SE Mean", "Effective Sample Size (N_eff)", "Rhat")

# Generate the LaTeX table correctly
sink(output_file)  # Open output file

stargazer(subset_fit, summary = FALSE, float = TRUE, align = TRUE,
          title = "Summary MCMC Estimation",
          label = "tab:mcmc_summary",
          out = output_file)

sink()  # Close output file

cat("LaTeX table saved in:", output_file, "\n")


# Bayesian Model Diagnostic Plots
###############################################################################

# Ensure directory exists
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}
library(bayesplot)

# -----------------------------------------------------------------------------
# Extract posterior samples
# -----------------------------------------------------------------------------
posterior_samples <- rstan::extract(fit)

# Define key parameters to include in the traceplot
params_to_trace <- c("sigma_team", "c[1]", "c[2]", "beta[1]")  # Add more if needed

# Extract MCMC samples from the fit object
posterior_draws <- as.array(fit)  # Convert Stan output to array for bayesplot
color_scheme_set("viridisB")  # Change this to experiment with different palettes

# Create a combined traceplot
trace_plot <- mcmc_trace(posterior_draws, pars = params_to_trace) +
  theme_minimal()

# Save as a high-resolution PNG file
ggsave(filename = file.path(output_dir, "combined_traceplots.png"), 
       plot = trace_plot, dpi = 300, width = 12, height = 8)

# -----------------------------------------------------------------------------
# 2) Posterior Distribution: Team Strength of a Specific Team
# -----------------------------------------------------------------------------
team_idx_to_plot <- 1  # Change this to plot another team

df_team_strength <- data.frame(team_strength = posterior_samples$team_strength[, team_idx_to_plot])

ggplot(df_team_strength, aes(x = team_strength)) +
  geom_histogram(bins = 50, alpha = 0.7, fill = "darkgreen", color = "black") +
  geom_vline(xintercept = mean(df_team_strength$team_strength), linetype = "dashed", color = "red") +
  labs(title = paste("Posterior Distribution: Team Strength (Team Index", team_idx_to_plot, ")"),
       x = "Team Strength", y = "Density") +
  theme_minimal()

# Save plot
ggsave(filename = file.path(output_dir, paste0("posterior_team_strength_", team_idx_to_plot, ".png")),
       dpi = 300, width = 8, height = 6)

# -----------------------------------------------------------------------------
# 3) Posterior Distribution: Cutpoints (c1, c2)
# -----------------------------------------------------------------------------
c1_samples <- posterior_samples$c[,1]  # Samples for c[1]
c2_samples <- posterior_samples$c[,2]  # Samples for c[2]

# Create a data frame for ggplot
posterior_df <- data.frame(
  Cutpoint = rep(c("c[1]", "c[2]"), each = length(c1_samples)),
  Value = c(c1_samples, c2_samples)
)

# Plot overlapping histograms with transparency
ggplot(posterior_df, aes(x = Value, fill = Cutpoint)) +
  geom_histogram(alpha = 0.6, bins = 40, position = "identity") +
  scale_fill_manual(values = c("blue", "red")) +  # Adjust colors as needed
  labs(x = "Value",
       y = "Density") +
  theme_minimal()

ggsave(filename = file.path(output_dir, "posteriors.png"), dpi = 300, width = 8, height = 6)

# -----------------------------------------------------------------------------
# 4) Posterior Distribution: First Principal Component (PC1)
# -----------------------------------------------------------------------------
df_pc1 <- data.frame(PC1 = posterior_samples$beta[, 1])

ggplot(df_pc1, aes(x = PC1)) +
  geom_histogram(bins = 50, alpha = 0.7, fill = "red", color = "black") +
  geom_vline(xintercept = mean(df_pc1$PC1), linetype = "dashed", color = "red") +
  labs(title = "Posterior Distribution: First Principal Component (PC1)",
       x = "PC1", y = "Density") +
  theme_minimal()

ggsave(filename = file.path(output_dir, "posterior_PC1.png"), dpi = 300, width = 8, height = 6)

cat("Plots saved in:", output_dir, "\n")

# -----------------------------------------------------------------------------
# 5) OPTIONAL: TRAINING SET POSTERIOR PREDICTIVE CHECK
# -----------------------------------------------------------------------------
posterior <- rstan::extract(fit)
thin_factor <- 10

# Thinned posterior to reduce loops
posterior_thinned <- lapply(posterior, function(x) {
  if (is.matrix(x) || is.data.frame(x)) {
    x[seq(1, nrow(x), by = thin_factor), , drop = FALSE]
  } else if (is.vector(x)) {
    x[seq(1, length(x), by = thin_factor)]
  } else {
    x
  }
})

S <- dim(posterior_thinned$c)[1]     # thinned draws
N <- dim(posterior_thinned$eta)[2]   # matches

post_probs <- array(0, dim = c(N, 3, S))
for (s in 1:S) {
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

###############################################################################
# 6) TEST SET PREDICTIONS
###############################################################################
test_file <- "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/test_data.csv"
test_data <- read.csv(test_file, stringsAsFactors = FALSE)

# Recode outcome if necessary
test_data <- test_data %>%
  mutate(
    match_result_cat = as.integer(
      factor(match_result, levels = c(-1, 0, 1), labels = c(1, 2, 3))
    )
  )

# Must reuse the same teams for consistent indexing
teams <- sort(unique(c(test_data$home_team, test_data$away_team)))
T_num <- length(teams)
team_index <- setNames(seq_len(T_num), teams)

test_data <- test_data %>%
  mutate(
    home_team_idx = team_index[home_team],
    away_team_idx = team_index[away_team]
  )

# Build test predictors
predictor_cols <- c("PC1","PC2","PC3","PC4","PC5","PC6","PC7")
X_test <- as.matrix(test_data[, predictor_cols])
P_test <- ncol(X_test)

stan_data_test <- list(
  N = nrow(test_data),
  K = 3,
  T = T_num,
  home_team = test_data$home_team_idx,
  away_team = test_data$away_team_idx,
  y = test_data$match_result_cat,
  P = P_test,
  X = X_test
)

# Compute posterior predictive probabilities for the test set
posterior <- rstan::extract(fit)
S_full <- dim(posterior$c)[1]     # number of unthinned draws
N_test <- stan_data_test$N

post_probs_test <- array(0, dim = c(N_test, 3, S_full))
for (s in 1:S_full) {
  c1 <- posterior$c[s, 1]
  c2 <- posterior$c[s, 2]
  ts <- posterior$team_strength[s, ]
  beta_s <- posterior$beta[s, ]
  home_adv_s <- posterior$home_adv[s]
  
  # Vectorized latent score with home advantage
  eta_test <- (ts[test_data$home_team_idx] + home_adv_s) -
    ts[test_data$away_team_idx] +
    X_test %*% beta_s
  
  p1 <- plogis(c1 - eta_test) 
  p2 <- plogis(c2 - eta_test)
  
  post_probs_test[, 1, s] <- p1
  post_probs_test[, 2, s] <- p2 - p1
  post_probs_test[, 3, s] <- 1 - p2
}

# Mean probability across posterior draws
mean_probs_test <- apply(post_probs_test, c(1, 2), mean)  # Nx3
predicted_cat_test <- apply(mean_probs_test, 1, which.max)

true_cat_test <- test_data$match_result_cat
accuracy_test <- mean(predicted_cat_test == true_cat_test, na.rm = TRUE)
cat("Out-of-sample (test) accuracy:", accuracy_test, "\n")

# Simple confusion matrix
conf_matrix_test <- table(Actual = true_cat_test, Predicted = predicted_cat_test)
cat("Out-of-sample (test) confusion matrix:\n")
print(conf_matrix_test)

# Quick confusion matrix plot
conf_df_test <- as.data.frame(conf_matrix_test)
colnames(conf_df_test) <- c("Actual", "Predicted", "Freq")
conf_df_test$Actual <- factor(conf_df_test$Actual, levels = c(1, 2, 3), 
                              labels = c("Away Win", "Draw", "Home Win"))
conf_df_test$Predicted <- factor(conf_df_test$Predicted, levels = c(1, 2, 3), 
                                 labels = c("Away Win", "Draw", "Home Win"))

ggplot(conf_df_test, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "grey70") +
  geom_text(aes(label = Freq), color = "black") +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(x="Predicted Result", y="Actual Result",
       title="Out-of-sample Confusion Matrix") +
  theme_minimal()

###############################################################################
# 7) ADD CREDIBLE INTERVALS & CREATE RESULTS DATAFRAME
###############################################################################
# reload the test data 
test_data <- read.csv(test_file, stringsAsFactors = FALSE)
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

write.csv(results, "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/results_bayes_base.csv", row.names = FALSE)
cat("Saved 'results.csv' with mean probabilities and 50% credible intervals.\n")
