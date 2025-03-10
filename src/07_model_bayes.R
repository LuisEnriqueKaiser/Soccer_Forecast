# for the bayesian model work in R as it has a better package support and was taught throughout the course
# idea is to use stan, what we used in the bayesina modelling course 
# offers exactly the advantages i was trying to accomplish in my project
# Load required packages
# Load required packages
rm(list = ls())
library(rstan)
library(dplyr)
library(ggplot2)
# Load required packages
library(rstan)
library(dplyr)
library(ggplot2)

# Use multiple cores for faster sampling and auto-write compiled models to disk
options(mc.cores = parallel::detectCores())

# -----------------------------------------------------------------------------
# Updated Stan model code with additional predictors (X) and coefficients (beta)
# -----------------------------------------------------------------------------
stan_model_code <- "
data {
  int<lower=1> N;                // number of matches
  int<lower=2> K;                // number of outcome categories (e.g., 3 for away win, draw, home win)
  int<lower=1> T;                // number of teams
  int<lower=1,upper=T> home_team[N]; // index of home team for each match
  int<lower=1,upper=T> away_team[N]; // index of away team for each match
  int<lower=1,upper=K> y[N];       // observed outcome (as integers 1, 2, ..., K)
  
  int<lower=0> P;                // number of additional predictors
  matrix[N, P] X;                // predictor matrix (match-specific covariates known on game day)
}
parameters {
  vector[T] team_strength;       // latent strength for each team
  ordered[K-1] c;                // threshold (cutpoint) parameters (ensures ordering)
  real<lower=0> sigma_team;      // standard deviation (hyperparameter) for team strengths
  
  vector[P] beta;              // coefficients for additional predictors
}
model {
  // Priors
  sigma_team ~ normal(0, 5);
  team_strength ~ normal(0, sigma_team);
  c ~ normal(0, 5);
  beta ~ normal(0, 5);  // weakly informative prior for covariate coefficients
  
  // Likelihood: latent score = team strength difference + linear predictor from X
  for (n in 1:N) {
    real eta = team_strength[home_team[n]] - team_strength[away_team[n]] + dot_product(X[n], beta);
    y[n] ~ ordered_logistic(eta, c);
  }
}
generated quantities {
  vector[N] eta;
  for (n in 1:N) {
    eta[n] = team_strength[home_team[n]] - team_strength[away_team[n]] + dot_product(X[n], beta);
  }
}
"

# -----------------------------------------------------------------------------
# Data Preparation
# -----------------------------------------------------------------------------
# Read in the dataset (assumed to be the final processed file)
data_raw <- read.csv("/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/train_data.csv", 
                     stringsAsFactors = FALSE)


# Recode match outcome directly from match_result.
# We assume match_result is coded as -1 (away win), 0 (draw), and 1 (home win).
# The recoding converts these to an ordered integer variable: 1, 2, 3.
data_raw <- data_raw %>%
  mutate(match_result_cat = as.integer(factor(match_result,
                                              levels = c(-1, 0, 1),
                                              labels = c(1, 2, 3))))

# Drop non-feature columns that are not available on game day
# drop everything except the 6 principal components and the outcome variable 

principal_component_names = c("home_team", "away_team","match_result_cat","PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7",
                              "PC8", "PC9")
data_raw <- data_raw[, (names(data_raw) %in% principal_component_names)]
#non_feature_cols <- c("home_win", "date", 
#                      "day", "score", "time", "Match_report", "notes", 
 #                     "venue", "referee", "game_id")

# Create a team index: get the unique team names from home_team and away_team
teams <- sort(unique(c(data_raw$home_team, data_raw$away_team)))
T_num <- length(teams)
team_index <- setNames(1:T_num, teams)

# Map team names to their integer indices
data_raw <- data_raw %>%
  mutate(home_team_idx = team_index[home_team],
         away_team_idx = team_index[away_team])

# Identify predictor columns to include in X.
# Exclude outcome-related columns and team identifiers.
predictor_cols <- setdiff(names(data_raw), c("match_result", "match_result_cat", "home_team", "away_team",
                                             "home_team_idx", "away_team_idx"))

# Drop rows with missing values in predictor columns (Stan does not support NA values)
data_raw <- data_raw[complete.cases(data_raw[, predictor_cols]), ]

# Convert the predictors to a matrix.
X <- as.matrix(data_raw[, predictor_cols])

P <- ncol(X)  # number of predictors

# -----------------------------------------------------------------------------
# Prepare data list for Stan
# -----------------------------------------------------------------------------
stan_data <- list(
  N = nrow(data_raw),
  K = 3,  # number of outcome categories (1, 2, 3)
  T = T_num,
  home_team = data_raw$home_team_idx,
  away_team = data_raw$away_team_idx,
  y = data_raw$match_result_cat,
  P = P,
  X = X
)

# -----------------------------------------------------------------------------
# Model Fitting using RStan
# -----------------------------------------------------------------------------
# Compile the model
stan_model <- stan_model(model_code = stan_model_code)

# Fit the model via MCMC sampling
fit <- sampling(stan_model,
                data = stan_data,
                iter = 30000,       # total iterations per chain
                warmup = 2000,     # warmup (burn-in) iterations
                chains = 4,        # number of chains
                seed = 123,        # for reproducibility
                control = list(adapt_delta = 0.95),
                refresh = 100)

# Print a summary of the posterior distributions
print(fit, pars = c("team_strength", "c", "sigma_team", "beta"))

# -----------------------------------------------------------------------------
# Post-processing: extract and plot team strength estimates
# -----------------------------------------------------------------------------
posterior_samples <- extract(fit)
traceplot(fit, pars = c("sigma_team", "team_strength[1]", "beta[2]"))

# Calculate the posterior mean for team strengths
team_strength_means <- colMeans(posterior_samples$team_strength)
team_strength_df <- data.frame(
  team = teams,
  strength = team_strength_means
)

# Plot team strengths
ggplot(team_strength_df, aes(x = reorder(team, strength), y = strength)) +
  geom_point() +
  coord_flip() +
  xlab("Team") +
  ylab("Estimated Strength") +
  ggtitle("Posterior Mean Estimates of Team Strengths")






posterior <- rstan::extract(fit)
thin_factor <- 10
posterior_thinned <- lapply(posterior, function(x) {
  if (is.matrix(x) || is.data.frame(x)) {
    x[seq(1, nrow(x), by = thin_factor), , drop = FALSE]
  } else if (is.vector(x)) {
    x[seq(1, length(x), by = thin_factor)]
  } else {
    x
  }
})
# For convenience:
S <- dim(posterior_thinned$c)[1]     # number of draws (posterior samples)
N <- dim(posterior_thinned$eta)[2]   # number of matches

# ---------------------------------------------------------------------------
# 2. Compute posterior predictive probabilities for each match
# ---------------------------------------------------------------------------
# For a 3-category ordered logistic, the probabilities are:
#   P(y=1) = logistic(c1 - eta)
#   P(y=2) = logistic(c2 - eta) - logistic(c1 - eta)
#   P(y=3) = 1 - logistic(c2 - eta)

# We'll store them in an array [N, 3, S].
post_probs <- array(0, dim = c(N, 3, S))

for (s in 1:S) {
  c1 <- posterior_thinned$c[s,1]  # first cutpoint
  c2 <- posterior_thinned$c[s,2]  # second cutpoint
  for (n in 1:N) {
    eta_n <- posterior_thinned$eta[s,n]
    p1 <- plogis(c1 - eta_n)
    p2 <- plogis(c2 - eta_n)
    post_probs[n,1,s] <- p1
    post_probs[n,2,s] <- p2 - p1
    post_probs[n,3,s] <- 1 - p2
  }
}

# ---------------------------------------------------------------------------
# 3. Average over posterior draws to get mean predicted probabilities
# ---------------------------------------------------------------------------
# mean_probs[n, k] = average probability of category k for match n
mean_probs <- apply(post_probs, c(1,2), mean)  # now [N, 3]

# Derive predicted category = argmax across the 3 categories
predicted_cat <- apply(mean_probs, 1, which.max)  # in {1,2,3}

# ---------------------------------------------------------------------------
# 4. Compare to actual outcome in your training data
# ---------------------------------------------------------------------------
# We assume your original data frame has 'match_result_cat' in {1,2,3}.
true_cat <- data_raw$match_result_cat  # must match the order in stan_data$y
accuracy <- mean(predicted_cat == true_cat, na.rm = TRUE)

cat("In-sample accuracy:", accuracy, "\n")

# If you want to see predicted probabilities for each match, 'mean_probs' is your Nx3 matrix.
# Each row sums to 1 and you can inspect them or compare them to the actual outcomes.










library(reshape2)


test_file <- "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/test_data.csv"
test_data <- read.csv(test_file, stringsAsFactors = FALSE)



# (Optional) Check a few rows
print(head(test_data))




# If necessary, recode match_result to match_result_cat.
# (Assuming it’s already done; if not, uncomment the next lines.)
test_data <- test_data %>%
   mutate(match_result_cat = as.integer(factor(match_result,
                                               levels = c(-1, 0, 1),
                                               labels = c(1, 2, 3))))

# -----------------------------------------------------------------------------
# Create Team Index for Test Data
# -----------------------------------------------------------------------------
# IMPORTANT: The team index must match that used in training.
# Here, we assume that the set of teams in test data is a subset of the training teams.
teams <- sort(unique(c(test_data$home_team, test_data$away_team)))
T_num <- length(teams)
team_index <- setNames(1:T_num, teams)
test_data <- test_data %>%
  mutate(home_team_idx = team_index[home_team],
         away_team_idx = team_index[away_team])

# -----------------------------------------------------------------------------
# Build Predictor Matrix for Test Data
# -----------------------------------------------------------------------------
# We assume that the model was built using the six principal components.
predictor_cols <- c("PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9")
X_test <- as.matrix(test_data[, predictor_cols])
P_test <- ncol(X_test)

# -----------------------------------------------------------------------------
# Build stan_data list for test predictions
# -----------------------------------------------------------------------------
stan_data_test <- list(
  N = nrow(test_data),
  K = 3,  # outcomes: 1 (away win), 2 (draw), 3 (home win)
  T = T_num,
  home_team = test_data$home_team_idx,
  away_team = test_data$away_team_idx,
  y = test_data$match_result_cat,  # observed outcomes (for evaluation)
  P = P_test,
  X = X_test
)

# -----------------------------------------------------------------------------
# In-Sample (Test) Prediction Using the Fitted Stan Model
# -----------------------------------------------------------------------------
# We extract the posterior samples from the fitted model (fit)
posterior <- rstan::extract(fit)

# Dimensions:
S <- dim(posterior$c)[1]     # number of posterior draws
N_test <- stan_data_test$N   # number of test matches

# We compute the latent score for each test match and each posterior draw.
# For each draw s and match n, the latent score is:
#   eta_test = team_strength[home_team] - team_strength[away_team] + dot_product(X_test[n,], beta)
# Then, for an ordered logistic model with 3 categories, the probabilities are:
#   P(y=1) = plogis(c1 - eta_test)
#   P(y=2) = plogis(c2 - eta_test) - plogis(c1 - eta_test)
#   P(y=3) = 1 - plogis(c2 - eta_test)

# We'll compute these in a vectorized manner using loops over draws only (to avoid a nested S x N loop).
post_probs_test <- array(0, dim = c(N_test, 3, S))

for (s in 1:S) {
  # For draw s, get the cutpoints and parameters:
  c1 <- posterior$c[s, 1]
  c2 <- posterior$c[s, 2]
  ts <- posterior$team_strength[s, ]  # vector of team strengths
  beta_s <- posterior$beta[s, ]        # vector of coefficients
  
  # Compute the latent score for all test matches in one vectorized operation:
  # Note: X_test is a matrix of size N_test x P.
  eta_test <- ts[test_data$home_team_idx] - ts[test_data$away_team_idx] + X_test %*% beta_s
  
  # Compute probabilities using vectorized plogis (which is essentially logistic function)
  p1 <- plogis(c1 - eta_test)         # probability of outcome 1 for all matches
  p2 <- plogis(c2 - eta_test)         # probability threshold for outcome 2
  
  # Store probabilities for each test match:
  post_probs_test[, 1, s] <- p1
  post_probs_test[, 2, s] <- p2 - p1
  post_probs_test[, 3, s] <- 1 - p2
}

# Average over posterior draws to obtain mean predicted probabilities for each test match
mean_probs_test <- apply(post_probs_test, c(1, 2), mean)  # matrix [N_test x 3]

# Predicted category for each match: choose the category with highest mean probability
predicted_cat_test <- apply(mean_probs_test, 1, which.max)
# -----------------------------------------------------------------------------
# Evaluate Out-of-Sample Predictions: Confusion Matrix & Accuracy
# -----------------------------------------------------------------------------
true_cat_test <- test_data$match_result_cat  # numeric categories: 1=Home Win, 2=Draw, 3=Away Win
accuracy_test <- mean(predicted_cat_test == true_cat_test, na.rm = TRUE)
cat("Out-of-sample (test) accuracy:", accuracy_test, "\n")

# Create a standard confusion matrix: rows = Actual, cols = Predicted
conf_matrix_test <- table(Actual = true_cat_test, Predicted = predicted_cat_test)
cat("Out-of-sample (test) confusion matrix:\n")
print(conf_matrix_test)

# Convert to data frame for plotting
conf_df_test <- as.data.frame(conf_matrix_test)
colnames(conf_df_test) <- c("Actual", "Predicted", "Freq")

# Optional: Relabel numeric outcomes as factors for nicer plotting
conf_df_test$Actual <- factor(conf_df_test$Actual,
                              levels = c(1, 2, 3),
                              labels = c("Home Win", "Draw", "Away Win"))
conf_df_test$Predicted <- factor(conf_df_test$Predicted,
                                 levels = c(1, 2, 3),
                                 labels = c("Home Win", "Draw", "Away Win"))

# Plot confusion matrix: x-axis = Predicted, y-axis = Actual
ggplot(conf_df_test, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "grey70") +
  geom_text(aes(label = Freq), color = "black") +
  scale_fill_gradient(low = "white", high = "blue") +
  xlab("Predicted Result") +
  ylab("Actual Result") +
  ggtitle("Out-of-sample Confusion Matrix") +
  theme_minimal()

