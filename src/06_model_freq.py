#!/usr/bin/env python3
"""
soccer_forecast.py

A complete framework for soccer betting forecasting:
 - Loads and preprocesses the data from separate training and testing sources.
 - Tunes a Random Forest classifier using TimeSeriesSplit.
 - Evaluates the tuned Random Forest and a baseline multinomial logistic regression.
 - Saves plots (PNG) and tables (LaTeX) to designated output directories.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import parallel_backend

# Check for the optional dependency 'jinja2'
try:
    import jinja2
except ImportError:
    raise ImportError("Missing optional dependency 'jinja2'. Please install it using 'pip install jinja2'.")

# Define output directories for tables and figures.
TABLES_DIR = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/results/tables"
FIGURES_DIR = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/results/figures"

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# -------------------------------
# Data Preprocessing Functions
# -------------------------------
def drop_old_incomplete_rows(df, date_col, frac_threshold=0.5):
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    now = pd.to_datetime("today")
    frac_missing = df.isna().sum(axis=1) / df.shape[1]
    cond_incomplete = frac_missing >= frac_threshold
    cond_in_past = df[date_col] < now
    df_clean = df[~(cond_incomplete & cond_in_past)].copy()
    df_clean.reset_index(drop=True, inplace=True)
    return df_clean

def impute_missing_with_columnmean_up_until_that_date(df):
    df_imputed = df.copy()
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        cum_mean = df_imputed[col].expanding(min_periods=1).mean().shift(1)
        df_imputed[col] = df_imputed[col].fillna(cum_mean)
    return df_imputed

def load_data(csv_path):
    """
    Load the soccer dataset from a CSV file and preprocess key columns.
    Assumes:
      - "match_result" is encoded as 1 (Home win), 0 (Draw), -1 (Away win).
      - "date" indicates when each match occurred.
    Processing:
      - Converts "date" to datetime and sorts by date.
      - Creates binary variable "home_win".
      - Maps "match_result" to a categorical variable "match_result_cat": -1 -> 0, 0 -> 1, 1 -> 2.
      - Drops rows with missing match_result to avoid unmapped (NaN) outcomes.
    """
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    # Drop rows where match_result is missing
    df = df.dropna(subset=["match_result"])
    df['home_win'] = (df['match_result'] == 1).astype(int)
    mapping = {-1: 0, 0: 1, 1: 2}
    df['match_result_cat'] = df['match_result'].map(mapping).astype('category')
    return df

# -------------------------------
# Output Functions
# -------------------------------
def save_latex_table(df, caption, label, filename, table_name=""):
    latex_str = df.to_latex(index=True, caption=caption, label=label)
    full_path = os.path.join(TABLES_DIR, filename)
    with open(full_path, "a") as f:
        f.write(f"% {table_name}\n")
        f.write(latex_str)
        f.write("\n\n")

def plot_and_save_confusion_matrix(y_true, y_pred, class_names, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    full_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(full_path)
    plt.close()
    return cm_df

def plot_accuracy_by_matchweek(df_test, y_true, y_pred, week_col='week'):
    df_plot = df_test.copy()
    df_plot['correct'] = (y_true == y_pred).astype(int)
    weekly_accuracy = df_plot.groupby(week_col)['correct'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=week_col, y='correct', data=weekly_accuracy, palette='viridis')
    plt.xlabel("Matchweek")
    plt.ylabel("Average Accuracy")
    plt.title("Average Prediction Accuracy per Matchweek")
    plt.ylim(0, 1)
    plt.tight_layout()
    full_path = os.path.join(FIGURES_DIR, "accuracy_by_matchweek.png")
    plt.savefig(full_path)
    plt.close()
    return weekly_accuracy

# -------------------------------
# Model Functions
# -------------------------------
def tune_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 250],#, 300, 250],
        'max_depth': [8],
        'criterion': ['gini'],
        'min_samples_split': [20],
        'min_samples_leaf': [ 8],
        'max_features': ['sqrt']
    }
    tscv = TimeSeriesSplit(n_splits=5)
    rf = RandomForestClassifier(random_state=42)
    with parallel_backend('loky', n_jobs=-1):
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0,
            pre_dispatch='2*n_jobs'
        )
        grid_search.fit(X_train, y_train)
    print("Best parameters found:", grid_search.best_params_)
    print("Best CV accuracy:", grid_search.best_score_)
    return grid_search.best_estimator_

def evaluate_rf_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    # Now y_test should only contain classes [0, 1, 2]
    ll = log_loss(y_test, y_proba)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Random Forest: Accuracy = {acc:.3f}, Log Loss = {ll:.3f}, Weighted F1 = {f1:.3f}")
    return y_pred, y_proba, acc, ll, f1

def run_logistic_regression(X_train, y_train, X_test, y_test):
    pipe_logit = make_pipeline(
        StandardScaler(),
        LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    )
    pipe_logit.fit(X_train, y_train)
    y_pred_logit = pipe_logit.predict(X_test)
    y_proba_logit = pipe_logit.predict_proba(X_test)
    acc_logit = accuracy_score(y_test, y_pred_logit)
    ll_logit = log_loss(y_test, y_proba_logit)
    f1_logit = f1_score(y_test, y_pred_logit, average='weighted')
    print(f"Logistic Regression: Accuracy = {acc_logit:.3f}, Log Loss = {ll_logit:.3f}, Weighted F1 = {f1_logit:.3f}")
    return pipe_logit, y_pred_logit, y_proba_logit, acc_logit, ll_logit, f1_logit

# -------------------------------
# Main Routine
# -------------------------------
def main():
    # Use separate CSV files for training and testing
    train_csv_path = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/train_data.csv"
    test_csv_path  = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/test_data.csv"
    
    df_train = load_data(train_csv_path)
    df_test = load_data(test_csv_path)
    
    # Preprocess each dataset: drop rows with too many missing values and impute missing values
    df_train = drop_old_incomplete_rows(df_train, date_col="date", frac_threshold=0.5)
    df_test = drop_old_incomplete_rows(df_test, date_col="date", frac_threshold=0.5)
    
    # Ensure dates are before today
    df_train = df_train[df_train["date"] < pd.to_datetime("today")]
    df_test = df_test[df_test["date"] < pd.to_datetime("2025-02-15")]
    
    df_train = impute_missing_with_columnmean_up_until_that_date(df_train)
    df_test = impute_missing_with_columnmean_up_until_that_date(df_test)
    
    # Define non-feature columns to exclude
    non_feature_cols = ["match_result", "match_result_cat", "home_win", "date", 
                        "day", "score", "time", "Match_report", "notes", "away_win",
                        "venue", "referee", "game_id", "home_team", "away_team"]

    # Identify feature columns (ensure train and test have common features)
    feature_cols_train = [col for col in df_train.columns if col not in non_feature_cols]
    feature_cols_test = [col for col in df_test.columns if col not in non_feature_cols]
    common_feature_cols = list(set(feature_cols_train) & set(feature_cols_test))
    
    X_train = df_train[common_feature_cols].values
    X_test = df_test[common_feature_cols].values
    
    # Prepare target variable using the mapped category
    y_train = df_train["match_result_cat"].cat.codes.values
    y_test = df_test["match_result_cat"].cat.codes.values
    
    print("Training set date range:", df_train["date"].min(), "to", df_train["date"].max())
    print("Test set date range:", df_test["date"].min(), "to", df_test["date"].max())
    
    # Tune and evaluate Random Forest model
    print("\n--- Random Forest Model ---")
    best_rf = tune_random_forest(X_train, y_train)
    print("evaluating model")
    y_pred_rf, y_proba_rf, acc_rf, ll_rf, f1_rf = evaluate_rf_model(best_rf, X_test, y_test)
    
    # Save confusion matrix plot for Random Forest
    cm_rf = plot_and_save_confusion_matrix(y_test, y_pred_rf, 
                   class_names=["Away win", "Draw", "Home win"], 
                   title="Random Forest Confusion Matrix", 
                   filename="rf_confusion_matrix.png")
    
    # Optionally, plot accuracy by matchweek if the test data contains a 'week' column
    if 'week' in df_test.columns:
        weekly_acc = plot_accuracy_by_matchweek(df_test, y_test, y_pred_rf)
    else:
        print("Test data does not contain a 'week' column. Skipping matchweek accuracy plot.")
        weekly_acc = None
    
    # Run Logistic Regression model
    pipe_logit, y_pred_logit, y_proba_logit, acc_logit, ll_logit, f1_logit = run_logistic_regression(X_train, y_train, X_test, y_test)
    
    # Save Logistic Regression confusion matrix plot
    cm_logit = plot_and_save_confusion_matrix(y_test, y_pred_logit, 
                   class_names=["Away win", "Draw", "Home win"], 
                   title="Logistic Regression Confusion Matrix", 
                   filename="logit_confusion_matrix.png")
    
    # Generate and save classification reports as LaTeX tables
    report_rf = classification_report(y_test, y_pred_rf, target_names=["Away win", "Draw", "Home win"], output_dict=True)
    report_logit = classification_report(y_test, y_pred_logit, target_names=["Away win", "Draw", "Home win"], output_dict=True)
    df_report_rf = pd.DataFrame(report_rf).transpose().round(3)
    df_report_logit = pd.DataFrame(report_logit).transpose().round(3)
    
    latex_filename = os.path.join(TABLES_DIR, "results_tables.tex")
    with open(latex_filename, "w") as f:
        f.write("% Tables generated from soccer_forecast.py\n\n")
    
    save_latex_table(cm_rf, "Random Forest Confusion Matrix", "tab:rf_cm", "rf_confusion_matrix.tex", "RF Confusion Matrix")
    save_latex_table(cm_logit, "Logistic Regression Confusion Matrix", "tab:logit_cm", "logit_confusion_matrix.tex", "Logit Confusion Matrix")
    save_latex_table(df_report_rf, "Random Forest Classification Report", "tab:rf_report", "rf_classification_report.tex", "RF Classification Report")
    save_latex_table(df_report_logit, "Logistic Regression Classification Report", "tab:logit_report", "logit_classification_report.tex", "Logit Classification Report")
    
    if weekly_acc is not None:
        save_latex_table(weekly_acc, "Average Accuracy per Matchweek", "tab:weekly_acc", "weekly_accuracy.tex", "Weekly Accuracy")
    
    print("\n--- Summary Metrics ---")
    print(f"Random Forest: Accuracy = {acc_rf:.3f}, Log Loss = {ll_rf:.3f}, Weighted F1 = {f1_rf:.3f}")
    print(f"Logistic Regression: Accuracy = {acc_logit:.3f}, Log Loss = {ll_logit:.3f}, Weighted F1 = {f1_logit:.3f}")

if __name__ == "__main__":
    main()
