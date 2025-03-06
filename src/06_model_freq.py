#!/usr/bin/env python3
"""
soccer_forecast.py

A complete framework for soccer betting forecasting:
 - Loads and preprocesses the data.
 - Splits data in a time-aware manner.
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

# Check for the optional dependency 'jinja2' (required for DataFrame.to_latex())
try:
    import jinja2
except ImportError:
    raise ImportError("Missing optional dependency 'jinja2'. Please install it using 'pip install jinja2'.")

# Define output directories for tables and figures.
TABLES_DIR = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/results/tables"
FIGURES_DIR = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/results/figures"

# Create directories if they do not exist.
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# -------------------------------
# Data Preprocessing Functions
# -------------------------------
def drop_old_incomplete_rows(df, date_col, frac_threshold=0.5):
    """
    Drops rows where the fraction of missing values is >= frac_threshold and the date is in the past.
    """
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    now = pd.to_datetime("today")
    frac_missing = df.isna().sum(axis=1) / df.shape[1]
    cond_incomplete = frac_missing >= frac_threshold
    cond_in_past = df[date_col] < now
    df_clean = df[~(cond_incomplete & cond_in_past)].copy()
    df_clean.reset_index(drop=True, inplace=True)
    return df_clean

def impute_missing_with_columnmean_up_until_that_date(df):
    """
    Impute missing numeric values with the expanding mean computed from all prior rows.
    Assumes df is sorted by date.
    """
    df_imputed = df.copy()
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        cum_mean = df_imputed[col].expanding(min_periods=1).mean().shift(1)
        df_imputed[col] = df_imputed[col].fillna(cum_mean)
    return df_imputed

def load_data(csv_path):
    """
    Load the soccer dataset from a CSV file.
    Assumptions:
      - "match_result" is encoded as 1 (Home win), 0 (Draw), -1 (Away win).
      - "date" indicates when each match occurred.
    
    Processing:
      - Convert "date" to datetime and sort by date.
      - Create binary variable "home_win".
      - Map "match_result" to a categorical variable "match_result_cat": -1 -> 0, 0 -> 1, 1 -> 2.
    """
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['home_win'] = (df['match_result'] == 1).astype(int)
    mapping = {-1: 0, 0: 1, 1: 2}
    df['match_result_cat'] = df['match_result'].map(mapping).astype('category')
    return df

# -------------------------------
# Output Functions
# -------------------------------
def save_latex_table(df, caption, label, filename, table_name=""):
    """
    Save a DataFrame as a LaTeX table to the specified filename.
    """
    latex_str = df.to_latex(index=True, caption=caption, label=label)
    full_path = os.path.join(TABLES_DIR, filename)
    with open(full_path, "a") as f:
        f.write(f"% {table_name}\n")
        f.write(latex_str)
        f.write("\n\n")

def plot_and_save_confusion_matrix(y_true, y_pred, class_names, title, filename):
    """
    Plot and save the confusion matrix as a PNG file.
    """
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
    """
    Compute the average prediction accuracy per matchweek and plot as a barplot.
    Assumes df_test contains a column (default 'week') indicating the matchweek.
    """
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
    """
    Tune hyperparameters for RandomForestClassifier using TimeSeriesSplit.
    """
    param_grid = {
        'n_estimators': [100, 250, 300, 250],
        'max_depth': [2, 4, 6, 8],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [15, 17, 20],
        'min_samples_leaf': [5, 7,8],
        'max_features': ['sqrt', 'log2']
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
    """
    Evaluate the Random Forest model and return predictions and key metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Random Forest: Accuracy = {acc:.3f}, Log Loss = {ll:.3f}, Weighted F1 = {f1:.3f}")
    return y_pred, y_proba, acc, ll, f1

def run_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Fit and evaluate a multinomial logistic regression model.
    Returns the fitted pipeline and predictions.
    """
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
    # File path (update as needed)
    csv_path = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/PL_final_dataset.csv"
    df = load_data(csv_path)
    
    # Preprocess: drop rows with too many missing values and impute missing values
    df = drop_old_incomplete_rows(df, date_col="date", frac_threshold=0.5)
    df = df[df["date"] < pd.to_datetime("today")]
    df = impute_missing_with_columnmean_up_until_that_date(df)
    
    # Define non-feature columns
    non_feature_cols = ["match_result", "match_result_cat", "home_win", "date", 
                        "day", "score", "time", "Match_report", "notes", 
                        "venue", "referee", "game_id", "home_team", "away_team"]
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    # For simplicity, assume feature_cols are numeric (or already processed).
    df_features = df[feature_cols]
    
    # Time-aware split (90% train, 10% test)
    n_total = len(df_features)
    train_end = int(0.9 * n_total)
    df_features_train = df_features.iloc[:train_end].reset_index(drop=True)
    df_features_test = df_features.iloc[train_end:].reset_index(drop=True)
    
    # Target variable from original DataFrame (numeric codes: {0,1,2})
    y = df["match_result_cat"].cat.codes.values
    y_train = y[:train_end]
    y_test = y[train_end:]
    
    # Retrieve test set info (including date and week columns)
    df_test_info = df.iloc[train_end:].reset_index(drop=True)
    
    print("Training set date range:", df.iloc[:train_end]["date"].min(), "to", df.iloc[:train_end]["date"].max())
    print("Test set date range:", df.iloc[train_end:]["date"].min(), "to", df.iloc[train_end:]["date"].max())
    
    # Tune and evaluate Random Forest model
    best_rf = tune_random_forest(df_features_train.values, y_train)
    y_pred_rf, y_proba_rf, acc_rf, ll_rf, f1_rf = evaluate_rf_model(best_rf, df_features_test.values, y_test)
    
    # Save Random Forest confusion matrix plot
    cm_rf = plot_and_save_confusion_matrix(y_test, y_pred_rf, 
                   class_names=["Away win", "Draw", "Home win"], 
                   title="Random Forest Confusion Matrix", 
                   filename="rf_confusion_matrix.png")
    
    # Plot and save average accuracy per matchweek (assumes a 'week' column exists in df_test_info)
    if 'week' in df_test_info.columns:
        weekly_acc = plot_accuracy_by_matchweek(df_test_info, y_test, y_pred_rf)
    else:
        print("Test data does not contain a 'week' column. Skipping matchweek accuracy plot.")
        weekly_acc = None
    
    # Run Logistic Regression model
    pipe_logit, y_pred_logit, y_proba_logit, acc_logit, ll_logit, f1_logit = run_logistic_regression(
        df_features_train.values, y_train, df_features_test.values, y_test)
    
    # Save Logistic Regression confusion matrix plot
    cm_logit = plot_and_save_confusion_matrix(y_test, y_pred_logit, 
                   class_names=["Away win", "Draw", "Home win"], 
                   title="Logistic Regression Confusion Matrix", 
                   filename="logit_confusion_matrix.png")
    
    # Generate classification reports as DataFrames
    report_rf = classification_report(y_test, y_pred_rf, target_names=["Away win", "Draw", "Home win"], output_dict=True)
    report_logit = classification_report(y_test, y_pred_logit, target_names=["Away win", "Draw", "Home win"], output_dict=True)
    df_report_rf = pd.DataFrame(report_rf).transpose().round(3)
    df_report_logit = pd.DataFrame(report_logit).transpose().round(3)
    
    # Define LaTeX output file for tables.
    latex_filename = os.path.join(TABLES_DIR, "results_tables.tex")
    # Clear file
    with open(latex_filename, "w") as f:
        f.write("% Tables generated from soccer_forecast.py\n\n")
    
    save_latex_table(cm_rf, "Random Forest Confusion Matrix", "tab:rf_cm", "rf_confusion_matrix.tex", "RF Confusion Matrix")
    save_latex_table(cm_logit, "Logistic Regression Confusion Matrix", "tab:logit_cm", "logit_confusion_matrix.tex", "Logit Confusion Matrix")
    save_latex_table(df_report_rf, "Random Forest Classification Report", "tab:rf_report", "rf_classification_report.tex", "RF Classification Report")
    save_latex_table(df_report_logit, "Logistic Regression Classification Report", "tab:logit_report", "logit_classification_report.tex", "Logit Classification Report")
    
    if weekly_acc is not None:
        save_latex_table(weekly_acc, "Average Accuracy per Matchweek", "tab:weekly_acc", "weekly_accuracy.tex", "Weekly Accuracy")
    
    # Print summary metrics to console.
    print("\n--- Summary Metrics ---")
    print(f"Random Forest: Accuracy = {acc_rf:.3f}, Log Loss = {ll_rf:.3f}, Weighted F1 = {f1_rf:.3f}")
    print(f"Logistic Regression: Accuracy = {acc_logit:.3f}, Log Loss = {ll_logit:.3f}, Weighted F1 = {f1_logit:.3f}")

if __name__ == "__main__":
    main()
