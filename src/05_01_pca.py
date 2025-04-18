#!/usr/bin/env python3
"""
pca_transform.py

This script:
  - Loads the training and test datasets (from your feature-engineered files).
  - Inspects the training data (prints shape, columns, and a head sample).
  - Selects numeric columns (excluding identifiers/outcomes) for PCA.
  - Fills any remaining NaNs (with zero here, but you can change the strategy).
  - Fits a StandardScaler and PCA on the training set only.
  - Transforms the test set using the same scaler and PCA.
  - Appends the PCA components to the original datasets.
  - Saves the resulting datasets as separate CSV files.
  
Adjust the number of PCA components as needed.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from project_specifics import TRAIN_OUTPUT_PATH, TEST_OUTPUT_PATH

def main():
    # File paths: adjust as needed
    train_file = TRAIN_OUTPUT_PATH
    test_file  = TEST_OUTPUT_PATH
    output_train_file =TRAIN_OUTPUT_PATH
    output_test_file  = TEST_OUTPUT_PATH
    
    # Load datasets
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    # Inspect the training data
    print("Training data shape:", df_train.shape)
    #print("Training data columns:", df_train.columns.tolist())
    #print("Training data (first 5 rows):")
    #print(df_train.head())
    
    # Define columns to exclude from PCA (these are typically identifiers or outcomes)
    exclude_cols = [
        'home_team', 'away_team', 'date', 'match_result', 'match_result_cat',
        'home_win', 'away_win', 'score', 'home_win', 'away_win', "match_report", 
        "home_points_previous_season", "away_points_previous_season" ]
    
    
    # Select numeric columns from training data that are not in the exclude list
    numeric_cols = [col for col in df_train.select_dtypes(include=[np.number]).columns if col not in exclude_cols]
    #print("Numeric columns for PCA:", numeric_cols)
    
    # Check for leftover NaNs in training numeric columns
    num_nans = df_train[numeric_cols].isna().sum().sum()
    print("Total number of NaNs in numeric columns (train):", num_nans)
    
    # Fill any leftover NaNs with zero (or use another imputation strategy)
    df_train[numeric_cols] = df_train[numeric_cols].fillna(0)
    df_test[numeric_cols] = df_test[numeric_cols].fillna(0)
    
    # Standardize features on training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train[numeric_cols])
    
    # Fit PCA on training set (choose the number of components as desired)
    n_components = 60
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    # Print explained variance ratio for inspection
    print("Explained variance ratio of PCA components:")
    print(pca.explained_variance_ratio_)
    print("Total explained variance:", pca.explained_variance_ratio_.sum())
    # Transform test set using the same scaler and PCA
    X_test_scaled = scaler.transform(df_test[numeric_cols])
    X_test_pca = pca.transform(X_test_scaled)
    
    # Create DataFrames for the principal components
    pc_cols = [f'PC{i+1}' for i in range(n_components)]
    df_train_pca = pd.DataFrame(X_train_pca, columns=pc_cols, index=df_train.index)
    df_test_pca = pd.DataFrame(X_test_pca, columns=pc_cols, index=df_test.index)
    
    # Option: Keep both the original numeric features and add the PCA columns
    df_train_final = pd.concat([df_train, df_train_pca], axis=1)
    df_test_final = pd.concat([df_test, df_test_pca], axis=1)
    # drop everything except for the pca columns and the following list 
    to_keep = [
        "match_report",'home_team', 'away_team', 'date', 'match_result_cat', "season_number",
        "home_team_integer", "away_team_integer", "home_points_prev_season", "away_points_prev_season",]
    df_train_final = df_train_final[to_keep + pc_cols]
    df_test_final = df_test_final[to_keep + pc_cols]
    
    # Save the final datasets
    df_train_final.to_csv("/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/PCA_Train.csv", index=False)
    df_test_final.to_csv("/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/PCA_Test.csv", index=False)
    

    print(f"Saved training data with PCA components to: {output_train_file}")
    print(f"Saved test data with PCA components to: {output_test_file}")

if __name__ == "__main__":
    main()
