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

def main():
    # File paths: adjust as needed
    train_file = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/train_data.csv"
    test_file  = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/test_data.csv"
    output_train_file = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/train_data.csv"
    output_test_file  = "/Users/luisenriquekaiser/Documents/soccer_betting_forecast/data/final/test_data.csv"
    
    # Load datasets
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    # Inspect the training data
    print("Training data shape:", df_train.shape)
    print("Training data columns:", df_train.columns.tolist())
    print("Training data (first 5 rows):")
    print(df_train.head())
    
    # Define columns to exclude from PCA (these are typically identifiers or outcomes)
    exclude_cols = [
        'home_team', 'away_team', 'date', 'match_result', 'match_result_cat',
        'home_win', 'away_win', 'score', 'home_win', 'away_win' ]
    
    
    # Select numeric columns from training data that are not in the exclude list
    numeric_cols = [col for col in df_train.select_dtypes(include=[np.number]).columns if col not in exclude_cols]
    print("Numeric columns for PCA:", numeric_cols)
    
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
    n_components = 9
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    # Print explained variance ratio for inspection
    print("Explained variance ratio of PCA components:")
    print(pca.explained_variance_ratio_)
    
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
    
    # Save the final datasets
    df_train_final.to_csv(output_train_file, index=False)
    df_test_final.to_csv(output_test_file, index=False)
    
    print(f"Saved training data with PCA components to: {output_train_file}")
    print(f"Saved test data with PCA components to: {output_test_file}")

if __name__ == "__main__":
    main()
