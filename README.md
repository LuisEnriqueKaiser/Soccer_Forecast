NOTE: THIS REPO IS UPDATED CONSTANTLY. FOR EVALUATORS OF THE BAYESIAN MODELLING COURSE: PLEASE LOOK UP THE COMMIT OF AROUND THE 20th OF MARCH. 
This repository contains the replication files for my final term paper in the 2nd-year PhD Bayesian Modeling course at Goethe University Frankfurt. 

My paper explores the advantages of using the Bayesian framework in order to improve on prediction accuracy, probability accuracy and uncertainty qunatification. 
It also contains a simulation of betting strategies, showcasing the potential of this approach. 

Please note that at multiple places of the code one has to change up hardcoded paths. So prior to running after cloning the repo, one should ensure for each file that the right libraries are installed (see below) and that
the paths are changed accordingly. 
It is important for replication that one runs the codes in the order I numbered them as they are building on each other. 

This repo also contains the basic data files of the initial download and the closing odds I simply downloaded from https://www.football-data.co.uk/, so that one might skip the first script 01_download_data.py


#### Overview over the files: 

01_download_data.py --> Downloads the baseline data

02_data_matching.py --> Creates match level data

03_data_merge.py --> merges all the individual match level data sets 

04_00_data_prep.py --> prepares data for feature engineering

05_00_feature_engineering.py --> builds features for the frequentist approaches

05_01_pca.py --> build principal components for the bayesian models

06_model_freq.py --> frequentist models and predictions

07_01_model_bayes.R --> baseline bayesian model 

07_02_season_based_prior_model.R --> advanced bayesian model 

08_01_odds_data_built.py --> prepares the odds data for the later results

08_02_odds_merge.py --> merges the odds data together in one file 

08_03_prepare_for_plotting.py --> prepares the data for the plots BEWARE: THIS IS BASICALLY THE SAME SCRIPT AS FEATURE ENGINEERING; AFTER YOU RAN THIS SCRIPT YOU CANNOT JUST RUN THE SCRIPTS BEFOREHAND AGAIN DUE TO THE ODDS BEING THEN IN THE TRAIN AND TEST DATASET

08_04_plots_comparison.py --> creates plots and final dataset with odds and credible intervalls and predictions of each algorithm

08_05_plotting_credible_intervalls.py --> creates credible intervall plots 

09_betting_strategies.R --> employs a betting strategy backtest 


R libraries: rstan, dplyr, ggplot2, stargazer, bayesplot
Python libraries: os, soccerdata, pandas, numpy, skicit-learn, joblib, matplotlib

## Odds Mispricing Streamlit App

This repo now includes a lightweight Streamlit app for comparing Polymarket odds to up to 10 bookmaker lines and logging the results locally.

### Run locally

1. Install dependencies (including Streamlit):
   ```bash
   pip install -r requirements.txt
   ```
2. Start the app:
   ```bash
   streamlit run streamlit_app.py
   ```

The app stores results in `data/bet_requests.csv` so you can revisit prior bets.


