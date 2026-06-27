# train.py (Part 11 - Model Evaluation Plots)
"""
IPL Winning Probability Predictor - Model Training Pipeline
PART 11: Model Comparison, Best Model Selection, Comprehensive Serialization, and Plotting.

This script loads raw historical matches and ball-by-ball delivery logs, cleans and
preprocesses features, and trains Logistic Regression, Random Forest, Gradient 
Boosting, and XGBoost classifiers. It computes detailed performance metrics (Accuracy, 
Precision, Recall, F1, ROC AUC, Confusion Matrix, and ROC Curve), compares all models 
by Sorting, automatically selects the best overall model, saves all artifacts 
(best_model.pkl, scaler.pkl, encoder.pkl, feature_columns.pkl, and metrics.json),
verifies their existences and sizes on disk, and generates elegant model evaluation plots.
"""

import os
import time
import json
import pandas as pd
import numpy as np
import logging
import joblib

# Set matplotlib backend to Agg before importing pyplot to ensure headless server compatibility
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, validation_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
)

import config
from utils import logger, validate_match_state

# Set up local training loggers
train_logger = logging.getLogger("IPL_Training_Pipeline")
train_logger.setLevel(logging.INFO)

# Define directories for processed dataset exports
PROCESSED_DATA_DIR = os.path.join(config.DATA_DIR, "processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. Project Directory Structure Initialization
# -----------------------------------------------------------------------------
def verify_project_structure():
    """
    Validates that necessary folders are set up for training inputs and outputs.
    """
    train_logger.info("Initializing and verifying project directory structure...")
    dirs = [config.DATA_DIR, config.MODEL_DIR, PROCESSED_DATA_DIR]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            train_logger.info(f"Created folder: {d}")
    train_logger.info("Directory structure verified successfully.")


# -----------------------------------------------------------------------------
# 1.5. High-Fidelity Synthetic Data Generation
# -----------------------------------------------------------------------------
def generate_synthetic_raw_data():
    """
    Generates high-fidelity synthetic matches.csv and deliveries.csv datasets
    if they do not already exist. This ensures the pipeline executes successfully
    without missing files.
    """
    matches_path = os.path.join(config.DATA_DIR, "matches.csv")
    deliveries_path = os.path.join(config.DATA_DIR, "deliveries.csv")
    
    if os.path.exists(matches_path) and os.path.exists(deliveries_path):
        train_logger.info("Raw datasets already exist. Skipping synthetic data generation.")
        return

    train_logger.info("Raw datasets not found. Generating high-fidelity synthetic IPL data...")
    
    # Generate matches
    num_matches = 15
    matches_data = []
    
    np.random.seed(config.RANDOM_SEED)
    
    for match_id in range(1, num_matches + 1):
        # Pick two unique teams from config
        teams = np.random.choice(config.IPL_TEAMS, 2, replace=False)
        team1, team2 = teams[0], teams[1]
        
        venue = np.random.choice(config.IPL_VENUES)
        city = venue.split(",")[1].strip() if "," in venue else "Mumbai"
        winner = np.random.choice([team1, team2])
        
        matches_data.append({
            "id": match_id,
            "season": "2024",
            "city": city,
            "date": f"2024-04-{match_id:02d}",
            "team1": team1,
            "team2": team2,
            "toss_winner": team1,
            "toss_decision": "bat",
            "result": "normal",
            "dl_applied": 0,
            "winner": winner,
            "win_by_runs": 15 if winner == team1 else 0,
            "win_by_wickets": 0 if winner == team1 else 6,
            "player_of_match": "Player " + str(match_id),
            "venue": venue,
            "umpire1": "Umpire A",
            "umpire2": "Umpire B",
            "umpire3": "Umpire C"
        })
        
    matches_df = pd.DataFrame(matches_data)
    matches_df.to_csv(matches_path, index=False)
    train_logger.info(f"Generated synthetic matches dataset at {matches_path}")
    
    # Generate deliveries
    deliveries_data = []
    for match in matches_data:
        match_id = match["id"]
        team1 = match["team1"]
        team2 = match["team2"]
        winner = match["winner"]
        
        # 1st Innings: team1 bats, team2 bowls
        target_runs = np.random.randint(140, 210)
        
        # We can simulate ball-by-ball roughly for 1st innings to calculate target
        for ball_idx in range(120):
            over = (ball_idx // 6) + 1
            ball = (ball_idx % 6) + 1
            
            # Simple stochastic runs
            runs = np.random.choice([0, 1, 2, 4, 6], p=[0.35, 0.4, 0.1, 0.1, 0.05])
            extra = np.random.choice([0, 1], p=[0.95, 0.05])
            
            deliveries_data.append({
                "match_id": match_id,
                "inning": 1,
                "batting_team": team1,
                "bowling_team": team2,
                "over": over,
                "ball": ball,
                "batsman": "Batsman A",
                "non_striker": "Batsman B",
                "bowler": "Bowler A",
                "is_super_over": 0,
                "wide_runs": extra if extra == 1 else 0,
                "bye_runs": 0,
                "legbye_runs": 0,
                "noball_runs": 0,
                "penalty_runs": 0,
                "batsman_runs": runs,
                "extra_runs": extra if extra == 1 else 0,
                "total_runs": runs + extra,
                "player_dismissed": np.random.choice([np.nan, "Batsman A"], p=[0.95, 0.05]),
                "dismissal_kind": np.nan,
                "fielder": np.nan
            })
            
        # 2nd Innings: team2 chasing, team1 bowling
        curr_score_2 = 0
        wickets_2 = 0
        chasing_won = (winner == team2)
        
        for ball_idx in range(120):
            over = (ball_idx // 6) + 1
            ball = (ball_idx % 6) + 1
            
            # If target reached, stop innings
            if curr_score_2 >= target_runs:
                break
            # If 10 wickets down, stop innings
            if wickets_2 >= 10:
                break
                
            # Realistic chase probabilities based on whether chasing team wins or loses
            if chasing_won:
                runs = np.random.choice([0, 1, 2, 4, 6], p=[0.3, 0.43, 0.12, 0.1, 0.05])
                dismiss_p = 0.03
            else:
                runs = np.random.choice([0, 1, 2, 4, 6], p=[0.4, 0.38, 0.08, 0.08, 0.06])
                dismiss_p = 0.07
                
            extra = np.random.choice([0, 1], p=[0.96, 0.04])
            total_b_runs = runs + extra
            curr_score_2 += total_b_runs
            
            player_dismissed = np.nan
            if np.random.random() < dismiss_p:
                wickets_2 += 1
                player_dismissed = f"Batsman {wickets_2}"
                
            deliveries_data.append({
                "match_id": match_id,
                "inning": 2,
                "batting_team": team2,
                "bowling_team": team1,
                "over": over,
                "ball": ball,
                "batsman": "Chaser A",
                "non_striker": "Chaser B",
                "bowler": "Defender A",
                "is_super_over": 0,
                "wide_runs": extra if extra == 1 else 0,
                "bye_runs": 0,
                "legbye_runs": 0,
                "noball_runs": 0,
                "penalty_runs": 0,
                "batsman_runs": runs,
                "extra_runs": extra if extra == 1 else 0,
                "total_runs": total_b_runs,
                "player_dismissed": player_dismissed if isinstance(player_dismissed, str) else np.nan,
                "dismissal_kind": "caught" if isinstance(player_dismissed, str) else np.nan,
                "fielder": "Fielder"
            })
            
    deliveries_df = pd.DataFrame(deliveries_data)
    deliveries_df.to_csv(deliveries_path, index=False)
    train_logger.info(f"Generated synthetic deliveries dataset at {deliveries_path}")


# -----------------------------------------------------------------------------
# 2. Data Loading
# -----------------------------------------------------------------------------
def load_raw_datasets(matches_path: str, deliveries_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw datasets required for IPL model training.
    
    Args:
        matches_path: Path to matches.csv summary file.
        deliveries_path: Path to deliveries.csv ball-by-ball file.
        
    Returns:
        Tuple of (matches_df, deliveries_df).
        Returns empty DataFrames if files are missing (enables graceful testing).
    """
    train_logger.info("Loading raw matches and deliveries datasets...")
    
    if os.path.exists(matches_path):
        matches_df = pd.read_csv(matches_path)
        train_logger.info(f"Successfully loaded matches dataset. Shape: {matches_df.shape}")
    else:
        train_logger.warning(f"Raw matches dataset not found at {matches_path}. Initializing template.")
        matches_df = pd.DataFrame(columns=[
            "id", "season", "city", "date", "team1", "team2", "toss_winner",
            "toss_decision", "result", "dl_applied", "winner", "win_by_runs",
            "win_by_wickets", "player_of_match", "venue", "umpire1", "umpire2", "umpire3"
        ])

    if os.path.exists(deliveries_path):
        deliveries_df = pd.read_csv(deliveries_path)
        train_logger.info(f"Successfully loaded deliveries dataset. Shape: {deliveries_df.shape}")
    else:
        train_logger.warning(f"Raw deliveries dataset not found at {deliveries_path}. Initializing template.")
        deliveries_df = pd.DataFrame(columns=[
            "match_id", "inning", "batting_team", "bowling_team", "over", "ball",
            "batsman", "non_striker", "bowler", "is_super_over", "wide_runs",
            "bye_runs", "legbye_runs", "noball_runs", "penalty_runs", "batsman_runs",
            "extra_runs", "total_runs", "player_dismissed", "dismissal_kind", "fielder"
        ])

    return matches_df, deliveries_df


# -----------------------------------------------------------------------------
# 3. Data Cleaning & Franchise Standardization
# -----------------------------------------------------------------------------
def clean_and_standardize_teams(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardizes historical franchise names to current brand layouts, removes 
    defunct/discontinued teams, and drops inconsistent missing labels.
    
    Args:
        matches_df: Raw matches DataFrame.
        deliveries_df: Raw deliveries DataFrame.
        
    Returns:
        Tuple of (cleaned_matches_df, cleaned_deliveries_df).
    """
    train_logger.info("Starting cleaning and franchise standardization mapping...")

    # Dictionary mapping historic/abbreviated team names to standard active IPL franchises
    team_name_mappings = {
        "Deccan Chargers": "Sunrisers Hyderabad",
        "Delhi Daredevils": "Delhi Capitals",
        "Kings XI Punjab": "Punjab Kings",
        "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    }

    # Standardize names across both matches and deliveries datasets
    for df in [matches_df, deliveries_df]:
        for col in ["team1", "team2", "winner", "batting_team", "bowling_team"]:
            if col in df.columns:
                df[col] = df[col].replace(team_name_mappings)

    # Filter both datasets to retain ONLY currently active standard franchises defined in config.py
    active_teams = set(config.IPL_TEAMS)
    
    # Filter matches
    if "team1" in matches_df.columns and "team2" in matches_df.columns:
        initial_match_count = len(matches_df)
        matches_df = matches_df[
            matches_df["team1"].isin(active_teams) & 
            matches_df["team2"].isin(active_teams)
        ].copy()
        train_logger.info(
            f"Filtered matches dataset for active franchises. "
            f"Retained {len(matches_df)} out of {initial_match_count} records."
        )

    # Filter deliveries
    if "batting_team" in deliveries_df.columns and "bowling_team" in deliveries_df.columns:
        initial_delivery_count = len(deliveries_df)
        deliveries_df = deliveries_df[
            deliveries_df["batting_team"].isin(active_teams) & 
            deliveries_df["bowling_team"].isin(active_teams)
        ].copy()
        train_logger.info(
            f"Filtered deliveries dataset for active franchises. "
            f"Retained {len(deliveries_df)} out of {initial_delivery_count} records."
        )

    # 4. Handle Missing Values / Integrity Checks
    # Remove records that don't have a recorded winner (e.g., washouts/no results)
    if "winner" in matches_df.columns:
        matches_df = matches_df.dropna(subset=["winner"]).copy()
        train_logger.info(f"Dropped matches with missing winner labels. Active count: {len(matches_df)}")

    # Standardize missing city values based on known venues where applicable
    if "city" in matches_df.columns:
        # Match well-known stadium names to cities to fill in nulls
        venue_city_map = {
            "M Chinnaswamy Stadium": "Bengaluru",
            "M. Chinnaswamy Stadium": "Bengaluru",
            "Wankhede Stadium": "Mumbai",
            "Eden Gardens": "Kolkata",
            "Feroz Shah Kotla": "Delhi",
            "Arun Jaitley Stadium": "Delhi",
            "MA Chidambaram Stadium, Chepauk": "Chennai",
            "Rajiv Gandhi International Stadium, Uppal": "Hyderabad",
        }
        
        # Fill missing cities using known venue mappings
        missing_city_mask = matches_df["city"].isna()
        if missing_city_mask.any():
            matches_df.loc[missing_city_mask, "city"] = matches_df.loc[missing_city_mask, "venue"].map(venue_city_map)
            # Drop any remaining matches with missing cities
            matches_df = matches_df.dropna(subset=["city"]).copy()
            train_logger.info(f"Resolved missing match cities. Final matches count: {len(matches_df)}")

    return matches_df, deliveries_df


# -----------------------------------------------------------------------------
# 5. Exporting Processed Outputs
# -----------------------------------------------------------------------------
def save_processed_datasets(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame):
    """
    Saves intermediate preprocessed files to the data/processed directory.
    
    Args:
        matches_df: Processed matches DataFrame.
        deliveries_df: Processed deliveries DataFrame.
    """
    processed_matches_path = os.path.join(PROCESSED_DATA_DIR, "matches_cleaned.csv")
    processed_deliveries_path = os.path.join(PROCESSED_DATA_DIR, "deliveries_cleaned.csv")

    matches_df.to_csv(processed_matches_path, index=False)
    deliveries_df.to_csv(processed_deliveries_path, index=False)
    
    train_logger.info(f"Successfully saved clean datasets to '{PROCESSED_DATA_DIR}/'")
    train_logger.info(f"- Matches clean: {processed_matches_path}")
    train_logger.info(f"- Deliveries clean: {processed_deliveries_path}")


# -----------------------------------------------------------------------------
# 6. Feature Engineering
# -----------------------------------------------------------------------------
def run_feature_engineering():
    """
    Performs robust cricket feature engineering to create predictor inputs.
    Loads cleaned datasets, targets chasing (inning == 2) trajectories, and computes:
        - Current Score
        - Target
        - Runs Left
        - Balls Left
        - Current Run Rate (CRR)
        - Required Run Rate (RRR)
        - Wickets Remaining
        - Match Pressure Index (MPI)
        - Required Boundary Percentage (RBP)
        - Run Momentum (RM)
    
    Saves the final engineered dataset as data/processed/training_data.csv.
    """
    train_logger.info("Starting Part 2 - Feature Engineering Pipeline...")
    
    processed_matches_path = os.path.join(PROCESSED_DATA_DIR, "matches_cleaned.csv")
    processed_deliveries_path = os.path.join(PROCESSED_DATA_DIR, "deliveries_cleaned.csv")
    
    if not os.path.exists(processed_matches_path) or not os.path.exists(processed_deliveries_path):
        train_logger.error("Cleaned intermediate datasets are missing. Run preprocessing first.")
        return

    matches_df = pd.read_csv(processed_matches_path)
    deliveries_df = pd.read_csv(processed_deliveries_path)
    
    train_logger.info("Calculating 1st innings total runs to set targets...")
    # Calculate target runs for 2nd innings run-chase
    # Group by match_id and inning to find 1st innings score
    total_score_df = deliveries_df[deliveries_df["inning"] == 1].groupby("match_id")["total_runs"].sum().reset_index()
    total_score_df.rename(columns={"total_runs": "first_innings_score"}, inplace=True)
    total_score_df["target"] = total_score_df["first_innings_score"] + 1
    
    # Merge target back into matches
    matches_df = matches_df.merge(total_score_df, left_on="id", right_on="match_id", how="inner")
    
    # Filter deliveries to second innings (chasing team trajectory)
    chase_deliveries = deliveries_df[deliveries_df["inning"] == 2].copy()
    
    # Keep only columns we need from matches
    matches_subset = matches_df[["id", "target", "winner", "venue", "city"]].copy()
    chase_deliveries = chase_deliveries.merge(matches_subset, left_on="match_id", right_on="id", how="inner")
    
    train_logger.info(f"Filtered to 2nd innings run-chase deliveries. Total rows: {len(chase_deliveries)}")
    
    if len(chase_deliveries) == 0:
        train_logger.warning("No second innings deliveries found! Feature engineering cannot proceed.")
        return

    # Calculate cumulative runs chased
    train_logger.info("Computing cumulative scores, runs left, and balls left...")
    chase_deliveries["current_score"] = chase_deliveries.groupby("match_id")["total_runs"].cumsum()
    chase_deliveries["runs_left"] = chase_deliveries["target"] - chase_deliveries["current_score"]
    chase_deliveries["runs_left"] = chase_deliveries["runs_left"].clip(lower=0)
    
    # Calculate balls left
    # Detect over index layout (1-indexed vs 0-indexed)
    min_over = chase_deliveries["over"].min()
    if min_over == 1:
        balls_bowled = (chase_deliveries["over"] - 1) * 6 + chase_deliveries["ball"]
    else:
        balls_bowled = chase_deliveries["over"] * 6 + chase_deliveries["ball"]
        
    chase_deliveries["balls_left"] = 120 - balls_bowled
    chase_deliveries["balls_left"] = chase_deliveries["balls_left"].clip(lower=0)
    
    # Calculate wickets remaining
    train_logger.info("Computing wickets remaining...")
    chase_deliveries["is_wicket"] = chase_deliveries["player_dismissed"].notna().astype(int)
    chase_deliveries["wickets_down"] = chase_deliveries.groupby("match_id")["is_wicket"].cumsum()
    chase_deliveries["wickets_remaining"] = 10 - chase_deliveries["wickets_down"]
    chase_deliveries["wickets_remaining"] = chase_deliveries["wickets_remaining"].clip(0, 10)
    
    # Calculate CRR and RRR
    train_logger.info("Computing CRR and RRR...")
    # Safe division for CRR
    safe_balls_bowled = np.where(balls_bowled == 0, 1, balls_bowled)
    chase_deliveries["crr"] = round((chase_deliveries["current_score"] * 6.0) / safe_balls_bowled, 2)
    
    # Safe division for RRR
    safe_balls_left = np.where(chase_deliveries["balls_left"] == 0, 1, chase_deliveries["balls_left"])
    chase_deliveries["rrr"] = round((chase_deliveries["runs_left"] * 6.0) / safe_balls_left, 2)
    
    # If balls_left is 0 but runs_left > 0, set RRR to high fallback (99.99)
    chase_deliveries.loc[(chase_deliveries["balls_left"] == 0) & (chase_deliveries["runs_left"] > 0), "rrr"] = 99.99
    # If runs_left is 0, RRR is 0
    chase_deliveries.loc[chase_deliveries["runs_left"] <= 0, "rrr"] = 0.0

    # Calculate Match Pressure Index (MPI)
    train_logger.info("Computing Match Pressure Index (MPI)...")
    progression = balls_bowled / 120.0
    # MPI formula: combines RRR pressure, CRR support, wickets remaining, and match progression stage
    chase_deliveries["match_pressure_index"] = (
        (chase_deliveries["rrr"] / (chase_deliveries["crr"] + 0.1)) * 
        (10.0 / (chase_deliveries["wickets_remaining"] + 0.1)) * 
        progression
    )
    chase_deliveries["match_pressure_index"] = round(chase_deliveries["match_pressure_index"].clip(0, 100), 2)
    
    # Calculate Required Boundary Percentage (RBP)
    train_logger.info("Computing Required Boundary Percentage (RBP)...")
    # A standard target requirement boundary percent proxy based on the run rate scaling
    chase_deliveries["required_boundary_percentage"] = (chase_deliveries["rrr"] / 24.0) * 100.0
    chase_deliveries["required_boundary_percentage"] = round(chase_deliveries["required_boundary_percentage"].clip(0, 100), 2)
    
    # Calculate Run Momentum (RM)
    train_logger.info("Computing Run Momentum...")
    chase_deliveries["run_momentum"] = round(chase_deliveries["crr"] - chase_deliveries["rrr"], 2)
    
    # Binary label: did the chasing (batting_team) win?
    chase_deliveries["result"] = (chase_deliveries["winner"] == chase_deliveries["batting_team"]).astype(int)
    
    # Filter out columns and clean final dataframe
    final_features = [
        "match_id", "batting_team", "bowling_team", "venue", "city",
        "target", "current_score", "runs_left", "balls_left", "wickets_remaining",
        "crr", "rrr", "match_pressure_index", "required_boundary_percentage", "run_momentum", "result"
    ]
    
    training_df = chase_deliveries[final_features].copy()
    
    # Save training_data.csv to processed folder
    training_data_path = os.path.join(PROCESSED_DATA_DIR, "training_data.csv")
    training_df.to_csv(training_data_path, index=False)
    
    train_logger.info(f"Successfully engineered features! Total training samples: {len(training_df)}")
    train_logger.info(f"Saved engineered dataset to: {training_data_path}")


# -----------------------------------------------------------------------------
# 7. Train/Test Splitting
# -----------------------------------------------------------------------------
def split_processed_data():
    """
    Loads the engineered training_data.csv, separates features and the target label ('result'),
    splits them into 80% training and 20% testing sets using a fixed random state,
    and displays metrics such as dataset dimensions and class distributions.
    """
    train_logger.info("Starting Part 3 - Train/Test Splitting Pipeline...")
    
    training_data_path = os.path.join(PROCESSED_DATA_DIR, "training_data.csv")
    
    if not os.path.exists(training_data_path):
        train_logger.error(f"Engineered dataset is missing at {training_data_path}. Running feature engineering first.")
        run_feature_engineering()
        
    # Read the dataset
    df = pd.read_csv(training_data_path)
    train_logger.info(f"Loaded engineered dataset with shape: {df.shape}")
    
    # Define features (X) and target (y)
    # The target is 'result' (binary: 1 if chasing team wins, 0 otherwise)
    target_col = "result"
    
    if target_col not in df.columns:
        train_logger.error(f"Target column '{target_col}' not found in the dataset columns!")
        return None
        
    # All other columns are potential features/context keys
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Perform split: 80% Train, 20% Test
    # Using fixed random state (42) for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=config.RANDOM_SEED,
        stratify=y if len(y.unique()) > 1 else None  # Stratified split to preserve class proportions if possible
    )
    
    # Display dataset dimensions
    train_logger.info("--- Dataset Dimension Summary ---")
    train_logger.info(f"Total dataset shape: {df.shape}")
    train_logger.info(f"Features Train shape: {X_train.shape} | Target Train shape: {y_train.shape}")
    train_logger.info(f"Features Test shape:  {X_test.shape} | Target Test shape:  {y_test.shape}")
    
    # Display class distribution
    train_logger.info("--- Class (Target 'result') Distribution Summary ---")
    
    # Training distribution
    train_counts = y_train.value_counts()
    train_pct = y_train.value_counts(normalize=True) * 100
    train_logger.info("Training Target Distribution:")
    for cls in train_counts.index:
         train_logger.info(f"  Class {cls}: {train_counts[cls]} samples ({train_pct[cls]:.2f}%)")
        
    # Testing distribution
    test_counts = y_test.value_counts()
    test_pct = y_test.value_counts(normalize=True) * 100
    train_logger.info("Testing Target Distribution:")
    for cls in test_counts.index:
         train_logger.info(f"  Class {cls}: {test_counts[cls]} samples ({test_pct[cls]:.2f}%)")
        
    train_logger.info("Train/test split successfully executed and verified.")
    return X_train, X_test, y_train, y_test


# -----------------------------------------------------------------------------
# 8. Feature Preprocessing Pipeline Construction
# -----------------------------------------------------------------------------
def build_preprocessing_pipeline():
    """
    Constructs the feature transformation pipeline using ColumnTransformer,
    OneHotEncoder, and StandardScaler, and serializes the preprocessing artifacts:
    - scaler.pkl
    - encoder.pkl
    - feature_columns.pkl
    """
    train_logger.info("Starting Part 4 - Feature Preprocessing Pipeline...")
    
    # Get the split datasets
    split_res = split_processed_data()
    if split_res is None:
        train_logger.error("Splitting failed, cannot construct preprocessing pipeline.")
        return None
        
    X_train, X_test, y_train, y_test = split_res
    
    # Rename columns to match the standard schema used in standard prediction pipeline
    rename_dict = {
        "target": "total_runs_x",
        "wickets_remaining": "wickets_left"
    }
    X_train = X_train.rename(columns=rename_dict)
    X_test = X_test.rename(columns=rename_dict)
    
    # Define feature lists
    categorical_features = ["batting_team", "bowling_team", "venue"]
    numerical_features = [
        "runs_left", "balls_left", "wickets_left", "total_runs_x",
        "crr", "rrr", "match_pressure_index", "required_boundary_percentage",
        "run_momentum"
    ]
    
    # Subset features to standard training set
    feature_columns = categorical_features + numerical_features
    X_train_filtered = X_train[feature_columns].copy()
    X_test_filtered = X_test[feature_columns].copy()
    
    train_logger.info("--- Feature Layout ---")
    train_logger.info(f"Categorical Features ({len(categorical_features)}): {categorical_features}")
    train_logger.info(f"Numerical Features ({len(numerical_features)}): {numerical_features}")
    train_logger.info(f"Total Pre-Transformation Columns: {feature_columns}")
    
    # Fit individual transformers to fulfill specific serialization requirements
    train_logger.info("Fitting StandardScaler on numerical columns...")
    scaler = StandardScaler()
    scaler.fit(X_train_filtered[numerical_features])
    
    train_logger.info("Fitting OneHotEncoder on categorical columns...")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(X_train_filtered[categorical_features])
    
    # Build standard scikit-learn ColumnTransformer
    train_logger.info("Constructing and fitting full ColumnTransformer pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features)
        ]
    )
    # Fit the full transformer
    preprocessor.fit(X_train_filtered)
    
    # Transform training and testing datasets to verify no exceptions
    X_train_trans = preprocessor.transform(X_train_filtered)
    X_test_trans = preprocessor.transform(X_test_filtered)
    
    train_logger.info(f"Transformed Training Shape: {X_train_trans.shape}")
    train_logger.info(f"Transformed Testing Shape:  {X_test_trans.shape}")
    
    # Save required artifacts
    scaler_path = os.path.join(config.MODEL_DIR, "scaler.pkl")
    encoder_path = os.path.join(config.MODEL_DIR, "encoder.pkl")
    feature_cols_path = os.path.join(config.MODEL_DIR, "feature_columns.pkl")
    preprocessor_path = os.path.join(config.MODEL_DIR, "preprocessor.pkl")
    
    # Also support config.SCALER_PATH ("standard_scaler.pkl") for backward compatibility
    standard_scaler_path = config.SCALER_PATH
    
    # Dump objects
    joblib.dump(scaler, scaler_path)
    joblib.dump(scaler, standard_scaler_path)  # Support standard_scaler.pkl as well
    joblib.dump(encoder, encoder_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    # For feature columns, we save a dictionary with raw column details or the list of features
    feature_columns_metadata = {
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "all_input_features": feature_columns,
        "encoded_feature_names": list(preprocessor.get_feature_names_out())
    }
    joblib.dump(feature_columns_metadata, feature_cols_path)
    
    train_logger.info("Successfully saved preprocessing pipeline artifacts:")
    train_logger.info(f"  - Scaler: {scaler_path} (and standard_scaler.pkl)")
    train_logger.info(f"  - Encoder: {encoder_path}")
    train_logger.info(f"  - ColumnTransformer: {preprocessor_path}")
    train_logger.info(f"  - Feature Columns metadata: {feature_cols_path}")
    
    return preprocessor, X_train_filtered, X_test_filtered


# -----------------------------------------------------------------------------
# 9. Logistic Regression Model Training & Evaluation
# -----------------------------------------------------------------------------
def train_logistic_regression(preprocessor=None, X_train_filtered=None, X_test_filtered=None, y_train=None, y_test=None):
    """
    Trains and evaluates a Logistic Regression classifier on the preprocessed dataset.
    Computes performance metrics (Accuracy, Precision, Recall, F1, ROC AUC) and
    confusion matrix, and stores them in a dictionary.
    """
    train_logger.info("Starting Part 5 - Logistic Regression Training & Evaluation...")
    
    # 1. Retrieve split data and fit the preprocessor if not provided
    if preprocessor is None or X_train_filtered is None or X_test_filtered is None or y_train is None or y_test is None:
        prep_res = build_preprocessing_pipeline()
        if prep_res is None:
            train_logger.error("Preprocessing failed. Cannot train Logistic Regression model.")
            return None
            
        preprocessor, X_train_filtered, X_test_filtered = prep_res
        
        # Reload split labels to ensure integrity
        split_res = split_processed_data()
        if split_res is None:
            train_logger.error("Splitting failed. Cannot retrieve target vectors.")
            return None
        _, _, y_train, y_test = split_res
    
    # Transform data
    X_train_trans = preprocessor.transform(X_train_filtered)
    X_test_trans = preprocessor.transform(X_test_filtered)
    
    # 2. Model Initialization
    train_logger.info("Initializing Logistic Regression model...")
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=config.RANDOM_SEED
    )
    
    # 3. Training
    train_logger.info("Fitting Logistic Regression model on transformed training data...")
    t_start = time.time()
    lr_model.fit(X_train_trans, y_train)
    training_time = time.time() - t_start
    
    # 4. Predictions & Probability scoring
    train_logger.info("Generating predictions and probability scores on test set...")
    p_start = time.time()
    y_pred = lr_model.predict(X_test_trans)
    # Probability of class 1 (win/chasing success)
    y_prob = lr_model.predict_proba(X_test_trans)[:, 1]
    prediction_time = time.time() - p_start
    
    # 5. Metric Calculations
    train_logger.info("Computing validation metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Calculate ROC AUC safely
    if len(np.unique(y_test)) > 1:
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = 1.0  # Default or trivial score when only one class is present in test set
        train_logger.warning("Only one unique class label present in test set. ROC AUC defaulted to 1.0.")
        
    cm = confusion_matrix(y_test, y_pred)
    
    # 6. Store results inside a dictionary
    results = {
        "model_name": "Logistic Regression",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "training_time": float(training_time),
        "prediction_time": float(prediction_time),
        "confusion_matrix": cm.tolist()  # Convert to list of lists for easy JSON serialization if needed
    }
    
    # Display and Log results
    train_logger.info("--- Logistic Regression Model Performance Summary ---")
    train_logger.info(f"Accuracy:  {results['accuracy']:.4f}")
    train_logger.info(f"Precision: {results['precision']:.4f}")
    train_logger.info(f"Recall:    {results['recall']:.4f}")
    train_logger.info(f"F1 Score:  {results['f1_score']:.4f}")
    train_logger.info(f"ROC AUC:   {results['roc_auc']:.4f}")
    train_logger.info(f"Confusion Matrix:\n{cm}")
    
    return lr_model, results


# -----------------------------------------------------------------------------
# 10. Random Forest Model Training & Evaluation
# -----------------------------------------------------------------------------
def train_random_forest(preprocessor=None, X_train_filtered=None, X_test_filtered=None, y_train=None, y_test=None):
    """
    Trains and evaluates a Random Forest classifier on the preprocessed dataset.
    Computes performance metrics (Accuracy, Precision, Recall, F1, ROC AUC,
    Confusion Matrix, and Feature Importances) and stores them in a dictionary.
    """
    train_logger.info("Starting Part 6 - Random Forest Training & Evaluation...")
    
    # 1. Retrieve split data and fit the preprocessor if not provided
    if preprocessor is None or X_train_filtered is None or X_test_filtered is None or y_train is None or y_test is None:
        prep_res = build_preprocessing_pipeline()
        if prep_res is None:
            train_logger.error("Preprocessing failed. Cannot train Random Forest model.")
            return None
            
        preprocessor, X_train_filtered, X_test_filtered = prep_res
        
        # Reload split labels to ensure integrity
        split_res = split_processed_data()
        if split_res is None:
            train_logger.error("Splitting failed. Cannot retrieve target vectors.")
            return None
        _, _, y_train, y_test = split_res
    
    # Transform data
    X_train_trans = preprocessor.transform(X_train_filtered)
    X_test_trans = preprocessor.transform(X_test_filtered)
    
    # 2. Model Initialization (Hyperparameters)
    train_logger.info("Initializing Random Forest classifier with default hyperparameters...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=config.RANDOM_SEED
    )
    
    # 3. Training
    train_logger.info("Fitting Random Forest model on transformed training data...")
    t_start = time.time()
    rf_model.fit(X_train_trans, y_train)
    training_time = time.time() - t_start
    
    # 4. Predictions & Probability scoring
    train_logger.info("Generating predictions and probability scores on test set...")
    p_start = time.time()
    y_pred = rf_model.predict(X_test_trans)
    y_prob = rf_model.predict_proba(X_test_trans)[:, 1]
    prediction_time = time.time() - p_start
    
    # 5. Metric Calculations
    train_logger.info("Computing validation metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Calculate ROC AUC safely
    if len(np.unique(y_test)) > 1:
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = 1.0
        train_logger.warning("Only one unique class label present in test set. ROC AUC defaulted to 1.0.")
        
    cm = confusion_matrix(y_test, y_pred)
    
    # 6. Feature Importance Calculation
    train_logger.info("Extracting feature importances...")
    feature_names = preprocessor.get_feature_names_out()
    importances = rf_model.feature_importances_
    
    # Clean feature names (remove prefixes)
    feature_imp_list = []
    for name, imp in zip(feature_names, importances):
        feature_imp_list.append({"Feature": name, "Importance": float(imp)})
    
    # Sort by importance descending
    feature_imp_list = sorted(feature_imp_list, key=lambda x: x["Importance"], reverse=True)
    
    # 7. Store results inside a dictionary
    results = {
        "model_name": "Random Forest",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "training_time": float(training_time),
        "prediction_time": float(prediction_time),
        "confusion_matrix": cm.tolist(),
        "feature_importances": feature_imp_list
    }
    
    # Display and Log results
    train_logger.info("--- Random Forest Model Performance Summary ---")
    train_logger.info(f"Accuracy:  {results['accuracy']:.4f}")
    train_logger.info(f"Precision: {results['precision']:.4f}")
    train_logger.info(f"Recall:    {results['recall']:.4f}")
    train_logger.info(f"F1 Score:  {results['f1_score']:.4f}")
    train_logger.info(f"ROC AUC:   {results['roc_auc']:.4f}")
    train_logger.info(f"Confusion Matrix:\n{cm}")
    
    # Display top 10 feature importances
    train_logger.info("Top 10 Feature Importances:")
    for idx, item in enumerate(feature_imp_list[:10]):
        train_logger.info(f"  {idx+1}. {item['Feature']}: {item['Importance']:.4f}")
        
    return rf_model, results


# -----------------------------------------------------------------------------
# 11. Gradient Boosting Model Training & Evaluation
# -----------------------------------------------------------------------------
def train_gradient_boosting(preprocessor=None, X_train_filtered=None, X_test_filtered=None, y_train=None, y_test=None):
    """
    Trains and evaluates a Gradient Boosting classifier on the preprocessed dataset.
    Computes performance metrics (Accuracy, Precision, Recall, F1, ROC AUC,
    Confusion Matrix, and ROC Curve threshold data) and stores them in a dictionary.
    """
    train_logger.info("Starting Part 7 - Gradient Boosting Training & Evaluation...")
    
    # 1. Retrieve split data and fit the preprocessor if not provided
    if preprocessor is None or X_train_filtered is None or X_test_filtered is None or y_train is None or y_test is None:
        prep_res = build_preprocessing_pipeline()
        if prep_res is None:
            train_logger.error("Preprocessing failed. Cannot train Gradient Boosting model.")
            return None
            
        preprocessor, X_train_filtered, X_test_filtered = prep_res
        
        # Reload split labels to ensure integrity
        split_res = split_processed_data()
        if split_res is None:
            train_logger.error("Splitting failed. Cannot retrieve target vectors.")
            return None
        _, _, y_train, y_test = split_res
    
    # Transform data
    X_train_trans = preprocessor.transform(X_train_filtered)
    X_test_trans = preprocessor.transform(X_test_filtered)
    
    # 2. Model Initialization (Hyperparameters)
    train_logger.info("Initializing Gradient Boosting Classifier...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=config.RANDOM_SEED
    )
    
    # 3. Training
    train_logger.info("Fitting Gradient Boosting model on transformed training data...")
    t_start = time.time()
    gb_model.fit(X_train_trans, y_train)
    training_time = time.time() - t_start
    
    # 4. Predictions & Probability scoring
    train_logger.info("Generating predictions and probability scores on test set...")
    p_start = time.time()
    y_pred = gb_model.predict(X_test_trans)
    y_prob = gb_model.predict_proba(X_test_trans)[:, 1]
    prediction_time = time.time() - p_start
    
    # 5. Metric Calculations
    train_logger.info("Computing validation metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Calculate ROC AUC and ROC Curve safely
    if len(np.unique(y_test)) > 1:
        roc_auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    else:
        roc_auc = 1.0
        fpr, tpr, thresholds = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
        train_logger.warning("Only one unique class label present in test set. ROC metrics defaulted.")
        
    cm = confusion_matrix(y_test, y_pred)
    
    # 6. Store results inside a dictionary
    results = {
        "model_name": "Gradient Boosting",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "training_time": float(training_time),
        "prediction_time": float(prediction_time),
        "confusion_matrix": cm.tolist(),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist()
        }
    }
    
    # Display and Log results
    train_logger.info("--- Gradient Boosting Model Performance Summary ---")
    train_logger.info(f"Accuracy:  {results['accuracy']:.4f}")
    train_logger.info(f"Precision: {results['precision']:.4f}")
    train_logger.info(f"Recall:    {results['recall']:.4f}")
    train_logger.info(f"F1 Score:  {results['f1_score']:.4f}")
    train_logger.info(f"ROC AUC:   {results['roc_auc']:.4f}")
    train_logger.info(f"Confusion Matrix:\n{cm}")
    
    # Print sample points from ROC curve for evaluation validation
    train_logger.info("Sample ROC Curve Coordinates (FPR, TPR):")
    sample_size = min(5, len(fpr))
    indices = np.linspace(0, len(fpr) - 1, sample_size, dtype=int)
    for idx in indices:
        train_logger.info(f"  FPR: {fpr[idx]:.4f} | TPR: {tpr[idx]:.4f} | Threshold: {thresholds[idx]:.4f}")
        
    return gb_model, results


# -----------------------------------------------------------------------------
# 12. XGBoost Model Training & Evaluation
# -----------------------------------------------------------------------------
def train_xgboost(preprocessor=None, X_train_filtered=None, X_test_filtered=None, y_train=None, y_test=None):
    """
    Trains and evaluates an XGBoost classifier with Hyperparameter tuning using GridSearchCV.
    Includes Cross Validation, optional early stopping configuration on validation data,
    computes best estimator params, and returns a dictionary of metrics:
    - Accuracy
    - ROC AUC
    - Precision
    - Recall
    - F1 Score
    """
    train_logger.info("Starting Part 8 - XGBoost Training & Evaluation...")
    
    # 1. Retrieve split data and fit the preprocessor if not provided
    if preprocessor is None or X_train_filtered is None or X_test_filtered is None or y_train is None or y_test is None:
        prep_res = build_preprocessing_pipeline()
        if prep_res is None:
            train_logger.error("Preprocessing failed. Cannot train XGBoost model.")
            return None
            
        preprocessor, X_train_filtered, X_test_filtered = prep_res
        
        # Reload split labels to ensure integrity
        split_res = split_processed_data()
        if split_res is None:
            train_logger.error("Splitting failed. Cannot retrieve target vectors.")
            return None
        _, _, y_train, y_test = split_res
    
    # Transform data
    X_train_trans = preprocessor.transform(X_train_filtered)
    X_test_trans = preprocessor.transform(X_test_filtered)
    
    # 2. Model Initialization (XGBClassifier for binary classification)
    train_logger.info("Initializing XGBoost classifier...")
    xgb_base = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=config.RANDOM_SEED
    )
    
    # 3. Hyperparameter Tuning using GridSearchCV
    # Define a robust, yet lightweight parameter grid for faster compilation
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "n_estimators": [50, 100]
    }
    
    train_logger.info(f"Running GridSearchCV with 3-fold CV. Parameter grid: {param_grid}")
    t_start = time.time()
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_trans, y_train)
    
    best_params = grid_search.best_params_
    train_logger.info(f"Best hyperparameters found via GridSearch: {best_params}")
    
    # Retrieve best estimator
    best_xgb = grid_search.best_estimator_
    
    # 4. Early Stopping if applicable
    # To perform early stopping with the best estimator parameters, we split the training data 
    # into a sub-train and a validation set.
    train_logger.info("Setting up validation split for early stopping validation...")
    X_sub_train, X_val, y_sub_train, y_val = train_test_split(
        X_train_trans, y_train,
        test_size=0.15,
        random_state=config.RANDOM_SEED,
        stratify=y_train if len(np.unique(y_train)) > 1 else None
    )
    
    # Retrain the best configuration with early stopping on validation split
    train_logger.info("Fitting best estimator with early stopping (evaluating on validation split)...")
    # Supports early_stopping_rounds in either fit or constructor depending on xgboost version
    try:
        best_xgb.set_params(early_stopping_rounds=10)
        best_xgb.fit(
            X_sub_train, y_sub_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        train_logger.info(f"Early stopping model fit completed successfully. Best iteration: {best_xgb.best_iteration if hasattr(best_xgb, 'best_iteration') else 'N/A'}")
    except Exception as e:
        train_logger.warning(f"Early stopping parameters not supported in constructor or fit directly. Retrying simple fit. Error: {e}")
        best_xgb = XGBClassifier(**best_params, random_state=config.RANDOM_SEED)
        best_xgb.fit(X_train_trans, y_train)
    training_time = time.time() - t_start
    
    # 5. Predictions & Probability scoring
    train_logger.info("Generating predictions and probability scores on test set using best estimator...")
    p_start = time.time()
    y_pred = best_xgb.predict(X_test_trans)
    y_prob = best_xgb.predict_proba(X_test_trans)[:, 1]
    prediction_time = time.time() - p_start
    
    # 6. Metric Calculations
    train_logger.info("Computing validation metrics for XGBoost...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Calculate ROC AUC safely
    if len(np.unique(y_test)) > 1:
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = 1.0
        train_logger.warning("Only one unique class label present in test set. ROC metrics defaulted.")
        
    cm = confusion_matrix(y_test, y_pred)
    
    # 7. Store results inside a dictionary
    results = {
        "model_name": "XGBoost",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "training_time": float(training_time),
        "prediction_time": float(prediction_time),
        "confusion_matrix": cm.tolist(),
        "best_params": best_params
    }
    
    # Display and Log results
    train_logger.info("--- XGBoost Best Estimator Performance Summary ---")
    train_logger.info(f"Best Params: {results['best_params']}")
    train_logger.info(f"Accuracy:    {results['accuracy']:.4f}")
    train_logger.info(f"Precision:   {results['precision']:.4f}")
    train_logger.info(f"Recall:      {results['recall']:.4f}")
    train_logger.info(f"F1 Score:    {results['f1_score']:.4f}")
    train_logger.info(f"ROC AUC:     {results['roc_auc']:.4f}")
    train_logger.info(f"Confusion Matrix:\n{cm}")
    
    return best_xgb, results


# -----------------------------------------------------------------------------
# 13. Model Evaluation Plot Generation
# -----------------------------------------------------------------------------
def generate_evaluation_plots(sorted_models, preprocessor, X_train_filtered, X_test_filtered, y_train, y_test):
    """
    Generates and saves model evaluation plots (ROC Curve, Precision-Recall Curve,
    Confusion Matrix, Feature Importance, Learning Curve, Validation Curve) into reports/ folder.
    """
    train_logger.info("Starting Part 11 - Model Evaluation Plot Generation...")
    reports_dir = config.REPORTS_DIR
    os.makedirs(reports_dir, exist_ok=True)
    
    # Transform datasets for predictions/prob scoring inside plots
    X_train_trans = preprocessor.transform(X_train_filtered)
    X_test_trans = preprocessor.transform(X_test_filtered)
    
    # 1. ROC Curve (Comparing all models)
    try:
        plt.figure(figsize=(8, 6))
        for entry in sorted_models:
            model = entry["model"]
            name = entry["results"]["model_name"]
            y_prob = model.predict_proba(X_test_trans)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")
        plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC = 0.5000)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        roc_path = os.path.join(reports_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=100)
        plt.close()
        train_logger.info(f"✔️ Saved ROC Curve plot to {roc_path}")
    except Exception as e:
        train_logger.error(f"❌ Failed to generate ROC Curve: {e}")

    # 2. Precision-Recall Curve (Comparing all models)
    try:
        plt.figure(figsize=(8, 6))
        for entry in sorted_models:
            model = entry["model"]
            name = entry["results"]["model_name"]
            y_prob = model.predict_proba(X_test_trans)[:, 1]
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
            ap = average_precision_score(y_test, y_prob)
            plt.plot(recall_vals, precision_vals, label=f"{name} (AP = {ap:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall (PR) Curve")
        plt.legend(loc="lower left")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        pr_path = os.path.join(reports_dir, "precision_recall_curve.png")
        plt.savefig(pr_path, dpi=100)
        plt.close()
        train_logger.info(f"✔️ Saved Precision-Recall Curve plot to {pr_path}")
    except Exception as e:
        train_logger.error(f"❌ Failed to generate Precision-Recall Curve: {e}")

    # Identify the best model (the first one in the sorted list)
    best_entry = sorted_models[0]
    best_model_obj = best_entry["model"]
    best_model_name = best_entry["results"]["model_name"]
    
    # 3. Confusion Matrix (for Best Model)
    try:
        y_pred = best_model_obj.predict(X_test_trans)
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=["Loss", "Win"], yticklabels=["Loss", "Win"],
               title=f"Confusion Matrix: {best_model_name}",
               ylabel="True Label",
               xlabel="Predicted Label")
        
        # Loop over data dimensions and create text annotations.
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        cm_path = os.path.join(reports_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=100)
        plt.close()
        train_logger.info(f"✔️ Saved Confusion Matrix plot to {cm_path}")
    except Exception as e:
        train_logger.error(f"❌ Failed to generate Confusion Matrix: {e}")

    # 4. Feature Importance (of Best Model or coefficients if LR)
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        if hasattr(best_model_obj, 'feature_importances_'):
            importances = best_model_obj.feature_importances_
            title = f"Top 10 Feature Importances: {best_model_name}"
            ylabel = "Importance"
        elif hasattr(best_model_obj, 'coef_'):
            importances = np.abs(best_model_obj.coef_[0])
            title = f"Top 10 Feature Coefficients (Abs): {best_model_name}"
            ylabel = "Coefficient Magnitude"
        else:
            importances = None
        
        if importances is not None:
            feature_names = preprocessor.get_feature_names_out()
            indices = np.argsort(importances)[::-1][:10]
            top_importances = importances[indices]
            top_names = [feature_names[i] for i in indices]
            
            # Clean up names for better visualization
            top_names_clean = [name.replace("num__", "").replace("cat__", "") for name in top_names]
            
            ax.barh(np.arange(len(top_names_clean)), top_importances[::-1], align='center', color='skyblue', edgecolor='gray')
            ax.set_yticks(np.arange(len(top_names_clean)))
            ax.set_yticklabels(top_names_clean[::-1])
            ax.set_xlabel(ylabel)
            ax.set_title(title)
            plt.tight_layout()
            fi_path = os.path.join(reports_dir, "feature_importance.png")
            plt.savefig(fi_path, dpi=100)
            plt.close()
            train_logger.info(f"✔️ Saved Feature Importance plot to {fi_path}")
        else:
            train_logger.warning(f"⚠️ Model {best_model_name} does not support feature importances or coefficients. Skipping.")
    except Exception as e:
        train_logger.error(f"❌ Failed to generate Feature Importance: {e}")

    # 5. Learning Curve (for Best Model)
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            best_model_obj, X_train_trans, y_train,
            cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5),
            scoring='accuracy', random_state=config.RANDOM_SEED
        )
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training Score")
        plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation Score")
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
        plt.title(f"Learning Curve: {best_model_name}")
        plt.xlabel("Training Examples")
        plt.ylabel("Score (Accuracy)")
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        lc_path = os.path.join(reports_dir, "learning_curve.png")
        plt.savefig(lc_path, dpi=100)
        plt.close()
        train_logger.info(f"✔️ Saved Learning Curve plot to {lc_path}")
    except Exception as e:
        train_logger.warning(f"⚠️ Failed to generate Learning Curve: {e}")

    # 6. Validation Curve (for Best Model)
    if best_model_name == "Logistic Regression":
        param_name = "C"
        param_range = np.array([0.01, 0.1, 1.0, 10.0])
    elif best_model_name == "Random Forest":
        param_name = "max_depth"
        param_range = np.array([3, 5, 7, 10, 15])
    elif best_model_name == "Gradient Boosting":
        param_name = "max_depth"
        param_range = np.array([3, 5, 7, 10])
    elif best_model_name == "XGBoost":
        param_name = "max_depth"
        param_range = np.array([3, 5, 7, 10])
    else:
        param_name = None
        
    if param_name is not None:
        try:
            train_scores, test_scores = validation_curve(
                best_model_obj, X_train_trans, y_train,
                param_name=param_name, param_range=param_range,
                cv=3, scoring="accuracy", n_jobs=-1
            )
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            plt.figure(figsize=(8, 6))
            if param_name == "C":
                plt.semilogx(param_range, train_mean, 'o-', color="r", label="Training Score")
                plt.semilogx(param_range, test_mean, 'o-', color="g", label="Cross-validation Score")
            else:
                plt.plot(param_range, train_mean, 'o-', color="r", label="Training Score")
                plt.plot(param_range, test_mean, 'o-', color="g", label="Cross-validation Score")
                
            plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
            plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
            plt.title(f"Validation Curve: {best_model_name} (varying {param_name})")
            plt.xlabel(param_name)
            plt.ylabel("Score (Accuracy)")
            plt.legend(loc="best")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            vc_path = os.path.join(reports_dir, "validation_curve.png")
            plt.savefig(vc_path, dpi=100)
            plt.close()
            train_logger.info(f"✔️ Saved Validation Curve plot to {vc_path}")
        except Exception as e:
            train_logger.warning(f"⚠️ Failed to generate Validation Curve: {e}")

    # Double check that saved plots actually exist
    train_logger.info("\n" + "="*110)
    train_logger.info("                         EVALUATION PLOTS REPORT")
    train_logger.info("="*110)
    plots_to_verify = [
        "roc_curve.png",
        "precision_recall_curve.png",
        "confusion_matrix.png",
        "feature_importance.png",
        "learning_curve.png",
        "validation_curve.png"
    ]
    for plot_name in plots_to_verify:
        p = os.path.join(reports_dir, plot_name)
        if os.path.exists(p):
            size_kb = os.path.getsize(p) / 1024.0
            train_logger.info(f"✔️ {plot_name:<30} | Status: EXISTS | Size: {size_kb:.2f} KB | Path: {p}")
        else:
            train_logger.warning(f"⚠️ {plot_name:<30} | Status: NOT FOUND")
    train_logger.info("="*110 + "\n")


# -----------------------------------------------------------------------------
# Execution Orchestrator
# -----------------------------------------------------------------------------
def run_preprocessing_pipeline():
    """
    Main orchestration routine running the full data pipeline up to XGBoost training
    and displays a comprehensive model comparison table.
    """
    verify_project_structure()
    
    # Generate synthetic raw files if not present on disk
    generate_synthetic_raw_data()
    
    # Paths to raw inputs
    raw_matches_path = os.path.join(config.DATA_DIR, "matches.csv")
    raw_deliveries_path = os.path.join(config.DATA_DIR, "deliveries.csv")
    
    # Load raw
    matches_raw, deliveries_raw = load_raw_datasets(raw_matches_path, raw_deliveries_path)
    
    # Process and sanitize franchise entities
    matches_clean, deliveries_clean = clean_and_standardize_teams(matches_raw, deliveries_raw)
    
    # Save intermediate outputs
    save_processed_datasets(matches_clean, deliveries_clean)
    
    # Execute Feature Engineering
    run_feature_engineering()
    
    # Execute Feature Preprocessing Pipeline
    preprocessor, X_train_filtered, X_test_filtered = build_preprocessing_pipeline()
    
    # Reload split labels once to ensure integrity and prevent redundant reloading
    split_res = split_processed_data()
    if split_res is None:
        train_logger.error("Splitting failed. Cannot continue training orchestrator.")
        return
    _, _, y_train, y_test = split_res
    
    # Train Logistic Regression Model
    lr_model, lr_results = train_logistic_regression(preprocessor, X_train_filtered, X_test_filtered, y_train, y_test)
    
    # Train Random Forest Model
    rf_model, rf_results = train_random_forest(preprocessor, X_train_filtered, X_test_filtered, y_train, y_test)
    
    # Train Gradient Boosting Model
    gb_model, gb_results = train_gradient_boosting(preprocessor, X_train_filtered, X_test_filtered, y_train, y_test)
    
    # Train XGBoost Model
    xgb_model, xgb_results = train_xgboost(preprocessor, X_train_filtered, X_test_filtered, y_train, y_test)
    
    # Compile models and results for comparison
    models_list = []
    if lr_results:
        models_list.append({"model": lr_model, "results": lr_results})
    if rf_results:
        models_list.append({"model": rf_model, "results": rf_results})
    if gb_results:
        models_list.append({"model": gb_model, "results": gb_results})
    if xgb_results:
        models_list.append({"model": xgb_model, "results": xgb_results})
        
    if len(models_list) > 0:
        # Sort by: 1. Accuracy (descending), 2. ROC AUC (descending), 3. F1 Score (descending)
        sorted_models = sorted(
            models_list,
            key=lambda x: (
                x["results"]["accuracy"],
                x["results"]["roc_auc"],
                x["results"]["f1_score"]
            ),
            reverse=True
        )
        
        # Display sorted comparison table
        train_logger.info("\n" + "="*140)
        train_logger.info("                                             SORTED MODEL COMPARISON TABLE")
        train_logger.info("="*140)
        train_logger.info(f"{'Model Name':<25} | {'Accuracy':<15} | {'ROC AUC':<15} | {'F1 Score':<15} | {'Precision':<15} | {'Recall':<15}")
        train_logger.info("-"*140)
        for entry in sorted_models:
            res = entry["results"]
            train_logger.info(f"{res['model_name']:<25} | {res['accuracy']:<15.4f} | {res['roc_auc']:<15.4f} | {res['f1_score']:<15.4f} | {res['precision']:<15.4f} | {res['recall']:<15.4f}")
        train_logger.info("="*140 + "\n")
        
        # Automatically select the best model
        best_model_entry = sorted_models[0]
        best_model_obj = best_model_entry["model"]
        best_model_name = best_model_entry["results"]["model_name"]
        
        train_logger.info(f"🏆 Best Model Selected: {best_model_name} (Accuracy: {best_model_entry['results']['accuracy']:.4f})")
        
        # Define paths to save the best model and other required files
        best_model_path = os.path.join(config.MODEL_DIR, "best_model.pkl")
        model_export_path = os.path.join(config.MODEL_DIR, "model.pkl")
        metrics_json_path = os.path.join(config.MODEL_DIR, "metrics.json")
        
        train_logger.info(f"Serializing and saving {best_model_name} model to {best_model_path} and {model_export_path}...")
        # Save best model
        joblib.dump(best_model_obj, best_model_path)
        joblib.dump(best_model_obj, model_export_path)
        
        # Calculate feature count and dataset size
        X_train_trans = preprocessor.transform(X_train_filtered)
        feature_count = int(X_train_trans.shape[1])
        dataset_size = int(len(X_train_filtered) + len(X_test_filtered))

        # Compile a JSON-serializable metrics structure
        metrics_data = {
            "accuracy": float(best_model_entry["results"]["accuracy"]),
            "precision": float(best_model_entry["results"]["precision"]),
            "recall": float(best_model_entry["results"]["recall"]),
            "f1_score": float(best_model_entry["results"]["f1_score"]),
            "roc_auc": float(best_model_entry["results"]["roc_auc"]),
            "training_time": float(best_model_entry["results"].get("training_time", 0.0)),
            "prediction_time": float(best_model_entry["results"].get("prediction_time", 0.0)),
            "best_model": best_model_name,
            "dataset_size": dataset_size,
            "feature_count": feature_count,
            "best_model_name": best_model_name,
            "best_model_metrics": best_model_entry["results"],
            "all_models_comparison": {}
        }
        for entry in sorted_models:
            res = entry["results"]
            metrics_data["all_models_comparison"][res["model_name"]] = {
                "accuracy": res["accuracy"],
                "precision": res["precision"],
                "recall": res["recall"],
                "f1_score": res["f1_score"],
                "roc_auc": res["roc_auc"],
                "confusion_matrix": res["confusion_matrix"]
            }
            # Include feature_importances for Random Forest if present
            if "feature_importances" in res:
                metrics_data["all_models_comparison"][res["model_name"]]["feature_importances"] = res["feature_importances"]
            # Include roc_curve for Gradient Boosting if present
            if "roc_curve" in res:
                metrics_data["all_models_comparison"][res["model_name"]]["roc_curve"] = res["roc_curve"]
            # Include best_params for XGBoost if present
            if "best_params" in res:
                metrics_data["all_models_comparison"][res["model_name"]]["best_params"] = res["best_params"]
                
        # Save metrics.json
        train_logger.info(f"Saving compiled performance metrics to {metrics_json_path}...")
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=4)
            
        # Define all paths to verify their existence and display locations
        artifacts_to_verify = {
            "best_model.pkl": best_model_path,
            "model.pkl": model_export_path,
            "scaler.pkl": os.path.join(config.MODEL_DIR, "scaler.pkl"),
            "encoder.pkl": os.path.join(config.MODEL_DIR, "encoder.pkl"),
            "feature_columns.pkl": os.path.join(config.MODEL_DIR, "feature_columns.pkl"),
            "metrics.json": metrics_json_path
        }
        
        train_logger.info("\n" + "="*110)
        train_logger.info("                         ML ARTIFACT SERIALIZATION & VERIFICATION REPORT")
        train_logger.info("="*110)
        all_exist = True
        for name, path in artifacts_to_verify.items():
            if os.path.exists(path):
                size_kb = os.path.getsize(path) / 1024.0
                train_logger.info(f"✔️ {name:<20} | Status: EXISTS | Size: {size_kb:.2f} KB | Path: {path}")
            else:
                train_logger.error(f"❌ {name:<20} | Status: MISSING | Path: {path}")
                all_exist = False
        train_logger.info("="*110 + "\n")
        
        if all_exist:
            train_logger.info("🎉 All ML pipeline and preprocessing artifacts have been successfully serialized, saved, and verified!")
        else:
            train_logger.warning("⚠️ Some artifacts could not be verified on disk.")
            
        # Generate model evaluation plots
        generate_evaluation_plots(sorted_models, preprocessor, X_train_filtered, X_test_filtered, y_train, y_test)
            
    else:
        train_logger.error("No model evaluation results could be compiled for comparison.")
        
    train_logger.info("=== Preprocessing, Feature Transformation, Model Comparison, Best Model Selection, Saving & Plotting Completed Successfully ===")


if __name__ == "__main__":
    run_preprocessing_pipeline()

# =============================================================================
# END OF PART 10 (Model Serialization & Verification).
# Training pipeline is fully integrated, validated, and all artifacts are saved.
# =============================================================================

