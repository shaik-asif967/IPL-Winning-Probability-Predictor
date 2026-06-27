# utils.py
"""
Utility and helper functions for data manipulation, calculations, logging,
model serialization, and input validation.
"""

import os
import sys
import logging
import pickle
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional

# -----------------------------------------------------------------------------
# Logger Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("IPL_Predictor_Utils")


# -----------------------------------------------------------------------------
# CRR & RRR Calculation Utilities
# -----------------------------------------------------------------------------
def calculate_crr(score: float, overs: float) -> float:
    """
    Calculate the Current Run Rate (CRR).
    
    Args:
        score: Current score of the batting team.
        overs: Overs bowled (float, e.g., 12.4 for 12 overs and 4 balls).
        
    Returns:
        Current run rate as a float rounded to 2 decimal places.
    """
    total_overs = convert_overs_to_fractional(overs)
    if total_overs <= 0:
        return 0.0
    return round(score / total_overs, 2)


def calculate_rrr(runs_needed: float, balls_left: int) -> float:
    """
    Calculate the Required Run Rate (RRR).
    
    Args:
        runs_needed: Total runs remaining to win.
        balls_left: Total balls remaining in the innings.
        
    Returns:
        Required run rate as a float rounded to 2 decimal places.
    """
    if balls_left <= 0:
        return 99.99 if runs_needed > 0 else 0.0
    return round((runs_needed * 6) / balls_left, 2)


def convert_overs_to_fractional(overs: float) -> float:
    """
    Convert a cricket over representation (e.g., 12.4) to its mathematical
    fractional value (12 + 4/6 = 12.666...).
    
    Args:
        overs: Cricket over representation (float)
        
    Returns:
        The exact mathematical over count as a float.
    """
    overs_str = f"{overs:.1f}"
    if "." in overs_str:
        completed_overs, balls = overs_str.split(".")
        completed_overs = int(completed_overs)
        balls = int(balls)
        if balls >= 6:
            completed_overs += balls // 6
            balls = balls % 6
        return completed_overs + (balls / 6.0)
    return float(overs)


def get_balls_left(overs: float) -> int:
    """
    Calculate balls left in a standard 20-over IPL innings.
    
    Args:
        overs: Overs bowled (e.g., 12.4)
        
    Returns:
        The remaining balls (integer).
    """
    fractional_overs = convert_overs_to_fractional(overs)
    balls_bowled = int(round(fractional_overs * 6))
    return max(0, 120 - balls_bowled)


# -----------------------------------------------------------------------------
# Data & Serialization Utilities
# -----------------------------------------------------------------------------
def load_csv_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load a CSV file into a pandas DataFrame with error handling.
    
    Args:
        file_path: Absolute or relative path to the CSV file.
        
    Returns:
        DataFrame if successful, None otherwise.
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"CSV File not found at path: {file_path}")
            return None
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded CSV data from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV data from {file_path}: {e}", exc_info=True)
        return None


def save_serialized_model(model: Any, file_path: str, use_joblib: bool=True) -> bool:
    """
    Serialize and save a trained machine learning model or preprocessor.
    
    Args:
        model: Trained model or scaler object.
        file_path: Path where the model should be saved.
        use_joblib: Use joblib (recommended for NumPy arrays) instead of pickle.
        
    Returns:
        True if save was successful, False otherwise.
    """
    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        if use_joblib:
            joblib.dump(model, file_path)
        else:
            with open(file_path, "wb") as f:
                pickle.dump(model, f)
                
        logger.info(f"Model successfully saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to serialize and save model to {file_path}: {e}", exc_info=True)
        return False


def load_serialized_model(file_path: str, use_joblib: bool=True) -> Optional[Any]:
    """
    Load a serialized machine learning model or preprocessor.
    
    Args:
        file_path: Path to the serialized model file.
        use_joblib: Load using joblib instead of pickle.
        
    Returns:
        The loaded model/object if successful, None otherwise.
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"Serialized model file does not exist: {file_path}")
            return None
            
        if use_joblib:
            model = joblib.load(file_path)
        else:
            with open(file_path, "rb") as f:
                model = pickle.load(f)
                
        logger.info(f"Model successfully loaded from {file_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading serialized model from {file_path}: {e}", exc_info=True)
        return None


# -----------------------------------------------------------------------------
# Input Validation & State Sanitization
# -----------------------------------------------------------------------------
def validate_match_state(
    batting_team: str,
    bowling_team: str,
    target: int,
    current_score: int,
    wickets: int,
    overs: float,
    allowed_teams: Optional[list]=None
) -> Tuple[bool, str]:
    """
    Validates current match inputs to ensure they match logical cricket bounds
    and can safely be passed into predicting models.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if allowed_teams:
        if batting_team not in allowed_teams:
            return False, f"Invalid batting team: {batting_team}"
        if bowling_team not in allowed_teams:
            return False, f"Invalid bowling team: {bowling_team}"

    if batting_team == bowling_team:
        return False, "Batting team and Bowling team cannot be identical."

    if target < 1:
        return False, "Target score must be at least 1."

    if current_score < 0:
        return False, "Current score cannot be negative."

    if current_score >= target:
        return False, "Chasing team has already reached or exceeded the target."

    if wickets < 0 or wickets > 10:
        return False, "Wickets down must be between 0 and 10."

    if wickets == 10:
        return False, "Innings completed. Chasing team is all-out."

    # Validate overs
    if overs < 0.0 or overs > 20.0:
        return False, "Overs must be between 0.0 and 20.0."

    # Validate fractional ball integrity
    overs_str = f"{overs:.1f}"
    if "." in overs_str:
        balls = int(overs_str.split(".")[1])
        if balls > 5:
            return False, f"Overs ball count cannot exceed 5. Got .{balls} (cricket uses base-6)."

    return True, "Match state is valid."
