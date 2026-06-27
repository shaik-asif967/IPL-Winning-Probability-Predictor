# config.py
"""
Configuration settings, constants, and theme parameters for the
IPL Winning Probability Predictor model pipeline and Streamlit UI.
"""

import os

# -----------------------------------------------------------------------------
# App Constants & Project Meta
# -----------------------------------------------------------------------------
APP_NAME = "IPL Winning Probability Predictor"
APP_DESCRIPTION = (
    "An advanced machine learning decision-support system analyzing "
    "live IPL run-chase trajectories, venue biases, and team strengths."
)
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# Directory & File Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Ensure critical directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Serialized Model & Scaler Paths
MODEL_PATH = os.path.join(MODEL_DIR, "ipl_xgboost_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "standard_scaler.pkl")
DATASET_PATH = os.path.join(DATA_DIR, "ipl_matches_cleaned.csv")

# -----------------------------------------------------------------------------
# IPL Sports Metadata
# -----------------------------------------------------------------------------
IPL_TEAMS = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Rajasthan Royals",
    "Gujarat Titans",
    "Lucknow Super Giants",
    "Delhi Capitals",
    "Punjab Kings",
    "Sunrisers Hyderabad"
]

IPL_VENUES = [
    "Wankhede Stadium, Mumbai",
    "M. Chinnaswamy Stadium, Bengaluru",
    "MA Chidambaram Stadium, Chennai",
    "Eden Gardens, Kolkata",
    "Narendra Modi Stadium, Ahmedabad",
    "Arun Jaitley Stadium, Delhi",
    "Rajiv Gandhi Intl. Stadium, Hyderabad",
    "Sawai Mansingh Stadium, Jaipur",
    "Punjab Cricket Association IS Bindra Stadium, Mohali",
    "BRSABV Ekana Cricket Stadium, Lucknow"
]

# -----------------------------------------------------------------------------
# Simulating XGBoost Feature Parameters / Learned Coefficients
# -----------------------------------------------------------------------------
TEAM_STRENGTH_COEFFICIENTS = {
    "Chennai Super Kings": {"batting": 0.15, "bowling":-0.1},
    "Mumbai Indians": {"batting": 0.1, "bowling":-0.05},
    "Kolkata Knight Riders": {"batting": 0.12, "bowling":-0.08},
    "Royal Challengers Bengaluru": {"batting": 0.05, "bowling": 0.12},
    "Rajasthan Royals": {"batting": 0.08, "bowling":-0.05},
    "Gujarat Titans": {"batting": 0.06, "bowling":-0.04},
    "Lucknow Super Giants": {"batting": 0.04, "bowling":-0.02},
    "Delhi Capitals": {"batting": 0.02, "bowling": 0.05},
    "Punjab Kings": {"batting":-0.02, "bowling": 0.06},
    "Sunrisers Hyderabad": {"batting": 0.20, "bowling":-0.05},
}

VENUE_BIAS_COEFFICIENTS = {
    "Wankhede Stadium, Mumbai": 0.22,  # Promotes chasing
    "M. Chinnaswamy Stadium, Bengaluru": 0.28,  # High-scoring chasing venue
    "MA Chidambaram Stadium, Chennai":-0.25,  # Spins, dry track; favors defending
    "Eden Gardens, Kolkata": 0.15,  # Chasing friendly
    "Narendra Modi Stadium, Ahmedabad": 0.08,  # Slightly favors chasing
    "Arun Jaitley Stadium, Delhi": 0.12,  # Small boundary size; favors chasing
    "Rajiv Gandhi Intl. Stadium, Hyderabad": 0.05,  # Fairly balanced
    "Sawai Mansingh Stadium, Jaipur":-0.08,  # Slow outfield, defense favors
    "Punjab Cricket Association IS Bindra Stadium, Mohali": 0.14,  # Fast outfield; chasing
    "BRSABV Ekana Cricket Stadium, Lucknow":-0.32,  # Low-scoring, sticky track; favors defense
}

# -----------------------------------------------------------------------------
# UI Color Schemes & Theme Configuration
# -----------------------------------------------------------------------------
THEME_COLORS = {
    "background": "#0b0f19",
    "card_background": "#0f172a",
    "primary": "#10b981",  # Emerald
    "secondary": "#ef4444",  # Red / Rose
    "text": "#f1f5f9",
    "accent": "#f59e0b"  # Amber
}

# Brand Colors corresponding to individual IPL teams
TEAM_BRAND_COLORS = {
    "Chennai Super Kings": {"primary": "#F7C305", "secondary": "#0081E4"},
    "Mumbai Indians": {"primary": "#004BA0", "secondary": "#D1AB3E"},
    "Royal Challengers Bengaluru": {"primary": "#EC1C24", "secondary": "#000000"},
    "Kolkata Knight Riders": {"primary": "#3A225D", "secondary": "#ECC542"},
    "Rajasthan Royals": {"primary": "#EA1B76", "secondary": "#254AA5"},
    "Gujarat Titans": {"primary": "#1B2F55", "secondary": "#D1AB3E"},
    "Lucknow Super Giants": {"primary": "#005780", "secondary": "#E2B300"},
    "Delhi Capitals": {"primary": "#00008F", "secondary": "#EF3A38"},
    "Punjab Kings": {"primary": "#DD1F26", "secondary": "#D7A22A"},
    "Sunrisers Hyderabad": {"primary": "#FF8225", "secondary": "#000000"}
}
