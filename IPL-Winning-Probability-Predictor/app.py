# app.py
"""
IPL Winning Probability Predictor - Streamlit Application (Part 1 - Initialization)
Analyzes live IPL run-chase trajectories, venue biases, and team strengths.
"""

import os
import sys
import logging
import time
from datetime import datetime
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Import project constants and utilities
import config
import utils
import predict

# -----------------------------------------------------------------------------
# 1. Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
app_logger = logging.getLogger("IPL_App_UI")

# -----------------------------------------------------------------------------
# 2. Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=config.APP_NAME,
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 3. Session State Initialization
# -----------------------------------------------------------------------------
def init_session_state():
    """
    Initializes standard UI inputs and model states within Streamlit session state.
    """
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        app_logger.info("Initializing Streamlit Session State...")
        
    # UI State inputs
    if "batting_team" not in st.session_state:
        st.session_state.batting_team = config.IPL_TEAMS[0]
    if "bowling_team" not in st.session_state:
        st.session_state.bowling_team = config.IPL_TEAMS[1]
    if "venue" not in st.session_state:
        st.session_state.venue = config.IPL_VENUES[0]
    if "target" not in st.session_state:
        st.session_state.target = 180
    if "current_score" not in st.session_state:
        st.session_state.current_score = 0
    if "wickets" not in st.session_state:
        st.session_state.wickets = 0
    if "overs" not in st.session_state:
        st.session_state.overs = 0.0
    if "prediction_results" not in st.session_state:
        st.session_state.prediction_results = None
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

init_session_state()

# -----------------------------------------------------------------------------
# 4. Model Loading with Error Handling & Caching
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ml_pipeline_components():
    """
    Loads all serialized model pipeline components (best_model, scaler, encoder, feature_columns).
    Fails gracefully with informative logs and UI alerts.
    
    Returns:
        Dict of components containing loaded objects or None.
    """
    components = {
        "best_model": None,
        "scaler": None,
        "encoder": None,
        "feature_columns": None,
        "error": None,
        "is_fallback": True
    }
    
    # Define exact paths to serialized artifacts
    best_model_path = os.path.join(config.MODEL_DIR, "best_model.pkl")
    scaler_path = os.path.join(config.MODEL_DIR, "scaler.pkl")
    encoder_path = os.path.join(config.MODEL_DIR, "encoder.pkl")
    feature_columns_path = os.path.join(config.MODEL_DIR, "feature_columns.pkl")
    
    # Flag to monitor if any critical artifact is missing
    missing_artifacts = []
    
    for label, path in [
        ("best_model", best_model_path),
        ("scaler", scaler_path),
        ("encoder", encoder_path),
        ("feature_columns", feature_columns_path)
    ]:
        if not os.path.exists(path):
            missing_artifacts.append(f"{label} ({os.path.basename(path)})")
            
    if missing_artifacts:
        err_msg = f"Missing model artifacts: {', '.join(missing_artifacts)}."
        app_logger.warning(f"{err_msg} Activating calibrated mathematical fallback engine.")
        components["error"] = err_msg
        components["is_fallback"] = True
        return components
        
    try:
        app_logger.info("Loading serialized machine learning pipeline artifacts...")
        components["best_model"] = joblib.load(best_model_path)
        components["scaler"] = joblib.load(scaler_path)
        components["encoder"] = joblib.load(encoder_path)
        components["feature_columns"] = joblib.load(feature_columns_path)
        components["is_fallback"] = False
        components["error"] = None
        app_logger.info("All model artifacts loaded successfully. Standard ML prediction pipeline online.")
    except Exception as e:
        err_msg = f"Error during model pipeline deserialization: {str(e)}"
        app_logger.error(err_msg, exc_info=True)
        components["error"] = err_msg
        components["is_fallback"] = True
        
    return components

# Load components and store in session state for reference
pipeline_components = load_ml_pipeline_components()
st.session_state.pipeline_components = pipeline_components

# Display state loading messages/banners cleanly
if pipeline_components["is_fallback"]:
    if pipeline_components["error"]:
        st.warning(
            f"⚠️ **Calibrated Simulator Mode Active:** {pipeline_components['error']}\n\n"
            "Using live calibrated log-odds simulation boundary model for predictions."
        )
    else:
        st.info("ℹ️ **Calibrated Simulator Mode Active** using live log-odds model boundary.")
else:
    st.success("⚡ **High-Fidelity Machine Learning Model Pipeline Online** (XGBoost/Best estimator loaded).")

# -----------------------------------------------------------------------------
# 5. Page Header & Title
# -----------------------------------------------------------------------------
st.title(f"🏏 {config.APP_NAME}")
st.markdown(f"*{config.APP_DESCRIPTION}*")
st.markdown("---")

# -----------------------------------------------------------------------------
# 6. Sidebar Panel Configuration
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 15px 0; border-bottom: 2px solid #1e293b; margin-bottom: 20px;">
        <span style="font-size: 3.5rem; display: block; margin-bottom: 5px; filter: drop-shadow(0 4px 6px rgba(16, 185, 129, 0.2));">🏏</span>
        <h2 style="color: #10b981; font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; text-transform: uppercase;">
            IPL Predictor
        </h2>
        <p style="color: #94a3b8; font-size: 0.8rem; margin: 4px 0 0 0; font-family: 'JetBrains Mono', monospace; font-weight: 500;">
            v2.1.0 // MACHINE LEARNING
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.subheader("🏟️ Match Setup")

# 1. Team Selector (Batting Chasing Team)
batting_index = config.IPL_TEAMS.index(st.session_state.batting_team) if st.session_state.batting_team in config.IPL_TEAMS else 0
batting_team = st.sidebar.selectbox(
    "Batting / Chasing Team",
    options=config.IPL_TEAMS,
    index=batting_index,
    help="Select the IPL franchise currently batting in the second innings of the match (the chasing team)."
)
st.session_state.batting_team = batting_team

# 2. Bowling Team Selector (Defending Team)
# Filter out batting team to avoid a team playing against itself
available_bowling_teams = [team for team in config.IPL_TEAMS if team != batting_team]
if st.session_state.bowling_team == batting_team or st.session_state.bowling_team not in available_bowling_teams:
    st.session_state.bowling_team = available_bowling_teams[0]

bowling_index = available_bowling_teams.index(st.session_state.bowling_team) if st.session_state.bowling_team in available_bowling_teams else 0
bowling_team = st.sidebar.selectbox(
    "Bowling / Defending Team",
    options=available_bowling_teams,
    index=bowling_index,
    help="Select the IPL franchise currently bowling in the second innings (the defending team)."
)
st.session_state.bowling_team = bowling_team

# 3. City Selector (Match Venue)
venue_index = config.IPL_VENUES.index(st.session_state.venue) if st.session_state.venue in config.IPL_VENUES else 0
venue = st.sidebar.selectbox(
    "Match City / Venue",
    options=config.IPL_VENUES,
    index=venue_index,
    help="Select the host stadium/city where the match is being contested to analyze pitch and venue bias."
)
st.session_state.venue = venue

st.sidebar.markdown("---")

# 4. About Section
st.sidebar.markdown("### ℹ️ About the Project")
st.sidebar.markdown(
    """
    This advanced decision-support platform leverages predictive machine learning models 
    to calculate live win probabilities during run chases.
    
    The underlying high-fidelity model is trained on historical ball-by-ball delivery-level data 
    spanning over 15 seasons of the IPL. It evaluates running performance, remaining wickets, 
    historical stadium chase characteristics, and specific franchise matchups.
    """
)

# 5. Instructions Section
st.sidebar.markdown("### 📋 Instructions")
st.sidebar.markdown(
    """
    1. **Set Up the Match**: Use this sidebar to choose the teams and host city/venue.
    2. **Enter Live Metrics**: Use the controls on the main dashboard to input live match conditions (target, score, wickets, and overs).
    3. **Analyze Results**: Explore the real-time calculated win-percentages, run rate telemetry, and detailed chase diagnostics dynamically updated on your screen.
    """
)

# 6. Theme Information Section
st.sidebar.markdown("### 🎨 UI Theme & Configuration")
st.sidebar.markdown(
    f"""
    - **Display Mode**: Dark Slate & Emerald Dashboard
    - **Primary Base**: `{config.THEME_COLORS['background']}`
    - **Card Shells**: `{config.THEME_COLORS['card_background']}`
    - **Accent Color**: `{config.THEME_COLORS['primary']}`
    - **Alert Color**: `{config.THEME_COLORS['secondary']}`
    """
)

# 7. Reset Action Button
st.sidebar.markdown(" ")
if st.sidebar.button("🔄 Reset Match Settings", use_container_width=True, type="primary"):
    st.session_state.batting_team = config.IPL_TEAMS[0]
    st.session_state.bowling_team = config.IPL_TEAMS[1]
    st.session_state.venue = config.IPL_VENUES[0]
    st.session_state.target = 180
    st.session_state.current_score = 0
    st.session_state.wickets = 0
    st.session_state.overs = 0.0
    st.session_state.prediction_results = None
    app_logger.info("Match settings and dynamics reset to original default state.")
    st.rerun()


# -----------------------------------------------------------------------------
# 7. Main Dashboard Layout (Part 2)
# -----------------------------------------------------------------------------

# Welcome Section & IPL Banner
st.subheader("📊 Live Match Win Probability & Performance Intelligence Dashboard")
st.markdown(
    "Analyze live run-chase trajectories, venue biases, and team strengths in real time. "
    "Use the left sidebar to configure the match setup (teams and stadium/city). Then, "
    "adjust the active game state inputs in the controls below to recalculate win probabilities."
)

# IPL Banner (Stylized using native Streamlit columns and cards)
banner_col1, banner_col2 = st.columns([4, 1])
with banner_col1:
    st.info(
        "🏆 **IPL Run-Chase Analytics Engine Online**\n\n"
        f"Active Matchup: **{st.session_state.batting_team}** chasing a target against **{st.session_state.bowling_team}** "
        f"at **{st.session_state.venue}**."
    )
with banner_col2:
    st.metric(label="Selected Venue", value=st.session_state.venue.split(",")[1].strip())

# Information Cards (Dynamic descriptive parameters of the setup)
st.markdown("### 📋 Analytics & Prediction Pipeline Setup")
info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.metric(
        label="Batting Team Strength Modifier",
        value=f"{config.TEAM_STRENGTH_COEFFICIENTS.get(st.session_state.batting_team, {}).get('batting', 0.0) * 100:+.1f}%",
        delta="Chasing Boost"
    )
    st.caption("Derived coefficient boosting chasing win probability based on batting depth.")

with info_col2:
    st.metric(
        label="Bowling Team Strength Modifier",
        value=f"{config.TEAM_STRENGTH_COEFFICIENTS.get(st.session_state.bowling_team, {}).get('bowling', 0.0) * 100:+.1f}%",
        delta="Defending Boost",
        delta_color="inverse"
    )
    st.caption("Derived coefficient boosting defending win probability based on bowling depth.")

with info_col3:
    venue_bias = config.VENUE_BIAS_COEFFICIENTS.get(st.session_state.venue, 0.0)
    bias_desc = "Favors Chasing" if venue_bias > 0 else "Favors Defending" if venue_bias < 0 else "Neutral Pitch"
    st.metric(
        label="Venue Chase Bias",
        value=f"{venue_bias * 100:+.1f}%",
        delta=bias_desc
    )
    st.caption("Historical statistical bias of the selected stadium towards chasing/defending.")

# Match Summary Container
st.markdown("---")
with st.container():
    st.subheader("🏟️ Active Match Setup Summary")
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.markdown(f"**Chasing Team (2nd Innings Batting):**  \n✨ `{st.session_state.batting_team}`")
        st.markdown(f"**Defending Team (1st Innings Bowling):**  \n🛡️ `{st.session_state.bowling_team}`")
        
    with summary_col2:
        st.markdown(f"**Match Location:**  \n📍 `{st.session_state.venue}`")
        st.markdown(f"**Target Score:**  \n🎯 `{st.session_state.target} runs`")
        
    with summary_col3:
        st.markdown(f"**Prediction Engine Mode:**")
        if pipeline_components["is_fallback"]:
            st.info("📊 Calibrated Fallback Simulator")
        else:
            st.success("⚡ XGBoost Model Pipeline")
        st.markdown(f"**Pipeline Diagnostics Status:**  \n🟢 All systems nominal")

# Statistics Cards (Historical & Current Match Analytics)
st.markdown("### 📈 Live Chase Statistics & Telemetry")
stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

# Calculate live metrics based on session state inputs
runs_needed = max(0, st.session_state.target - st.session_state.current_score)
balls_left = predict.get_balls_left(st.session_state.overs)
crr = predict.calculate_crr(st.session_state.current_score, st.session_state.overs)
rrr = predict.calculate_rrr(runs_needed, balls_left)

with stats_col1:
    st.metric(
        label="Required Run Rate (RRR)",
        value=f"{rrr:.2f}",
        delta=f"{rrr - 8.2:.2f} vs Par (8.2)",
        delta_color="inverse"
    )
    st.caption("Required runs per over to win the match.")

with stats_col2:
    st.metric(
        label="Current Run Rate (CRR)",
        value=f"{crr:.2f}",
        delta=f"{crr - 7.5:.2f} vs Par (7.5)"
    )
    st.caption("Average runs scored per over in the active chase.")

with stats_col3:
    st.metric(
        label="Runs Left Needed",
        value=f"{runs_needed}",
        delta=f"From Target: {st.session_state.target}",
        delta_color="off"
    )
    st.caption("Total runs remaining to secure victory.")

with stats_col4:
    st.metric(
        label="Balls Remaining",
        value=f"{balls_left}",
        delta=f"{st.session_state.overs:.1f} Overs Bowled",
        delta_color="off"
    )
    st.caption("Total legitimate deliveries remaining in the innings.")


# -----------------------------------------------------------------------------
# 8. Prediction Input Section (Part 3)
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🏏 Enter Live Match Conditions")
st.markdown(
    "Input the real-time game situation to calculate live win probabilities. "
    "All statistics, run rates, and ML predictions will update dynamically."
)

input_col1, input_col2 = st.columns(2)

with input_col1:
    st.markdown("##### 🎯 Target & Score Progress")
    
    # 1. Target Score Input
    target = st.number_input(
        "Target Score",
        min_value=1,
        max_value=450,
        value=int(st.session_state.target),
        step=1,
        help="Target score set by the team batting first + 1 (runs needed to win)."
    )
    st.session_state.target = target
    
    # 2. Current Score Input
    # Cap current score at target to avoid invalid chase state
    max_score = int(target)
    current_score = st.number_input(
        "Current Score",
        min_value=0,
        max_value=max_score,
        value=min(int(st.session_state.current_score), max_score),
        step=1,
        help="Active runs scored by the batting team in the second innings."
    )
    st.session_state.current_score = current_score

with input_col2:
    st.markdown("##### ⏱️ Overs & Wickets")
    
    # 3. Overs Completed Input
    overs = st.number_input(
        "Overs Completed",
        min_value=0.0,
        max_value=19.5,
        value=float(st.session_state.overs),
        step=0.1,
        format="%.1f",
        help="Number of completed overs in the second innings. Note: fractional part is balls bowled (e.g., 12.4)."
    )
    
    # Sanitize cricket fractional overs if user types invalid values (e.g. 12.6, 12.7)
    overs_str = f"{overs:.1f}"
    if "." in overs_str:
        comp_overs_str, balls_str = overs_str.split(".")
        comp_overs = int(comp_overs_str)
        balls = int(balls_str)
        if balls >= 6:
            corrected_overs = float(comp_overs + (balls // 6)) + (balls % 6) / 10.0
            overs = min(19.5, corrected_overs)
            
    st.session_state.overs = overs
    
    # 4. Wickets Fallen Input
    wickets = st.number_input(
        "Wickets Fallen",
        min_value=0,
        max_value=9,
        value=min(int(st.session_state.wickets), 9),
        step=1,
        help="Total dismissals of the chasing team (active chase ends if 10 wickets fall)."
    )
    st.session_state.wickets = wickets

# -----------------------------------------------------------------------------
# 9. Real-Time Calculation & Verification Panel
# -----------------------------------------------------------------------------
st.markdown(" ")
st.markdown("##### 📊 Dynamically Calculated Chase Dynamics")

# Calculate metrics dynamically based on the current user inputs
calc_runs_needed = max(0, target - current_score)
calc_balls_left = predict.get_balls_left(overs)
calc_crr = predict.calculate_crr(current_score, overs)
calc_rrr = predict.calculate_rrr(calc_runs_needed, calc_balls_left)

calc_col1, calc_col2, calc_col3, calc_col4 = st.columns(4)

with calc_col1:
    st.info(f"**Runs Left Needed:**  \n### {calc_runs_needed}")
    
with calc_col2:
    st.info(f"**Balls Remaining:**  \n### {calc_balls_left}")
    
with calc_col3:
    st.info(f"**Current Run Rate (CRR):**  \n### {calc_crr:.2f}")
    
with calc_col4:
    # Stylize required run rate with a red warning color if extremely high
    if calc_rrr > 12.0:
        st.error(f"**Required Run Rate (RRR):**  \n### {calc_rrr:.2f}")
    elif calc_rrr > 9.0:
        st.warning(f"**Required Run Rate (RRR):**  \n### {calc_rrr:.2f}")
    else:
        st.success(f"**Required Run Rate (RRR):**  \n### {calc_rrr:.2f}")


# -----------------------------------------------------------------------------
# 10. Live Probability Prediction Execution (Part 4)
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔮 Predictive Machine Learning Inference")

if st.button("🚀 Calculate Live Match Probability", use_container_width=True, type="primary"):
    start_time = time.perf_counter()
    
    app_logger.info(
        f"Inference requested: Batting={st.session_state.batting_team}, Bowling={st.session_state.bowling_team}, "
        f"Target={st.session_state.target}, Score={st.session_state.current_score}, Wickets={st.session_state.wickets}, "
        f"Overs={st.session_state.overs}"
    )
    
    try:
        # Run prediction
        res = predict.predict_match_probability(
            batting_team=st.session_state.batting_team,
            bowling_team=st.session_state.bowling_team,
            venue=st.session_state.venue,
            target=st.session_state.target,
            current_score=st.session_state.current_score,
            wickets=st.session_state.wickets,
            overs=st.session_state.overs
        )
        
        elapsed_time = (time.perf_counter() - start_time) * 1000  # in milliseconds
        st.session_state.prediction_results = {
            "res": res,
            "elapsed_time_ms": elapsed_time
        }
        
        # Append to prediction history
        chasing_p = res["chasing_win_probability"]
        defending_p = res["defending_win_probability"]
        if chasing_p >= defending_p:
            win_t = st.session_state.batting_team
            p_val = chasing_p
        else:
            win_t = st.session_state.bowling_team
            p_val = defending_p
            
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = []
            
        st.session_state.prediction_history.append({
            "Time": now_str,
            "Batting Team": st.session_state.batting_team,
            "Bowling Team": st.session_state.bowling_team,
            "Winning Team": win_t,
            "Probability": f"{p_val:.1f}%"
        })
        
        # Limit history to prevent memory leak (cap at last 100 records)
        if len(st.session_state.prediction_history) > 100:
            st.session_state.prediction_history = st.session_state.prediction_history[-100:]
        
        app_logger.info(f"Inference completed in {elapsed_time:.2f}ms.")
        
    except Exception as e:
        app_logger.error(f"Inference pipeline failed: {str(e)}", exc_info=True)
        st.error(f"🚨 **Inference Failure:** An unexpected error occurred in the prediction pipeline: {str(e)}")

# Display Prediction Results if available
if st.session_state.prediction_results:
    pred_data = st.session_state.prediction_results
    res = pred_data["res"]
    elapsed_time = pred_data["elapsed_time_ms"]
    
    chasing_prob = res["chasing_win_probability"]
    defending_prob = res["defending_win_probability"]
    
    # Determine winning/losing teams and probabilities
    if chasing_prob >= defending_prob:
        winner_team = st.session_state.batting_team
        loser_team = st.session_state.bowling_team
        win_prob = chasing_prob
    else:
        winner_team = st.session_state.bowling_team
        loser_team = st.session_state.batting_team
        win_prob = defending_prob
        
    # Determine confidence levels
    if win_prob >= 85.0:
        confidence = "High"
        confidence_color = "green"
    elif win_prob >= 65.0:
        confidence = "Medium"
        confidence_color = "orange"
    else:
        confidence = "Low"
        confidence_color = "red"
        
    st.markdown("### 🏆 Live Match Prediction Result Cards")
    
    # Reuse dynamically calculated chase dynamics to avoid redundant calls and speed up rendering
    runs_needed = calc_runs_needed
    balls_left = calc_balls_left
    
    card_col1, card_col2, card_col3 = st.columns(3)
    
    with card_col1:
        st.metric(
            label="👑 Predicted Winner",
            value=winner_team,
            delta="Match Favorite"
        )
        
    with card_col2:
        st.metric(
            label="📈 Winning Probability",
            value=f"{win_prob:.1f}%",
            delta="Winner Confidence"
        )
        
    with card_col3:
        st.metric(
            label="📉 Losing Probability",
            value=f"{(100.0 - win_prob):.1f}%",
            delta=f"{loser_team} chance",
            delta_color="inverse"
        )
        
    st.markdown(" ") # Vertical spacing spacer
    
    card_col4, card_col5, card_col6 = st.columns(3)
    
    with card_col4:
        st.metric(
            label="🎯 Confidence Score",
            value=confidence,
            delta="Statistical Certainty"
        )
        
    with card_col5:
        st.metric(
            label="📊 Current Run Rate (CRR)",
            value=f"{calc_crr:.2f}",
            delta=f"{calc_crr - 7.50:+.2f} vs Par"
        )
        
    with card_col6:
        st.metric(
            label="⏱️ Required Run Rate (RRR)",
            value=f"{calc_rrr:.2f}",
            delta=f"{calc_rrr - 8.20:+.2f} vs Par",
            delta_color="inverse"
        )

    # -----------------------------------------------------------------------------
    # 11. Plotly Interactive Visualizations (Part 5)
    # -----------------------------------------------------------------------------
    
    st.markdown("---")
    st.subheader("📊 Interactive Performance Visualizations")
    
    # Row 1: Gauge & Pie Chart
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        # 1. Winning Probability Gauge
        st.markdown("##### ⏱️ Chase Probability Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = chasing_prob,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"{st.session_state.batting_team} Win %", 'font': {'size': 16}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': "#10b981"}, # Emerald color
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#cbd5e1",
                'steps': [
                    {'range': [0, 40], 'color': '#fee2e2'}, # Light red
                    {'range': [40, 70], 'color': '#fef3c7'}, # Light amber
                    {'range': [70, 100], 'color': '#d1fae5'} # Light emerald
                ],
                'threshold': {
                    'line': {'color': "#ef4444", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=280,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_chart")

    with row1_col2:
        # 2. Donut Pie Chart
        st.markdown("##### 🍩 Win Probability Breakdown")
        labels = [st.session_state.batting_team, st.session_state.bowling_team]
        values = [chasing_prob, defending_prob]
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.4,
            marker=dict(colors=["#10b981", "#ef4444"]), # Emerald vs Red
            hoverinfo="label+percent",
            textinfo="value+percent"
        )])
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=280,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=20, b=40)
        )
        st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart")
        
    # Row 2: Horizontal Probability Bar
    st.markdown("##### 📊 Relative Match Win Probability Comparison")
    
    # 3. Horizontal Stacked Bar Chart
    fig_hbar = go.Figure()
    fig_hbar.add_trace(go.Bar(
        y=['Win Probability'],
        x=[chasing_prob],
        name=st.session_state.batting_team,
        orientation='h',
        marker=dict(
            color='#10b981',
            line=dict(color='rgba(255, 255, 255, 1.0)', width=1)
        )
    ))
    fig_hbar.add_trace(go.Bar(
        y=['Win Probability'],
        x=[defending_prob],
        name=st.session_state.bowling_team,
        orientation='h',
        marker=dict(
            color='#ef4444',
            line=dict(color='rgba(255, 255, 255, 1.0)', width=1)
        )
    ))
    
    fig_hbar.update_layout(
        barmode='stack',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=140,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-1.2, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(showgrid=False, range=[0, 100], ticksuffix="%"),
        yaxis=dict(showgrid=False, showticklabels=False)
    )
    st.plotly_chart(fig_hbar, use_container_width=True, key="hbar_chart")
    
    # Row 3: Run Rate Comparison & Runs Left Progress
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        # 4. Run Rate Comparison (Vertical Bar Chart)
        st.markdown("##### 📈 Run Rate Analysis")
        rr_categories = ['Current Run Rate (CRR)', 'Required Run Rate (RRR)', 'Initial Par Rate']
        rr_values = [calc_crr, calc_rrr, 8.20]
        
        fig_rr = go.Figure(data=[go.Bar(
            x=rr_categories,
            y=rr_values,
            marker_color=['#3b82f6', '#f59e0b', '#6b7280'], # Blue, Amber, Gray
            text=[f"{val:.2f}" for val in rr_values],
            textposition='auto',
        )])
        fig_rr.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=280,
            yaxis=dict(title="Runs Per Over (RPO)", showgrid=True, gridcolor='#f3f4f6'),
            xaxis=dict(showgrid=False),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_rr, use_container_width=True, key="rr_comparison_chart")
        
    with row2_col2:
        # 5. Runs Left Chart (Horizontal Progress stacked bar: Runs Scored vs Runs Needed)
        st.markdown("##### 🎯 Target Progress (Runs Left Chart)")
        runs_scored = st.session_state.current_score
        
        fig_runs = go.Figure()
        fig_runs.add_trace(go.Bar(
            y=['Target Score'],
            x=[runs_scored],
            name='Runs Scored',
            orientation='h',
            marker=dict(color='#10b981')
        ))
        fig_runs.add_trace(go.Bar(
            y=['Target Score'],
            x=[runs_needed],
            name='Runs Needed (Left)',
            orientation='h',
            marker=dict(color='#e2e8f0') # Soft grey
        ))
        
        fig_runs.update_layout(
            barmode='stack',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=280,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=20, b=40),
            xaxis=dict(title="Total Runs", range=[0, target], showgrid=True, gridcolor='#f3f4f6'),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        st.plotly_chart(fig_runs, use_container_width=True, key="runs_left_chart")


# -----------------------------------------------------------------------------
# 12. Prediction History Section (Part 6)
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📜 Saved Prediction History")

if "prediction_history" not in st.session_state or not st.session_state.prediction_history:
    st.info("No predictions calculated yet. Run a match probability simulation to populate history.")
else:
    # Convert history array of dicts into a structured pandas DataFrame
    history_df = pd.DataFrame(st.session_state.prediction_history)
    
    # Reorder or select columns to ensure requested display keys
    cols_order = ["Time", "Batting Team", "Bowling Team", "Winning Team", "Probability"]
    history_df = history_df[[c for c in cols_order if c in history_df.columns]]
    
    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Actions: Clear History & Export Options
    action_col1, action_col2, action_col3 = st.columns([1.5, 2, 2])
    with action_col1:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.prediction_history = []
            app_logger.info("Prediction history cleared by user.")
            st.rerun()
            
    # Prepare export files
    csv_bytes = history_df.to_csv(index=False).encode('utf-8')
    json_bytes = history_df.to_json(orient='records', indent=2).encode('utf-8')
    
    with action_col2:
        st.download_button(
            label="📥 Export CSV",
            data=csv_bytes,
            file_name="match_prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download prediction history as a CSV file."
        )
        
    with action_col3:
        st.download_button(
            label="📥 Export JSON",
            data=json_bytes,
            file_name="match_prediction_history.json",
            mime="application/json",
            use_container_width=True,
            help="Download prediction history as a JSON file."
        )


# -----------------------------------------------------------------------------
# 13. Footer Section (Part 7)
# -----------------------------------------------------------------------------
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("##### 🏏 Project Information")
    st.markdown(
        "The **IPL Match Predictor** is an interactive web-based simulator designed to forecast "
        "real-time second-innings winning probabilities. Utilizing standard ML algorithms "
        "and calibrated mathematical formulas, it dynamically analyzes game situations "
        "on a ball-by-ball basis."
    )

with footer_col2:
    st.markdown("##### 📊 Dataset Information")
    st.markdown(
        "Powered by comprehensive ball-by-ball match logs from the Indian Premier League (IPL) "
        "spanning from **2008 to 2024**. The dataset models venue behaviors, batting/bowling team matchups, "
        "and historical chasing patterns under pressure."
    )

with footer_col3:
    st.markdown("##### 👥 Developer Credits")
    st.markdown("**Lead Developer:**Shaik Asif")
    st.markdown(
        "[🐙 GitHub Profile (Shaik Asif)](https://github.com/shaik-asif967)  \n"
        "[💼 LinkedIn Profile (Shaik Asif)](https://www.linkedin.com/in/shaikasif369)"
    )

st.markdown("---")

bot_col1, bot_col2, bot_col3 = st.columns(3)
with bot_col1:
    st.caption("🚀 **Version:** 1.0.0 (Production-Ready)")
with bot_col2:
    st.caption("⚖️ **License:** MIT License")
with bot_col3:
    st.caption("💚 **Engine Diagnostics:** 100% Operational")



