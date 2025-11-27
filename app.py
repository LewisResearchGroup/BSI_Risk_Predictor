import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Baseline event rate (for contextualizing predictions)
BASELINE_RATE = 0.15

# Risk strata for human-readable grouping
RISK_BINS = [
    ("Low", 0.0, 0.10, "#cfd8dc"),           # light gray
    ("Moderate", 0.10, 0.25, "#ffcc80"), # soft amber
    ("High", 0.25, 0.40, "#ff8a65"),         # orange-red
    ("Very high", 0.40, 1.0, "#d32f2f"),     # deep red
]

# Ticks to annotate compartment borders (fractions of 1.0)
BOUNDARY_TICKS = [0.10, 0.25, 0.40]


def categorize_risk(p):
    """Return (label, color) for a probability between 0 and 1."""
    for label, lo, hi, color in RISK_BINS:
        if lo <= p < hi:
            return label, color
    # Fallback if outside [0,1]
    return "Uncategorized", "#7391f5"


# ------------------------
# Model and Scaler Loaders
# ------------------------
@st.cache_resource
def load_model():
    """Load the trained machine learning model from disk."""
    with open("model.pkl", "rb") as f:
        return pickle.load(f)
    return None


@st.cache_resource
def load_imputer():
    """Load the fitted imputer from disk."""
    with open("imputer.pkl", "rb") as f:
        return pickle.load(f)
    return None


@st.cache_data
def load_calibration_data(path: str):
    """Load calibration CSV ensuring required columns are present."""
    df = pd.read_csv(path)
    required_cols = {"Age", "Calibrated_prob", "Calibrated_prob_mean", "q30"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in calibration file: {', '.join(sorted(missing_cols))}")
    return df


model = load_model()
imputer = load_imputer()

# ------------------------
# Session state for resettable fields
# ------------------------
default_values = dict(age=79, cci=3, sofa=5, pbs=0)
for k, v in default_values.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_form():
    """Reset all user-editable fields to defaults."""
    for k, v in default_values.items():
        st.session_state[k] = v


# ------------------------
# Sidebar: Branding, About, Disclaimer
# ------------------------
with st.sidebar:
    st.image("LRG.png", width=120)
    st.markdown(
        "<div style='margin-top: -10px; font-size:1.05em; color: #666;'>"
        "Developed at <a href='https://www.lewisresearchgroup.org/' target='_blank' "
        "style='color: #7391f5; text-decoration: none;'>Lewis Research Group</a>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.header("About this Tool")
    st.write(
        "This tool provides a model-based estimate of 30-day mortality probability. "
        "It uses four inputs: Age, Charlson Comorbidity Index (CCI), Pitt Bacteremia Score (PBS), "
        "and SOFA score to generate a probability derived from a machine learning model trained "
        "on historical data. "
        "This estimate should be interpreted as a statistical output, not a clinical determination. "
        "For background on these indices: "
        "[CCI](https://www.mdcalc.com/calc/3917/charlson-comorbidity-index-cci), "
        "[PBS](https://m.medicalalgorithms.com/pitt-bacteremia-score-of-paterson-et-al), "
        "[SOFA](https://www.mdcalc.com/calc/691/sequential-organ-failure-assessment-sofa-score)."
    )

# ------------------------
# Main App Interface
# ------------------------
st.title("Model-Based Estimate of 30-Day Mortality Probability")
st.header("Enter Patient Data")

# Tooltips for each feature
cci_info = (
    "Charlson Comorbidity Index (CCI) is a widely used scoring system "
    "that predicts ten-year mortality based on the presence of comorbidity conditions. "
    "[Learn more.](https://www.mdcalc.com/calc/3917/charlson-comorbidity-index-cci)"
)
pbs_info = (
    "Pitt Bacteremia Score (PBS) is a tool used in infectious disease research to assess the severity of acute "
    "illness and predict mortality. "
    "[Learn more.](https://m.medicalalgorithms.com/pitt-bacteremia-score-of-paterson-et-al)"
)
sofa_info = (
    "SOFA (Sequential Organ Failure Assessment) score quantifies the extent of a patient's organ function or rate of "
    "failure. [Learn more.](https://www.mdcalc.com/calc/691/sequential-organ-failure-assessment-sofa-score)"
)
age_info = "Patient's age in years."

# Feature inputs (linked to session_state for reset functionality)
age = st.number_input("Age (years)", min_value=0, max_value=120, key="age", help=age_info)
cci = st.number_input("Charlson Comorbidity Index (CCI)", min_value=0, max_value=37, key="cci", help=cci_info)
pbs = st.number_input("PBS Score", min_value=0, max_value=14, key="pbs", help=pbs_info)
sofa = st.number_input("SOFA Score", min_value=0, max_value=24, key="sofa", help=sofa_info)

# Calibration dataset path (used for plotting age vs calibrated probabilities)

calibration_df = load_calibration_data("calibration_data.csv")

# --------------------------------
# Training ranges (from your dataset)
# --------------------------------
TRAINING_RANGES = {
    "age": {"min": 0, "max": 100},   # adjust if needed
    "cci": {"min": 0, "max": 17},
    "pbs": {"min": 0, "max": 14},
    "sofa": {"min": 0, "max": 24},
}

# Check if any inputs are outside training range
violations = []
if age > TRAINING_RANGES["age"]["max"]:
    violations.append(f"Age={age} (trained up to {TRAINING_RANGES['age']['max']})")
if cci > TRAINING_RANGES["cci"]["max"]:
    violations.append(f"CCI={cci} (trained up to {TRAINING_RANGES['cci']['max']})")
if pbs > TRAINING_RANGES["pbs"]["max"]:
    violations.append(f"PBS={pbs} (trained up to {TRAINING_RANGES['pbs']['max']})")
if sofa > TRAINING_RANGES["sofa"]["max"]:
    violations.append(f"SOFA={sofa} (trained up to {TRAINING_RANGES['sofa']['max']})")

if violations:
    st.warning(
        "⚠️ One or more inputs are outside the range observed in the model's training data:\n\n"
        + "\n".join([f"- {v}" for v in violations])
        + "\n\nPredictions in this region may be less reliable."
    )

# Predict and Reset buttons side-by-side
col1, col2 = st.columns([1, 1])
with col1:
    predict = st.button("Estimate Probability")
with col2:
    reset = st.button("Reset", on_click=reset_form)

# ------------------------
# Prediction & Results
# ------------------------
if predict:
    # Prepare and scale features for prediction
    X = np.array([[age, cci, pbs, sofa]])
    X = imputer.transform(X)

    # Predict mortality probability
    proba = model.predict_proba(X)[0][1]
    relative = proba / BASELINE_RATE if BASELINE_RATE > 0 else float("nan")
    risk_label, bar_color = categorize_risk(proba)


    # Calibration plot comparing dataset calibration vs current prediction
    if calibration_df is not None and not calibration_df.empty:
        plot_df = calibration_df.sort_values("Age")
        age_match = plot_df[plot_df["Age"] == age]
        if age_match.empty:
            nearest_idx = (plot_df["Age"] - age).abs().idxmin()
            age_mean = float(plot_df.loc[nearest_idx, "Calibrated_prob_mean"])
            age_q30 = float(plot_df.loc[nearest_idx, "q30"])
            matched_age = float(plot_df.loc[nearest_idx, "Age"])
            st.caption(f"No exact age match; using nearest age in data: {matched_age:.0f}.")
        else:
            age_mean = float(age_match["Calibrated_prob_mean"].mean())
            age_q30 = float(age_match["q30"].mean())
            matched_age = age

        delta_pct_points = (proba - age_mean) * 100
        direction = "higher" if delta_pct_points >= 0 else "lower"
        delta_q30 = (proba - age_q30) * 100
        direction_q30 = "higher" if delta_q30 >= 0 else "lower"

    # Display probability
    st.write("### Estimated Probability")
    st.info(f"Model-estimated 30-day mortality probability is **{proba * 100:.1f}%**, which is:\n"
            f"- {abs(delta_pct_points):.1f} percentage points {direction} than the age-matched mean ({age_mean*100:.1f}%)\n"
            f"- {abs(delta_q30):.1f} percentage points {direction_q30} than the age-matched US baseline ({age_q30*100:.1f}%)")

    # Segmented severity bar (four compartments with marker)
    marker_left = min(max(proba * 100, 0), 100)
    segments_html = "".join(
        f"<div style='flex: {hi - lo}; background: {color}; height: 28px;'></div>"
        for _, lo, hi, color in RISK_BINS
    )
    ticks_html = "".join(
        f"""
        <div style='position: absolute; left: {tick*100}%; top: 28px; transform: translateX(-50%); width: 1px; height: 10px; background: #666;'></div>
        <div style='position: absolute; left: {tick*100}%; top: 40px; transform: translateX(-50%); font-size: 11px; color: #444;'>
            {tick:.2f}
        </div>
        """
        for tick in BOUNDARY_TICKS
    )
    st.markdown(
        f"""
        <div style='width: 100%; position: relative; margin: 8px 0 72px 0;'>
          <div style='display: flex; border-radius: 12px; overflow: hidden; border: 1px solid #e0e0e0; height: 28px;'>
            {segments_html}
          </div>
          <div style='position: absolute; left: {BASELINE_RATE*100}%; top: 32px; transform: translateX(-50%); display: flex; flex-direction: column; align-items: center; gap: 4px; pointer-events: none;'>
            
          </div>
          <div style='position: absolute; left: {marker_left}%; top: -8px; transform: translateX(-50%); display: flex; flex-direction: column; align-items: center; gap: 2px;'>
            <div style='width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-bottom: 8px solid #000;'></div>
            <div style='font-size: 11px; color: #000; white-space: nowrap; background: rgba(255,255,255,0.85); padding: 0 2px; border-radius: 3px;'>{proba * 100:.1f}%</div>
          </div>
          {ticks_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Calibration plot comparing dataset calibration vs current prediction
    if calibration_df is not None and not calibration_df.empty:
        plot_df = calibration_df.sort_values("Age")
        age_match = plot_df[plot_df["Age"] == age]
        if age_match.empty:
            nearest_idx = (plot_df["Age"] - age).abs().idxmin()
            age_mean = float(plot_df.loc[nearest_idx, "Calibrated_prob_mean"])
            matched_age = float(plot_df.loc[nearest_idx, "Age"])
            st.caption(f"No exact age match; using nearest age in data: {matched_age:.0f}.")
        else:
            age_mean = float(age_match["Calibrated_prob_mean"].mean())
            matched_age = age

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.pointplot(data=plot_df, x="Age", y="Calibrated_prob", color="#bbbbbb", errorbar=None, ax=ax, linewidth=0.3)
        sns.pointplot(data=plot_df, x="Age", y="q30", color="#bbbbbb", errorbar=None, ax=ax, linewidth=0.3)

        ax.tick_params(axis='y', which='both', length=0)
        sns.despine(left=True, bottom=True)

        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xticklabels(['0', '20', '40', '60', '80', '100'])
        
        ax.scatter(
            [age],
            [proba],
            color=bar_color,
            s=90,
            zorder=4,
            label="Current patient",
            edgecolor="white",
            linewidth=0.8,
            
        )
        ax.scatter(
            [matched_age],
            [age_mean],
            color="#bbbbbb",
            s=70,
            zorder=5,
            label=f"Age {matched_age:.0f} mean",
            edgecolor="white",
            linewidth=0.8,
        )
        ax.scatter(
            [matched_age],
            [age_q30],
            color="#2ca02c",
            s=70,
            zorder=5,
            label=f"Age {matched_age:.0f} US baseline (q30)",
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_xlabel("Age (Years)", labelpad=10)
        ax.set_ylabel("Calibrated probability", labelpad=10)
        ax.legend()
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        st.pyplot(fig)
    else:
        st.info("Provide a calibration CSV path to see the calibration point plot and comparison to the dataset mean.")

    # Risk chip and bands under the bar
    st.markdown(
        f"""
        <div style='margin: 10px 0 6px 0; padding: 10px 12px; border-radius: 10px; background: #eef6ff; border: 1px solid #d9e7ff;'>
          <div style='display: inline-flex; align-items: center; gap: 8px; font-weight: 600;'>
            <span>Risk category:</span>
            <span style='padding: 4px 10px; border-radius: 999px; background: {bar_color}; color: #1a1a1a; border: 1px solid rgba(0,0,0,0.05); box-shadow: 0 1px 2px rgba(0,0,0,0.08);'>
              {risk_label}
            </span>
          </div>
          <div style='margin-top: 6px; font-size: 12px; color: #444;'>
            Bands: Low &lt;0.10 · Moderate 0.10–0.25 · High 0.25–0.40 · Very high &gt;0.40
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # ================================
# Additional Details (Expandable)
# ================================
with st.expander("Additional Details"):
    st.markdown("### Model Validation Summary")
    st.markdown(
        f"""
        - **AUC (uncalibrated):** 0.767  
        - **AUC (calibrated):** 0.763  
        - **Brier score (calibrated):** 0.113  
        - **Calibration intercept:** 0.05  
        - **Calibration slope:** 1.02 
        - **Baseline event rate (30-day mortality):** 15%  
        """
    )

    st.markdown("### Methodological Notes & Citations")
    st.markdown(
        """
        This risk model was trained on historical bloodstream infection (BSI) cases using:

        - **XGBoost** classifier with class weighting  
        - **Isotonic regression calibration** on a held-out 15% validation set  
        - **Evaluation using the TRIPOD framework** principles for predictive modeling  

        **Key sources:**

        - Platt J. *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods* (1999).  
        - Zadrozny & Elkan. *Transforming Classifier Scores into Accurate Multiclass Probability Estimates* (2002).  
        - Van Calster et al. *Calibration: the Achilles heel of predictive analytics* (2019).  
        """
    )

    st.markdown("### Version Information")
    st.markdown(
        """
        - **Model version:** 1.0.0  
        - **Calibration method:** Isotonic regression  
        - **App version:** 1.0.0  
        - **Developer:** Lewis Research Group  
        """
    )


# ------------------------
# Footer: Disclaimer
# ------------------------
st.markdown(
    """
    <hr style="margin-top:2em; margin-bottom:1em">
    <span style="color:gray; font-size:0.95em;">
    <b>Disclaimer:</b> This tool is for research and educational purposes only. 
    It does not constitute medical advice. For clinical use, consult a licensed healthcare professional.
    </span>
    """,
    unsafe_allow_html=True,
)
