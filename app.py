import streamlit as st
import numpy as np
import pickle
import pandas as pd

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
    """Load the fitted scaler from disk."""
    with open("imputer.pkl", "rb") as f:
        return pickle.load(f)
    return None


model = load_model()
imputer = load_imputer()

# ------------------------
# Session state for resettable fields
# ------------------------
default_values = dict(age=65, cci=2, sofa=4, pbs=4, inputs_scaled=False)
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
        "⚠️ One or more inputs are outside the range observed in the model’s training data:\n\n"
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

    # Display probability
    st.write("### Estimated Probability")
    st.info(f"Model-estimated 30-day mortality probability: {proba * 100:.1f}%")

    # Probability bar
    st.markdown(
        f"""
        <div style='width: 100%; background: #f0f2f6; border-radius: 10px; height: 30px; margin-bottom: 10px;'>
          <div style='width: {proba * 100:.1f}%; background: #7391f5; height: 30px; border-radius: 10px; text-align: right; color: black; padding-right: 10px; font-weight: bold;'>
            {proba * 100:.1f}%
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Threshold-based grouping
    thresh = 0.5
    if proba > thresh:
        st.warning("Risk group (threshold = 0.5): HIGH")
    else:
        st.info("Risk group (threshold = 0.5): LOW")

    # Download results as CSV
    result_dict = {
        "Age": [age],
        "CCI": [cci],
        "PBS": [pbs],
        "SOFA": [sofa],
        "Model-estimated probability (%)": [proba * 100],
        "Threshold (for grouping)": [thresh],
        "Risk group (based on threshold)": ["HIGH" if proba > thresh else "LOW"],
    }
    result_df = pd.DataFrame(result_dict)

    csv = result_df.to_csv(index=False)
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name="mortality_probability.csv",
        mime="text/csv",
    )

    # Advanced/Debug Details
    with st.expander("Show advanced details"):
        st.write(f"Inputs: Age={age}, CCI={cci}, PBS={pbs}, SOFA={sofa}")
        st.write(f"Output probability: {proba * 100:.1f}%")
        st.write(f"Threshold for grouping: {thresh} ({thresh*100:.1f}%)")

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
