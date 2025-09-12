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


@st.cache_resource
def load_scaler():
    """Load the fitted scaler from disk."""
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()
scaler = load_scaler()

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
# Sidebar: Branding and About
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
        "This calculator estimates the risk of 30-day mortality based on Age, "
        "Charlson Comorbidity Index (CCI), Pitt Bacteremia Score (PBS) and SOFA score, using a machine learning model. "
        "For more on these indices: "
        "[CCI](https://www.mdcalc.com/calc/3917/charlson-comorbidity-index-cci), "
        "[PBS](https://m.medicalalgorithms.com/pitt-bacteremia-score-of-paterson-et-al), "
        "[SOFA](https://www.mdcalc.com/calc/691/sequential-organ-failure-assessment-sofa-score)"

    )

# ------------------------
# Main App Interface
# ------------------------
st.title("30-Day Mortality Prediction")
st.header("Enter Patient Data")

# Tooltips for each feature
cci_info = (
    "Charlson Comorbidity Index (CCI) is a widely-used scoring system "
    "that predicts ten-year mortality based on the presence of comorbidity conditions. "
    "[Learn more.](https://www.mdcalc.com/calc/3917/charlson-comorbidity-index-cci)"
)
pbs_info = (
    "Pitt Bacteremia Score (PBS) is a widely used tool in infectious disease research to assess the severity of acute "
    "illness and predict mortality."
    "[Learn more.](https://m.medicalalgorithms.com/pitt-bacteremia-score-of-paterson-et-al)"
)
sofa_info = (
    "SOFA (Sequential Organ Failure Assessment) score quantifies the extent of a patient's organ function or rate of "
    "failure. [Learn more.](https://www.mdcalc.com/calc/691/sequential-organ-failure-assessment-sofa-score)"
)
age_info = "Patient's age in years."

# Feature inputs (linked to session_state for reset functionality)
age = st.number_input("Age (years)", min_value=0, max_value=120, key="age", help=age_info)
cci = st.number_input("Charlson Comorbidity Index (CCI)", min_value=0, max_value=20, key="cci", help=cci_info)
pbs = st.number_input("PBS Score", min_value=0, max_value=14, key="pbs", help=pbs_info)
sofa = st.number_input("SOFA Score", min_value=0, max_value=24, key="sofa", help=sofa_info)
inputs_scaled = st.toggle("Scale Inputs", value=st.session_state["inputs_scaled"], key="inputs_scaled")

# Predict and Reset buttons side-by-side
col1, col2 = st.columns([1, 1])
with col1:
    predict = st.button("Predict 30-Day Mortality")
with col2:
    reset = st.button("Reset", on_click=reset_form)

# ------------------------
# Prediction & Results
# ------------------------
if predict:
    # Prepare features for prediction
    X = np.array([[age, cci, pbs, sofa]])
    if not inputs_scaled:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    # Predict mortality risk
    proba = model.predict_proba(X_scaled)[0][1]

    # Risk Indicator
    st.write("### Risk Indicator")
    st.success(f"Estimated 30-day mortality risk: {proba * 100:.1f}%")

    def risk_color(prob):
        if prob < 0.33:
            return "#65c18c"  # Green
        elif prob < 0.67:
            return "#FFD700"  # Yellow
        else:
            return "#ff4b4b"  # Red

    bar_color = risk_color(proba)
    st.markdown(f"""
    <div style='width: 100%; background: #f0f2f6; border-radius: 10px; height: 30px; margin-bottom: 10px;'>
      <div style='width: {proba * 100:.1f}%; background: {bar_color}; height: 30px; border-radius: 10px; text-align: right; color: black; padding-right: 10px; font-weight: bold;'>
        {proba * 100:.1f}%
      </div>
    </div>
    """, unsafe_allow_html=True)

    if proba > 0.5:
        st.warning("Predicted outcome: HIGH RISK of 30-day mortality")
    else:
        st.info("Predicted outcome: LOW RISK of 30-day mortality")

    # Download Results as CSV
    result_dict = {
        "Age": [age],
        "CCI": [cci],
        "PBS": [pbs],
        "SOFA": [sofa],
        "Inputs scaled": [inputs_scaled],
        "Model input": [X.tolist() if inputs_scaled else scaler.transform(X).tolist()],
        "Predicted risk (%)": [proba * 100],
        "High risk threshold": [0.5],
        "Risk label": ["HIGH" if proba > 0.5 else "LOW"]
    }
    result_df = pd.DataFrame(result_dict)

    csv = result_df.to_csv(index=False)
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name="mortality_prediction.csv",
        mime="text/csv"
    )

    # Advanced/Debug Details
    with st.expander("Show advanced details"):
        st.write(f"Raw inputs: Age={age}, CCI={cci}, PBS={pbs}, SOFA={sofa}")
        st.write(f"Inputs were NOT {'scaled' if not inputs_scaled else 'scaled'} before prediction.")
        st.write(f"Model input: {X.tolist() if  not inputs_scaled else scaler.transform(X).tolist()}")
        st.write(f"Output probability (risk): {proba * 100:.1f}%")
        st.write(f"Threshold for HIGH RISK: 0.5")

# ------------------------
# Footer: Disclaimer
# ------------------------
st.markdown(
    """
    <hr style="margin-top:2em; margin-bottom:1em">
    <span style="color:gray; font-size:0.95em;">
    <b>Disclaimer:</b> This tool is for research and educational purposes only and does not constitute medical advice. 
    For clinical decisions, consult a licensed healthcare professional.
    </span>
    """,
    unsafe_allow_html=True
)
