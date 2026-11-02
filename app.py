"""
Diabetes Risk Screening - Streamlit frontend
Uses the models saved by diabetes_prediction.py
(logistic_model.pkl and random_forest_model.pkl).

Run with:  streamlit run app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

FEATURES = ["Glucose", "BMI", "Insulin", "Age"]
BASE_DIR = Path(__file__).parent

st.set_page_config(page_title="Diabetes Risk Screening", page_icon="🩺", layout="centered")

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container {max-width: 760px; padding-top: 2.2rem;}
    h1 {font-weight: 700; letter-spacing: -0.5px; margin-bottom: 0.2rem;}
    .subtitle {color: #5b6b73; margin-bottom: 1.6rem;}
    .meter-wrap {margin: 0.4rem 0 0.2rem 0;}
    .meter-track {
        position: relative; height: 14px; border-radius: 7px;
        background: linear-gradient(90deg, #2f9e75 0%, #e9c46a 50%, #d1495b 100%);
    }
    .meter-pin {
        position: absolute; top: -5px; width: 4px; height: 24px;
        background: #12232b; border-radius: 2px; transform: translateX(-2px);
    }
    .meter-scale {display: flex; justify-content: space-between;
        font-size: 0.8rem; color: #5b6b73; margin-top: 6px;}
    .model-name {font-weight: 600; margin-bottom: 0;}
    .big {font-size: 2.2rem; font-weight: 700; line-height: 1.1;}
    .note {font-size: 0.85rem; color: #5b6b73;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Load models ----------
@st.cache_resource
def load_models():
    lr = joblib.load(BASE_DIR / "logistic_model.pkl")
    rf = joblib.load(BASE_DIR / "random_forest_model.pkl")
    return lr, rf


try:
    lr_model, rf_model = load_models()
except FileNotFoundError:
    st.error(
        "Model files not found. Run diabetes_prediction.py first so that "
        "logistic_model.pkl and random_forest_model.pkl are saved in the "
        "same folder as app.py."
    )
    st.stop()


def meter(probability: float) -> str:
    pos = max(0.0, min(1.0, probability)) * 100
    return f"""
    <div class="meter-wrap">
      <div class="meter-track"><div class="meter-pin" style="left:{pos:.1f}%"></div></div>
      <div class="meter-scale"><span>Low</span><span>50%</span><span>High</span></div>
    </div>
    """


def show_result(name: str, model, patient: pd.DataFrame):
    prediction = int(model.predict(patient)[0])
    probability = float(model.predict_proba(patient)[0][1])

    st.markdown(f"<p class='model-name'>{name}</p>", unsafe_allow_html=True)
    label = "Diabetic" if prediction == 1 else "Non-diabetic"
    st.markdown(f"<div class='big'>{label}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<span class='note'>Estimated probability of diabetes: "
        f"<b>{probability * 100:.1f}%</b></span>",
        unsafe_allow_html=True,
    )
    st.markdown(meter(probability), unsafe_allow_html=True)
    return prediction, probability


# ---------- Header ----------
st.title("Diabetes Risk Screening")
st.markdown(
    "<div class='subtitle'>Enter four measurements to see what two models "
    "trained on the Pima Indians Diabetes Dataset predict.</div>",
    unsafe_allow_html=True,
)

tab_predict, tab_about = st.tabs(["Predict", "About the models"])

# ---------- Predict tab ----------
with tab_predict:
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            glucose = st.number_input("Glucose (mg/dL)", min_value=1.0, max_value=300.0,
                                      value=120.0, step=1.0)
            bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0,
                                  value=30.0, step=0.1)
        with col2:
            age = st.number_input("Age (years)", min_value=18, max_value=100,
                                  value=35, step=1)
            insulin = st.number_input("Insulin (µU/mL)", min_value=0.0, max_value=900.0,
                                      value=80.0, step=1.0)

        insulin_unknown = st.checkbox(
            "Insulin value is not available",
            help="The models handle a missing insulin value by using the "
                 "typical value from the training data.",
        )
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        insulin_value = np.nan if (insulin_unknown or insulin == 0) else insulin
        patient = pd.DataFrame([[glucose, bmi, insulin_value, age]], columns=FEATURES)

        st.divider()
        left, right = st.columns(2)
        with left:
            lr_pred, lr_prob = show_result("Logistic Regression", lr_model, patient)
        with right:
            rf_pred, rf_prob = show_result("Random Forest", rf_model, patient)

        st.divider()
        if lr_pred == 1 and rf_pred == 1:
            st.warning("Both models flag a high risk. A fasting glucose or HbA1c test "
                       "with a doctor would be the next step.")
        elif lr_pred == 1 or rf_pred == 1:
            st.info("The models disagree. Since missing a real case is the costlier error, "
                    "treat this as a borderline result and consider a follow-up test.")
        else:
            st.success("Both models predict non-diabetic.")

        if insulin_unknown or insulin == 0:
            st.caption("Insulin was treated as missing and replaced with the training median.")

# ---------- About tab ----------
with tab_about:
    st.markdown("### How the models were evaluated")
    st.markdown(
        "Both models use Glucose, BMI, Insulin and Age and were evaluated "
        "with 5-fold cross-validation on 768 records."
    )
    st.table(
        pd.DataFrame(
            {
                "Model": ["Logistic Regression", "Random Forest"],
                "Accuracy": ["74.4%", "74.7%"],
                "Recall (diabetic)": ["69.4%", "61.2%"],
            }
        ).set_index("Model")
    )
    st.markdown(
        "Recall is the share of real diabetic patients the model catches. "
        "Logistic Regression catches more of them, so it is the better "
        "choice for early detection."
    )
    st.markdown("### Limits")
    st.markdown(
        "- Trained on 768 Pima Indian women aged 21 and over, so results may "
        "not apply to other groups.\n"
        "- Nearly half of the Insulin values in the training data were missing.\n"
        "- Even the better model misses roughly 3 in 10 diabetic patients."
    )

st.divider()
st.caption("Educational project. This tool is a screening aid, not a medical diagnosis.")
