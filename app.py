import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------------------
# PAGE CONFIGURATION & ENTERPRISE STYLING
# -------------------------------------------------------------
st.set_page_config(
    page_title="Claim Triage | Clinical & Financial Adjudication",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Claim Triage: Holistic Clinical & Financial Adjudication Engine")
st.markdown(
    "Native enterprise underwriting platform combining WHO clinical health"
    " baselines with insurance policy rules."
)


# -------------------------------------------------------------
# 1. CLINICAL RISK EVALUATION FUNCTION (Fixed Smoking Logic)
# -------------------------------------------------------------
def evaluate_clinical_risk(
    gender,
    age,
    hypertension,
    heart_disease,
    avg_glucose_level,
    smoking_history,
    bmi,
):
  risk_points = 0
  risk_reasons = []

  # A. Age & Gender Baseline Risk Weighting
  gender_lower = gender.strip().lower()

  if age >= 65:
    risk_points += 3
    risk_reasons.append(
        f"Advanced senior age demographic ({age} years - High baseline"
        " vulnerability)"
    )
  elif age >= 50:
    risk_points += 2
    risk_reasons.append(f"Elevated age bracket ({age} years)")
  elif gender_lower == "male" and age >= 45:
    risk_points += 1
    risk_reasons.append(f"Male demographic risk threshold met ({age} years)")
  elif gender_lower == "female" and age >= 55:
    risk_points += 1
    risk_reasons.append(
        f"Post-menopausal female demographic risk threshold met ({age} years)"
    )

  # B. BMI Evaluation (WHO Standard: Normal 18.5 - 24.9)
  if bmi > 30.0:
    risk_points += 2
    risk_reasons.append(
        f"Obesity flagged (BMI: {bmi} > 30.0 - High metabolic/cardiac risk)"
    )
  elif bmi < 18.5:
    risk_points += 1
    risk_reasons.append(f"Underweight flagged (BMI: {bmi} < 18.5)")

  # C. Glucose Evaluation (Normal: 70 - 99 mg/dL)
  if avg_glucose_level >= 126.0:
    risk_points += 3
    risk_reasons.append(
        f"Elevated Glucose ({avg_glucose_level} mg/dL - Diabetes indicator)"
    )
  elif avg_glucose_level > 99.0:
    risk_points += 1
    risk_reasons.append(
        f"Pre-diabetic glucose range ({avg_glucose_level} mg/dL)"
    )

  # D. Chronic Conditions & History (Hypertension / Heart Disease)
  if hypertension == 1:
    risk_points += 2
    risk_reasons.append("Diagnosed Hypertension (High Blood Pressure)")

  if heart_disease == 1:
    risk_points += 4
    risk_reasons.append("Prior History of Cardiovascular Disease")

  # E. Smoking History (FIXED: Differentiating Active vs. Former)
  smoking_lower = smoking_history.strip().lower()
  if smoking_lower == "smokes":
    risk_points += 2
    risk_reasons.append(
        "Active smoker profile (Elevated respiratory/cardiac risk)"
    )
  elif smoking_lower == "former":
    risk_points += (
        1  # Lower risk point for former smokers compared to active ones
    )
    risk_reasons.append(
        "Former smoker profile (Moderate historical risk adjustment)"
    )
  # 'never' adds 0 points

  # Holistic Tier Classification
  is_high_risk = risk_points >= 5
  risk_tier = (
      "HIGH RISK"
      if is_high_risk
      else ("MODERATE RISK" if risk_points >= 2 else "LOW RISK")
  )

  return {
      "total_risk_score": risk_points,
      "risk_tier": risk_tier,
      "clinical_flags": risk_reasons,
  }


# -------------------------------------------------------------
# 2. SIDEBAR CONTROLS (Dataset Attributes)
# -------------------------------------------------------------
st.sidebar.header("📋 Patient Clinical Profile")

in_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
in_age = st.sidebar.slider("Patient Age", 18, 90, 45)
in_bmi = st.sidebar.slider("Body Mass Index (BMI)", 15.0, 45.0, 27.5)
in_glucose = st.sidebar.number_input(
    "Avg Glucose Level (mg/dL)", 60.0, 300.0, 95.0
)

st.sidebar.markdown("---")
st.sidebar.header("🏥 Medical History & Lifestyle")
in_hypertension = st.sidebar.selectbox(
    "Hypertension",
    [0, 1],
    format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
)
in_heart_disease = st.sidebar.selectbox(
    "Heart Disease",
    [0, 1],
    format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
)
in_smoking = st.sidebar.selectbox(
    "Smoking History", ["never", "former", "smokes"]
)

st.sidebar.markdown("---")
st.sidebar.header("💰 Policy & Claim Financials")
in_claim = st.sidebar.number_input("Claim Amount ($)", 100.0, 50000.0, 4500.0)
in_limit = st.sidebar.selectbox(
    "Policy Coverage Limit ($)", [5000, 10000, 25000, 50000], index=1
)
in_tenure = st.sidebar.slider("Policy Tenure (Months)", 1, 60, 18)

# -------------------------------------------------------------
# 3. RUN ADJUDICATION ENGINE
# -------------------------------------------------------------
st.markdown("### **Live Adjudication & Clinical Risk Dashboard**")

if st.sidebar.button("Run Claim Adjudication", type="primary"):
  clinical_result = evaluate_clinical_risk(
      gender=in_gender,
      age=in_age,
      hypertension=in_hypertension,
      heart_disease=in_heart_disease,
      avg_glucose_level=in_glucose,
      smoking_history=in_smoking,
      bmi=in_bmi,
  )

  is_excess = in_claim > in_limit
  is_early = in_tenure < 6

  approval_prob = 0.90
  if clinical_result["risk_tier"] == "HIGH RISK":
    approval_prob -= 0.55
  elif clinical_result["risk_tier"] == "MODERATE RISK":
    approval_prob -= 0.25

  if is_excess or is_early:
    approval_prob = 0.0

  final_prob_str = f"{max(0.0, approval_prob) * 100:.1f}%"

  col1, col2 = st.columns(2)

  with col1:
    st.markdown("#### 🩺 Clinical Risk Assessment")
    st.metric(label="Calculated Risk Tier", value=clinical_result["risk_tier"])
    st.metric(
        label="Total Risk Score", value=clinical_result["total_risk_score"]
    )

    if clinical_result["clinical_flags"]:
      st.warning("⚠️ **Detected Clinical Risk Flags:**")
      for flag in clinical_result["clinical_flags"]:
        st.write(f"- {flag}")
    else:
      st.success("✅ Patient profile is within normal medical baselines.")

  with col2:
    st.markdown("#### ⚖️ Financial & Adjudication Decision")
    if approval_prob < 0.5 or is_excess or is_early:
      reason = (
          "Claim amount exceeds coverage limit."
          if is_excess
          else (
              "Policy tenure under 6-month waiting period."
              if is_early
              else "High clinical risk profile breached underwriting"
              " thresholds."
          )
      )
      st.error(
          f"**STATUS: FLAGGED FOR AUDIT / DENIED**\n\n**Reason:**"
          f" {reason}\n\n**Approval Probability:** {final_prob_str}"
      )
    else:
      st.success(
          "**STATUS: AUTO-APPROVED (Code: PA_100)**\n\nClaim verified against"
          f" medical baselines and policy terms.\n\n**Approval Probability:**"
          f" {final_prob_str}"
      )
else:
  st.info(
      "👈 Configure patient health stats and policy parameters in the left"
      " sidebar, then click **Run Claim Adjudication**."
  )
