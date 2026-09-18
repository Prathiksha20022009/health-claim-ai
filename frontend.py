import requests
import streamlit as st

st.set_page_config(
    page_title="Claim Triage | Patient & Hospital Portal", page_icon="🛡️"
)

st.title("🛡️ Claim Triage: Decoupled Portal Interface")
st.markdown("Client UI sending asynchronous payloads to the FastAPI backend.")

# Sidebar Input Controls
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

if st.sidebar.button("Submit Claim to Backend", type="primary"):
  payload = {
      "gender": in_gender,
      "age": in_age,
      "hypertension": in_hypertension,
      "heart_disease": in_heart_disease,
      "avg_glucose_level": in_glucose,
      "smoking_history": in_smoking,
      "bmi": in_bmi,
      "claim_amount": in_claim,
      "coverage_limit": in_limit,
      "policy_tenure_months": in_tenure,
  }

  try:
    # Send request to local FastAPI backend
    response = requests.post("http://127.0.0.1:8000/api/adjudicate", json=payload)
    if response.status_code == 200:
      res_data = response.json()

      col1, col2 = st.columns(2)
      with col1:
        st.markdown("#### 🩺 Backend Clinical Assessment")
        risk_data = res_data["clinical_assessment"]
        st.metric(label="Calculated Risk Tier", value=risk_data["risk_tier"])
        st.metric(label="Total Risk Score", value=risk_data["total_risk_score"])
        if risk_data["clinical_flags"]:
          st.warning("⚠️ **Flags:**")
          for flag in risk_data["clinical_flags"]:
            st.write(f"- {flag}")
        else:
          st.success("✅ Normal baselines verified.")

      with col2:
        st.markdown("#### ⚖️ Backend Adjudication Decision")
        if res_data["decision"] == "AUTO-APPROVED":
          st.success(
              f"**{res_data['decision']} (Code: {res_data['status_code']})**\n\n"
              f"**Reason:** {res_data['reason']}\n\n**Approval Probability:**"
              f" {res_data['approval_probability']}"
          )
        else:
          st.error(
              f"**{res_data['decision']} (Code: {res_data['status_code']})**\n\n"
              f"**Reason:** {res_data['reason']}\n\n**Approval Probability:**"
              f" {res_data['approval_probability']}"
          )
    else:
      st.error(f"Backend Error: {response.text}")
  except requests.exceptions.ConnectionError:
    st.error(
        "Could not connect to backend. Make sure FastAPI is running on port"
        " 8000!"
    )
else:
  st.info("👈 Enter claim parameters in the sidebar and click submit.")