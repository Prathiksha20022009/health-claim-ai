import requests
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="ClaimTriage | Enterprise Health Platform",
    page_icon="🛡️",
    layout="wide",
)

# Hero Header Banner
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 25px; border-radius: 12px; color: white; margin-bottom: 25px;">
        <h1 style="margin: 0; font-size: 26px; color: #ffffff;">🛡️ ClaimTriage Enterprise Hub</h1>
        <p style="margin: 5px 0 0 0; color: #94a3b8; font-size: 14px;">Decoupled Unified Health Aggregation & Autonomous Adjudication Engine</p>
    </div>
""",
    unsafe_allow_html=True,
)

# Focused 2-Tab Layout (User-Facing Portals Only)
tab1, tab2 = st.tabs([
    "🧑‍⚕️ Patient Portal (Scheme Matching)",
    "🏥 Hospital Portal (Claim Adjudication)",
])

# =====================================================================
# TAB 1: PATIENT PORTAL
# =====================================================================
with tab1:
  st.markdown("### **Patient Eligibility & Welfare Scheme Discovery**")
  st.markdown(
      "Evaluate public safety nets and private commercial insurance"
      " side-by-side."
  )

  col_p1, col_p2 = st.columns(2)
  with col_p1:
    p_age = st.slider("Patient Age", 18, 90, 35, key="p_age")
    p_income = st.number_input(
        "Annual Household Income ($ / ₹)",
        50000.0,
        3000000.0,
        350000.0,
        step=10000.0,
        key="p_income",
    )
  with col_p2:
    p_pre = st.selectbox(
        "Pre-existing Medical Conditions?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        key="p_pre",
    )
    p_location = st.text_input("State / Region Pincode", "110001", key="p_loc")

  if st.button("Check Scheme Eligibility", key="btn_scheme"):
    st.markdown("#### **Available Coverage & Scheme Matches:**")
    c_s1, c_s2 = st.columns(2)
    with c_s1:
      if p_income <= 500000:
        st.success(
            "🟢 **Ayushman Bharat PM-JAY (Public Safety Net)**\n\n- **Coverage:"
            "** Up to ₹5,00,000 Cashless Cover\n- **Match:** 100% Eligible"
        )
      else:
        st.info(
            "🔵 **Standard Private Health Shield**\n\n- **Coverage:**"
            " Customizable up to ₹1,000,000\n- **Match:** Standard Market"
        )
    with c_s2:
      if p_age >= 60:
        st.success(
            "🟢 **Senior Citizen Health Protection Plan**\n\n- **Coverage:**"
            " Specialized Geriatric Care\n- **Match:** Age Demographic Match"
        )
      else:
        st.success(
            "🟢 **Comprehensive Family Floater**\n\n- **Coverage:** Outpatient"
            " & Inpatient Shield\n- **Match:** Fully Qualified"
        )

# =====================================================================
# TAB 2: HOSPITAL PORTAL (Adjudication Engine)
# =====================================================================
with tab2:
  st.markdown("### **Hospital Billing & Autonomous Adjudication Portal**")

  with st.sidebar:
    st.markdown("### 📋 Hospital Submission Controls")
    st.markdown("---")
    in_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    in_age = st.slider("Patient Age", 18, 90, 45, key="h_age")
    in_income = st.number_input(
        "Annual Income ($ / ₹)",
        50000.0,
        3000000.0,
        350000.0,
        step=10000.0,
        key="h_income",
    )
    in_bmi = st.slider("Body Mass Index (BMI)", 15.0, 45.0, 27.5)
    in_glucose = st.number_input(
        "Avg Glucose Level (mg/dL)", 60.0, 300.0, 95.0
    )

    st.markdown("---")
    in_hypertension = st.selectbox(
        "Hypertension",
        [0, 1],
        format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
    )
    in_heart_disease = st.selectbox(
        "Heart Disease",
        [0, 1],
        format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
    )
    in_smoking = st.selectbox("Smoking History", ["never", "former", "smokes"])

    st.markdown("---")
    in_claim = st.number_input(
        "Claim Request Amount ($)", 100.0, 100000.0, 4500.0
    )
    in_limit = st.selectbox(
        "Policy Coverage Limit ($)", [5000, 10000, 25000, 50000], index=1
    )
    in_tenure = st.slider("Policy Tenure (Months)", 1, 60, 18)

    submit_adjudication = st.button("Run Adjudication Review", type="primary")

  if submit_adjudication:
    payload = {
        "gender": in_gender,
        "age": in_age,
        "income": in_income,
        "hypertension": in_hypertension,
        "heart_disease": in_heart_disease,
        "avg_glucose_level": in_glucose,
        "smoking_history": in_smoking,
        "bmi": in_bmi,
        "claim_amount": in_claim,
        "coverage_limit": in_limit,
        "policy_tenure_months": in_tenure,
    }

    with st.spinner("Executing FastAPI microservice evaluation..."):
      try:
        response = requests.post(
            "http://127.0.0.1:8000/api/adjudicate", json=payload
        )
        if response.status_code == 200:
          res_data = response.json()
          res_col1, res_col2 = st.columns(2)

          with res_col1:
            st.markdown("#### 🩺 Clinical Risk Audit")
            risk_data = res_data["clinical_assessment"]
            m1, m2 = st.columns(2)
            with m1:
              st.metric("Risk Tier", risk_data["risk_tier"])
            with m2:
              st.metric("Risk Score", risk_data["total_risk_score"])

            st.markdown("<br>", unsafe_allow_html=True)
            if risk_data["clinical_flags"]:
              st.warning("⚠️ **Active Clinical Flags:**")
              for flag in risk_data["clinical_flags"]:
                st.write(f"- {flag}")
            else:
              st.success("✅ Patient metrics clear standard WHO baselines.")

          with res_col2:
            st.markdown("#### ⚖️ Adjudication Decision Output")
            decision = res_data["decision"]
            if decision == "AUTO-APPROVED":
              st.success(
                  f"**STATUS: AUTO-APPROVED**\n\n"
                  f"**Authorization Code:** `{res_data['status_code']}`\n\n"
                  f"**Reason:** {res_data['reason']}\n\n"
                  f"**Approval Confidence:**"
                  f" {res_data['approval_probability']}"
              )
            else:
              st.error(
                  f"**STATUS: FLAGGED FOR AUDIT / DENIED**\n\n"
                  f"**Audit Code:** `{res_data['status_code']}`\n\n"
                  f"**Reason:** {res_data['reason']}\n\n"
                  f"**Approval Probability:**"
                  f" {res_data['approval_probability']}"
              )
        else:
          st.error(f"Backend Server Error: {response.text}")
      except requests.exceptions.ConnectionError:
        st.error(
            "⚠️ **Backend Offline:** Please ensure FastAPI is running via"
            " `uvicorn backend:app --reload`."
        )
  else:
    st.info(
        "👈 Configure patient metrics and claim financials in the left sidebar,"
        " then click **Run Adjudication Review** to see results."
    )
