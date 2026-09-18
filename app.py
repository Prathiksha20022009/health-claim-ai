import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -------------------------------------------------------------
# PAGE CONFIGURATION & ENTERPRISE STYLING
# -------------------------------------------------------------
st.set_page_config(
    page_title="Claim Triage | Enterprise Underwriting Platform",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Claim Triage: Automated Insurance Adjudication Engine")
st.markdown(
    "Enterprise-grade risk profiling and policy validation platform combining"
    " clinical baselines with financial underwriting rules."
)


# -------------------------------------------------------------
# 1. CACHED DATA LOADING & MODEL TRAINING
# -------------------------------------------------------------
@st.cache_resource
def load_and_train_model():
  np.random.seed(42)
  n_samples = 2500

  age = np.random.randint(18, 85, size=n_samples)
  policy_tenure_months = np.random.randint(1, 120, size=n_samples)
  annual_premium = np.random.uniform(300, 3500, size=n_samples).round(2)
  pre_existing_conditions = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
  claim_amount = np.random.uniform(200, 25000, size=n_samples).round(2)
  policy_coverage_limit = np.random.choice(
      [5000, 10000, 20000, 50000], size=n_samples
  )

  treatment_types = np.random.choice(
      ["Outpatient", "Inpatient", "Dental", "Emergency", "Elective Surgery"],
      size=n_samples,
      p=[0.35, 0.25, 0.15, 0.15, 0.10],
  )
  hospital_network = np.random.choice(
      ["In-Network", "Out-Network"], size=n_samples, p=[0.75, 0.25]
  )
  prior_claims_count = np.random.poisson(lam=1.2, size=n_samples)

  exceeds_limit = claim_amount > policy_coverage_limit
  early_preexisting = (policy_tenure_months < 6) & (
      pre_existing_conditions == 1
  )
  unauthorized_elective = (hospital_network == "Out-Network") & (
      treatment_types == "Elective Surgery"
  )
  too_elderly = age > 70  # NEW RULE: Deny coverage for age > 70

  claim_approved = []
  claim_status_codes = []
  adjudication_reasons = []

  for i in range(n_samples):
    if (
        exceeds_limit[i]
        or early_preexisting[i]
        or unauthorized_elective[i]
        or too_elderly[i]
    ):
      approved = np.random.choice([0, 1], p=[0.88, 0.12])
    else:
      approved = np.random.choice([1, 0], p=[0.85, 0.15])

    claim_approved.append(approved)

    if approved == 1:
      claim_status_codes.append("PA_100")
      adjudication_reasons.append("Approved: Claim met coverage guidelines")
    else:
      if too_elderly[i]:
        claim_status_codes.append("D5_AGE")
        adjudication_reasons.append(
            "Denied: Age exceeds maximum underwriting demographic limit (>70"
            " yrs)"
        )
      elif exceeds_limit[i]:
        claim_status_codes.append("D1_EXP")
        adjudication_reasons.append(
            "Denied: Amount exceeds policy coverage limit"
        )
      elif early_preexisting[i]:
        claim_status_codes.append("D2_PRX")
        adjudication_reasons.append(
            "Denied: Pre-existing condition within waiting period (<6 mo)"
        )
      elif unauthorized_elective[i]:
        claim_status_codes.append("D3_OON")
        adjudication_reasons.append(
            "Denied: Unauthorized out-of-network elective procedure"
        )
      else:
        claim_status_codes.append("D4_GEN")
        adjudication_reasons.append(
            "Denied: Incomplete documentation or standard policy exclusions"
        )

  df = pd.DataFrame({
      "age": age,
      "policy_tenure_months": policy_tenure_months,
      "annual_premium": annual_premium,
      "pre_existing_conditions": pre_existing_conditions,
      "treatment_type": treatment_types,
      "hospital_network": hospital_network,
      "claim_amount": claim_amount,
      "policy_coverage_limit": policy_coverage_limit,
      "prior_claims_count": prior_claims_count,
      "claim_approved": claim_approved,
      "claim_status_code": claim_status_codes,
      "adjudication_reason": adjudication_reasons,
  })

  df["claim_to_limit_ratio"] = (
      df["claim_amount"] / df["policy_coverage_limit"]
  ).round(4)

  feature_cols = [
      "age",
      "policy_tenure_months",
      "annual_premium",
      "pre_existing_conditions",
      "treatment_type",
      "hospital_network",
      "claim_amount",
      "policy_coverage_limit",
      "prior_claims_count",
      "claim_to_limit_ratio",
  ]

  X = df[feature_cols]
  y = df["claim_approved"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.20, stratify=y, random_state=42
  )

  numerical_features = [
      "age",
      "policy_tenure_months",
      "annual_premium",
      "claim_amount",
      "policy_coverage_limit",
      "prior_claims_count",
      "claim_to_limit_ratio",
  ]
  categorical_features = [
      "treatment_type",
      "hospital_network",
      "pre_existing_conditions",
  ]

  preprocessor = ColumnTransformer(
      transformers=[
          ("num", StandardScaler(), numerical_features),
          (
              "cat",
              OneHotEncoder(handle_unknown="ignore"),
              categorical_features,
          ),
      ]
  )

  model_pipeline = Pipeline(
      steps=[
          ("preprocessor", preprocessor),
          (
              "classifier",
              RandomForestClassifier(
                  n_estimators=150,
                  max_depth=10,
                  class_weight="balanced",
                  random_state=42,
              ),
          ),
      ]
  )

  model_pipeline.fit(X_train, y_train)

  y_pred = model_pipeline.predict(X_test)
  y_proba = model_pipeline.predict_proba(X_test)[:, 1]
  roc_score = roc_auc_score(y_test, y_proba)

  return df, model_pipeline, roc_score, y_test, y_pred


with st.spinner("Compiling enterprise underwriting model..."):
  df, model_pipeline, roc_score, y_test, y_pred = load_and_train_model()

# -------------------------------------------------------------
# 2. SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.header("📋 Claim & Policy Parameters")

in_age = st.sidebar.slider("Patient Age", 18, 85, 34)
in_tenure = st.sidebar.slider("Policy Tenure (Months)", 1, 120, 36)
in_premium = st.sidebar.number_input("Annual Premium ($)", 300.0, 3500.0, 1200.0)
in_pre_existing = st.sidebar.selectbox(
    "Pre-existing Conditions?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
)
in_treatment = st.sidebar.selectbox(
    "Treatment Type",
    [
        "Outpatient",
        "Inpatient",
        "Dental",
        "Emergency",
        "Elective Surgery",
    ],
)
in_network = st.sidebar.selectbox(
    "Hospital Network", ["In-Network", "Out-Network"]
)
in_claim = st.sidebar.number_input(
    "Claim Amount ($)", 200.0, 50000.0, 3200.0
)
in_limit = st.sidebar.selectbox(
    "Policy Coverage Limit ($)", [5000, 10000, 20000, 50000], index=1
)
in_prior_claims = st.sidebar.slider("Prior Claims Count", 0, 10, 1)

single_claim = pd.DataFrame([{
    "age": in_age,
    "policy_tenure_months": in_tenure,
    "annual_premium": in_premium,
    "pre_existing_conditions": in_pre_existing,
    "treatment_type": in_treatment,
    "hospital_network": in_network,
    "claim_amount": in_claim,
    "policy_coverage_limit": in_limit,
    "prior_claims_count": in_prior_claims,
    "claim_to_limit_ratio": round(in_claim / in_limit, 4),
}])


# -------------------------------------------------------------
# 3. HELPER FUNCTION FOR REASON CODES
# -------------------------------------------------------------
def adjudicate_claim(row, prediction):
  if prediction == 1:
    return "PA_100", "Approved: Claim meets coverage terms"

  if row["age"] > 70:
    return (
        "D5_AGE",
        "Denied: Age exceeds maximum underwriting demographic limit (>70 yrs)",
    )
  if row["claim_amount"] > row["policy_coverage_limit"]:
    return "D1_EXP", "Denied: Exceeds maximum policy limit"
  if row["policy_tenure_months"] < 6 and row["pre_existing_conditions"] == 1:
    return (
        "D2_PRX",
        "Denied: Pre-existing condition restriction (<6 months tenure)",
    )
  if (
      row["hospital_network"] == "Out-Network"
      and row["treatment_type"] == "Elective Surgery"
  ):
    return (
        "D3_OON",
        "Denied: Non-emergency elective service at out-of-network facility",
    )
  return "D4_GEN", "Denied: High risk profile / Policy exclusions"


# -------------------------------------------------------------
# 4. DASHBOARD METRICS & INFERENCE OUTPUT
# -------------------------------------------------------------
c1, c2, c3 = st.columns(3)
with c1:
  st.metric(
      label="MODEL ROC-AUC",
      value=f"{roc_score:.4f}",
      delta="Validated Pipeline",
  )
with c2:
  st.metric(
      label="DATASET SCALE",
      value=f"{len(df):,} Records",
      delta="Training Corpus",
  )
with c3:
  st.metric(label="ENGINE STATUS", value="Operational", delta="Low Latency")

st.markdown("---")
st.markdown("### **Underwriting Decision & Risk Analysis**")

if st.sidebar.button("Run Adjudication Review", type="primary"):
  prediction = model_pipeline.predict(single_claim)[0]
  prediction_proba = model_pipeline.predict_proba(single_claim)[0][1]
  code, reason = adjudicate_claim(single_claim.iloc[0], prediction)

  if prediction == 1:
    st.success(
        f"**STATUS: AUTO-APPROVED (Code: {code})**\n\n**Reason:** {reason}\n\n"
        f"Payout authorized. (Approval Confidence: {prediction_proba * 100:.1f}%)"
    )
  else:
    st.error(
        f"**STATUS: FLAGGED FOR AUDIT / DENIED (Code: {code})**\n\n**Reason:**"
        f" {reason}\n\nImmediate review routed to supervisor. (Denial Risk"
        f" Score: {(1 - prediction_proba) * 100:.1f}%)"
    )
else:
  st.info(
      "👈 Adjust claim parameters in the left sidebar and click **Run"
      " Adjudication Review** to evaluate a submission."
  )

st.markdown("---")
with st.expander("🔍 System Diagnostics & Confusion Matrix"):
  st.text(str(confusion_matrix(y_test, y_pred)))
  st.text(
      classification_report(
          y_test, y_pred, target_names=["Denied (0)", "Approved (1)"]
      )
  )
