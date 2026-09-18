import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# -------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="Health Claim Triage AI",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Intelligent Healthcare Claim Risk & Triage System")
st.markdown("Automating medical claim reviews using machine learning pipelines to flag high-risk or fraudulent submissions.")

# -------------------------------------------------------------
# 1. GENERATE SYNTHETIC HEALTHCARE CLAIM DATASET
# -------------------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    n_samples = 2500
    np.random.seed(42)

    age = np.random.randint(18, 85, size=n_samples)
    policy_tenure_months = np.random.randint(1, 120, size=n_samples)
    annual_premium = np.random.uniform(300, 3500, size=n_samples).round(2)
    pre_existing_conditions = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    claim_amount = np.random.uniform(200, 25000, size=n_samples).round(2)
    policy_coverage_limit = np.random.choice([5000, 10000, 20000, 50000], size=n_samples)

    treatment_types = np.random.choice(
        ['Outpatient', 'Inpatient', 'Dental', 'Emergency', 'Elective Surgery'],
        size=n_samples,
        p=[0.35, 0.25, 0.15, 0.15, 0.10]
    )
    hospital_network = np.random.choice(['In-Network', 'Out-Network'], size=n_samples, p=[0.75, 0.25])
    prior_claims_count = np.random.poisson(lam=1.2, size=n_samples)

    exceeds_limit = claim_amount > policy_coverage_limit
    early_preexisting = (policy_tenure_months < 6) & (pre_existing_conditions == 1)
    unauthorized_elective = (hospital_network == 'Out-Network') & (treatment_types == 'Elective Surgery')

    claim_approved = []
    for i in range(n_samples):
        if exceeds_limit[i] or early_preexisting[i] or unauthorized_elective[i]:
            approved = np.random.choice([0, 1], p=[0.88, 0.12])
        else:
            approved = np.random.choice([1, 0], p=[0.85, 0.15])
        claim_approved.append(approved)

    df = pd.DataFrame({
        'age': age,
        'policy_tenure_months': policy_tenure_months,
        'annual_premium': annual_premium,
        'pre_existing_conditions': pre_existing_conditions,
        'treatment_type': treatment_types,
        'hospital_network': hospital_network,
        'claim_amount': claim_amount,
        'policy_coverage_limit': policy_coverage_limit,
        'prior_claims_count': prior_claims_count,
        'claim_approved': claim_approved
    })
    
    # Feature Engineering
    df['claim_to_limit_ratio'] = (df['claim_amount'] / df['policy_coverage_limit']).round(4)
    return df

df = load_and_prepare_data()

# Separate Target and Features
X = df.drop(columns=['claim_approved'])
y = df['claim_approved']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# -------------------------------------------------------------
# 2. PIPELINE SETUP & MODEL TRAINING
# -------------------------------------------------------------
numerical_features = [
    'age', 'policy_tenure_months', 'annual_premium',
    'claim_amount', 'policy_coverage_limit',
    'prior_claims_count', 'claim_to_limit_ratio'
]
categorical_features = ['treatment_type', 'hospital_network', 'pre_existing_conditions']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    ))
])

with st.spinner("Training the Random Forest Pipeline..."):
    model_pipeline.fit(X_train, y_train)

# Calculate Model Metrics for Display
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]
roc_score = roc_auc_score(y_test, y_proba)

# -------------------------------------------------------------
# 3. STREAMLIT SIDEBAR & LIVE INFERENCE
# -------------------------------------------------------------
st.sidebar.header("📝 Evaluate New Claim")

in_age = st.sidebar.slider("Patient Age", 18, 85, 34)
in_tenure = st.sidebar.slider("Policy Tenure (Months)", 1, 120, 36)
in_premium = st.sidebar.number_input("Annual Premium ($)", 300.0, 3500.0, 1200.0)
in_pre_existing = st.sidebar.selectbox("Pre-existing Conditions?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
in_treatment = st.sidebar.selectbox("Treatment Type", ['Outpatient', 'Inpatient', 'Dental', 'Emergency', 'Elective Surgery'])
in_network = st.sidebar.selectbox("Hospital Network", ['In-Network', 'Out-Network'])
in_claim = st.sidebar.number_input("Claim Amount ($)", 200.0, 25000.0, 3200.0)
in_limit = st.sidebar.selectbox("Policy Coverage Limit ($)", [5000, 10000, 20000, 50000], index=1)
in_prior_claims = st.sidebar.slider("Prior Claims Count", 0, 10, 1)

if st.sidebar.button("Run Claim Triage Prediction", type="primary"):
    # Build single row dataframe matching feature structure
    single_claim = pd.DataFrame([{
        'age': in_age,
        'policy_tenure_months': in_tenure,
        'annual_premium': in_premium,
        'pre_existing_conditions': in_pre_existing,
        'treatment_type': in_treatment,
        'hospital_network': in_network,
        'claim_amount': in_claim,
        'policy_coverage_limit': in_limit,
        'prior_claims_count': in_prior_claims,
        'claim_to_limit_ratio': round(in_claim / in_limit, 4)
    }])
    
    pred = model_pipeline.predict(single_claim)[0]
    proba = model_pipeline.predict_proba(single_claim)[0][1]
    
    st.subheader("🎯 Evaluation Decision Output")
    col1, col2 = st.columns(2)
    
    with col1:
        if pred == 1:
            st.success(f"### Decision: APPROVED\n**Approval Probability:** {proba:.2%}")
        else:
            st.error(f"### Decision: DENIED (High Risk)\n**Approval Probability:** {proba:.2%}")
            
    with col2:
        st.metric(label="Calculated Utilization Ratio", value=f"{single_claim['claim_to_limit_ratio'].values[0]:.2f}")

# -------------------------------------------------------------
# 4. DASHBOARD METRICS VIEW
# -------------------------------------------------------------
st.divider()
st.subheader("📊 System Performance Overview (Test Set)")
m1, m2 = st.columns(2)
with m1:
    st.metric(label="Model ROC-AUC Score", value=f"{roc_score:.4f}")
with m2:
    st.metric(label="Dataset Size", value=f"{len(df)} Records")

with st.expander("🔍 View Detailed Confusion Matrix & Classification Report"):
    st.text(confusion_matrix(y_test, y_pred))
    st.text(classification_report(y_test, y_pred, target_names=['Denied (0)', 'Approved (1)']))