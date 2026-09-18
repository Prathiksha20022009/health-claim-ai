import pandas as pd
import numpy as np
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Claim Assessment",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD HEALTH DATA
# =========================================================

@st.cache_data
def load_health_data():

    df = pd.read_csv("stroke_prediction.csv")

    df = df.dropna()

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    if "stroke" not in df.columns:
        st.error("stroke_prediction.csv must contain a 'stroke' column.")
        st.stop()

    df["target_flag"] = df["stroke"]

    df = df.drop(columns=["stroke"])

    return df


health_df = load_health_data()

X_health = health_df.drop(columns=["target_flag"])
y_health = health_df["target_flag"]


# =========================================================
# TRAIN HEALTH MODEL
# =========================================================

@st.cache_resource
def train_health_model(X, y):

    numeric_features = X.select_dtypes(
        include=[
            "int64",
            "float64",
            "int32",
            "float32"
        ]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features
            ),
            (
                "categorical",
                encoder,
                categorical_features
            )
        ]
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    test_probability = model.predict_proba(
        X_test
    )[:, 1]

    try:
        auc = roc_auc_score(
            y_test,
            test_probability
        )
    except:
        auc = 0.0

    return model, auc


health_model, health_auc = train_health_model(
    X_health,
    y_health
)


# =========================================================
# CREATE CLAIM DATASET USING NUMPY + PANDAS
# =========================================================

@st.cache_data
def create_claim_dataset(n=5000):

    np.random.seed(42)

    df = pd.DataFrame({

        "policy_tenure_months":
            np.random.randint(
                3,
                121,
                n
            ),

        "annual_premium":
            np.random.randint(
                10000,
                100001,
                n
            ),

        "pre_existing_conditions":
            np.random.randint(
                0,
                4,
                n
            ),

        "treatment_type":
            np.random.choice(
                [
                    "General",
                    "Emergency",
                    "Surgery",
                    "Chronic",
                    "Maternity"
                ],
                n
            ),

        "hospital_network":
            np.random.choice(
                [
                    "Network",
                    "Out-of-Network"
                ],
                n,
                p=[0.75, 0.25]
            ),

        "claim_amount":
            np.random.randint(
                5000,
                5000001,
                n
            ),

        "policy_coverage_limit":
            np.random.randint(
                100000,
                10000001,
                n
            ),

        "prior_claims_count":
            np.random.randint(
                0,
                7,
                n
            )
    })

    df["claim_to_limit_ratio"] = (
        df["claim_amount"]
        /
        df["policy_coverage_limit"]
    )

    score = np.zeros(n)

    score += np.where(
        df["claim_to_limit_ratio"] <= 0.30,
        1.2,
        np.where(
            df["claim_to_limit_ratio"] <= 0.60,
            0.5,
            np.where(
                df["claim_to_limit_ratio"] <= 1.00,
                -0.2,
                -1.2
            )
        )
    )

    score += np.where(
        df["policy_tenure_months"] >= 60,
        0.8,
        np.where(
            df["policy_tenure_months"] >= 24,
            0.3,
            -0.5
        )
    )

    score += np.where(
        df["pre_existing_conditions"] == 0,
        0.5,
        np.where(
            df["pre_existing_conditions"] == 1,
            0.1,
            -0.5
        )
    )

    score += np.select(
        [
            df["treatment_type"] == "Emergency",
            df["treatment_type"] == "General",
            df["treatment_type"] == "Chronic",
            df["treatment_type"] == "Maternity",
            df["treatment_type"] == "Surgery"
        ],
        [
            0.7,
            0.4,
            -0.1,
            0.2,
            -0.3
        ],
        default=0
    )

    score += np.where(
        df["hospital_network"] == "Network",
        0.6,
        -0.5
    )

    score += np.where(
        df["prior_claims_count"] == 0,
        0.5,
        np.where(
            df["prior_claims_count"] <= 2,
            0.1,
            -0.6
        )
    )

    score += np.random.normal(
        0,
        0.7,
        n
    )

    probability = 1 / (
        1 + np.exp(-score)
    )

    df["claim_approved"] = (
        np.random.random(n) < probability
    ).astype(int)

    return df


claim_df = create_claim_dataset()


# =========================================================
# SHOW GENERATED CLAIM DATA
# =========================================================

with st.expander("View Generated Claim Dataset"):

    st.write(
        "This is the synthetic dataset generated using NumPy and Pandas."
    )

    st.dataframe(
        claim_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# TRAIN CLAIM MODEL
# =========================================================

@st.cache_resource
def train_claim_model(claim_df):

    X_claim = claim_df.drop(
        columns=["claim_approved"]
    )

    y_claim = claim_df["claim_approved"]

    numeric_features = X_claim.select_dtypes(
        include=[
            "int64",
            "float64",
            "int32",
            "float32"
        ]
    ).columns.tolist()

    categorical_features = X_claim.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features
            ),
            (
                "categorical",
                encoder,
                categorical_features
            )
        ]
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_claim,
        y_claim,
        test_size=0.2,
        random_state=42,
        stratify=y_claim
    )

    model.fit(
        X_train,
        y_train
    )

    test_probability = model.predict_proba(
        X_test
    )[:, 1]

    try:
        auc = roc_auc_score(
            y_test,
            test_probability
        )
    except:
        auc = 0.0

    return (
        model,
        X_claim.columns.tolist(),
        auc
    )


claim_model, claim_columns, claim_auc = train_claim_model(
    claim_df
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Claim Assessment")


# =========================================================
# INPUT FORM
# =========================================================

with st.sidebar.form("claim_assessment_form"):

    st.markdown("### Health Information")

    input_data = {}

    for col in X_health.columns:

        label = col.replace(
            "_",
            " "
        ).title()

        # -------------------------------------------------
        # AGE
        # -------------------------------------------------

        if col.strip().lower() == "age":

            min_age = int(
                X_health[col].min()
            )

            max_age = int(
                X_health[col].max()
            )

            default_age = int(
                round(
                    X_health[col].mean()
                )
            )

            input_data[col] = st.slider(
                "Age",
                min_value=min_age,
                max_value=max_age,
                value=default_age,
                step=1,
                format="%d"
            )

        # -------------------------------------------------
        # OTHER NUMERIC FEATURES
        # -------------------------------------------------

        elif col in X_health.select_dtypes(
            include=[
                "int64",
                "float64",
                "int32",
                "float32"
            ]
        ).columns:

            min_value = float(
                X_health[col].min()
            )

            max_value = float(
                X_health[col].max()
            )

            default_value = float(
                X_health[col].mean()
            )

            if min_value == max_value:

                input_data[col] = min_value

            else:

                input_data[col] = st.slider(
                    label,
                    min_value=min_value,
                    max_value=max_value,
                    value=default_value,
                    step=0.1
                )

        # -------------------------------------------------
        # CATEGORICAL FEATURES
        # -------------------------------------------------

        else:

            options = (
                X_health[col]
                .dropna()
                .unique()
                .tolist()
            )

            input_data[col] = st.selectbox(
                label,
                options
            )


    # =====================================================
    # INSURANCE INFORMATION
    # =====================================================

    st.markdown("### Insurance Information")

    policy_tenure = st.number_input(
        "Policy Tenure (Months)",
        min_value=3,
        max_value=120,
        value=36,
        step=1
    )

    annual_premium = st.number_input(
        "Annual Premium",
        min_value=10000,
        max_value=1000000,
        value=30000,
        step=1000
    )

    pre_existing = st.number_input(
        "Pre-existing Conditions",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    treatment = st.selectbox(
        "Treatment Type",
        [
            "General",
            "Emergency",
            "Surgery",
            "Chronic",
            "Maternity"
        ]
    )

    hospital = st.selectbox(
        "Hospital Network",
        [
            "Network",
            "Out-of-Network"
        ]
    )

    claim_amount = st.number_input(
        "Claim Amount",
        min_value=1000,
        max_value=50000000,
        value=100000,
        step=5000,
        format="%d"
    )

    coverage_limit = st.number_input(
        "Policy Coverage Limit",
        min_value=10000,
        max_value=100000000,
        value=500000,
        step=10000,
        format="%d"
    )

    prior_claims = st.number_input(
        "Previous Claims",
        min_value=0,
        max_value=50,
        value=0,
        step=1
    )

    assess = st.form_submit_button(
        "ASSESS CLAIM",
        use_container_width=True
    )


# =========================================================
# CLAIM RATIO
# =========================================================

claim_ratio = (
    float(claim_amount)
    /
    float(coverage_limit)
)


# =========================================================
# MAIN PAGE
# =========================================================

st.title("Claim Assessment")

st.write(
    "Health risk and insurance claim assessment"
)


# =========================================================
# ASSESSMENT
# =========================================================

if assess:

    try:

        # =================================================
        # HEALTH PREDICTION
        # =================================================

        health_input_df = pd.DataFrame(
            [input_data]
        )

        health_input_df = health_input_df[
            X_health.columns
        ]

        health_probability = (
            health_model.predict_proba(
                health_input_df
            )[0]
        )

        health_risk_probability = float(
            health_probability[1]
        )

        high_health_risk = (
            health_risk_probability >= 0.50
        )

        low_health_risk = (
            not high_health_risk
        )


        # =================================================
        # CLAIM MODEL INPUT
        # =================================================

        claim_input = pd.DataFrame({

            "policy_tenure_months":
                [int(policy_tenure)],

            "annual_premium":
                [float(annual_premium)],

            "pre_existing_conditions":
                [int(pre_existing)],

            "treatment_type":
                [treatment],

            "hospital_network":
                [hospital],

            "claim_amount":
                [float(claim_amount)],

            "policy_coverage_limit":
                [float(coverage_limit)],

            "prior_claims_count":
                [int(prior_claims)],

            "claim_to_limit_ratio":
                [float(claim_ratio)]
        })

        claim_input = claim_input[
            claim_columns
        ]


        # =================================================
        # CLAIM MODEL PREDICTION
        # =================================================

        claim_probability = (
            claim_model.predict_proba(
                claim_input
            )[0]
        )

        claim_approval_probability = float(
            claim_probability[1]
        )


        # =================================================
        # COVERAGE RULE
        # =================================================

        if claim_ratio > 1:

            claim_approved = False

        else:

            claim_approved = True


        # =================================================
        # FINAL DECISION
        # =================================================

        final_approved = (
            low_health_risk
            and
            claim_approved
        )


        # =================================================
        # RESULTS
        # =================================================

        st.markdown("---")

        st.header("Assessment Result")

        col1, col2, col3 = st.columns(3)


        # =================================================
        # HEALTH
        # =================================================

        with col1:

            st.subheader("Health Risk")

            st.metric(
                "Risk Probability",
                f"{health_risk_probability * 100:.2f}%"
            )

            if high_health_risk:

                st.error(
                    "HIGH HEALTH RISK"
                )

            else:

                st.success(
                    "LOW HEALTH RISK"
                )


        # =================================================
        # CLAIM
        # =================================================

        with col2:

            st.subheader("Claim Assessment")

            st.metric(
                "Claim / Coverage",
                f"{claim_ratio * 100:.2f}%"
            )

            if claim_ratio > 1:

                st.error(
                    "NOT APPROVED"
                )

                st.write(
                    "Claim amount exceeds policy coverage."
                )

            else:

                st.success(
                    "APPROVED"
                )

                st.write(
                    "Claim amount is within policy coverage."
                )


        # =================================================
        # FINAL
        # =================================================

        with col3:

            st.subheader("Final Decision")

            if final_approved:

                st.success(
                    "CLAIM APPROVED"
                )

                st.write(
                    "Both criteria passed."
                )

            else:

                st.error(
                    "CLAIM NOT APPROVED"
                )

                if not low_health_risk:

                    st.write(
                        "Health-risk criterion failed."
                    )

                if claim_ratio > 1:

                    st.write(
                        "Claim exceeds coverage limit."
                    )


        # =================================================
        # CLAIM DETAILS
        # =================================================

        st.markdown("---")

        st.subheader("Claim Details")

        d1, d2, d3 = st.columns(3)

        with d1:

            st.metric(
                "Claim Amount",
                f"₹{claim_amount:,.0f}"
            )

        with d2:

            st.metric(
                "Coverage Limit",
                f"₹{coverage_limit:,.0f}"
            )

        with d3:

            st.metric(
                "Claim / Coverage Ratio",
                f"{claim_ratio:.2f}"
            )


        # =================================================
        # DECISION BREAKDOWN
        # =================================================

        st.markdown("---")

        st.subheader("Decision Breakdown")

        result_table = pd.DataFrame({

            "Criterion": [
                "Health Risk",
                "Claim vs Coverage",
                "Final Decision"
            ],

            "Result": [

                "PASS"
                if low_health_risk
                else "FAIL",

                "APPROVED"
                if claim_ratio <= 1
                else "NOT APPROVED",

                "APPROVED"
                if final_approved
                else "NOT APPROVED"
            ]
        })

        st.dataframe(
            result_table,
            use_container_width=True,
            hide_index=True
        )


    except Exception as e:

        st.error(
            "Something went wrong while processing the claim."
        )

        st.exception(e)


else:

    st.info(
        "Enter the details and click "
        "**ASSESS CLAIM** to process the claim."
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

with st.expander("Model Information"):

    st.write("### Health Risk Model")

    st.write(
        "Random Forest trained on the original "
        "stroke_prediction.csv dataset."
    )

    st.write(
        f"Health Model ROC-AUC: {health_auc:.3f}"
    )

    st.write(
        "Age handling is preserved from the original model."
    )

    st.write("### Claim Model")

    st.write(
        "Random Forest using policy, treatment, "
        "hospital, claim and coverage information."
    )

    st.write(
        f"Claim Model ROC-AUC: {claim_auc:.3f}"
    )

    st.write("### Final Rule")

    st.code(
        """
Claim / Coverage > 1
    NOT APPROVED

Claim / Coverage <= 1
    APPROVED

Final Approval =
Low Health Risk AND Claim/Coverage <= 1
"""
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Claim Assessment Prototype"
)
