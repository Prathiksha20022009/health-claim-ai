import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ClaimTriage AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_and_prepare_data():

    df = pd.read_csv("stroke_prediction.csv")

    # Remove missing values
    df = df.dropna()

    # Remove ID column if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Make sure target exists
    if "stroke" not in df.columns:
        st.error("The dataset must contain a 'stroke' column.")
        st.stop()

    # Target variable
    df["target_flag"] = df["stroke"]

    # Remove original target
    df = df.drop(columns=["stroke"])

    return df


# =========================================================
# LOAD DATA
# =========================================================

df = load_and_prepare_data()

X = df.drop(columns=["target_flag"])
y = df["target_flag"]


# =========================================================
# IDENTIFY COLUMN TYPES
# =========================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


# =========================================================
# PREPROCESSING
# =========================================================

try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
except TypeError:
    # For older versions of scikit-learn
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


# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=150,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# TRAIN MODEL
# =========================================================

model.fit(X_train, y_train)


# =========================================================
# MODEL EVALUATION
# =========================================================

y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)[:, 1]

try:
    roc_auc = roc_auc_score(y_test, y_test_prob)
except:
    roc_auc = 0.0


# =========================================================
# HEADER
# =========================================================

st.title("⚡ ClaimTriage Intelligence")

st.caption(
    "Clinical Dataset Integration • AI-Powered Health Risk Assessment"
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Patient Information")
st.sidebar.caption("Enter patient parameters")

input_data = {}


for col in X.columns:

    # -----------------------------------------------------
    # NUMERIC VARIABLES
    # -----------------------------------------------------

    if col in numeric_features:

        # AGE MUST BE AN INTEGER
        if col.strip().lower() == "age":

            min_age = int(X[col].min())
            max_age = int(X[col].max())
            default_age = int(round(X[col].mean()))

            input_data[col] = st.sidebar.slider(
                "Age",
                min_value=min_age,
                max_value=max_age,
                value=default_age,
                step=1,
                format="%d"
            )

        # Other numerical variables
        else:

            min_value = float(X[col].min())
            max_value = float(X[col].max())
            default_value = float(X[col].mean())

            input_data[col] = st.sidebar.slider(
                col.replace("_", " ").title(),
                min_value=min_value,
                max_value=max_value,
                value=default_value,
                step=0.1
            )

    # -----------------------------------------------------
    # CATEGORICAL VARIABLES
    # -----------------------------------------------------

    elif col in categorical_features:

        options = X[col].dropna().unique().tolist()

        input_data[col] = st.sidebar.selectbox(
            col.replace("_", " ").title(),
            options
        )


# =========================================================
# PREDICTION BUTTON
# =========================================================

run_prediction = st.sidebar.button(
    "🔍 Execute Risk Assessment",
    use_container_width=True
)


# =========================================================
# TOP INFORMATION
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model ROC-AUC",
        f"{roc_auc:.3f}"
    )

with col2:
    st.metric(
        "Dataset Size",
        f"{len(df):,}"
    )

with col3:
    st.metric(
        "Engine Status",
        "ONLINE"
    )


st.divider()


# =========================================================
# MAIN CONTENT
# =========================================================

st.header("Live Clinical Risk Profiling & Inference")


if run_prediction:

    # Convert entered data into DataFrame
    input_df = pd.DataFrame([input_data])

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0]

    # Probability of class 1
    risk_probability = probability[1]


    # =====================================================
    # RESULT SECTION
    # =====================================================

    result_col, chart_col = st.columns([1.1, 1])


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    with result_col:

        if prediction == 1:

            st.error("🔴 HIGH HEALTH RISK")

            st.write(
                "The model has flagged the patient as having "
                "a higher predicted likelihood of stroke based "
                "on the entered parameters."
            )

        else:

            st.success("🟢 LOW HEALTH RISK")

            st.write(
                "The patient is not flagged by the model. "
                "The model predicts a lower likelihood of stroke "
                "based on the entered parameters."
            )


    # -----------------------------------------------------
    # RISK CHART
    # -----------------------------------------------------

    with chart_col:

        st.subheader("Risk Confidence Distribution")

        chart_values = [
            probability[0],
            probability[1]
        ]

        fig, ax = plt.subplots()

        ax.barh(
            ["Low Risk", "High Risk"],
            chart_values
        )

        ax.set_xlim(0, 1)

        ax.set_xlabel("Probability")

        ax.set_title("Model Risk Distribution")

        st.pyplot(fig)

        plt.close(fig)


    # =====================================================
    # RISK PROBABILITY
    # =====================================================

    st.subheader("Calculated Health Risk Probability")

    st.metric(
        "Stroke Risk Probability",
        f"{risk_probability:.1%}"
    )


else:

    st.info(
        "Enter the patient information in the sidebar "
        "and click **Execute Risk Assessment**."
    )


# =========================================================
# ADVANCED MODEL DIAGNOSTICS
# =========================================================

with st.expander("🔬 Advanced Model Diagnostics & Confusion Matrix"):

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_test_pred,
        output_dict=False
    )

    st.text(report)


    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_test_pred
    )

    fig2, ax2 = plt.subplots()

    ax2.imshow(cm)

    ax2.set_title("Confusion Matrix")

    ax2.set_xlabel("Predicted")

    ax2.set_ylabel("Actual")

    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])

    ax2.set_xticklabels(["Low Risk", "High Risk"])
    ax2.set_yticklabels(["Low Risk", "High Risk"])

    # Display numbers inside matrix
    for i in range(2):
        for j in range(2):
            ax2.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    st.pyplot(fig2)

    plt.close(fig2)
