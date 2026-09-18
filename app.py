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
# DARK UI
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #151922;
    border-right: 1px solid #292e38;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

h1, h2, h3 {
    color: white !important;
}

p, label {
    color: #d5d8de;
}

div[data-testid="stMetric"] {
    background-color: #151922;
    border: 1px solid #292e38;
    border-radius: 10px;
    padding: 15px;
}

div[data-testid="stMetricLabel"] {
    color: #9da4b0 !important;
}

div[data-testid="stMetricValue"] {
    color: white !important;
}

.stButton > button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1d4ed8;
}

div[data-baseweb="select"] > div {
    background-color: #1b202b;
    border-color: #343a46;
}

div[data-testid="stExpander"] {
    background-color: #151922;
    border: 1px solid #292e38;
    border-radius: 8px;
}

hr {
    border-color: #292e38;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_and_prepare_data():

    df = pd.read_csv("stroke_prediction.csv")

    df = df.dropna()

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    if "stroke" not in df.columns:
        st.error("The dataset must contain a 'stroke' column.")
        st.stop()

    # Original target
    df["target_flag"] = df["stroke"]

    df = df.drop(columns=["stroke"])

    return df


df = load_and_prepare_data()


# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df.drop(columns=["target_flag"])

y = df["target_flag"]


numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()


categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


# =========================================================
# ONE HOT ENCODER
# =========================================================

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


# =========================================================
# PREPROCESSOR
# =========================================================

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
# RANDOM FOREST MODEL
# =========================================================

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
                random_state=42
            )
        )

    ]
)


# =========================================================
# TRAIN TEST SPLIT
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

model.fit(
    X_train,
    y_train
)


# =========================================================
# MODEL TESTING
# =========================================================

y_test_pred = model.predict(X_test)

y_test_prob = model.predict_proba(X_test)[:, 1]


try:

    roc_auc = roc_auc_score(
        y_test,
        y_test_prob
    )

except:

    roc_auc = 0.0


# =========================================================
# TITLE
# =========================================================

st.title("⚡ ClaimTriage AI")

st.caption(
    "AI-powered insurance claim assessment"
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📋 Claim Information")

st.sidebar.caption(
    "Enter customer information"
)


input_data = {}


for col in X.columns:

    # -----------------------------------------------------
    # NUMERIC FEATURES
    # -----------------------------------------------------

    if col in numeric_features:

        # AGE
        if col.strip().lower() == "age":

            min_age = int(X[col].min())

            max_age = int(X[col].max())

            default_age = int(
                round(X[col].mean())
            )

            input_data[col] = st.sidebar.slider(

                "Age",

                min_value=min_age,

                max_value=max_age,

                value=default_age,

                step=1,

                format="%d"

            )


        # TENURE
        elif col.strip().lower() == "tenure":

            min_tenure = int(X[col].min())

            max_tenure = int(X[col].max())

            default_tenure = int(
                round(X[col].mean())
            )

            input_data[col] = st.sidebar.slider(

                "Policy Tenure",

                min_value=min_tenure,

                max_value=max_tenure,

                value=default_tenure,

                step=1,

                format="%d"

            )


        # OTHER NUMERIC FEATURES
        else:

            min_value = float(
                X[col].min()
            )

            max_value = float(
                X[col].max()
            )

            default_value = float(
                X[col].mean()
            )

            input_data[col] = st.sidebar.slider(

                col.replace(
                    "_",
                    " "
                ).title(),

                min_value=min_value,

                max_value=max_value,

                value=default_value,

                step=0.1

            )


    # -----------------------------------------------------
    # CATEGORICAL FEATURES
    # -----------------------------------------------------

    elif col in categorical_features:

        options = (
            X[col]
            .dropna()
            .unique()
            .tolist()
        )

        input_data[col] = st.sidebar.selectbox(

            col.replace(
                "_",
                " "
            ).title(),

            options

        )


# =========================================================
# BUTTON
# =========================================================

st.sidebar.divider()

run_prediction = st.sidebar.button(

    "⚡ Assess Claim",

    use_container_width=True

)


# =========================================================
# TOP METRICS
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Model ROC-AUC",
        f"{roc_auc:.3f}"
    )


with col2:

    st.metric(
        "Total Records",
        f"{len(df):,}"
    )


with col3:

    st.metric(
        "System",
        "ONLINE"
    )


st.divider()


# =========================================================
# CLAIM ASSESSMENT
# =========================================================

st.header(
    "Insurance Claim Assessment"
)


if run_prediction:

    input_df = pd.DataFrame(
        [input_data]
    )


    # -----------------------------------------------------
    # ORIGINAL MODEL PROBABILITIES
    # -----------------------------------------------------

    probability = model.predict_proba(
        input_df
    )[0]


    original_not_approved_probability = probability[0]

    original_approved_probability = probability[1]


    # =====================================================
    # REVERSE THE ORIGINAL DECISION
    #
    # ORIGINAL:
    # approved     -> APPROVED
    # not approved -> NOT APPROVED
    #
    # NEW:
    # approved     -> NOT APPROVED
    # not approved -> APPROVED
    # =====================================================

    if original_approved_probability > original_not_approved_probability:

        # Original model said APPROVED
        # New decision must be NOT APPROVED

        claim_approved = False

    else:

        # Original model said NOT APPROVED
        # New decision must be APPROVED

        claim_approved = True


    # =====================================================
    # DISPLAY
    # =====================================================

    result_col, graph_col = st.columns(
        [1, 1]
    )


    # =====================================================
    # DECISION
    # =====================================================

    with result_col:

        st.subheader(
            "Decision"
        )


        if claim_approved:

            st.success(
                "🟢 CLAIM APPROVED"
            )

        else:

            st.error(
                "🔴 CLAIM NOT APPROVED"
            )


        st.divider()


        # -------------------------------------------------
        # DISPLAY THE REVERSED DECISION PROBABILITIES
        # -------------------------------------------------

        reversed_approved_probability = (
            original_not_approved_probability
        )

        reversed_not_approved_probability = (
            original_approved_probability
        )


        st.metric(

            "Claim Approval Probability",

            f"{reversed_approved_probability:.1%}"

        )


        st.metric(

            "Claim Not Approval Probability",

            f"{reversed_not_approved_probability:.1%}"

        )


    # =====================================================
    # GRAPH
    # =====================================================

    with graph_col:

        st.subheader(
            "Claim Decision Probability"
        )


        values = [

            reversed_not_approved_probability,

            reversed_approved_probability

        ]


        labels = [

            "Claim Not Approved",

            "Claim Approved"

        ]


        fig, ax = plt.subplots()


        fig.patch.set_facecolor(
            "#151922"
        )

        ax.set_facecolor(
            "#151922"
        )


        ax.barh(
            labels,
            values
        )


        ax.set_xlim(
            0,
            1
        )


        ax.set_xlabel(
            "Probability",
            color="white"
        )


        ax.set_title(
            "Claim Decision Distribution",
            color="white"
        )


        ax.tick_params(
            colors="white"
        )


        for spine in ax.spines.values():

            spine.set_visible(False)


        for i, value in enumerate(values):

            ax.text(

                value + 0.02,

                i,

                f"{value:.1%}",

                va="center",

                color="white"

            )


        st.pyplot(
            fig
        )


        plt.close(
            fig
        )


else:

    st.info(

        "Enter the claim information on the left "
        "and click **Assess Claim**."

    )


# =========================================================
# MODEL DIAGNOSTICS
# =========================================================

with st.expander(
    "🔬 Model Diagnostics"
):


    st.subheader(
        "Classification Report"
    )


    report = classification_report(

        y_test,

        y_test_pred,

        output_dict=False

    )


    st.text(
        report
    )


    st.subheader(
        "Confusion Matrix"
    )


    cm = confusion_matrix(

        y_test,

        y_test_pred

    )


    fig2, ax2 = plt.subplots()


    fig2.patch.set_facecolor(
        "#151922"
    )

    ax2.set_facecolor(
        "#151922"
    )


    ax2.imshow(
        cm
    )


    ax2.set_title(

        "Claim Decision Confusion Matrix",

        color="white"

    )


    ax2.set_xlabel(
        "Predicted",
        color="white"
    )


    ax2.set_ylabel(
        "Actual",
        color="white"
    )


    ax2.set_xticks(
        [0, 1]
    )


    ax2.set_yticks(
        [0, 1]
    )


    ax2.set_xticklabels(

        [
            "Not Approved",
            "Approved"
        ],

        color="white"

    )


    ax2.set_yticklabels(

        [
            "Not Approved",
            "Approved"
        ],

        color="white"

    )


    for i in range(2):

        for j in range(2):

            ax2.text(

                j,

                i,

                cm[i, j],

                ha="center",

                va="center",

                color="white"

            )


    st.pyplot(
        fig2
    )


    plt.close(
        fig2
    )
