import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="ClaimTriage AI | Clinical Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# ELITE VIBE-CODED CSS & STYLING
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #f8fafc;
        background-color: #07090e;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.3);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.03em;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    [data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        border: none;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    </style>

    <div class="hero-container">
        <div class="hero-title">ClaimTriage Intelligence</div>
        <div class="hero-subtitle">Official Clinical Dataset Integration — Health Risk & Underwriting Engine</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 1. LOAD OFFICIAL DATASET
# ---------------------------------------------------------
@st.cache_data
def load_and_prepare_data():
  try:
    df = pd.read_csv("stroke_prediction.csv")
  except FileNotFoundError:
    st.error(
        "Error: 'stroke_prediction.csv' not found in your folder. Make sure"
        " it's in your workspace directory!"
    )
    st.stop()

  df = df.dropna()
  if "id" in df.columns:
    df = df.drop(columns=["id"])

  if "stroke" in df.columns:
    df["target_flag"] = df["stroke"]
    df = df.drop(columns=["stroke"])

  return df


df = load_and_prepare_data()

# ---------------------------------------------------------
# 2. MACHINE LEARNING PIPELINE DYNAMIC SETUP
# ---------------------------------------------------------
X = df.drop(columns=["target_flag"])
y = df["target_flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
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
                n_estimators=150, class_weight="balanced", random_state=42
            ),
        ),
    ]
)

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
y_prob = model_pipeline.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

# ---------------------------------------------------------
# 3. DYNAMIC SIDEBAR CONTROLS BASED ON CSV COLUMNS
# ---------------------------------------------------------
st.sidebar.markdown(
    "### **Patient Parameter Config**\nAdjust clinical inputs based on official"
    " schema."
)

input_data = {}
for col in numeric_features:
  min_val = float(X[col].min())
  max_val = float(X[col].max())
  mean_val = float(X[col].mean())
  input_data[col] = st.sidebar.slider(
      f"{col.replace('_', ' ').title()}", min_val, max_val, mean_val
  )

for col in categorical_features:
  unique_vals = X[col].unique().tolist()
  input_data[col] = st.sidebar.selectbox(
      f"{col.replace('_', ' ').title()}", unique_vals
  )

input_df = pd.DataFrame([input_data])

# ---------------------------------------------------------
# 4. DASHBOARD LAYOUT & INFERENCE
# ---------------------------------------------------------
c1, c2, c3 = st.columns(3)
with c1:
  st.metric(
      label="MODEL ROC-AUC",
      value=f"{roc_auc:.4f}",
      delta="Official Dataset Pipeline",
  )
with c2:
  st.metric(
      label="DATASET SIZE", value=f"{len(df)} Records", delta="Official Source"
  )
with c3:
  st.metric(label="ENGINE STATUS", value="Live / Ready", delta="100% Uptime")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### **Live Clinical Risk Profiling & Inference**")

if st.sidebar.button("Execute Risk Assessment", type="primary"):
  prediction = model_pipeline.predict(input_df)[0]
  prediction_proba = model_pipeline.predict_proba(input_df)[0][1]

  res_col1, res_col2 = st.columns([1.2, 1])

  with res_col1:
    if prediction == 1:
      st.error(
          "**HIGH HEALTH RISK — FLAG FOR CASE MANAGEMENT**\n\nClinical"
          " indicators exceed safety thresholds. Immediate medical review"
          " recommended."
      )
    else:
      st.success(
          "**LOW CLINICAL RISK — AUTO-CLEARED**\n\nPatient metrics clear all"
          " automated risk and triage filters."
      )

    st.metric(
        label="Calculated Health Risk Probability",
        value=f"{prediction_proba * 100:.1f}%",
    )

  with res_col2:
    fig, ax = plt.subplots(figsize=(4, 2.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    categories = ["Low Risk", "High Risk"]
    probabilities = [1 - prediction_proba, prediction_proba]
    colors = ["#10b981", "#ef4444" if prediction == 1 else "#3b82f6"]

    ax.barh(categories, probabilities, color=colors, height=0.5)
    ax.set_xlim(0, 1.0)
    ax.tick_params(colors="white", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")
    plt.title("Risk Confidence Distribution", color="white", fontsize=10, pad=10)

    st.pyplot(fig)

else:
  st.info(
      "Adjust patient parameters in the left sidebar and click **Execute Risk"
      " Assessment** to run live model inference."
  )

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("Advanced Model Diagnostics & Confusion Matrix"):
  diag_col1, diag_col2 = st.columns(2)
  with diag_col1:
    st.markdown("**Classification Metrics**")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
  with diag_col2:
    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        columns=["Pred: Low Risk", "Pred: High Risk"],
        index=["Actual: Low Risk", "Actual: High Risk"],
    )
    st.dataframe(cm_df, use_container_width=True)
