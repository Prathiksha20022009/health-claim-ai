"""ClaimTriage Enterprise Hub - Streamlit frontend.

Presentation layer only. All request payloads, endpoints, eligibility rules
and response handling are unchanged from the original version.
"""

import base64
import html

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/api/adjudicate"

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ClaimTriage | Enterprise Health Platform",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Illustrations (inline SVG, so no external image files are required)
# ---------------------------------------------------------------------------
HERO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 180" fill="none">
  <rect x="150" y="20" width="190" height="140" rx="14" fill="#ffffff" fill-opacity="0.06" stroke="#ffffff" stroke-opacity="0.18"/>
  <rect x="168" y="38" width="64" height="8" rx="4" fill="#ffffff" fill-opacity="0.35"/>
  <rect x="168" y="54" width="110" height="6" rx="3" fill="#ffffff" fill-opacity="0.15"/>
  <rect x="172" y="118" width="16" height="26" rx="3" fill="#ffffff" fill-opacity="0.16"/>
  <rect x="200" y="106" width="16" height="38" rx="3" fill="#ffffff" fill-opacity="0.16"/>
  <rect x="228" y="112" width="16" height="32" rx="3" fill="#ffffff" fill-opacity="0.16"/>
  <rect x="256" y="96" width="16" height="48" rx="3" fill="#ffffff" fill-opacity="0.16"/>
  <rect x="284" y="102" width="16" height="42" rx="3" fill="#ffffff" fill-opacity="0.16"/>
  <rect x="312" y="84" width="16" height="60" rx="3" fill="#ffffff" fill-opacity="0.16"/>
  <polyline points="180,100 208,88 236,94 264,74 292,80 320,62" stroke="#5eead4" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="320" cy="62" r="5" fill="#5eead4"/>
  <path d="M100 28 L152 46 V90 C152 118 128 136 100 148 C72 136 48 118 48 90 V46 Z" fill="#12324a" stroke="#5eead4" stroke-width="2.5" stroke-linejoin="round"/>
  <rect x="92" y="62" width="16" height="48" rx="3" fill="#5eead4"/>
  <rect x="76" y="78" width="48" height="16" rx="3" fill="#5eead4"/>
</svg>
"""

PATIENT_ICON_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96">
  <rect width="96" height="96" rx="20" fill="#e3f2ef"/>
  <circle cx="44" cy="34" r="13" fill="#0f6b66"/>
  <path d="M18 78 C18 62 30 54 44 54 C58 54 70 62 70 78 Z" fill="#0f6b66"/>
  <circle cx="70" cy="68" r="14" fill="#ffffff" stroke="#13858a" stroke-width="3"/>
  <path d="M63 68 L68 73 L77 63" stroke="#13858a" stroke-width="3.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

HOSPITAL_ICON_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96">
  <rect width="96" height="96" rx="20" fill="#e6ecf7"/>
  <rect x="26" y="22" width="44" height="54" rx="3" fill="#1b365d"/>
  <rect x="44" y="30" width="8" height="18" rx="1" fill="#ffffff"/>
  <rect x="39" y="35" width="18" height="8" rx="1" fill="#ffffff"/>
  <rect x="32" y="54" width="8" height="8" rx="1" fill="#8fb0dc"/>
  <rect x="56" y="54" width="8" height="8" rx="1" fill="#8fb0dc"/>
  <rect x="42" y="58" width="12" height="18" rx="1" fill="#8fb0dc"/>
  <rect x="18" y="76" width="60" height="3" rx="1.5" fill="#1b365d"/>
</svg>
"""

EMPTY_STATE_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 150" fill="none">
  <circle cx="120" cy="76" r="62" fill="#eef2f7"/>
  <rect x="82" y="22" width="76" height="102" rx="8" fill="#ffffff" stroke="#c5d0de" stroke-width="2"/>
  <rect x="102" y="14" width="36" height="14" rx="4" fill="#1b365d"/>
  <rect x="94" y="44" width="52" height="6" rx="3" fill="#c5d0de"/>
  <rect x="94" y="58" width="40" height="6" rx="3" fill="#dbe3ee"/>
  <rect x="94" y="72" width="48" height="6" rx="3" fill="#dbe3ee"/>
  <rect x="94" y="86" width="30" height="6" rx="3" fill="#dbe3ee"/>
  <circle cx="156" cy="108" r="20" fill="#13858a"/>
  <path d="M147 108 L154 115 L166 101" stroke="#ffffff" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');

body .stApp { font-family: 'IBM Plex Sans', 'Segoe UI', -apple-system, Helvetica, Arial, sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.5rem; max-width: 1200px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid rgba(148,163,184,.35); }
.stTabs [data-baseweb="tab"] { height: 46px; padding: 0 18px; }
.stTabs [data-baseweb="tab"] p { font-size: 15px; font-weight: 500; }

/* Widget labels */
[data-testid="stWidgetLabel"] p { font-size: 13.5px; font-weight: 500; }

/* Buttons */
div.stButton > button {
    background: #13858a; color: #ffffff; border: 1px solid #13858a;
    border-radius: 8px; font-weight: 500; padding: .55rem 1.5rem;
}
div.stButton > button:hover { background: #0f6b66; border-color: #0f6b66; color: #ffffff; }
div.stButton > button p { color: #ffffff; }
[data-testid="stSidebar"] div.stButton > button { width: 100%; }

/* Metrics */
[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #dfe6ee; border-radius: 10px; padding: 14px 16px;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color: #5b6b7f; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div { color: #0b2545; font-weight: 600; }

/* Hero */
.ct-hero {
    display: flex; align-items: center; justify-content: space-between; gap: 24px;
    background: #0b2545; border-radius: 14px; padding: 26px 34px; margin-bottom: 26px;
}
.stApp .ct-hero h1 { margin: 0; padding: 0; font-size: 27px; font-weight: 600; color: #ffffff; }
.stApp .ct-hero p { margin: 8px 0 0 0; color: #a9b8cf; font-size: 14.5px; max-width: 560px; line-height: 1.5; }
.ct-hero img { height: 128px; flex-shrink: 0; }
@media (max-width: 800px) { .ct-hero img { display: none; } }

/* Section headers */
.ct-section { display: flex; align-items: center; gap: 16px; margin: 6px 0 22px 0; }
.ct-section img { width: 56px; height: 56px; flex-shrink: 0; }
.stApp .ct-section h2 { margin: 0; padding: 0; font-size: 21px; font-weight: 600; }
.stApp .ct-section p { margin: 3px 0 0 0; font-size: 14px; opacity: .7; }

/* Group labels and sub headings */
.ct-group { font-size: 14px; font-weight: 600; margin: 4px 0 12px 0; padding-bottom: 8px; border-bottom: 1px solid rgba(148,163,184,.35); }
.ct-subhead { font-size: 17px; font-weight: 600; margin: 20px 0 8px 0; }

/* Result cards */
.ct-card {
    background: #ffffff; color: #0b2545; border: 1px solid #dfe6ee;
    border-left: 4px solid var(--accent, #13858a); border-radius: 8px;
    padding: 16px 20px; margin: 12px 0 0 0;
}
.ct-green { --accent: #1a8a5a; }
.ct-blue  { --accent: #2b5cab; }
.ct-amber { --accent: #b9770e; }
.ct-red   { --accent: #b3382c; }
.ct-tag { font-size: 12.5px; font-weight: 600; color: var(--accent); }
.stApp .ct-card h4 { margin: 3px 0 12px 0; padding: 0; font-size: 16.5px; font-weight: 600; color: #0b2545; line-height: 1.35; }
.ct-grid { display: grid; grid-template-columns: 140px 1fr; gap: 7px 14px; font-size: 13.5px; }
.ct-k { color: #5b6b7f; }
.ct-v { color: #0b2545; word-break: break-word; }
.ct-mono { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace; background: #eef2f7; padding: 1px 6px; border-radius: 4px; font-size: 12.5px; }
.ct-list { margin: 0; padding-left: 18px; font-size: 13.5px; line-height: 1.75; color: #0b2545; }

/* Empty state */
.ct-empty { text-align: center; padding: 34px 20px; border: 1px dashed rgba(148,163,184,.6); border-radius: 12px; }
.ct-empty img { height: 130px; margin-bottom: 8px; }
.stApp .ct-empty h4 { margin: 0 0 4px 0; padding: 0; font-size: 16.5px; font-weight: 600; }
.stApp .ct-empty p { margin: 0 auto; max-width: 480px; font-size: 14px; opacity: .7; line-height: 1.5; }

/* Sidebar header */
.ct-side-head { display: flex; align-items: center; gap: 12px; margin: 0 0 18px 0; }
.ct-side-head img { width: 40px; height: 40px; flex-shrink: 0; }
.stApp .ct-side-head h3 { margin: 0; padding: 0; font-size: 16px; font-weight: 600; line-height: 1.3; }
</style>
"""


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def esc(value) -> str:
    """HTML-escape any value so backend responses can be embedded safely."""
    return html.escape(str(value))


def render_html(markup: str) -> None:
    """Render an HTML fragment.

    Leading whitespace and blank lines are stripped so Markdown never treats
    indented lines as a code block.
    """
    compact = "\n".join(
        line.strip() for line in markup.strip().splitlines() if line.strip()
    )
    st.markdown(compact, unsafe_allow_html=True)


def svg_data_uri(svg: str) -> str:
    encoded = base64.b64encode(svg.strip().encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def section_header(icon_svg: str, title: str, description: str) -> None:
    render_html(
        f"""
        <div class="ct-section">
        <img src="{svg_data_uri(icon_svg)}" alt="">
        <div><h2>{esc(title)}</h2><p>{esc(description)}</p></div>
        </div>
        """
    )


def group_label(text: str) -> None:
    render_html(f'<div class="ct-group">{esc(text)}</div>')


def subhead(text: str) -> None:
    render_html(f'<div class="ct-subhead">{esc(text)}</div>')


def info_card(tag: str, title: str, rows: list, accent: str) -> None:
    """Card with a label/value grid. Rows are (label, value[, monospace])."""
    cells = []
    for label, value, *mono in rows:
        text = esc(value)
        if mono and mono[0]:
            text = f'<span class="ct-mono">{text}</span>'
        cells.append(
            f'<div class="ct-k">{esc(label)}</div><div class="ct-v">{text}</div>'
        )
    cells_html = "".join(cells)
    render_html(
        f"""
        <div class="ct-card ct-{accent}">
        <div class="ct-tag">{esc(tag)}</div>
        <h4>{esc(title)}</h4>
        <div class="ct-grid">{cells_html}</div>
        </div>
        """
    )


def list_card(tag: str, title: str, items: list, accent: str) -> None:
    bullets = "".join(f"<li>{esc(item)}</li>" for item in items)
    render_html(
        f"""
        <div class="ct-card ct-{accent}">
        <div class="ct-tag">{esc(tag)}</div>
        <h4>{esc(title)}</h4>
        <ul class="ct-list">{bullets}</ul>
        </div>
        """
    )


def empty_state(title: str, message: str) -> None:
    render_html(
        f"""
        <div class="ct-empty">
        <img src="{svg_data_uri(EMPTY_STATE_SVG)}" alt="">
        <h4>{esc(title)}</h4>
        <p>{esc(message)}</p>
        </div>
        """
    )


# ---------------------------------------------------------------------------
# Global styles and hero banner
# ---------------------------------------------------------------------------
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

render_html(
    f"""
    <div class="ct-hero">
    <div>
    <h1>ClaimTriage Enterprise Hub</h1>
    <p>Decoupled Unified Health Aggregation &amp; Autonomous Adjudication Engine</p>
    </div>
    <img src="{svg_data_uri(HERO_SVG)}" alt="">
    </div>
    """
)

# Focused 2-tab layout (user-facing portals only)
tab1, tab2 = st.tabs(
    [
        "Patient Portal (Scheme Matching)",
        "Hospital Portal (Claim Adjudication)",
    ]
)

# =====================================================================
# TAB 1: PATIENT PORTAL
# =====================================================================
with tab1:
    section_header(
        PATIENT_ICON_SVG,
        "Patient Eligibility & Welfare Scheme Discovery",
        "Evaluate public safety nets and private commercial insurance"
        " side-by-side.",
    )

    group_label("Patient profile")
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
        p_location = st.text_input(
            "State / Region Pincode", "110001", key="p_loc"
        )

    if st.button("Check Scheme Eligibility", key="btn_scheme"):
        subhead("Available Coverage & Scheme Matches")
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            if p_income <= 500000:
                info_card(
                    "Public safety net",
                    "Ayushman Bharat PM-JAY",
                    [
                        ("Coverage", "Up to ₹5,00,000 Cashless Cover"),
                        ("Match", "100% Eligible"),
                    ],
                    "green",
                )
            else:
                info_card(
                    "Private insurance",
                    "Standard Private Health Shield",
                    [
                        ("Coverage", "Customizable up to ₹1,000,000"),
                        ("Match", "Standard Market"),
                    ],
                    "blue",
                )
        with c_s2:
            if p_age >= 60:
                info_card(
                    "Senior citizen",
                    "Senior Citizen Health Protection Plan",
                    [
                        ("Coverage", "Specialized Geriatric Care"),
                        ("Match", "Age Demographic Match"),
                    ],
                    "green",
                )
            else:
                info_card(
                    "Family coverage",
                    "Comprehensive Family Floater",
                    [
                        ("Coverage", "Outpatient & Inpatient Shield"),
                        ("Match", "Fully Qualified"),
                    ],
                    "green",
                )
    else:
        empty_state(
            "No results yet",
            "Enter the patient profile above, then select Check Scheme"
            " Eligibility to see matching schemes.",
        )

# =====================================================================
# TAB 2: HOSPITAL PORTAL (Adjudication Engine)
# =====================================================================
with tab2:
    section_header(
        HOSPITAL_ICON_SVG,
        "Hospital Billing & Autonomous Adjudication Portal",
        "Submit patient clinical data and claim financials for automated"
        " review.",
    )

    with st.sidebar:
        render_html(
            f"""
            <div class="ct-side-head">
            <img src="{svg_data_uri(HOSPITAL_ICON_SVG)}" alt="">
            <h3>Hospital Submission Controls</h3>
            </div>
            """
        )

        group_label("Patient demographics")
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

        group_label("Clinical history")
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
        in_smoking = st.selectbox(
            "Smoking History", ["never", "former", "smokes"]
        )

        group_label("Claim and policy")
        in_claim = st.number_input(
            "Claim Request Amount ($)", 100.0, 100000.0, 4500.0
        )
        in_limit = st.selectbox(
            "Policy Coverage Limit ($)", [5000, 10000, 25000, 50000], index=1
        )
        in_tenure = st.slider("Policy Tenure (Months)", 1, 60, 18)

        submit_adjudication = st.button(
            "Run Adjudication Review", type="primary"
        )

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
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    res_data = response.json()
                    res_col1, res_col2 = st.columns(2)

                    with res_col1:
                        subhead("Clinical Risk Audit")
                        risk_data = res_data["clinical_assessment"]
                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric("Risk Tier", risk_data["risk_tier"])
                        with m2:
                            st.metric(
                                "Risk Score", risk_data["total_risk_score"]
                            )

                        if risk_data["clinical_flags"]:
                            list_card(
                                "Attention required",
                                "Active Clinical Flags",
                                risk_data["clinical_flags"],
                                "amber",
                            )
                        else:
                            info_card(
                                "Clear",
                                "Clinical Baseline",
                                [
                                    (
                                        "Result",
                                        "Patient metrics clear standard WHO"
                                        " baselines.",
                                    )
                                ],
                                "green",
                            )

                    with res_col2:
                        subhead("Adjudication Decision Output")
                        decision = res_data["decision"]
                        if decision == "AUTO-APPROVED":
                            info_card(
                                "Status",
                                "AUTO-APPROVED",
                                [
                                    (
                                        "Authorization Code",
                                        res_data["status_code"],
                                        True,
                                    ),
                                    ("Reason", res_data["reason"]),
                                    (
                                        "Approval Confidence",
                                        res_data["approval_probability"],
                                    ),
                                ],
                                "green",
                            )
                        else:
                            info_card(
                                "Status",
                                "FLAGGED FOR AUDIT / DENIED",
                                [
                                    (
                                        "Audit Code",
                                        res_data["status_code"],
                                        True,
                                    ),
                                    ("Reason", res_data["reason"]),
                                    (
                                        "Approval Probability",
                                        res_data["approval_probability"],
                                    ),
                                ],
                                "red",
                            )
                else:
                    st.error(f"Backend Server Error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error(
                    "**Backend Offline:** Please ensure FastAPI is running via"
                    " `uvicorn backend:app --reload`."
                )
    else:
        empty_state(
            "No claim submitted",
            "Configure patient metrics and claim financials in the left"
            " sidebar, then click Run Adjudication Review to see results.",
        )
