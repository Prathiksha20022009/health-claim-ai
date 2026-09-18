from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="ClaimTriage Enterprise Backend",
    description="Clinical Risk & Underwriting Adjudication Engine",
    version="3.4",
)


class AssessmentRequest(BaseModel):
  gender: str
  age: int
  income: float
  hypertension: int
  heart_disease: int
  avg_glucose_level: float
  smoking_history: str
  bmi: float
  claim_amount: float
  coverage_limit: int
  policy_tenure_months: int


def evaluate_scheme_eligibility(
    age: int, income: float, pre_existing: int, hypertension: int, heart_disease: int
):
  eligible_schemes = []

  if income <= 500000 or age >= 70:
    eligible_schemes.append({
        "scheme_code": "PUB_PMJAY",
        "name": "Ayushman Bharat PM-JAY (Public Safety Net)",
        "coverage": "Cashless Hospitalization Protection",
        "eligibility_reason": (
            "Qualified via Senior Citizen Provision (Age >= 70)"
            if age >= 70
            else "Qualified via Lower-Income Threshold Criteria"
        ),
    })

  if age >= 60:
    eligible_schemes.append({
        "scheme_code": "PUB_SENIOR",
        "name": "Senior Citizen Health Protection Plan",
        "coverage": "Specialized Geriatric Inpatient Care",
        "eligibility_reason": f"Matched Age Demographic ({age} years)",
    })

  if income > 300000:
    eligible_schemes.append({
        "scheme_code": "PVT_SHIELD",
        "name": "Comprehensive Private Health Shield",
        "coverage": "Customizable Annual Sum Insured",
        "eligibility_reason": (
            "Standard Market Eligibility (Subject to Pre-Existing Waiting"
            " Period)"
            if (hypertension == 1 or heart_disease == 1 or pre_existing == 1)
            else "Standard Market Eligibility (Clean Profile)"
        ),
    })

  return eligible_schemes


@app.post("/api/adjudicate")
def adjudicate_claim_endpoint(data: AssessmentRequest):
  risk_points = 0
  risk_reasons = []

  # 1. Age & Demographic Risk Weighting
  if data.age >= 80:
    risk_points += 5
    risk_reasons.append(
        f"Advanced geriatric age demographic ({data.age} years - Mandatory"
        " audit threshold)"
    )
  elif data.age >= 65:
    risk_points += 3
    risk_reasons.append(
        f"Advanced senior age demographic ({data.age} years - Elevated baseline)"
    )
  elif data.age >= 50:
    risk_points += 2
    risk_reasons.append(f"Elevated age bracket ({data.age} years)")

  # 2. BMI Evaluation
  if data.bmi > 30.0:
    risk_points += 2
    risk_reasons.append(f"Obesity indicator flagged (BMI: {data.bmi})")
  elif data.bmi < 18.5:
    risk_points += 1
    risk_reasons.append(f"Underweight indicator flagged (BMI: {data.bmi})")

  # 3. Glucose Evaluation
  if data.avg_glucose_level >= 126.0:
    risk_points += 3
    risk_reasons.append(
        f"Elevated Glucose ({data.avg_glucose_level} mg/dL - Hyperglycemia)"
    )
  elif data.avg_glucose_level > 99.0:
    risk_points += 1
    risk_reasons.append(
        f"Pre-diabetic glucose range ({data.avg_glucose_level} mg/dL)"
    )

  # 4. Chronic Conditions
  has_pre_existing = 0
  if data.hypertension == 1:
    risk_points += 2
    risk_reasons.append("Diagnosed Hypertension Profile")
    has_pre_existing = 1
  if data.heart_disease == 1:
    risk_points += 4
    risk_reasons.append("Cardiovascular Disease History")
    has_pre_existing = 1

  # 5. Smoking Behavior
  smoking_lower = data.smoking_history.strip().lower()
  if smoking_lower == "smokes":
    risk_points += 2
    risk_reasons.append("Active Smoker Risk Profile")
  elif smoking_lower == "former":
    risk_points += 1
    risk_reasons.append("Former Smoker Risk Adjustment")

  # Holistic Risk Tier Assignment
  is_high_risk = risk_points >= 5
  risk_tier = (
      "HIGH RISK"
      if is_high_risk
      else ("MODERATE RISK" if risk_points >= 2 else "LOW RISK")
  )

  # Financial & Policy Rule Verification
  is_excess = data.claim_amount > data.coverage_limit
  is_ped_waiting_breach = data.policy_tenure_months < 24 and has_pre_existing == 1
  is_initial_wait_breach = data.policy_tenure_months < 1

  # Clean Probability Calculations (Ensuring non-zero scaling)
  if risk_tier == "HIGH RISK":
    approval_prob = 0.35
  elif risk_tier == "MODERATE RISK":
    approval_prob = 0.65
  else:
    approval_prob = 0.90

  # Override to 0 only on true contractual/legal breaches
  if is_excess or is_ped_waiting_breach or is_initial_wait_breach:
    approval_prob = 0.0

  final_prob_str = f"{approval_prob * 100:.1f}%"

  # Decision Code Matrix
  if is_excess:
    decision, code, reason = (
        "FLAGGED FOR AUDIT / DENIED",
        "D1_LIMIT_EXCEEDED",
        "Requested claim amount exceeds active policy coverage limit.",
    )
  elif is_ped_waiting_breach:
    decision, code, reason = (
        "FLAGGED FOR AUDIT / DENIED",
        "D2_WAITING_PERIOD",
        (
            "Pre-existing chronic condition disclosed within mandatory 24-month"
            " waiting window."
        ),
    )
  elif risk_tier == "HIGH RISK" or approval_prob < 0.5:
    decision, code, reason = (
        "FLAGGED FOR AUDIT / DENIED",
        "D3_RISK_THRESHOLD",
        (
            "Cumulative clinical risk score breached commercial underwriting"
            " limits."
        ),
    )
  else:
    decision, code, reason = (
        "AUTO-APPROVED",
        "PA_VERIFIED_100",
        (
            "Claim successfully verified against medical baselines and policy"
            " terms."
        ),
    )

  matched_schemes = evaluate_scheme_eligibility(
      age=data.age,
      income=data.income,
      pre_existing=has_pre_existing,
      hypertension=data.hypertension,
      heart_disease=data.heart_disease,
  )

  return {
      "decision": decision,
      "status_code": code,
      "reason": reason,
      "approval_probability": final_prob_str,
      "clinical_assessment": {
          "risk_tier": risk_tier,
          "total_risk_score": risk_points,
          "clinical_flags": risk_reasons,
      },
      "eligible_schemes": matched_schemes,
  }
