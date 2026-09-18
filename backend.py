from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="ClaimTriage Enterprise Backend",
    description=(
        "Decoupled FastAPI backend processing clinical risk and underwriting"
        " rules."
    ),
    version="3.0",
)


class ClinicalAssessmentRequest(BaseModel):
  gender: str
  age: int
  hypertension: int
  heart_disease: int
  avg_glucose_level: float
  smoking_history: str
  bmi: float
  claim_amount: float
  coverage_limit: int
  policy_tenure_months: int


@app.post("/api/adjudicate")
def adjudicate_claim_endpoint(data: ClinicalAssessmentRequest):
  risk_points = 0
  risk_reasons = []

  # 1. Age & Gender Risk Weighting
  gender_lower = data.gender.strip().lower()
  if data.age >= 65:
    risk_points += 3
    risk_reasons.append(
        f"Advanced senior age demographic ({data.age} years - High baseline"
        " vulnerability)"
    )
  elif data.age >= 50:
    risk_points += 2
    risk_reasons.append(f"Elevated age bracket ({data.age} years)")
  elif gender_lower == "male" and data.age >= 45:
    risk_points += 1
    risk_reasons.append(f"Male demographic risk threshold met ({data.age} years)")
  elif gender_lower == "female" and data.age >= 55:
    risk_points += 1
    risk_reasons.append(
        f"Post-menopausal female demographic risk threshold met ({data.age}"
        " years)"
    )

  # 2. BMI Evaluation (WHO Normal: 18.5 - 24.9)
  if data.bmi > 30.0:
    risk_points += 2
    risk_reasons.append(
        f"Obesity flagged (BMI: {data.bmi} > 30.0 - High metabolic/cardiac risk)"
    )
  elif data.bmi < 18.5:
    risk_points += 1
    risk_reasons.append(f"Underweight flagged (BMI: {data.bmi} < 18.5)")

  # 3. Glucose Evaluation (Normal: 70 - 99 mg/dL)
  if data.avg_glucose_level >= 126.0:
    risk_points += 3
    risk_reasons.append(
        f"Elevated Glucose ({data.avg_glucose_level} mg/dL - Diabetes indicator)"
    )
  elif data.avg_glucose_level > 99.0:
    risk_points += 1
    risk_reasons.append(
        f"Pre-diabetic glucose range ({data.avg_glucose_level} mg/dL)"
    )

  # 4. Chronic Conditions
  if data.hypertension == 1:
    risk_points += 2
    risk_reasons.append("Diagnosed Hypertension (High Blood Pressure)")
  if data.heart_disease == 1:
    risk_points += 4
    risk_reasons.append("Prior History of Cardiovascular Disease")

  # 5. Smoking History (Differentiating Active vs. Former)
  smoking_lower = data.smoking_history.strip().lower()
  if smoking_lower == "smokes":
    risk_points += 2
    risk_reasons.append(
        "Active smoker profile (Elevated respiratory/cardiac risk)"
    )
  elif smoking_lower == "former":
    risk_points += 1
    risk_reasons.append(
        "Former smoker profile (Moderate historical risk adjustment)"
    )

  # Holistic Tier Classification
  is_high_risk = risk_points >= 5
  risk_tier = (
      "HIGH RISK"
      if is_high_risk
      else ("MODERATE RISK" if risk_points >= 2 else "LOW RISK")
  )

  # Financial Policy Rules
  is_excess = data.claim_amount > data.coverage_limit
  is_early = data.policy_tenure_months < 6

  # Probability Calculation
  approval_prob = 0.90
  if risk_tier == "HIGH RISK":
    approval_prob -= 0.55
  elif risk_tier == "MODERATE RISK":
    approval_prob -= 0.25

  if is_excess or is_early:
    approval_prob = 0.0

  final_prob_str = f"{max(0.0, approval_prob) * 100:.1f}%"

  if approval_prob < 0.5 or is_excess or is_early:
    reason = (
        "Claim amount exceeds coverage limit."
        if is_excess
        else (
            "Policy tenure under 6-month waiting period."
            if is_early
            else "High clinical risk profile breached underwriting thresholds."
        )
    )
    decision = "FLAGGED FOR AUDIT / DENIED"
    code = "D_AUDIT"
  else:
    reason = "Claim verified against medical baselines and policy terms."
    decision = "AUTO-APPROVED"
    code = "PA_100"

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
  }