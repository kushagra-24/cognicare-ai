# services/hospital_finder.py

import os
from groq import Groq
from services.health_agent import load_history
from services.ai_analysis import get_lang_instruction



# ─── ADD YOUR GROQ API KEY in the .env file as: GROQ_API_KEY=your_key_here ───
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MARKER_SPECIALIST_MAP = {
    # Liver
    "sgpt": "Hepatologist (Liver Specialist)",
    "sgot": "Hepatologist (Liver Specialist)",
    "alt":  "Hepatologist (Liver Specialist)",
    "ast":  "Hepatologist (Liver Specialist)",
    "bilirubin": "Hepatologist (Liver Specialist)",
    "alkaline phosphatase": "Hepatologist (Liver Specialist)",
    # Kidney
    "creatinine": "Nephrologist (Kidney Specialist)",
    "urea": "Nephrologist (Kidney Specialist)",
    "bun": "Nephrologist (Kidney Specialist)",
    "egfr": "Nephrologist (Kidney Specialist)",
    "uric acid": "Nephrologist (Kidney Specialist)",
    # Blood / CBC
    "hemoglobin": "Hematologist (Blood Specialist)",
    "hb": "Hematologist (Blood Specialist)",
    "wbc": "Hematologist (Blood Specialist)",
    "rbc": "Hematologist (Blood Specialist)",
    "platelet": "Hematologist (Blood Specialist)",
    "hematocrit": "Hematologist (Blood Specialist)",
    "mcv": "Hematologist (Blood Specialist)",
    # Thyroid
    "tsh": "Endocrinologist (Thyroid / Hormone Specialist)",
    "t3":  "Endocrinologist (Thyroid / Hormone Specialist)",
    "t4":  "Endocrinologist (Thyroid / Hormone Specialist)",
    "thyroid": "Endocrinologist (Thyroid / Hormone Specialist)",
    # Diabetes
    "glucose": "Endocrinologist / Diabetologist",
    "hba1c": "Endocrinologist / Diabetologist",
    "insulin": "Endocrinologist / Diabetologist",
    "fasting sugar": "Endocrinologist / Diabetologist",
    # Heart / Lipids
    "cholesterol": "Cardiologist (Heart Specialist)",
    "triglycerides": "Cardiologist (Heart Specialist)",
    "ldl": "Cardiologist (Heart Specialist)",
    "hdl": "Cardiologist (Heart Specialist)",
    "vldl": "Cardiologist (Heart Specialist)",
    "troponin": "Cardiologist (Heart Specialist)",
    "ecg": "Cardiologist (Heart Specialist)",
    "echo": "Cardiologist (Heart Specialist)",
    # Vitamins / Minerals
    "vitamin d": "General Physician / Endocrinologist",
    "vitamin b12": "General Physician / Neurologist",
    "iron": "Hematologist (Blood Specialist)",
    "ferritin": "Hematologist (Blood Specialist)",
    "calcium": "Endocrinologist / Orthopedic Specialist",
    "sodium": "Nephrologist / General Physician",
    "potassium": "Nephrologist / Cardiologist",
    # Inflammation
    "crp": "Rheumatologist / General Physician",
    "esr": "Rheumatologist / General Physician",
    "c-reactive protein": "Rheumatologist / General Physician",
    # Urine
    "urine": "Nephrologist / Urologist",
    "protein in urine": "Nephrologist",
    "pus cells": "Urologist / General Physician",
    # Hormones
    "testosterone": "Endocrinologist",
    "estrogen": "Gynecologist / Endocrinologist",
    "progesterone": "Gynecologist / Endocrinologist",
    "prolactin": "Endocrinologist",
    "cortisol": "Endocrinologist",
    "fsh": "Gynecologist / Endocrinologist",
    "lh": "Gynecologist / Endocrinologist",
    # Imaging / Scans
    "x-ray": "Radiologist / Orthopedic Specialist",
    "mri": "Radiologist / Neurologist",
    "ct scan": "Radiologist",
    "ultrasound": "Radiologist / Sonologist",
    "usg": "Radiologist / Sonologist",
    "sonography": "Radiologist / Sonologist",
    "abdominal scan": "Radiologist / Gastroenterologist",
    "pelvic scan": "Radiologist / Gynecologist",
    "scan": "Radiologist",
    # Cancer markers
    "psa": "Urologist / Oncologist",
    "cea": "Oncologist",
    "ca-125": "Gynecologist / Oncologist",
    "afp": "Hepatologist / Oncologist",
    # Allergy
    "ige": "Allergist / Immunologist",
    "allergy": "Allergist / Immunologist",
    # Infection
    "covid": "Pulmonologist / General Physician",
    "dengue": "General Physician / Infectious Disease Specialist",
    "malaria": "General Physician / Infectious Disease Specialist",
    "typhoid": "General Physician / Infectious Disease Specialist",
    "widal": "General Physician / Infectious Disease Specialist",
    # Discharge / Prescription
    "discharge": "General Physician / Respective Specialist",
    "prescription": "General Physician / Respective Specialist",
}


def detect_specialists_from_report(report_text: str) -> list:
    report_lower = report_text.lower()
    specialists = set()
    for marker, specialist in MARKER_SPECIALIST_MAP.items():
        if marker in report_lower:
            specialists.add(specialist)
    specialists.add("General Physician / Internal Medicine")
    return sorted(specialists)


def find_hospitals_and_specialists(user_location: str, specialists: list, report_type: str = "Health Report", language: str = "English") -> str:
    lang_instr = get_lang_instruction(language)
    specialist_list = "\n".join(f"- {s}" for s in specialists)

    prompt = f"""You are CogniCare Hospital Finder.
Location: {user_location}
Report Type: {report_type}
LANGUAGE INSTRUCTION: {lang_instr}
IMPORTANT: Your ENTIRE response must be in the specified language only.

Specialists needed:
{specialist_list}

Give:
1. 4-5 best real hospitals near {user_location} for these specialists
   - Name | Why recommended | Area | Govt or Private
2. Top 2 specialists to book first
3. 2 tips for booking appointments in {user_location}

Be concise. Real hospitals only. End with a note to verify details before visiting.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800
    )
    return response.choices[0].message.content


def hospital_finder_agent(user_location: str, language: str = "English") -> dict:
    history = load_history()

    if not history["reports"]:
        return {
            "error": True,
            "message": "No health report found. Please upload and analyze a report first."
        }

    latest_report = history["reports"][-1]
    report_text   = latest_report["raw_text"]
    risk_level    = latest_report.get("risk_level", "Unknown")
    report_type   = latest_report.get("report_type", "Health Report")

    specialists     = detect_specialists_from_report(report_text)
    recommendations = find_hospitals_and_specialists(user_location, specialists, report_type, language)

    return {
        "error":              False,
        "location":           user_location,
        "risk_level":         risk_level,
        "report_type":        report_type,
        "specialists_needed": specialists,
        "recommendations":    recommendations
    }
