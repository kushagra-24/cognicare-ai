# services/ai_analysis.py

import os
from groq import Groq


# ─── ADD YOUR GROQ API KEY in the .env file as: GROQ_API_KEY=your_key_here ───
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
LANGUAGE_INSTRUCTION = {
    "English":    "Respond entirely in English.",
    "Hindi":      "Respond entirely in Hindi (हिंदी). Use Devanagari script.",
    "Bengali":    "Respond entirely in Bengali (বাংলা). Use Bengali script.",
    "Telugu":     "Respond entirely in Telugu (తెలుగు). Use Telugu script.",
    "Tamil":      "Respond entirely in Tamil (தமிழ்). Use Tamil script.",
    "Marathi":    "Respond entirely in Marathi (मराठी). Use Devanagari script.",
    "Gujarati":   "Respond entirely in Gujarati (ગુજરાતી). Use Gujarati script.",
    "Kannada":    "Respond entirely in Kannada (ಕನ್ನಡ). Use Kannada script.",
    "Malayalam":  "Respond entirely in Malayalam (മലയാളം). Use Malayalam script.",
    "Punjabi":    "Respond entirely in Punjabi (ਪੰਜਾਬੀ). Use Gurmukhi script.",
    "Odia":       "Respond entirely in Odia (ଓଡ଼ିଆ). Use Odia script.",
    "Urdu":       "Respond entirely in Urdu (اردو). Use Nastaliq/Arabic script.",
    "Assamese":   "Respond entirely in Assamese (অসমীয়া).",
    "Maithili":   "Respond entirely in Maithili (मैथिली).",
    "Sanskrit":   "Respond entirely in Sanskrit (संस्कृतम्).",
    "Spanish":    "Respond entirely in Spanish (Español).",
    "French":     "Respond entirely in French (Français).",
    "German":     "Respond entirely in German (Deutsch).",
    "Arabic":     "Respond entirely in Arabic (العربية). Use Arabic script.",
    "Chinese":    "Respond entirely in Simplified Chinese (中文). Use Chinese characters.",
    "Japanese":   "Respond entirely in Japanese (日本語). Use Japanese script.",
    "Korean":     "Respond entirely in Korean (한국어). Use Hangul script.",
    "Portuguese": "Respond entirely in Portuguese (Português).",
    "Russian":    "Respond entirely in Russian (Русский). Use Cyrillic script.",
    "Italian":    "Respond entirely in Italian (Italiano).",
    "Dutch":      "Respond entirely in Dutch (Nederlands).",
    "Turkish":    "Respond entirely in Turkish (Türkçe).",
    "Vietnamese": "Respond entirely in Vietnamese (Tiếng Việt).",
    "Thai":       "Respond entirely in Thai (ภาษาไทย). Use Thai script.",
    "Swahili":    "Respond entirely in Swahili (Kiswahili).",
}


def get_lang_instruction(language: str) -> str:
    return LANGUAGE_INSTRUCTION.get(language, "Respond entirely in English.")


def analyze_single_report(report_text, report_type="Auto Detect (Recommended)", language="English"):
    lang_instr = get_lang_instruction(language)

    prompt = f"""You are CogniCare — a fast AI Health Report Analyzer.
Report Type: {report_type}
LANGUAGE INSTRUCTION: {lang_instr}

Analyze this health report. If Auto Detect, identify the type first.

RULES: Bullet points only. No diagnosis. Educational only. Be concise.
IMPORTANT: Your ENTIRE response must be in the specified language only.

OUTPUT:
🔍 REPORT TYPE: [name it]

1️⃣ HIGH/ABNORMAL VALUES 🔺
- marker | value | normal range | possible cause | precaution

2️⃣ LOW VALUES 🔻
- marker | value | normal range | possible cause | precaution

3️⃣ NORMAL VALUES ✅
- list them briefly

4️⃣ KEY FINDINGS
- 3-4 most important points

5️⃣ LIFESTYLE TIPS
- diet, hydration, sleep, exercise (brief)

6️⃣ SEE A DOCTOR
- which specialist | how urgent

7️⃣ RISK LEVEL: 🟢 Low | 🟡 Moderate | 🔴 High

Report:
{report_text[:2000]}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt[:4000]}],
        temperature=0.2,
        max_tokens=1500
    )
    return response.choices[0].message.content


def compare_reports(previous_report, current_report, report_type="Health Report", language="English"):
    lang_instr = get_lang_instruction(language)

    prompt = f"""You are CogniCare Comparison Agent.
Report Type: {report_type}
LANGUAGE INSTRUCTION: {lang_instr}

Compare these TWO health reports. Be concise and clear.
IMPORTANT: Your ENTIRE response must be in the specified language only.

RULES: Bullet points. No diagnosis. Educational only.

OUTPUT:
1️⃣ IMPROVED ✅ — what got better
2️⃣ WORSENED 🔺 — what got worse
3️⃣ UNCHANGED ➡️ — stable values
4️⃣ NEW ABNORMAL VALUES — in current report
5️⃣ POSSIBLE REASONS — lifestyle/medication/infection
6️⃣ HEALTH RISKS — explain calmly
7️⃣ PRECAUTIONS — brief advice
8️⃣ SUMMARY — improving or declining? (2-3 lines)

Previous Report:
{previous_report[:1500]}

Current Report:
{current_report[:1500]}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1200
    )
    return response.choices[0].message.content


def generate_pdf_report(analysis_text, risk_score, risk_level, report_type, language="English"):
    """Generate a structured doctor-ready report summary."""
    lang_instr = get_lang_instruction(language)

    prompt = f"""You are CogniCare PDF Report Generator.
LANGUAGE INSTRUCTION: {lang_instr}
IMPORTANT: Respond entirely in the specified language.

Based on this health analysis, create a clean structured report:

PATIENT HEALTH SUMMARY REPORT
- Report Type: {report_type}
- Risk Score: {risk_score}
- Risk Level: {risk_level}

Analysis:
{analysis_text}

Format as a clean medical summary with sections:
1. Executive Summary (2-3 lines)
2. Key Abnormal Findings
3. Normal Findings
4. Recommended Specialists
5. Lifestyle Recommendations
6. Urgency Level & Next Steps

Keep it professional and concise."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000
    )
    return response.choices[0].message.content


def generate_risk_heatmap_data(report_text, language="English"):
    """Extract markers and their risk levels for heatmap visualization."""
    prompt = f"""You are CogniCare Risk Analyzer.
From this health report, extract ALL measurable markers and classify each.

Respond ONLY in this exact JSON format, nothing else:
{{
  "markers": [
    {{"name": "Hemoglobin", "value": "11.2", "unit": "g/dL", "status": "low", "severity": 60}},
    {{"name": "Glucose", "value": "95", "unit": "mg/dL", "status": "normal", "severity": 10}},
    {{"name": "Cholesterol", "value": "240", "unit": "mg/dL", "status": "high", "severity": 80}}
  ]
}}

Status must be: "high", "low", or "normal"
Severity is 0-100 (0=perfectly normal, 100=critically abnormal)

Report:
{report_text[:2000]}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=800
    )
    return response.choices[0].message.content
