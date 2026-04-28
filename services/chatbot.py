# services/chatbot.py

import os
import re
from groq import Groq
from dotenv import load_dotenv
from services.health_agent import load_history
from services.ai_analysis import get_lang_instruction

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

HOSPITAL_KEYWORDS = [
    "hospital", "doctor", "specialist", "clinic", "appointment",
    "consult", "near me", "nearby", "where to go", "which hospital",
    "best doctor", "which doctor", "whom to see", "who should i visit",
    "medical center", "find a doctor", "book appointment",
    "referral", "visit", "nearest", "find hospital",
]


def is_hospital_query(user_question: str) -> bool:
    return any(kw in user_question.lower() for kw in HOSPITAL_KEYWORDS)


def extract_location_from_query(user_question: str):
    patterns = [
        r'\bin ([A-Z][a-zA-Z\s]+)',
        r'\bnear ([A-Z][a-zA-Z\s]+)',
        r'\bat ([A-Z][a-zA-Z\s]+)',
        r'\baround ([A-Z][a-zA-Z\s]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, user_question)
        if match:
            return match.group(1).strip()
    return None


def chat_with_agent(
    user_question: str,
    user_location: str = None,
    language: str = "English",
) -> str:
    history   = load_history()
    lang_instr = get_lang_instruction(language)

    if not history["reports"]:
        return "Please upload and analyze a health report first."

    # ── Route to hospital finder ──────────────────────────────────────────────
    if is_hospital_query(user_question):
        from services.hospital_finder import hospital_finder_agent

        location = user_location or extract_location_from_query(user_question)

        if not location:
            return (
                "I can help find hospitals near you! 🏥\n\n"
                "Enter your city in the **Location** box in the sidebar, "
                "or type: *'Find hospitals in [your city]'*"
            )

        result = hospital_finder_agent(location, language=language)

        if result["error"]:
            return result["message"]

        specialists_text = "\n".join(f"- {s}" for s in result["specialists_needed"])

        return f"""## 🏥 Hospitals & Specialists — {result['location']}
**Risk Level:** {result['risk_level']}

### 👨‍⚕️ Recommended Specialists:
{specialists_text}

---
{result['recommendations']}

---
*Educational guidance only. Verify details before visiting.*"""

    # ── Standard health Q&A ───────────────────────────────────────────────────
    latest_report   = history["reports"][-1]
    report_text     = latest_report["raw_text"]
    report_analysis = latest_report["analysis"]
    report_type     = latest_report.get("report_type", "Health Report")

    prompt = f"""You are CogniCare Chat — a fast AI health assistant.
Report Type: {report_type}
LANGUAGE INSTRUCTION: {lang_instr}
Rules: Educational only. No diagnosis. No medicines. Clear and concise.
IMPORTANT: Your ENTIRE response must be in the specified language only.

Report Summary:
{report_text[:1500]}

Analysis Summary:
{report_analysis[:1000]}

Question: {user_question}

Answer in 3-5 bullet points. Be clear and brief."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=600,
    )

    return response.choices[0].message.content
