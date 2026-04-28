# services/health_agent.py

import json
import os
from datetime import datetime
from services.ai_analysis import analyze_single_report, compare_reports

HISTORY_PATH = "data/report_history.json"


def load_history():
    if not os.path.exists(HISTORY_PATH):
        return {"reports": []}
    try:
        with open(HISTORY_PATH, "r") as f:
            data = json.load(f)
            if "reports" not in data:
                return {"reports": []}
            return data
    except:
        return {"reports": []}


def save_history(history):
    os.makedirs("data", exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def calculate_risk_score(analysis_text):
    high_count = analysis_text.count("🔺")
    low_count  = analysis_text.count("🔻")
    score = (high_count * 15) + (low_count * 10)

    if score < 20:
        level = "Low Risk 🟢"
    elif score < 50:
        level = "Moderate Risk 🟡"
    else:
        level = "High Risk 🔴"

    return score, level


def health_agent(report_text, report_type="Auto Detect (Recommended)", language="English"):
    history   = load_history()
    report_id = len(history["reports"]) + 1

    # Always analyze single report only — comparison is manual
    analysis     = analyze_single_report(report_text, report_type, language)
    score, level = calculate_risk_score(analysis)

    new_report = {
        "report_id":   report_id,
        "date":        str(datetime.now().date()),
        "report_type": report_type,
        "raw_text":    report_text,
        "analysis":    analysis,
        "risk_score":  score,
        "risk_level":  level,
        "language":    language
    }

    history["reports"].append(new_report)
    save_history(history)

    # Always return single mode — user triggers comparison manually
    return {
        "mode":       "single",
        "analysis":   analysis,
        "risk_score": score,
        "risk_level": level,
        "history":    history
    }


def run_comparison(report_id_1, report_id_2, language="English"):
    """
    Called only when user clicks Compare button.
    Compares two reports by their IDs.
    """
    history = load_history()
    reports = history["reports"]

    r1 = next((r for r in reports if r["report_id"] == report_id_1), None)
    r2 = next((r for r in reports if r["report_id"] == report_id_2), None)

    if not r1 or not r2:
        return "Could not find the selected reports."

    return compare_reports(
        r1["raw_text"],
        r2["raw_text"],
        r2.get("report_type", "Health Report"),
        language
    )
