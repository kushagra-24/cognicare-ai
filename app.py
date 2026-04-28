# app.py — CogniCare | Full-Featured AI Health Analyzer

import streamlit as st
import os, json, tempfile, re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from services.pdf_extractor import extract_text_from_pdf
from services.health_agent import health_agent, load_history, save_history
from services.chatbot import chat_with_agent
from services.ai_analysis import generate_risk_heatmap_data, generate_pdf_report

st.set_page_config(
    page_title="CogniCare AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');
:root {
    --bg:#050c14; --bg2:#0a1628; --card:#0d1f35; --border:#1a3a5c;
    --cyan:#00e5ff; --green:#00ff9d; --amber:#ffb800; --red:#ff3d5a;
    --text:#c8dff0; --muted:#4a6880;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;font-family:'DM Sans',sans-serif;color:var(--text);}
[data-testid="stSidebar"]{background:var(--bg2)!important;border-right:1px solid var(--border);}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stToolbar"]{display:none;}
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}
.hero{text-align:center;padding:2rem 1rem 1rem;}
.hero-title{font-family:'Orbitron',sans-serif;font-size:clamp(1.8rem,4vw,3.2rem);font-weight:900;letter-spacing:.08em;background:linear-gradient(135deg,var(--cyan),var(--green),var(--amber));-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1;margin-bottom:.3rem;}
.hero-sub{color:var(--muted);font-size:.9rem;letter-spacing:.15em;text-transform:uppercase;font-weight:300;}
.hero-line{width:100px;height:2px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);margin:1rem auto;}
.glass-card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.2rem;margin-bottom:.8rem;position:relative;overflow:hidden;}
.glass-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);}
.metric-box{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:1rem .8rem;text-align:center;}
.metric-val{font-family:'Orbitron',sans-serif;font-size:1.8rem;font-weight:700;color:var(--cyan);}
.metric-label{color:var(--muted);font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;margin-top:.2rem;}
.badge-low{background:rgba(0,255,157,.12);color:var(--green);border:1px solid var(--green);border-radius:20px;padding:3px 12px;font-size:.82rem;font-weight:600;display:inline-block;}
.badge-mod{background:rgba(255,184,0,.12);color:var(--amber);border:1px solid var(--amber);border-radius:20px;padding:3px 12px;font-size:.82rem;font-weight:600;display:inline-block;}
.badge-high{background:rgba(255,61,90,.12);color:var(--red);border:1px solid var(--red);border-radius:20px;padding:3px 12px;font-size:.82rem;font-weight:600;display:inline-block;}
[data-testid="stTabs"] button{font-family:'Orbitron',sans-serif!important;font-size:.65rem!important;letter-spacing:.08em!important;color:var(--muted)!important;border-radius:8px 8px 0 0!important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--cyan)!important;border-bottom:2px solid var(--cyan)!important;background:rgba(0,229,255,.05)!important;}
.stButton>button{background:linear-gradient(135deg,rgba(0,229,255,.12),rgba(0,255,157,.08))!important;border:1px solid var(--cyan)!important;color:var(--cyan)!important;font-family:'Orbitron',sans-serif!important;font-size:.68rem!important;letter-spacing:.1em!important;border-radius:8px!important;transition:all .2s!important;}
.stButton>button:hover{background:rgba(0,229,255,.2)!important;box-shadow:0 0 20px rgba(0,229,255,.3)!important;}
[data-testid="stFileUploader"]{border:2px dashed var(--border)!important;border-radius:12px!important;background:var(--card)!important;}
input,textarea,[data-testid="stTextInput"] input{background:var(--card)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:8px!important;}
[data-testid="stSelectbox"]>div>div{background:var(--card)!important;border:1px solid var(--border)!important;color:var(--text)!important;}
[data-testid="stExpander"]{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:10px!important;}
.sec-heading{font-family:'Orbitron',sans-serif;font-size:.75rem;letter-spacing:.18em;text-transform:uppercase;color:var(--cyan);margin-bottom:.7rem;display:flex;align-items:center;gap:.5rem;}
.sec-heading::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent);}
.chat-user{background:rgba(0,229,255,.07);border:1px solid rgba(0,229,255,.18);border-radius:14px 14px 2px 14px;padding:.75rem 1rem;margin:.4rem 0;}
.chat-ai{background:rgba(0,255,157,.05);border:1px solid rgba(0,255,157,.13);border-radius:14px 14px 14px 2px;padding:.75rem 1rem;margin:.4rem 0;}
.sidebar-logo{font-family:'Orbitron',sans-serif;font-size:1.2rem;font-weight:900;background:linear-gradient(135deg,var(--cyan),var(--green));-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:.1em;}
.sidebar-tagline{color:var(--muted);font-size:.68rem;letter-spacing:.15em;text-transform:uppercase;}
.stProgress>div>div{background:linear-gradient(90deg,var(--cyan),var(--green))!important;border-radius:4px;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🧬 CogniCare</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">AI Health Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="sec-heading">⚙ Config</div>', unsafe_allow_html=True)

    report_type = st.selectbox("Report Type", [
        "Auto Detect (Recommended)","Blood Test / CBC","Liver Function Test",
        "Kidney Function Test","Thyroid Panel","Lipid Panel",
        "Diabetes / Blood Sugar","Urine Test","Hormonal Panel",
        "Imaging / Radiology","Ultrasound / USG","Discharge Summary",
        "Prescription","Other",
    ])

    language = st.selectbox("Language", [
        "English","Hindi","Bengali","Telugu","Tamil","Marathi","Gujarati",
        "Kannada","Malayalam","Punjabi","Odia","Urdu","Assamese",
        "Spanish","French","German","Arabic","Chinese","Japanese",
        "Korean","Portuguese","Russian","Italian","Dutch","Turkish",
    ])

    user_location = st.text_input("📍 Your City", placeholder="e.g. Bhubaneswar")
    st.markdown("---")

    history = load_history()
    n = len(history["reports"])
    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="metric-box"><div class="metric-val" style="font-size:1.4rem">{n}</div><div class="metric-label">Reports</div></div>', unsafe_allow_html=True)
    if n > 0:
        last_level = history["reports"][-1].get("risk_level","")
        icon = "🔴" if "High" in last_level else ("🟡" if "Moderate" in last_level else "🟢")
    else:
        icon = "⚪"
    c2.markdown(f'<div class="metric-box"><div class="metric-val" style="font-size:1.4rem">{icon}</div><div class="metric-label">Risk</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑 Clear History", use_container_width=True):
        save_history({"reports": []})
        st.rerun()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">COGNICARE AI</div>
  <div class="hero-sub">Advanced Health Intelligence Platform</div>
  <div class="hero-line"></div>
</div>""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🔬 ANALYZE","🌡 HEATMAP","📈 TRENDS","💬 CHAT","🔄 COMPARE","🏥 HOSPITALS","📄 PDF","📁 HISTORY"])

# ── TAB 1: ANALYZE ────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="sec-heading">📄 Upload Health Report PDF</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop PDF here", type=["pdf"], label_visibility="collapsed")

    if uploaded_file:
        col_a, col_b = st.columns([3,1])
        with col_a:
            st.markdown(f'<div class="glass-card" style="padding:.8rem"><span style="color:var(--green)">✅</span> <strong>{uploaded_file.name}</strong> <span style="color:var(--muted);font-size:.8rem">({round(uploaded_file.size/1024,1)} KB)</span></div>', unsafe_allow_html=True)
        with col_b:
            go = st.button("🔍 ANALYZE", use_container_width=True)

        if go:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            with st.spinner("Extracting text…"):
                report_text = extract_text_from_pdf(tmp_path)
            os.unlink(tmp_path)

            if not report_text.strip():
                st.error("❌ Could not extract text. PDF may be image-only.")
            else:
                with st.spinner("🧠 Running AI analysis…"):
                    result = health_agent(report_text, report_type, language)
                st.session_state["last_result"] = result
                st.rerun()

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        score  = result["risk_score"]
        level  = result["risk_level"]
        badge  = "badge-high" if "High" in level else ("badge-mod" if "Moderate" in level else "badge-low")

        m1,m2,m3,m4 = st.columns(4)
        m1.markdown(f'<div class="metric-box"><div class="metric-val">{score}</div><div class="metric-label">Risk Score</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-box"><div style="margin:.3rem 0"><span class="{badge}">{level}</span></div><div class="metric-label">Risk Level</div></div>', unsafe_allow_html=True)
        h_now = load_history()
        m3.markdown(f'<div class="metric-box"><div class="metric-val">{len(h_now["reports"])}</div><div class="metric-label">Total Reports</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-box"><div class="metric-val" style="font-size:1rem;color:var(--amber)">{datetime.now().strftime("%b %d")}</div><div class="metric-label">Date</div></div>', unsafe_allow_html=True)

        st.progress(min(score,100)/100)

        st.markdown('<div class="sec-heading" style="margin-top:1.2rem">📋 AI Analysis</div>', unsafe_allow_html=True)
        if result["mode"] == "single":
            st.markdown(f'<div class="glass-card">{result["analysis"]}</div>', unsafe_allow_html=True)
        else:
            ca, cb = st.columns(2)
            with ca:
                st.markdown('<div class="sec-heading">📋 Current</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="glass-card">{result["analysis"]}</div>', unsafe_allow_html=True)
            with cb:
                st.markdown('<div class="sec-heading">🔄 Vs Previous</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="glass-card">{result["comparison"]}</div>', unsafe_allow_html=True)

# ── TAB 2: HEATMAP ────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="sec-heading">🌡 Risk Heatmap</div>', unsafe_allow_html=True)
    h2 = load_history()
    if not h2["reports"]:
        st.info("Analyze a report first.")
    else:
        latest = h2["reports"][-1]
        if st.button("🌡 Generate Heatmap", use_container_width=True):
            with st.spinner("Building heatmap…"):
                raw = generate_risk_heatmap_data(latest["raw_text"])
            st.session_state["heatmap_raw"] = raw

        if "heatmap_raw" in st.session_state:
            try:
                raw_clean = re.sub(r"```json|```","",st.session_state["heatmap_raw"]).strip()
                hm = json.loads(raw_clean)
                markers = hm.get("markers",[])
                if markers:
                    st.markdown("🟢 Normal &nbsp; 🟡 Low &nbsp; 🔴 High", unsafe_allow_html=True)
                    cols_n = 4
                    rows = [markers[i:i+cols_n] for i in range(0,len(markers),cols_n)]
                    for row in rows:
                        rcols = st.columns(cols_n)
                        for ci,mk in enumerate(row):
                            sev = mk.get("severity",50)
                            st_val = mk.get("status","normal").lower()
                            if st_val=="high":   bg="rgba(255,61,90,0.15)";  bc="#ff3d5a"; ic="🔺"
                            elif st_val=="low":  bg="rgba(255,184,0,0.15)";  bc="#ffb800"; ic="🔻"
                            else:                bg="rgba(0,255,157,0.1)";   bc="#00ff9d"; ic="✅"
                            with rcols[ci]:
                                st.markdown(f"""
                                <div style="background:{bg};border:1px solid {bc};border-radius:10px;padding:10px 6px;text-align:center;margin:3px 0">
                                  <div style="font-size:1.1rem">{ic}</div>
                                  <div style="font-weight:700;color:#fff;font-size:.82rem;margin:3px 0">{mk['name']}</div>
                                  <div style="color:{bc};font-size:.88rem;font-weight:600">{mk.get('value','?')} <span style="font-size:.68rem;opacity:.6">{mk.get('unit','')}</span></div>
                                  <div style="margin-top:5px;background:rgba(0,0,0,.3);border-radius:3px;height:3px">
                                    <div style="width:{sev}%;background:{bc};height:3px;border-radius:3px"></div>
                                  </div>
                                  <div style="font-size:.6rem;color:rgba(255,255,255,.35);margin-top:2px">sev {sev}/100</div>
                                </div>""", unsafe_allow_html=True)
                else:
                    st.warning("No markers extracted.")
            except Exception as e:
                st.error(f"Parse error: {e}")

# ── TAB 3: TRENDS ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="sec-heading">📈 Health Trend Analysis</div>', unsafe_allow_html=True)
    h3 = load_history()
    if len(h3["reports"]) < 2:
        st.info("Upload at least 2 reports to see trends.")
    else:
        try:
            import plotly.graph_objects as go
            dates  = [r["date"] for r in h3["reports"]]
            scores = [r.get("risk_score",0) for r in h3["reports"]]
            levels = [r.get("risk_level","") for r in h3["reports"]]
            colors = ["#ff3d5a" if "High" in l else ("#ffb800" if "Moderate" in l else "#00ff9d") for l in levels]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=scores, mode="lines+markers",
                line=dict(color="#00e5ff",width=3),
                marker=dict(size=13,color=colors,line=dict(color="#00e5ff",width=2)),
                fill="tozeroy", fillcolor="rgba(0,229,255,0.05)",
                hovertemplate="<b>%{x}</b><br>Score: %{y}<extra></extra>",
            ))
            fig.add_hrect(y0=0,  y1=20,  fillcolor="rgba(0,255,157,0.04)",  line_width=0)
            fig.add_hrect(y0=20, y1=50,  fillcolor="rgba(255,184,0,0.04)",  line_width=0)
            fig.add_hrect(y0=50, y1=300, fillcolor="rgba(255,61,90,0.04)",  line_width=0)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c8dff0",family="DM Sans"),
                xaxis=dict(gridcolor="#1a3a5c",title="Date"),
                yaxis=dict(gridcolor="#1a3a5c",title="Risk Score"),
                margin=dict(l=10,r=10,t=10,b=10), hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.warning("Run `pip install plotly` for chart visualization.")

        st.markdown('<div class="sec-heading">📋 All Reports</div>', unsafe_allow_html=True)
        for r in reversed(h3["reports"]):
            rl = r.get("risk_level","Unknown")
            badge = "badge-high" if "High" in rl else ("badge-mod" if "Moderate" in rl else "badge-low")
            rt = r.get("report_type","Unknown")
            st.markdown(f"""
            <div class="glass-card" style="padding:.7rem 1rem;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.5rem">
              <span style="color:var(--muted);font-size:.78rem">#{r.get('report_id','?')}</span>
              <span>{r.get('date','')}</span>
              <span style="color:var(--muted);font-size:.78rem">{rt[:22]}</span>
              <span class="metric-val" style="font-size:1.1rem">{r.get('risk_score',0)}</span>
              <span class="{badge}">{rl}</span>
            </div>""", unsafe_allow_html=True)

# ── TAB 4: CHAT ───────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="sec-heading">💬 Chat with Agent</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:var(--muted);font-size:.83rem;margin-bottom:.8rem">Ask anything about your report · Try <em>"Find hospitals in [city]"</em></div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        cls = "chat-user" if msg["role"]=="user" else "chat-ai"
        icon = "👤" if msg["role"]=="user" else "🧬"
        st.markdown(f'<div class="{cls}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)

    ci, cs = st.columns([5,1])
    with ci:
        user_input = st.text_input("", placeholder="Ask about your health report…", label_visibility="collapsed", key="chat_input")
    with cs:
        send_btn = st.button("SEND ➤", use_container_width=True)

    if send_btn and user_input.strip():
        st.session_state.messages.append({"role":"user","content":user_input})
        with st.spinner("🧠 Thinking…"):
            reply = chat_with_agent(user_input, user_location or None, language)
        st.session_state.messages.append({"role":"assistant","content":reply})
        st.rerun()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── TAB 5: COMPARE ────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="sec-heading">🔄 Compare Two Reports</div>', unsafe_allow_html=True)
    h5 = load_history()
    if len(h5["reports"]) < 2:
        st.info("Upload and analyze at least 2 reports to use comparison.")
    else:
        labels = [f"#{r.get('report_id','?')} — {r.get('date','')} ({r.get('report_type','?')[:18]})" for r in h5["reports"]]
        ids    = [r.get("report_id") for r in h5["reports"]]

        c1, c2 = st.columns(2)
        with c1:
            sel_p = st.selectbox("Select Report 1 (Older)", labels, index=0, key="cmp_p")
        with c2:
            sel_c = st.selectbox("Select Report 2 (Newer)", labels, index=len(labels)-1, key="cmp_c")

        if st.button("🔄 Run Comparison", use_container_width=True):
            if sel_p == sel_c:
                st.error("Please select two different reports.")
            else:
                pi    = labels.index(sel_p)
                ci_idx = labels.index(sel_c)
                pr    = h5["reports"][pi]
                cr    = h5["reports"][ci_idx]

                from services.health_agent import run_comparison
                with st.spinner("Comparing reports…"):
                    cmp = run_comparison(pr["report_id"], cr["report_id"], language)

                st.session_state["comparison_result"] = cmp
                st.session_state["cmp_pr"] = pr
                st.session_state["cmp_cr"] = cr

        if "comparison_result" in st.session_state:
            pr = st.session_state["cmp_pr"]
            cr = st.session_state["cmp_cr"]
            cmp = st.session_state["comparison_result"]

            ra, rb = st.columns(2)
            for col, rep, label in [(ra, pr, "Report 1"), (rb, cr, "Report 2")]:
                rl    = rep.get("risk_level", "?")
                badge = "badge-high" if "High" in rl else ("badge-mod" if "Moderate" in rl else "badge-low")
                col.markdown(f'<div class="glass-card"><strong>{label}:</strong> #{rep.get("report_id","?")} &nbsp;{rep.get("date","")} &nbsp;<span class="{badge}">{rl}</span><br><span style="color:var(--muted);font-size:.78rem">{rep.get("report_type","?")}</span></div>', unsafe_allow_html=True)

            st.markdown('<div class="sec-heading" style="margin-top:1rem">📊 Comparison Result</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="glass-card">{cmp}</div>', unsafe_allow_html=True)

            if st.button("🗑 Clear Comparison"):
                del st.session_state["comparison_result"]
                del st.session_state["cmp_pr"]
                del st.session_state["cmp_cr"]
                st.rerun()

# ── TAB 6: HOSPITALS ──────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="sec-heading">🏥 Hospital & Specialist Finder</div>', unsafe_allow_html=True)
    loc_inp = st.text_input("Your city", value=user_location or "", placeholder="e.g. Bhubaneswar", key="hosp_loc")
    if st.button("🔍 Find Hospitals", use_container_width=True):
        h6 = load_history()
        if not h6["reports"]:
            st.error("Analyze a report first.")
        elif not loc_inp.strip():
            st.error("Enter your city.")
        else:
            from services.hospital_finder import hospital_finder_agent
            with st.spinner(f"Searching near {loc_inp}…"):
                res = hospital_finder_agent(loc_inp, language=language)
            if res.get("error"):
                st.error(res["message"])
            else:
                st.markdown(f'<div class="glass-card">📍 <strong>{res["location"]}</strong> &nbsp;|&nbsp; Risk: {res["risk_level"]}</div>', unsafe_allow_html=True)
                st.markdown('<div class="sec-heading">👨‍⚕️ Specialists Needed</div>', unsafe_allow_html=True)
                sc1,sc2 = st.columns(2)
                for i,sp in enumerate(res["specialists_needed"]):
                    (sc1 if i%2==0 else sc2).markdown(f'<div style="background:rgba(0,229,255,.05);border:1px solid rgba(0,229,255,.18);border-radius:8px;padding:7px 12px;margin:3px 0;font-size:.83rem">🩺 {sp}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="glass-card" style="margin-top:.8rem">{res["recommendations"]}</div>', unsafe_allow_html=True)

# ── TAB 7: PDF ────────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown('<div class="sec-heading">📄 Export Doctor-Ready Report</div>', unsafe_allow_html=True)
    h7 = load_history()
    if not h7["reports"]:
        st.info("Analyze a report first.")
    else:
        lat7 = h7["reports"][-1]
        st.markdown(f'<div class="glass-card">Ready to export: <strong>Report #{lat7.get("report_id","?")} — {lat7.get("date","")}</strong><br>Type: {lat7.get("report_type","?")} &nbsp;|&nbsp; Risk: {lat7.get("risk_level","?")}</div>', unsafe_allow_html=True)
        if st.button("📄 Generate Summary", use_container_width=True):
            with st.spinner("Generating…"):
                pdf_text = generate_pdf_report(
                    lat7["analysis"], lat7.get("risk_score",0),
                    lat7.get("risk_level","Unknown"),
                    lat7.get("report_type","Health Report"), language,
                )
            try:
                from fpdf import FPDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.set_font("Helvetica","B",18)
                pdf.set_text_color(0,180,200)
                pdf.cell(0,10,"CogniCare — Patient Health Report",ln=True,align="C")
                pdf.set_font("Helvetica","",9)
                pdf.set_text_color(100,120,140)
                pdf.cell(0,5,f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}",ln=True,align="C")
                pdf.ln(4)
                pdf.set_draw_color(0,180,200)
                pdf.line(10,pdf.get_y(),200,pdf.get_y())
                pdf.ln(5)
                for line in pdf_text.split("\n"):
                    clean = line.encode("latin-1","replace").decode("latin-1")
                    if line.strip().startswith("#") or (line.strip().isupper() and len(line.strip())>3):
                        pdf.set_font("Helvetica","B",11); pdf.set_text_color(0,100,150)
                    else:
                        pdf.set_font("Helvetica","",10); pdf.set_text_color(40,40,40)
                    pdf.multi_cell(0,6,clean)
                pdf_bytes = pdf.output(dest="S").encode("latin-1")
                st.download_button("⬇ Download PDF", data=pdf_bytes,
                    file_name=f"cognicare_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf", use_container_width=True)
            except ImportError:
                st.warning("Install fpdf2 for PDF: `pip install fpdf2`")
                st.download_button("⬇ Download Text", data=pdf_text,
                    file_name=f"cognicare_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain", use_container_width=True)
            st.markdown(f'<div class="glass-card">{pdf_text}</div>', unsafe_allow_html=True)

# ── TAB 8: HISTORY ────────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown('<div class="sec-heading">📁 Report History</div>', unsafe_allow_html=True)
    h8 = load_history()
    if not h8["reports"]:
        st.info("No reports yet.")
    else:
        for rep in reversed(h8["reports"]):
            rl  = rep.get("risk_level","Unknown")
            rt  = rep.get("report_type","Unknown")
            rd  = rep.get("date","Unknown")
            rid = rep.get("report_id","?")
            rs  = rep.get("risk_score",0)
            badge = "badge-high" if "High" in rl else ("badge-mod" if "Moderate" in rl else "badge-low")
            with st.expander(f"Report #{rid} — {rd} | {rt[:28]} | Score: {rs}"):
                st.markdown(f'<span class="{badge}">{rl}</span>', unsafe_allow_html=True)
                st.markdown("---")
                st.markdown(rep.get("analysis","No analysis."))

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:1.5rem 0 .8rem;color:var(--muted);font-size:.7rem;letter-spacing:.12em;">
  🧬 COGNICARE AI &nbsp;·&nbsp; EDUCATIONAL PURPOSES ONLY &nbsp;·&nbsp; NOT A SUBSTITUTE FOR MEDICAL ADVICE
</div>""", unsafe_allow_html=True)
