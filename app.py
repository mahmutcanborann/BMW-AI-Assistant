import streamlit as st
import requests
import base64
from pathlib import Path
import os
API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(
    page_title="BMW AI Assistant",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)
if "page" not in st.session_state:
    st.session_state.page = "home"
if "result" not in st.session_state:
    st.session_state.result = None

query = ""
ask_clicked = False
def img_to_base64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()

logo_base64 = img_to_base64("b.png")
car_base64 = img_to_base64("a.jpg")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {{
    font-family: 'Inter', sans-serif;
}}

html, body, [data-testid="stAppViewContainer"] {{
    background-color: #02060e !important;
}}

[data-testid="stHeader"] {{
    display: none !important;
}}


[data-testid="stToolbar"] {{
    display: none !important;
}}

[data-testid="stDecoration"] {{
    display: none !important;
}}

#MainMenu, footer {{
    display: none !important;
}}

[data-testid="stAppViewContainer"] {{
    background:
        linear-gradient(90deg, rgba(2,6,14,0.96) 0%, rgba(2,6,14,0.82) 35%, rgba(2,6,14,0.25) 68%, rgba(2,6,14,0.72) 100%),
        url("data:image/jpeg;base64,{car_base64}");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    color: white;
}}

[data-testid="stAppViewContainer"] > .main {{
    background: transparent !important;
}}
[data-testid="stSidebar"] {{
    background: rgba(3, 8, 18, 0.88);
    border-right: 1px solid rgba(255,255,255,0.08);
}}

[data-testid="stSidebar"] * {{
    color: white;
}}

.block-container {{
    padding-top: 0rem !important;
    padding-bottom: 1.5rem;
}}

.logo-title {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 38px;
}}

.logo-img {{
    width: 52px;
    height: 52px;
    border-radius: 50%;
    object-fit: cover;
    box-shadow: 0 0 20px rgba(0,136,255,0.35);
}}

.brand-text {{
    font-size: 21px;
    font-weight: 700;
}}

.hero-title {{
    font-size: 46px;
    font-weight: 800;
    line-height: 1.15;
    margin-top: 55px;
}}

.blue {{
    color: #1683ff;
}}

.subtitle {{
    font-size: 21px;
    margin-top: 25px;
    color: rgba(255,255,255,0.9);
}}

.input-wrapper {{
    margin-top: 28px;
    max-width: 610px;
}}

.example-title {{
    margin-top: 48px;
    margin-bottom: 16px;
    font-size: 13px;
    letter-spacing: 1.2px;
    color: #b7c9df;
    font-weight: 700;
}}

.example-card {{
    padding: 14px 18px;
    margin-bottom: 12px;
    max-width: 520px;
    border-radius: 14px;
    background: rgba(8,17,34,0.72);
    border: 1px solid rgba(255,255,255,0.1);
    color: white;
}}

.glass-card {{
    background: rgba(4, 12, 25, 0.84);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 22px;
    box-shadow: 0 18px 45px rgba(0,0,0,0.35);
    backdrop-filter: blur(14px);
    margin-bottom: 18px;
}}

.card-title {{
    font-size: 14px;
    color: #c8d7ec;
    letter-spacing: 1px;
    font-weight: 700;
    margin-bottom: 18px;
}}

.metric-row {{
    display: flex;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 13px;
}}

.metric-label {{
    color: #aebcd0;
}}

.metric-value {{
    color: #1683ff;
    font-weight: 700;
}}

.answer-text {{
    color: rgba(255,255,255,0.92);
    line-height: 1.65;
    font-size: 14px;
}}

.source-item {{
    padding: 8px 0;
    color: rgba(255,255,255,0.9);
    font-size: 13px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}}

.footer-card {{
    margin-top: 70px;
    padding: 17px 22px;
    max-width: 740px;
    border-radius: 16px;
    background: rgba(4,12,25,0.76);
    border: 1px solid rgba(255,255,255,0.1);
    color: #b8c8dc;
}}

.stTextInput > div > div > input {{
    background: rgba(5,13,28,0.9) !important;
    color: white !important;
    border-radius: 14px !important;
    border: 1px solid rgba(36,137,255,0.75) !important;
    height: 54px !important;
    box-shadow: 0 0 32px rgba(0,115,255,0.25);
}}

.stTextInput > div > div > input::placeholder {{
    color: rgba(255,255,255,0.35) !important;
}}

.stButton > button {{
    background: linear-gradient(135deg, #006eff, #1683ff);
    color: white;
    border: none;
    border-radius: 14px;
    height: 46px;
    font-weight: 700;
}}

.sidebar-item {{
    padding: 15px 18px;
    margin-bottom: 12px;
    border-radius: 14px;
    background: rgba(11,24,45,0.62);
    border: 1px solid rgba(255,255,255,0.08);
}}

.active {{
    border-left: 4px solid #1683ff;
    background: rgba(22,131,255,0.15);
}}
.active {{
    border-left: 4px solid #1683ff;
    background: rgba(22,131,255,0.15);
}}

/* 👇 BURAYA EKLE */
button, [role="button"] {{
    outline: none !important;
    box-shadow: none !important;
}}

button:focus, [role="button"]:focus {{
    outline: none !important;
    box-shadow: none !important;
}}

button:active, [role="button"]:active {{
    outline: none !important;
    box-shadow: none !important;
}}

</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"""
    <div class="logo-title">
        <img class="logo-img" src="data:image/png;base64,{logo_base64}">
        <div class="brand-text">BMW AI Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("💬 New Chat", use_container_width=True):
        st.session_state.page = "home"
        st.session_state.result = None

    if st.button("❓ Ask Question", use_container_width=True):
        st.session_state.page = "ask"

    if st.button("ℹ️ About", use_container_width=True):
        st.session_state.page = "about"

    st.markdown("<div style='height: 300px;'></div>", unsafe_allow_html=True)

    st.button("🌙 Dark Mode", use_container_width=True)
    st.button("⚙️ Settings", use_container_width=True)
left, right = st.columns([1.65, 1])

if st.session_state.page == "home":
    with left:
        st.markdown("""
        <div class="hero-title">
            Hello,<br>
            <span class="blue">BMW AI Assistant</span> is here.
        </div>
        <div class="subtitle">How can I help you today?</div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)

        col_input, col_btn = st.columns([6, 1])

        with col_input:
            query = st.text_input(
                "Ask your question",
                placeholder="Ask your question about your BMW...",
                label_visibility="collapsed",
                key="home_query"
            )
        with col_btn:
                ask_clicked = st.button("➜", use_container_width=True, key="home_ask_btn")
        

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="example-title">EXAMPLE QUESTIONS</div>', unsafe_allow_html=True)

        examples = [
            "How do I turn on the headlights in my BMW?",
            "Apple CarPlay setup for BMW 3 Series (G20)",
            "What does error code P0456 mean?",
            "How do I change the BMW X5 (G05) key battery?"
        ]

        for ex in examples:
            st.markdown(f'<div class="example-card">✦ &nbsp; {ex}</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="footer-card">
            🛡️ &nbsp; Queries are processed in real-time. No personal data is stored.
        </div>
        """, unsafe_allow_html=True)
elif st.session_state.page == "ask":
    with left:
        st.markdown("<div class='hero-title'>Ask your BMW question</div>", unsafe_allow_html=True)

        st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)

        col_input, col_btn = st.columns([6, 1])

        with col_input:
            query = st.text_input(
                "Ask your question",
                placeholder="BMW 320i 2020 how to reset oil light",
                label_visibility="collapsed",
                key="ask_query"
            )

        with col_btn:
            ask_clicked = st.button("➜", use_container_width=True,key="ask_btn")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <div class="card-title">ASK QUESTION GUIDE</div>
            <div class="answer-text">
                <b>Supported:</b> model-specific questions, BMW feature usage, warning messages, and error codes.<br>
                <b>Example:</b> BMW 320i 2020 how to reset oil light<br>
                <b>Scope:</b> Designed for BMW owner manual assistance, not mechanical repair advice.
            </div>
        </div>
        """, unsafe_allow_html=True)
elif st.session_state.page == "about":
    with left:
        st.markdown("<div class='hero-title'>About</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
        This is a RAG-based BMW assistant.<br><br>

        <b>Features:</b><br>
        • Hybrid retrieval<br>
        • Reranking<br>
        • Model/year filtering<br><br>

        <b>Developer:</b><br>
        Mahmut Can Boran
        </div>
        """, unsafe_allow_html=True)

with right:
    result = st.session_state.result

    if ask_clicked and query.strip():
        try:
            response = requests.post(API_URL, json={"query": query}, timeout=120)
            response.raise_for_status()
            st.session_state.result = response.json()
            result = st.session_state.result
        except Exception as e:
            st.error(f"API error: {e}")

    if result:
        route = result.get("route", "-")
        retrieval_strategy = result.get("retrieval_strategy", "-")
        model = result.get("model") or "-"
        year = result.get("year") or "-"
        error_code = result.get("error_code") or "-"
        warning_note = result.get("warning_note") or "-"
        answer = result.get("answer", "No answer.")
        sources = result.get("sources", [])

        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">RESPONSE OVERVIEW</div>
            <div class="metric-row"><span class="metric-label">ROUTE</span><span class="metric-value">{route}</span></div>
            <div class="metric-row"><span class="metric-label">RETRIEVAL STRATEGY</span><span class="metric-value">{retrieval_strategy}</span></div>
            <div class="metric-row"><span class="metric-label">DETECTED MODEL</span><span>{model}</span></div>
            <div class="metric-row"><span class="metric-label">DETECTED YEAR</span><span>{year}</span></div>
            <div class="metric-row"><span class="metric-label">DETECTED ERROR CODE</span><span style="color:#ff8a00;font-weight:700;">{error_code}</span></div>
            <div class="metric-row"><span class="metric-label">WARNING NOTE</span><span>{warning_note}</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">ANSWER</div>
            <div class="answer-text">{answer}</div>
        </div>
        """, unsafe_allow_html=True)

        source_html = ""

        source_html = ""

        for i, src in enumerate(sources, start=1):
            if isinstance(src, dict):
                source_name = os.path.basename(src.get("source", "unknown"))
                page = src.get("page") or "-"

                source_html += f'<div class="source-item">Doc {i}: {source_name} · Page {page}</div>'
            else:
                source_html += f'<div class="source-item">Doc {i}: {src}</div>'

        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">SOURCES</div>
            {source_html if source_html else "<div class='source-item'>No sources found.</div>"}
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">RESPONSE OVERVIEW</div>
            <div class="metric-row"><span class="metric-label">ROUTE</span><span class="metric-value">waiting</span></div>
            <div class="metric-row"><span class="metric-label">RETRIEVAL STRATEGY</span><span class="metric-value">waiting</span></div>
            <div class="metric-row"><span class="metric-label">DETECTED MODEL</span><span>-</span></div>
            <div class="metric-row"><span class="metric-label">DETECTED YEAR</span><span>-</span></div>
            <div class="metric-row"><span class="metric-label">DETECTED ERROR CODE</span><span style="color:#ff8a00;font-weight:700;">-</span></div>
        </div>

        <div class="glass-card">
            <div class="card-title">ANSWER</div>
            <div class="answer-text">
                Ask a BMW manual or diagnostic question to see the answer here.
            </div>
        </div>

        <div class="glass-card">
            <div class="card-title">SOURCES</div>
            <div class="source-item">Doc 1: source will appear here · Page -</div>
        </div>
        """, unsafe_allow_html=True)