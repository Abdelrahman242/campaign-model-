import streamlit as st
import pandas as pd
import pickle

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CampaignIQ · Predict",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

CHANNEL_OPTIONS = [
    "Facebook", "Instagram", "Google Ads", "TikTok", "Email",
    "Facebok", "Insta_gram", "Gogle", "Tik_Tok", "E-mail",
]
ACTIVE_OPTIONS = ["Yes", "No", "True", "False", "Y", "1", "0"]

CHANNEL_ICONS = {
    "Facebook": "📘", "Facebok": "📘",
    "Instagram": "📸", "Insta_gram": "📸",
    "Google Ads": "🔍", "Gogle": "🔍",
    "TikTok": "🎵", "Tik_Tok": "🎵",
    "Email": "✉️", "E-mail": "✉️",
}

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: #080C14 !important;
    font-family: 'Inter', sans-serif !important;
    color: #E2E8F0 !important;
}

[data-testid="stAppViewContainer"] > .main {
    background: #080C14 !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 3px; }

/* ── Typography utils ── */
.label-sm {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748B;
}

/* ── Top nav bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 40px;
    background: rgba(13,17,23,0.95);
    border-bottom: 1px solid #111827;
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(12px);
}
.topbar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
}
.topbar-logo-icon {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #0EA5E9, #06B6D4);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    box-shadow: 0 0 20px rgba(14,165,233,0.4);
}
.topbar-brand {
    font-size: 17px;
    font-weight: 700;
    color: #F1F5F9;
    letter-spacing: -0.3px;
}
.topbar-brand span { color: #0EA5E9; }
.topbar-pill {
    background: rgba(14,165,233,0.12);
    border: 1px solid rgba(14,165,233,0.25);
    color: #38BDF8;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    padding: 4px 10px;
    border-radius: 20px;
}

/* ── Hero section ── */
.hero {
    padding: 56px 40px 40px;
    max-width: 1200px;
    margin: 0 auto;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #0EA5E9;
    margin-bottom: 16px;
}
.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    background: #0EA5E9;
    border-radius: 50%;
    box-shadow: 0 0 8px #0EA5E9;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(1.4); }
}
.hero-title {
    font-size: 40px;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -1.2px;
    line-height: 1.1;
    margin-bottom: 14px;
}
.hero-title span {
    background: linear-gradient(90deg, #0EA5E9, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 16px;
    color: #64748B;
    font-weight: 400;
    line-height: 1.6;
    max-width: 560px;
}

/* ── Stats strip ── */
.stats-strip {
    display: flex;
    gap: 1px;
    background: #111827;
    border: 1px solid #111827;
    border-radius: 14px;
    overflow: hidden;
    margin: 32px 40px;
    max-width: 1200px;
    margin-left: auto;
    margin-right: auto;
}
.stat-cell {
    flex: 1;
    padding: 20px 28px;
    background: #0D1117;
    transition: background 0.2s;
}
.stat-cell:hover { background: #111827; }
.stat-val {
    font-size: 24px;
    font-weight: 700;
    color: #F1F5F9;
    letter-spacing: -0.5px;
}
.stat-lbl {
    font-size: 12px;
    font-weight: 500;
    color: #475569;
    margin-top: 2px;
}
.stat-delta {
    font-size: 11px;
    font-weight: 600;
    color: #10B981;
    margin-top: 4px;
}

/* ── Main grid ── */
.main-grid {
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 24px;
    padding: 0 40px 60px;
    max-width: 1200px;
    margin: 0 auto;
}

/* ── Card ── */
.card {
    background: #0D1117;
    border: 1px solid #1E293B;
    border-radius: 16px;
    overflow: hidden;
}
.card-header {
    padding: 22px 28px 18px;
    border-bottom: 1px solid #1E293B;
    display: flex;
    align-items: center;
    gap: 10px;
}
.card-header-icon {
    width: 36px; height: 36px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.card-header-icon.blue  { background: rgba(14,165,233,0.12); }
.card-header-icon.teal  { background: rgba(6,182,212,0.12); }
.card-header-icon.slate { background: rgba(100,116,139,0.12); }
.card-title {
    font-size: 15px;
    font-weight: 600;
    color: #F1F5F9;
    letter-spacing: -0.2px;
}
.card-subtitle {
    font-size: 12px;
    color: #475569;
    margin-top: 1px;
}
.card-body { padding: 24px 28px; }

/* ── Form inputs (override Streamlit defaults) ── */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] input {
    background: #131B2A !important;
    border: 1px solid #1E3A5F !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stTextInput"] input:focus {
    border-color: #0EA5E9 !important;
    box-shadow: 0 0 0 3px rgba(14,165,233,0.15) !important;
    outline: none !important;
}
[data-testid="stSelectbox"] svg { color: #64748B !important; }

/* Labels */
[data-testid="stWidgetLabel"] p,
label {
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: #64748B !important;
    margin-bottom: 6px !important;
}

/* Dropdown popup */
[data-baseweb="popover"] ul,
[data-baseweb="menu"] {
    background: #131B2A !important;
    border: 1px solid #1E3A5F !important;
    border-radius: 12px !important;
}
[data-baseweb="menu"] li {
    color: #CBD5E1 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}
[data-baseweb="menu"] li:hover {
    background: rgba(14,165,233,0.1) !important;
    color: #38BDF8 !important;
}

/* Number input arrows */
[data-testid="stNumberInput"] button {
    background: #1E293B !important;
    border: none !important;
    color: #64748B !important;
    border-radius: 6px !important;
}
[data-testid="stNumberInput"] button:hover {
    background: #1E3A5F !important;
    color: #0EA5E9 !important;
}

/* ── Submit button ── */
[data-testid="stFormSubmitButton"] button {
    width: 100% !important;
    padding: 14px 24px !important;
    background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: -0.2px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 24px rgba(14,165,233,0.35) !important;
    margin-top: 8px !important;
}
[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(14,165,233,0.5) !important;
    background: linear-gradient(135deg, #38BDF8 0%, #0EA5E9 100%) !important;
}
[data-testid="stFormSubmitButton"] button:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 12px rgba(14,165,233,0.3) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #0EA5E9 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0D1117 !important;
    border: 1px solid #1E293B !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    color: #64748B !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1E293B !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Column gap ── */
[data-testid="stHorizontalBlock"] { gap: 16px !important; }

/* ── Result cards ── */
.result-card {
    border-radius: 16px;
    padding: 32px 28px;
    animation: slide-up 0.35s cubic-bezier(0.16,1,0.3,1) both;
}
@keyframes slide-up {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-positive {
    background: linear-gradient(145deg, #052e16 0%, #064e3b 100%);
    border: 1px solid #059669;
    box-shadow: 0 0 40px rgba(5,150,105,0.2);
}
.result-negative {
    background: linear-gradient(145deg, #1c0a0a 0%, #2d1010 100%);
    border: 1px solid #DC2626;
    box-shadow: 0 0 40px rgba(220,38,38,0.2);
}
.result-icon {
    font-size: 40px;
    margin-bottom: 12px;
    display: block;
    animation: pop 0.4s cubic-bezier(0.34,1.56,0.64,1) 0.1s both;
}
@keyframes pop {
    from { transform: scale(0.5); opacity: 0; }
    to   { transform: scale(1);   opacity: 1; }
}
.result-title {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
}
.result-positive .result-title { color: #34D399; }
.result-negative .result-title { color: #F87171; }
.result-desc {
    font-size: 13px;
    line-height: 1.6;
}
.result-positive .result-desc { color: #6EE7B7; }
.result-negative .result-desc { color: #FCA5A5; }
.result-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 16px;
}
.result-positive .result-badge {
    background: rgba(16,185,129,0.2);
    color: #10B981;
    border: 1px solid rgba(16,185,129,0.35);
}
.result-negative .result-badge {
    background: rgba(239,68,68,0.15);
    color: #EF4444;
    border: 1px solid rgba(239,68,68,0.3);
}

/* ── Sidebar card (right panel) ── */
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid #1E293B;
}
.info-row:last-child { border-bottom: none; }
.info-row-key {
    font-size: 12px;
    color: #475569;
    font-weight: 500;
}
.info-row-val {
    font-size: 13px;
    color: #CBD5E1;
    font-weight: 600;
}
.feature-chip {
    display: inline-block;
    background: rgba(14,165,233,0.1);
    border: 1px solid rgba(14,165,233,0.2);
    color: #38BDF8;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 6px;
    margin: 2px;
}

/* ── Divider ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1E293B 20%, #1E293B 80%, transparent);
    margin: 0;
}

/* ── Tooltip helper ── */
[data-testid="stTooltipHoverTarget"] { color: #334155 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# NAV BAR
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="topbar">
  <div class="topbar-logo">
    <div class="topbar-logo-icon">⚡</div>
    <span class="topbar-brand">Campaign<span>IQ</span></span>
  </div>
  <span class="topbar-pill">ML · SVC Model</span>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">AI-Powered Prediction Engine</div>
  <div class="hero-title">Predict Campaign<br><span>Performance</span> Instantly</div>
  <div class="hero-sub">
    Enter your campaign parameters below. Our trained SVC model will classify the
    expected outcome in real time — no guesswork required.
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STATS STRIP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="stats-strip">
  <div class="stat-cell">
    <div class="stat-val">SVC</div>
    <div class="stat-lbl">Model Architecture</div>
    <div class="stat-delta">sklearn Pipeline</div>
  </div>
  <div class="stat-cell">
    <div class="stat-val">5</div>
    <div class="stat-lbl">Input Features</div>
    <div class="stat-delta">Num + Cat encoded</div>
  </div>
  <div class="stat-cell">
    <div class="stat-val">Binary</div>
    <div class="stat-lbl">Classification</div>
    <div class="stat-delta">Class 0 · Class 1</div>
  </div>
  <div class="stat-cell">
    <div class="stat-val">RBF</div>
    <div class="stat-lbl">Kernel</div>
    <div class="stat-delta">C=1.0, gamma=scale</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT  (form  |  right panel)
# ═══════════════════════════════════════════════════════════════════════════════
col_form, col_info = st.columns([2.2, 1], gap="large")

# ── RIGHT PANEL ───────────────────────────────────────────────────────────────
with col_info:
    st.markdown("""
    <div style="padding: 0 0 0 0; margin-top: 0;">
      <div class="card">
        <div class="card-header">
          <div class="card-header-icon slate">🧠</div>
          <div>
            <div class="card-title">Model Info</div>
            <div class="card-subtitle">Pre-trained · Read-only</div>
          </div>
        </div>
        <div class="card-body">
          <div class="info-row">
            <span class="info-row-key">Type</span>
            <span class="info-row-val">sklearn Pipeline</span>
          </div>
          <div class="info-row">
            <span class="info-row-key">Classifier</span>
            <span class="info-row-val">SVC (RBF kernel)</span>
          </div>
          <div class="info-row">
            <span class="info-row-key">Output</span>
            <span class="info-row-val">Class 0 / Class 1</span>
          </div>
          <div class="info-row">
            <span class="info-row-key">Num. preprocessing</span>
            <span class="info-row-val">Impute → Scale</span>
          </div>
          <div class="info-row">
            <span class="info-row-key">Cat. preprocessing</span>
            <span class="info-row-val">Impute → OHE</span>
          </div>
          <div style="margin-top: 20px;">
            <div class="label-sm" style="margin-bottom: 10px;">Feature Set</div>
            <span class="feature-chip">Channel</span>
            <span class="feature-chip">Impressions</span>
            <span class="feature-chip">Clicks</span>
            <span class="feature-chip">Spend</span>
            <span class="feature-chip">Active</span>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div class="card-header">
        <div class="card-header-icon teal">💡</div>
        <div>
          <div class="card-title">How it works</div>
          <div class="card-subtitle">3-step inference</div>
        </div>
      </div>
      <div class="card-body">
        <div style="display:flex;flex-direction:column;gap:16px;">
          <div style="display:flex;gap:12px;align-items:flex-start;">
            <div style="width:24px;height:24px;border-radius:50%;background:rgba(14,165,233,0.15);border:1px solid rgba(14,165,233,0.3);display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#0EA5E9;flex-shrink:0;">1</div>
            <div style="font-size:13px;color:#64748B;line-height:1.5;">Fill in all five campaign fields in the form.</div>
          </div>
          <div style="display:flex;gap:12px;align-items:flex-start;">
            <div style="width:24px;height:24px;border-radius:50%;background:rgba(14,165,233,0.15);border:1px solid rgba(14,165,233,0.3);display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#0EA5E9;flex-shrink:0;">2</div>
            <div style="font-size:13px;color:#64748B;line-height:1.5;">The pipeline preprocesses inputs — scales numerics, encodes categoricals.</div>
          </div>
          <div style="display:flex;gap:12px;align-items:flex-start;">
            <div style="width:24px;height:24px;border-radius:50%;background:rgba(14,165,233,0.15);border:1px solid rgba(14,165,233,0.3);display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#0EA5E9;flex-shrink:0;">3</div>
            <div style="font-size:13px;color:#64748B;line-height:1.5;">SVC classifies the input and returns a binary prediction.</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── FORM (LEFT) ───────────────────────────────────────────────────────────────
with col_form:
    st.markdown("""
    <div class="card">
      <div class="card-header">
        <div class="card-header-icon blue">📋</div>
        <div>
          <div class="card-title">Campaign Parameters</div>
          <div class="card-subtitle">All fields required for prediction</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    # Invisible card body — Streamlit form goes here
    st.markdown("<div style='background:#0D1117;border:1px solid #1E293B;border-top:none;border-radius:0 0 16px 16px;padding:24px 28px;'>", unsafe_allow_html=True)

    with st.form("predict_form"):
        r1c1, r1c2 = st.columns(2, gap="medium")
        with r1c1:
            channel = st.selectbox(
                "Channel",
                options=CHANNEL_OPTIONS,
                format_func=lambda x: f"{CHANNEL_ICONS.get(x,'')}  {x}",
                help="Marketing channel for the campaign",
            )
        with r1c2:
            active = st.selectbox(
                "Active",
                options=ACTIVE_OPTIONS,
                help="Is the campaign currently active?",
            )

        r2c1, r2c2, r2c3 = st.columns(3, gap="medium")
        with r2c1:
            impressions = st.number_input(
                "Impressions",
                min_value=0,
                value=10_000,
                step=500,
                help="Total ad impressions",
            )
        with r2c2:
            clicks = st.number_input(
                "Clicks",
                min_value=0,
                value=500,
                step=10,
                help="Total clicks on the ad",
            )
        with r2c3:
            spend = st.number_input(
                "Spend ($)",
                min_value=0.0,
                value=200.0,
                step=10.0,
                format="%.2f",
                help="Total spend in USD",
            )

        # Derived quick metrics
        ctr = (clicks / impressions * 100) if impressions > 0 else 0.0
        cpc = (spend / clicks) if clicks > 0 else 0.0
        st.markdown(f"""
        <div style="display:flex;gap:12px;margin:16px 0 20px;">
          <div style="flex:1;background:#131B2A;border:1px solid #1E293B;border-radius:10px;padding:14px 18px;">
            <div class="label-sm">CTR</div>
            <div style="font-size:22px;font-weight:700;color:#0EA5E9;margin-top:4px;">{ctr:.2f}%</div>
          </div>
          <div style="flex:1;background:#131B2A;border:1px solid #1E293B;border-radius:10px;padding:14px 18px;">
            <div class="label-sm">CPC</div>
            <div style="font-size:22px;font-weight:700;color:#06B6D4;margin-top:4px;">${cpc:.2f}</div>
          </div>
          <div style="flex:1;background:#131B2A;border:1px solid #1E293B;border-radius:10px;padding:14px 18px;">
            <div class="label-sm">Channel</div>
            <div style="font-size:22px;font-weight:700;color:#38BDF8;margin-top:4px;">{CHANNEL_ICONS.get(channel,'⚡')}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "⚡  Run Prediction",
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── RESULT ────────────────────────────────────────────────────────────────
    if submitted:
        input_df = pd.DataFrame({
            "Channel":     [channel],
            "Impressions": [impressions],
            "Clicks":      [clicks],
            "Spend":       [spend],
            "Active":      [active],
        })

        with st.spinner(""):
            prediction = model.predict(input_df)[0]

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-card result-positive">
              <span class="result-badge">Positive Signal</span>
              <span class="result-icon">🚀</span>
              <div class="result-title">High Performance Predicted</div>
              <div class="result-desc">
                The model classifies this campaign as <strong>Class 1</strong> — a positive outcome.
                Your {channel} campaign with {impressions:,} impressions and ${spend:,.2f} spend
                is predicted to perform well.
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-negative">
              <span class="result-badge">Needs Attention</span>
              <span class="result-icon">⚠️</span>
              <div class="result-title">Underperformance Predicted</div>
              <div class="result-desc">
                The model classifies this campaign as <strong>Class 0</strong> — a negative outcome.
                Consider adjusting budget, creative, or targeting for your {channel} campaign.
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        with st.expander("🔎  Inspect raw input payload"):
            st.dataframe(input_df, use_container_width=True, hide_index=True)
