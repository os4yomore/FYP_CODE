"""
app.py — SCD Pain Management Decision Support System

Streamlit clinician-facing interface for the trained CQL agent.
Implements the UI specified in Chapter 3, Section 3.5.2.

Run with:
    streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import altair as alt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit.watcher")
# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
DOSE_MG = {0: 0, 1: 5, 2: 10, 3: 15, 4: 20}
ACTION_LABELS = {
    0: "No dose",
    1: "Low dose",
    2: "Medium dose",
    3: "High dose",
    4: "Maximum dose"
}
ACTION_DESCRIPTIONS = {
    0: "Withhold opioid medication",
    1: "Administer 5 mg morphine equivalent",
    2: "Administer 10 mg morphine equivalent",
    3: "Administer 15 mg morphine equivalent",
    4: "Administer 20 mg morphine equivalent"
}
N_ACTIONS = 5
STATE_DIM = 5
MAX_DOSE_24H = 80
MAX_HOURS = 120
PATIENT_TYPE_LABELS = {0: "Mild", 1: "Moderate", 2: "Severe"}

# Clinical palette
C = {
    "primary":      "#2C5282",
    "primary_dark": "#1A365D",
    "primary_soft": "#EBF4FB",
    "text":         "#1A202C",
    "text_soft":    "#4A5568",
    "text_label":   "#2D3748",
    "border":       "#CBD5E0",
    "border_soft":  "#E2E8F0",
    "background":   "#F7FAFC",
    "card":         "#FFFFFF",
    "safe":         "#2F855A",
    "safe_soft":    "#E6F4EA",
    "caution":      "#B7791F",
    "caution_soft": "#FEF3C7",
    "danger":       "#9B2C2C",
    "danger_soft":  "#FED7D7",
}


# ─────────────────────────────────────────────────────────────
# Q-NETWORK
# ─────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x):
        return self.network(x)


@st.cache_resource
def load_agent(model_path: str = "cql_agent.pth"):
    net = QNetwork()
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()
    return net


def get_safe_actions(pain_score, tolerance, cumulative_dose):
    safe = []
    status = {}
    for action in range(N_ACTIONS):
        dose = DOSE_MG[action]
        violations = []
        if cumulative_dose + dose > MAX_DOSE_24H:
            violations.append(
                f"Would exceed 24h dose limit ({int(cumulative_dose + dose)} > 80 mg)"
            )
        if pain_score < 3.0 and action >= 3:
            violations.append("High dose not indicated at low pain (< 3)")
        if tolerance > 0.8 and action == 4:
            violations.append("Maximum dose not recommended at high tolerance")
        if not violations:
            safe.append(action)
            status[action] = ("safe", [])
        else:
            status[action] = ("unsafe", violations)
    if not safe:
        safe = [0]
        status[0] = ("safe", [])
    return safe, status


def normalise_state(pain_score, tolerance, cumulative_dose,
                    time_since_crisis, patient_type):
    return np.array([
        pain_score / 10.0,
        tolerance,
        cumulative_dose / MAX_DOSE_24H,
        time_since_crisis / MAX_HOURS,
        patient_type / 2.0
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SCD Pain Management — Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(f"""
<style>
    /* ─── BASE ────────────────────────────────────────── */
    .stApp {{
        background-color: {C["background"]};
    }}
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }}
    #MainMenu, footer, header {{ visibility: hidden; }}

    /* ─── TYPOGRAPHY ──────────────────────────────────── */
    html, body, [class*="css"] {{
        font-size: 16px;
        color: {C["text"]};
    }}

    /* Slider labels (the variable names above each slider) */
    .stSlider > label,
    .stSelectbox > label {{
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: {C["text_label"]} !important;
        margin-bottom: 0.4rem !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }}
    /* Slider tick value below the slider */
    .stSlider [data-baseweb="slider"] {{
        margin-top: 0.5rem;
    }}

    /* ─── HEADER ──────────────────────────────────────── */
    .app-header {{
        background: linear-gradient(135deg, {C["primary"]} 0%, {C["primary_dark"]} 100%);
        padding: 2rem 2.25rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 2px 8px rgba(44, 82, 130, 0.15);
    }}
    .app-header h1 {{
        color: white !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        letter-spacing: -0.02em;
    }}
    .app-header p {{
        color: #CBD5E0 !important;
        margin: 0.4rem 0 0 0 !important;
        font-size: 1.05rem !important;
        font-weight: 400;
    }}

    /* ─── SECTION HEADINGS ────────────────────────────── */
    .section-title {{
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {C["text_soft"]};
        margin-bottom: 1rem;
        margin-top: 0.5rem;
        border-bottom: 2px solid {C["border_soft"]};
        padding-bottom: 0.6rem;
    }}

    /* ─── INPUT PANEL ─────────────────────────────────── */
    .input-panel {{
        background-color: {C["card"]};
        border: 1px solid {C["border_soft"]};
        border-radius: 10px;
        padding: 1.5rem;
    }}

    /* ─── RECOMMENDATION CARD ─────────────────────────── */
    .rec-card {{
        background-color: {C["card"]};
        border-left: 6px solid {C["primary"]};
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }}
    .rec-card.rec-withhold {{
        border-left-color: {C["text_soft"]};
        background-color: #F8F9FA;
    }}
    .rec-card.rec-max {{
        border-left-color: {C["caution"]};
        background-color: {C["caution_soft"]};
    }}
    .rec-label {{
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {C["text_soft"]};
        margin: 0;
    }}
    .rec-dose {{
        font-size: 3rem;
        font-weight: 700;
        color: {C["text"]};
        margin: 0.3rem 0;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }}
    .rec-action {{
        font-size: 1.15rem;
        font-weight: 600;
        color: {C["primary"]};
        margin: 0.3rem 0 0.2rem 0;
    }}
    .rec-desc {{
        font-size: 0.95rem;
        color: {C["text_soft"]};
        margin: 0;
    }}

    /* ─── STATUS BADGES ───────────────────────────────── */
    .status-badge {{
        display: inline-block;
        padding: 0.45rem 0.9rem;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .status-safe {{
        background-color: {C["safe_soft"]};
        color: {C["safe"]};
        border: 1.5px solid {C["safe"]};
    }}
    .status-caution {{
        background-color: {C["caution_soft"]};
        color: {C["caution"]};
        border: 1.5px solid {C["caution"]};
    }}
    .status-danger {{
        background-color: {C["danger_soft"]};
        color: {C["danger"]};
        border: 1.5px solid {C["danger"]};
    }}

    /* ─── METRIC TILES ────────────────────────────────── */
    .metric-box {{
        background-color: {C["primary_soft"]};
        border: 1px solid {C["border_soft"]};
        border-radius: 8px;
        padding: 1.1rem 0.75rem;
        text-align: center;
        height: 100%;
    }}
    .metric-value {{
        font-size: 1.7rem;
        font-weight: 700;
        color: {C["primary"]};
        margin: 0;
        line-height: 1.2;
    }}
    .metric-label {{
        font-size: 0.78rem;
        color: {C["text_soft"]};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 0.3rem 0 0 0;
        font-weight: 600;
    }}

    /* ─── DISCLAIMER ──────────────────────────────────── */
    .disclaimer {{
        background-color: {C["caution_soft"]};
        border-left: 4px solid {C["caution"]};
        padding: 0.85rem 1.1rem;
        border-radius: 6px;
        font-size: 0.95rem;
        color: {C["text"]};
        margin: 0 0 1.5rem 0;
    }}

    /* ─── ACTION ROWS ─────────────────────────────────── */
    .action-row {{
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin: 0.35rem 0;
        font-size: 0.95rem;
        border: 1px solid transparent;
    }}
    .action-row.safe {{
        background-color: {C["safe_soft"]};
        color: {C["text"]};
        border-color: #B7EBC6;
    }}
    .action-row.unsafe {{
        background-color: #F1F3F5;
        color: {C["text_soft"]};
    }}
    .action-row.recommended {{
        background-color: {C["primary_soft"]};
        color: {C["primary"]};
        font-weight: 600;
        border: 1.5px solid {C["primary"]};
    }}
    .action-name {{
        font-weight: 600;
    }}
    .action-detail {{
        font-size: 0.85rem;
        color: {C["text_soft"]};
        margin-left: 0.5rem;
    }}

    /* ─── PATIENT SUMMARY ─────────────────────────────── */
    .summary-row {{
        display: flex;
        justify-content: space-between;
        padding: 0.55rem 0;
        border-bottom: 1px solid {C["border_soft"]};
        font-size: 0.95rem;
    }}
    .summary-row:last-child {{
        border-bottom: none;
    }}
    .summary-label {{
        color: {C["text_soft"]};
        font-weight: 500;
    }}
    .summary-value {{
        color: {C["text"]};
        font-weight: 600;
    }}

    /* ─── FOOTER ──────────────────────────────────────── */
    .app-footer {{
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid {C["border_soft"]};
        font-size: 0.85rem;
        color: {C["text_soft"]};
        text-align: center;
        line-height: 1.6;
    }}

    /* Streamlit element overrides */
    .stExpander {{
        border: 1px solid {C["border_soft"]};
        border-radius: 8px;
    }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>Sickle Cell Disease — Pain Management</h1>
    <p>Reinforcement Learning Decision Support System</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD AGENT
# ─────────────────────────────────────────────────────────────
try:
    agent = load_agent()
except FileNotFoundError:
    st.error(
        "Trained model file 'cql_agent.pth' not found. "
        "Please ensure the model has been trained and saved before running this app."
    )
    st.stop()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


# ─────────────────────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    <strong>Clinical decision support tool.</strong> All recommendations require clinician review before implementation. This system supports — but does not replace — clinical judgement.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN LAYOUT — 3 columns: inputs | recommendation | summary/safety
# ─────────────────────────────────────────────────────────────
col_input, col_rec, col_safety = st.columns([1, 1.1, 1], gap="large")

# ── INPUT COLUMN ─────────────────────────────────────────
with col_input:
    st.markdown('<div class="section-title">Patient Status</div>', unsafe_allow_html=True)

    pain_score = st.slider(
        "Pain score (NRS 0–10)",
        min_value=0.0, max_value=10.0, value=6.0, step=0.5,
        help="Numeric Rating Scale — patient's reported pain level."
    )

    tolerance = st.slider(
        "Estimated opioid tolerance",
        min_value=0.0, max_value=1.0, value=0.2, step=0.05,
        help="Clinical estimate: 0 = opioid-naive, 1 = highly tolerant."
    )

    cumulative_dose = st.slider(
        "Cumulative dose, last 24h (mg)",
        min_value=0, max_value=80, value=20, step=5,
        help="Total morphine-equivalent dose in the last 24 hours."
    )

    time_since_crisis = st.slider(
        "Hours since crisis onset",
        min_value=0, max_value=120, value=24, step=4,
        help="Time elapsed since the start of the VOC episode."
    )

    patient_type = st.selectbox(
        "Patient phenotype",
        options=[0, 1, 2],
        format_func=lambda x: PATIENT_TYPE_LABELS[x],
        index=1,
        help="Clinical assessment of SCD severity."
    )


# ─────────────────────────────────────────────────────────────
# COMPUTE RECOMMENDATION
# ─────────────────────────────────────────────────────────────
state_vec = normalise_state(
    pain_score, tolerance, cumulative_dose,
    time_since_crisis, patient_type
)
safe_actions, safety_status = get_safe_actions(
    pain_score, tolerance, cumulative_dose
)
state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
with torch.no_grad():
    q_values = agent(state_tensor).squeeze(0).numpy()

recommended_action = max(safe_actions, key=lambda a: q_values[a])
recommended_dose_mg = DOSE_MG[recommended_action]
n_safe = len(safe_actions)
headroom = MAX_DOSE_24H - cumulative_dose

# ── RECOMMENDATION COLUMN ────────────────────────────────
with col_rec:
    st.markdown('<div class="section-title">Recommendation</div>', unsafe_allow_html=True)

    if recommended_action == 0:
        card_class = "rec-card rec-withhold"
        dose_display = "Withhold"
    elif recommended_action == 4:
        card_class = "rec-card rec-max"
        dose_display = f"{recommended_dose_mg} mg"
    else:
        card_class = "rec-card"
        dose_display = f"{recommended_dose_mg} mg"

    st.markdown(f"""
    <div class="{card_class}">
        <p class="rec-label">Suggested Action</p>
        <p class="rec-dose">{dose_display}</p>
        <p class="rec-action">{ACTION_LABELS[recommended_action]}</p>
        <p class="rec-desc">{ACTION_DESCRIPTIONS[recommended_action]}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="metric-box">
            <p class="metric-value">{n_safe}/{N_ACTIONS}</p>
            <p class="metric-label">Actions Permitted</p>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-box">
            <p class="metric-value">{headroom} mg</p>
            <p class="metric-label">24h Headroom</p>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-box">
            <p class="metric-value">{q_values[recommended_action]:.2f}</p>
            <p class="metric-label">Agent Q-value</p>
        </div>
        """, unsafe_allow_html=True)

    # Patient summary below metric tiles
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Patient Summary</div>', unsafe_allow_html=True)

    summary_html = f"""
    <div style="background: white; border: 1px solid {C['border_soft']}; border-radius: 8px; padding: 0.5rem 1.1rem;">
        <div class="summary-row">
            <span class="summary-label">Phenotype</span>
            <span class="summary-value">{PATIENT_TYPE_LABELS[patient_type]}</span>
        </div>
        <div class="summary-row">
            <span class="summary-label">Current pain (NRS)</span>
            <span class="summary-value">{pain_score:.1f} / 10</span>
        </div>
        <div class="summary-row">
            <span class="summary-label">Estimated tolerance</span>
            <span class="summary-value">{tolerance:.2f}</span>
        </div>
        <div class="summary-row">
            <span class="summary-label">24h cumulative dose</span>
            <span class="summary-value">{cumulative_dose} mg</span>
        </div>
        <div class="summary-row">
            <span class="summary-label">Hours since onset</span>
            <span class="summary-value">{time_since_crisis} h</span>
        </div>
    </div>
    """
    st.markdown(summary_html, unsafe_allow_html=True)


# ── SAFETY COLUMN ────────────────────────────────────────
with col_safety:
    st.markdown('<div class="section-title">Safety Status</div>', unsafe_allow_html=True)

    if n_safe == N_ACTIONS:
        badge_class = "status-safe"
        badge_text = "All actions permitted"
    elif n_safe >= 3:
        badge_class = "status-caution"
        badge_text = f"{N_ACTIONS - n_safe} action(s) blocked"
    else:
        badge_class = "status-danger"
        badge_text = f"{N_ACTIONS - n_safe} action(s) blocked"

    st.markdown(f"""
    <div style="margin-bottom: 1rem;">
        <span class="status-badge {badge_class}">{badge_text}</span>
    </div>
    """, unsafe_allow_html=True)

    for action in range(N_ACTIONS):
        status, violations = safety_status[action]
        dose_label = f"{ACTION_LABELS[action]} ({DOSE_MG[action]} mg)"

        if action == recommended_action:
            row_class = "action-row recommended"
            detail = "  ←  recommended"
        elif "safe" in status:
            row_class = "action-row safe"
            detail = "  ✓ permitted"
        else:
            row_class = "action-row unsafe"
            detail = f"  ✗ {violations[0]}"

        st.markdown(f"""
        <div class="{row_class}">
            <span class="action-name">{dose_label}</span><span class="action-detail">{detail}</span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Q-VALUE TRANSPARENCY (full-width)
# ─────────────────────────────────────────────────────────────
st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Decision Transparency — Agent Q-Values</div>', unsafe_allow_html=True)

st.markdown(f"""
<p style="color: {C['text_soft']}; font-size: 0.95rem; margin-bottom: 1.25rem; line-height: 1.5;">
The agent computes a Q-value for each action — an estimate of expected long-term clinical outcome quality.
Higher values indicate the agent expects better outcomes. <strong style="color: {C['text']};">Unsafe actions are never selected,
regardless of their Q-value</strong> — they are filtered out by the structural safety layer before action selection.
</p>
""", unsafe_allow_html=True)

chart_data = pd.DataFrame({
    "Action":  [f"{ACTION_LABELS[a]}\n({DOSE_MG[a]} mg)" for a in range(N_ACTIONS)],
    "Q-value": [float(q_values[a]) for a in range(N_ACTIONS)],
    "Status":  ["Recommended" if a == recommended_action
                else ("Safe" if "safe" in safety_status[a][0] else "Blocked")
                for a in range(N_ACTIONS)],
    "Order":   list(range(N_ACTIONS)),
})

colour_scale = alt.Scale(
    domain=["Recommended", "Safe", "Blocked"],
    range=[C["primary"], C["safe"], "#718096"]
)

# Numeric Q-value labels above each bar
text_labels = alt.Chart(chart_data).mark_text(
    align="center",
    baseline="bottom",
    dy=-6,
    fontSize=12,
    fontWeight=600,
    color="#1A202C"
).encode(
    x=alt.X("Action:N", sort=alt.SortField(field="Order")),
    y=alt.Y("Q-value:Q"),
    text=alt.Text("Q-value:Q", format=".2f")
)

bars = alt.Chart(chart_data).mark_bar(
    cornerRadiusTopLeft=5,
    cornerRadiusTopRight=5,
).encode(
    x=alt.X("Action:N",
            title=None,
            sort=alt.SortField(field="Order"),
            axis=alt.Axis(
                labelAngle=0,
                labelFontSize=13,
                labelFontWeight="bold",
                labelColor="#1A202C",
                labelPadding=10,
                domain=False,
                ticks=False
            )),
    y=alt.Y("Q-value:Q",
            title="Q-value (expected outcome quality)",
            axis=alt.Axis(
                titleFontSize=13,
                titleFontWeight="bold",
                titleColor="#1A202C",
                titlePadding=15,
                labelFontSize=12,
                labelFontWeight=500,
                labelColor="#1A202C",
                gridColor="#CBD5E0"
            )),
    color=alt.Color("Status:N",
                    scale=colour_scale,
                    legend=alt.Legend(
                        title=None,
                        orient="top",
                        labelFontSize=13,
                        labelFontWeight="bold",
                        labelColor="#1A202C",
                        symbolSize=200,
                        symbolStrokeWidth=2,
                        padding=12,
                        rowPadding=4
                    )),
    tooltip=[
        alt.Tooltip("Action:N", title="Action"),
        alt.Tooltip("Q-value:Q", title="Q-value", format=".3f"),
        alt.Tooltip("Status:N", title="Status")
    ]
)

final_chart = (bars + text_labels).properties(
    height=420,
    background="white",
    padding={"left": 20, "right": 20, "top": 30, "bottom": 20}
).configure_view(
    strokeWidth=0,
    fill="white"
).configure_axis(
    domainColor="#CBD5E0"
)

st.altair_chart(final_chart, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# ABOUT
# ─────────────────────────────────────────────────────────────
with st.expander("About this system"):
    st.markdown(f"""
**System Architecture**

This decision support system uses a Conservative Q-Learning (CQL) agent trained on a synthetic
offline dataset of 2,000 patient trajectories. The agent learned dosing strategies by analysing
the consequences of historical decisions made under multiple behaviour policies.

**Two-Layer Safety Architecture**

1. **Conservative Q-Learning** (Kumar et al., 2020) provides pessimism against actions poorly
   represented in the training data, preventing overconfident recommendations for untested treatments.

2. **Hard constraint filter** at every recommendation excludes any action that would violate
   clinical safety rules — the 24-hour dose limit, dose-pain compatibility constraints, and
   tolerance-dose limits.

**Limitations**

- Recommendations are based on a synthetic simulator, not real patient data
- This is a proof-of-concept research tool, not a clinically validated system
- All recommendations require clinician review before implementation
- The system supports, but does not replace, clinical judgement

**Key References**

- Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). Conservative Q-learning for offline
  reinforcement learning. *NeurIPS, 33*, 1179–1191.
- Levine, S., Kumar, A., Tucker, G., & Fu, J. (2020). Offline reinforcement learning: Tutorial,
  review, and perspectives on open problems. *arXiv:2005.01643*.
- Gottesman, O., et al. (2019). Guidelines for reinforcement learning in healthcare.
  *Nature Medicine, 25*(1), 16–18.
""")


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <strong>Beyond Standardised Protocols</strong><br>
    A Safety-Constrained RL Framework for SCD Pain Management<br>
    Pan-Atlantic University · 2026
</div>
""", unsafe_allow_html=True)