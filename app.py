import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

MODEL_DIR = "../model" if os.path.exists("../model") else "model"
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.joblib")
MEANS_PATH = os.path.join(MODEL_DIR, "feature_means.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.joblib")
META_PATH = os.path.join(MODEL_DIR, "model_meta.joblib")

required = [MODEL_PATH, SCALER_PATH, ENCODERS_PATH, MEANS_PATH, FEATURES_PATH]
if not all(os.path.exists(p) for p in required):
    st.set_page_config(page_title="Fraud Detection — Error", layout="centered")
    st.error("Model files missing. Train the model first.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODERS_PATH)
feature_means = joblib.load(MEANS_PATH)
feature_columns = joblib.load(FEATURES_PATH)
model_meta = joblib.load(META_PATH) if os.path.exists(META_PATH) else {"probability_threshold": 0.5}
threshold = model_meta.get("probability_threshold", 0.5)
cat_cols = list(encoders.keys())

st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    :root{
        --bg-1: #0f172a;
        --bg-2: #0b1220;
        --accent: #6366f1;
        --accent-2: #06b6d4;
        --muted: #94a3b8;
        --card-radius: 12px;
        --glass: rgba(255,255,255,0.06);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background: radial-gradient(1200px 400px at 10% 10%, rgba(99,102,241,0.08), transparent 12%),
                    radial-gradient(1000px 300px at 90% 90%, rgba(6,182,212,0.04), transparent 12%),
                    linear-gradient(180deg, #f7fbff 0%, #ffffff 100%);
        color: #0f172a;
    }

    /* Remove the large default white block background Streamlit adds */
    [data-testid="stAppViewContainer"],
    [data-testid="stMainContent"],
    [data-testid="stBlock"] {
        background: transparent !important;
    }

    .app-container{
        max-width: 1100px;
        margin: 28px auto;
        padding: 18px;
    }

    .top-bar{
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:12px;
        margin-bottom: 14px;
    }

    .brand {
        display:flex;
        gap:12px;
        align-items:center;
    }
    .logo {
        width:54px;
        height:54px;
        border-radius:12px;
        background: linear-gradient(135deg, rgba(99,102,241,0.95), rgba(6,182,212,0.9));
        display:flex;
        align-items:center;
        justify-content:center;
        color:white;
        font-weight:700;
        box-shadow: 0 8px 30px rgba(99,102,241,0.12);
        font-size:20px;
    }

    .title {
        font-size:18px;
        font-weight:700;
        color:white;
    }
    .subtitle {
        color: var(--muted);
        font-size:13px;
    }

    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(250,250,255,0.85));
        border-radius: var(--card-radius);
        padding: 14px;
        box-shadow: 0 8px 30px rgba(15,23,42,0.06);
        border: 1px solid rgba(99,102,241,0.06);
    }

    .card-compact {
        padding: 12px;
        border-radius: 10px;
    }

    .inputs-grid {
        display: grid;
        grid-template-columns: 1fr 420px;
        gap: 18px;
        align-items: start;
    }

    .section-title{
        font-size:15px;
        font-weight:600;
        margin-bottom:8px;
    }

    .muted { color: var(--muted); font-size:13px; }

    .result-card {
        border-radius: 12px;
        padding: 14px;
    }

    .risk-high {
        background: linear-gradient(90deg,#fff5f5,#fff6f6);
        border: 1px solid rgba(185,28,28,0.06);
    }
    .risk-low {
        background: linear-gradient(90deg,#f0fdf4,#f8fef7);
        border: 1px solid rgba(16,185,129,0.06);
    }

    .big-number { font-size:26px; font-weight:700; color:white; margin-top:6px; }
    .small { font-size:13px; color:var(--muted); }

    div.stButton > button:first-child {
        background: linear-gradient(90deg,var(--accent), #3b82f6);
        color: white;
        border: none;
        padding: 10px 14px;
        border-radius: 10px;
        font-weight:600;
        box-shadow: 0 8px 22px rgba(59,130,246,0.14);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 28px rgba(59,130,246,0.20);
    }

    .card .stNumberInput, .card .stSelectbox, .card .stDateInput, .card .stTimeInput {
        margin-bottom: 8px;
    }

    .footer-note {
        text-align:center;
        color: #6b7280;
        margin-top: 20px;
    }

    @media(max-width:900px){
        .inputs-grid { grid-template-columns: 1fr; }
        .logo { width:46px; height:46px; font-size:18px }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='app-container'>", unsafe_allow_html=True)
st.markdown("<div class='top-bar'>", unsafe_allow_html=True)
st.markdown("<div class='brand'><div class='logo'>💳</div><div><div class='title'>Fraud Detection — Predict Transaction Risk</div><div class='subtitle'>Fast scoring • Actionable recommendations</div></div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

def safe_log1p(val):
    try:
        v = float(val)
        if v < 0:
            v = 0.0
        return np.log1p(v)
    except:
        return 0.0

def safe_map(cat_col, val):
    mapping = encoders.get(cat_col, {})
    if pd.isna(val):
        return feature_means.get(cat_col, np.mean(list(mapping.values())) if mapping else 0.0)
    return mapping.get(val, np.mean(list(mapping.values())) if mapping else 0.0)

def build_input_df(user_inputs: dict):
    row = {}
    for feature in feature_columns:
        if feature in user_inputs:
            row[feature] = user_inputs[feature]
        elif feature in cat_cols:
            row[feature] = feature_means.get(feature, 0.0)
        else:
            row[feature] = feature_means.get(feature, 0.0)
    for c in cat_cols:
        if c in user_inputs:
            row[c] = safe_map(c, user_inputs[c])
    return pd.DataFrame([row])

st.markdown("<div class='inputs-grid'>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='card card-compact'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Transaction details</div>", unsafe_allow_html=True)
    trans_amount = st.number_input("Transaction amount (USD)", min_value=0.0, value=float(feature_means.get("transactionAmount", 100.0)))
    tdate = st.date_input("Transaction date", value=datetime.now().date(), key="trans_date", help="Select date")
    ttime = st.time_input("Transaction time", value=datetime.now().time().replace(microsecond=0), key="trans_time", help="Select time")
    try:
        trans_dt = datetime.combine(tdate, ttime)
    except Exception:
        trans_dt = datetime.now()

    cols = st.columns(3)
    with cols[0]:
        trans_type = st.selectbox("Type", ["PURCHASE", "CASH_OUT", "TRANSFER", "DEBIT", "REFUND"], 0)
    with cols[1]:
        merchant_category = st.selectbox("Merchant category", ["retail", "food", "travel", "entertainment", "tech", "other"], 0)
    with cols[2]:
        acquiring_country = st.selectbox("Acquiring country", ["US", "CA", "GB", "AU", "IN", "other"], 0)

    pos_mode = st.selectbox("Entry method", ["swipe", "chip", "contactless", "manual", "online"], 2)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Optional financial details</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        current_balance = st.number_input("Current balance (USD)", value=float(feature_means.get("currentBalance", 0.0)))
    with col2:
        credit_limit = st.number_input("Credit limit (USD)", value=float(feature_means.get("creditLimit", 0.0)))

    with st.expander("More optional details", expanded=False):
        avail_money = st.number_input("Available funds (USD)", value=float(feature_means.get("availableMoney", 0.0)))
        days_since_open = st.number_input("Days since account opened", value=float(feature_means.get("days_since_account_open", 0.0)))
        days_since_addr = st.number_input("Days since address change", value=float(feature_means.get("days_since_address_change", 0.0)))

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Predict fraud risk", key="predict")

with st.container():
    st.markdown("<div class='section-title'>Prediction</div>", unsafe_allow_html=True)
    result_placeholder = st.empty()
    st.markdown("<div class='muted'>Ready — enter transaction details and click Predict</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

if predict_btn:
    user_inputs = {
        "transactionAmount": safe_log1p(trans_amount),
        "currentBalance": safe_log1p(current_balance),
        "creditLimit": safe_log1p(credit_limit),
        "availableMoney": safe_log1p(avail_money if 'avail_money' in locals() else feature_means.get("availableMoney", 0.0)),
        "trans_hour": int(trans_dt.hour),
        "trans_dayofweek": int(trans_dt.weekday()),
        "days_since_account_open": float(days_since_open if 'days_since_open' in locals() else feature_means.get("days_since_account_open", 0.0)),
        "days_since_address_change": float(days_since_addr if 'days_since_addr' in locals() else feature_means.get("days_since_address_change", 0.0)),
        "transactionType": trans_type,
        "merchantCategoryCode": merchant_category,
        "acqCountry": acquiring_country,
        "posEntryMode": pos_mode,
        "cardPresent": 1
    }

    input_df = build_input_df(user_inputs)
    X_scaled = scaler.transform(input_df)
    proba = float(model.predict_proba(X_scaled)[0, 1])
    pred = int(proba >= threshold)

    with result_placeholder.container():
        risk_class = "risk-high" if pred == 1 else "risk-low"
        st.markdown(f"<div class='card result-card {risk_class}'>", unsafe_allow_html=True)
        cols = st.columns([2, 1])
        with cols[0]:
            label = "High fraud risk" if pred == 1 else "Low fraud risk"
            color = "#ef4444" if pred == 1 else "#10b981"
            emoji = "⚠️" if pred == 1 else "✅"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px'>"
                f"<div style='width:56px;height:56px;border-radius:10px;color:white;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:20px'>{emoji}</div>"
                f"<div><div style='font-weight:700;font-size:18px'>{label}</div><div class='small muted'>Predicted probability</div></div></div>",
                unsafe_allow_html=True)
            st.markdown(f"<div class='big-number'>{proba*100:.2f}%</div>", unsafe_allow_html=True)
        with cols[1]:
            st.metric("Prediction", "FRAUD" if pred == 1 else "Legitimate", delta=None)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**Risk overview & recommended actions**")
        st.progress(min(100, max(0, int(proba * 100))))
        st.write(f"Predicted fraud probability: **{proba*100:.2f}%**")
        if pred == 1:
            st.error("High risk detected. Recommended immediate checks:")
            with st.expander("Suggested actions (click to expand)", expanded=True):
                st.markdown(
                    "- Hold the transaction and require additional verification (OTP/2FA/ID).\n"
                    "- Notify the cardholder and issuing bank.\n"
                    "- Escalate to fraud investigation for manual review.\n"
                    "- Monitor related merchant/card patterns and consider temporary blocks."
                )
        else:
            st.success("Low risk detected")
            with st.expander("Monitoring recommendations", expanded=False):
                st.markdown(
                    "- Proceed with the transaction.\n"
                    "- Continue passive monitoring for the next 24–48 hours.\n"
                    "- For unusually large amounts, consider lightweight verification."
                )
        st.markdown(f"**Transaction summary:**  \n- Amount: ${trans_amount:,.2f}  \n- Time: {trans_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer-note'>Tip: Use realistic values for better model accuracy.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
