"""
app.py
Streamlit dashboard pro predikci systemove odchylky SO CZ.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import sys, os, requests

sys.path.insert(0, os.path.dirname(__file__))
from src.data_fetcher import fetch_all
from src.feature_builder import build_features
from src.predictor import load_models, load_feature_cols, predict, HORIZONS, HORIZON_LABELS

# ── Auto-download modelu z GitHub Releases ────────────────────────────────────
RELEASE_URL = "https://github.com/stamvasek-png/modv1/releases/download/v1.0"
MODEL_FILES = [
    "tuned_reg_h1.json", "tuned_reg_h2.json", "tuned_reg_h3.json", "tuned_reg_h4.json",
    "tuned_clf_h1.json", "tuned_clf_h2.json", "tuned_clf_h3.json", "tuned_clf_h4.json",
    "feature_cols.json",
]

def ensure_models():
    os.makedirs("models", exist_ok=True)
    missing = [f for f in MODEL_FILES if not os.path.exists(f"models/{f}")]
    if not missing:
        return True
    with st.spinner(f"Stahuji modely z GitHub ({len(missing)} souboru)..."):
        for fname in missing:
            url  = f"{RELEASE_URL}/{fname}"
            dest = f"models/{fname}"
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                with open(dest, "wb") as fh:
                    fh.write(r.content)
            except Exception as e:
                st.error(f"Nepodarilo se stahnout {fname}: {e}")
                return False
    return True

# ── Konfigurace stranky ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SO Predikce | PP Dashboard",
    page_icon="⚡",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 16px;
    border-left: 4px solid #1F4E79;
    margin-bottom: 8px;
}
.status-ok    { color: #1E7B34; font-weight: bold; }
.status-warn  { color: #E87722; font-weight: bold; }
.status-error { color: #C00000; font-weight: bold; }
.big-number   { font-size: 2.5rem; font-weight: bold; line-height: 1; }
.label-small  { font-size: 0.8rem; color: #666; }
</style>
""", unsafe_allow_html=True)

# ── Nacteni modelu ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    models_reg, models_clf = load_models("models")
    feature_cols = load_feature_cols("models")
    return models_reg, models_clf, feature_cols

if not ensure_models():
    st.stop()

try:
    models_reg, models_clf, feature_cols = load_all_models()
    models_ok = True
except Exception as e:
    models_ok = False
    model_error = str(e)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("⚡ Predikce Systémové Odchylky CZ")
st.caption(f"Posledni aktualizace: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if not models_ok:
    st.error(f"Modely se nepodarily nacist: {model_error}")
    st.info("Nahraj modely do slozky `models/` (tuned_reg_h1.json az tuned_clf_h4.json + feature_cols.json)")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Ovladani")
    btn_fetch = st.button("🔄 Stáhnout data a predikovat", type="primary", use_container_width=True)
    st.divider()
    st.caption("Zdroje dat:")
    st.caption("• ČEPS SOAP API")
    st.caption("• ENTSO-E Transparency")
    st.divider()
    st.caption("Modely: XGBoost (Optuna tuned)")
    st.caption("Horizonty: T+1 až T+4 (15-60 min)")

# ── Session state ──────────────────────────────────────────────────────────────
if "fetch_status" not in st.session_state:
    st.session_state.fetch_status = []
if "predictions"  not in st.session_state:
    st.session_state.predictions  = None
if "so_history"   not in st.session_state:
    st.session_state.so_history   = None
if "current_so"   not in st.session_state:
    st.session_state.current_so   = None
if "fetch_time"   not in st.session_state:
    st.session_state.fetch_time   = None

# ── Stazeni dat ────────────────────────────────────────────────────────────────
if btn_fetch:
    st.session_state.fetch_status = []
    status_placeholder = st.empty()

    def status_cb(msg, ok=True):
        icon = "✅" if ok else "❌"
        st.session_state.fetch_status.append((icon, msg))
        with status_placeholder.container():
            for ico, m in st.session_state.fetch_status[-8:]:
                st.write(f"{ico} {m}")

    with st.spinner("Stahuji data ze ČEPS a ENTSO-E..."):
        data = fetch_all(status_callback=status_cb)

    status_placeholder.empty()

    if not data or "so" not in data:
        st.error("Nepodarilo se stahnout data SO z CEPS.")
    else:
        # Sestav features
        with st.spinner("Sestavuji features..."):
            X_row = build_features(data, feature_cols)

        if X_row is None:
            st.error("Nepodarilo se sestavit features.")
        else:
            # Predikce
            with st.spinner("Pocitam predikce..."):
                preds = predict(X_row, models_reg, models_clf)

            st.session_state.predictions = preds
            st.session_state.so_history  = data["so"]
            st.session_state.current_so  = float(data["so"]["SO_MW"].iloc[-1])
            st.session_state.fetch_time  = datetime.now()
            st.success("Predikce dokoncena!")

# ── Hlavni obsah ───────────────────────────────────────────────────────────────
col_main, col_status = st.columns([3, 1])

with col_main:

    # Aktualni SO
    if st.session_state.current_so is not None:
        so_now = st.session_state.current_so
        color  = "#1E7B34" if so_now > 50 else ("#C00000" if so_now < -50 else "#808080")
        st.markdown(f"""
        <div class="metric-card">
            <div class="label-small">Aktualni SO</div>
            <div class="big-number" style="color:{color}">{so_now:+.1f} MW</div>
            <div class="label-small">Posledni interval · {st.session_state.fetch_time.strftime('%H:%M:%S') if st.session_state.fetch_time else ''}</div>
        </div>
        """, unsafe_allow_html=True)

    # Predikce
    if st.session_state.predictions:
        st.subheader("Predikce SO")
        cols = st.columns(len(st.session_state.predictions))

        for i, pred in enumerate(st.session_state.predictions):
            with cols[i]:
                so_val = pred["so_mw"]
                color  = pred["direction_color"]
                arrow  = "▲" if pred["direction"] == 1 else ("▼" if pred["direction"] == -1 else "●")
                proba  = pred["proba"]

                st.markdown(f"""
                <div class="metric-card" style="border-left-color:{color}">
                    <div class="label-small">T+{pred['horizon']} · {pred['label']}</div>
                    <div class="big-number" style="color:{color}">{so_val:+.1f}</div>
                    <div class="label-small">MW</div>
                    <div style="color:{color}; font-size:1.1rem; margin-top:4px">{arrow} {pred['direction_label']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Pravdepodobnosti
                p_neg  = proba.get(-1, 0)
                p_neu  = proba.get(0,  0)
                p_pos  = proba.get(1,  0)
                st.caption(f"▼ {p_neg:.0%}  ● {p_neu:.0%}  ▲ {p_pos:.0%}")

        # Graf SO historie + predikce
        st.subheader("Graf SO")
        if st.session_state.so_history is not None:
            df_hist = st.session_state.so_history.copy()
            df_hist = df_hist.tail(96 * 2)  # posledni 2 dny

            last_time = df_hist.index[-1]
            pred_times = [last_time + pd.Timedelta(minutes=h*15)
                          for h in HORIZONS]
            pred_vals  = [p["so_mw"] for p in st.session_state.predictions]

            fig = go.Figure()

            # Historie
            fig.add_trace(go.Scatter(
                x=df_hist.index,
                y=df_hist["SO_MW"],
                name="Skutecnost",
                line=dict(color="#1F4E79", width=1.5),
            ))

            # Predikce
            fig.add_trace(go.Scatter(
                x=[last_time] + pred_times,
                y=[float(df_hist["SO_MW"].iloc[-1])] + pred_vals,
                name="Predikce",
                line=dict(color="#E87722", width=2, dash="dot"),
                mode="lines+markers",
                marker=dict(size=8),
            ))

            fig.add_hline(y=0,   line_dash="dash", line_color="gray", line_width=0.8)
            fig.add_hline(y=50,  line_dash="dot",  line_color="#1E7B34", line_width=0.5)
            fig.add_hline(y=-50, line_dash="dot",  line_color="#C00000", line_width=0.5)

            fig.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", y=1.05),
                yaxis_title="SO [MW]",
                xaxis_title="",
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
            fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

            st.plotly_chart(fig, use_container_width=True)

        # Tabulka predikcí
        st.subheader("Detail predikcí")
        rows = []
        for p in st.session_state.predictions:
            rows.append({
                "Horizont":  f"T+{p['horizon']} ({p['label']})",
                "SO [MW]":   f"{p['so_mw']:+.1f}",
                "Smer":      p["direction_label"],
                "P(zaporna)":f"{p['proba'].get(-1,0):.1%}",
                "P(neutral)":f"{p['proba'].get(0,0):.1%}",
                "P(kladna)": f"{p['proba'].get(1,0):.1%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    else:
        st.info("Stiskni tlacitko **Stáhnout data a predikovat** pro zobrazeni predikcí.")

# ── Panel dostupnosti dat ──────────────────────────────────────────────────────
with col_status:
    st.subheader("Stav dat")

    if st.session_state.fetch_status:
        for icon, msg in st.session_state.fetch_status:
            color = "status-ok" if icon == "✅" else "status-error"
            # Zkrat zpravy
            short = msg[:45] + "..." if len(msg) > 45 else msg
            st.markdown(f'<span class="{color}">{icon}</span> <small>{short}</small>',
                        unsafe_allow_html=True)
    else:
        st.caption("Data zatim nestazena.")
        st.caption("Stiskni tlacitko vlevo.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: XGBoost Regressor + Klasifikator | Optuna tuned | T+1 MAE ~46 MW | T+4 MAE ~55 MW")
