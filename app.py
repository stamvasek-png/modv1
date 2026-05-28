"""
app.py
Streamlit dashboard pro predikci systemove odchylky SO CZ.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys, os, requests

sys.path.insert(0, os.path.dirname(__file__))
from src.data_fetcher import fetch_all
from src.feature_builder import build_features
from src.predictor import load_models, load_feature_cols, predict, HORIZONS, HORIZON_LABELS

# ── Auto-download modelu z GitHub Releases ─────────────────────────────────────
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

# ── Limity zpozdenı pro validaci ───────────────────────────────────────────────
DELAY_LIMITS = {
    "SO":            15,
    "Load RT":       30,
    "Generation RT": 30,
    "CB flows":      30,
    "SVR":           30,
    "Cena odchylky": 30,
}

# Treninkove shifty pro expander
TRAINING_SHIFTS = [
    ("SO (systémová odchylka)",      "~6–15 min",    "shift(1) = 15 min",   "Lag 1–192 intervalů"),
    ("CB flows (přeshr. toky)",      "~6–21 min",    "shift(1) + shift(2)", "Oba lagy jako features"),
    ("Cena odchylky CZ",             "~6–15 min",    "shift(1) + shift(2)", ""),
    ("Load RT (zatížení)",           "~21 min",      "shift(2) = 30 min",   ""),
    ("Generation RT (výroba)",       "~21 min",      "shift(2) = 30 min",   ""),
    ("SVR aktivace (aFRR/mFRR)",     "~21 min",      "shift(2) = 30 min",   ""),
    ("DE Imbalance prices",          "~51 min",      "shift(4) = 60 min",   ""),
    ("Load forecast ČEPS/ENTSO-E",   "den předem",   "bez shiftu",          ""),
    ("Wind+Solar forecast CZ+DE",    "den předem",   "bez shiftu",          ""),
    ("DAP ceny",                     "po 13:00 D-1", "bez shiftu",          ""),
    ("Scheduled exchanges",          "den předem",   "bez shiftu",          ""),
]

def validate_data(fetch_status):
    issues = []
    for icon, msg in fetch_status:
        if "zpozdeni" not in msg:
            continue
        try:
            mins  = int(msg.split("zpozdeni")[1].split("min")[0].strip())
            zdroj = msg.split(":")[0].replace("CEPS - ", "").replace("ENTSO-E - ", "").strip()
            limit = None
            for key, lim in DELAY_LIMITS.items():
                if key in msg:
                    limit = lim
                    break
            if limit is not None:
                issues.append({
                    "zdroj":    zdroj,
                    "zpozdeni": mins,
                    "limit":    limit,
                    "ok":       mins <= limit,
                })
        except Exception:
            continue
    return issues

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
.val-warn     { background: #FFF3CD; border-left: 4px solid #E87722;
                border-radius: 4px; padding: 6px 10px; margin: 2px 0; font-size: 0.82rem; }
.val-ok       { background: #D4EDDA; border-left: 4px solid #1E7B34;
                border-radius: 4px; padding: 6px 10px; margin: 2px 0; font-size: 0.82rem; }
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
    st.info("Nahraj modely do slozky `models/`")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Ovládání")
    btn_fetch = st.button("🔄 Stáhnout data a predikovat", type="primary", use_container_width=True)
    st.divider()
    st.caption("Zdroje dat:")
    st.caption("• ČEPS SOAP API")
    st.caption("• ENTSO-E Transparency")
    st.divider()
    st.caption("Modely: XGBoost (Optuna tuned)")
    st.caption("Horizonty: T+1 až T+4 (15–60 min)")
    st.divider()
    with st.expander("ℹ️ Trénovací zpoždění (shifty)"):
        for row in TRAINING_SHIFTS:
            st.markdown(f"**{row[0]}**  \nReálné: {row[1]}  \nShift: `{row[2]}`"
                        + (f"  \n_{row[3]}_" if row[3] else ""))
            st.markdown("---")

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("fetch_status", []),
    ("predictions",  None),
    ("so_history",   None),
    ("current_so",   None),
    ("fetch_time",   None),
    ("validation",   []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

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
        with st.spinner("Sestavuji features..."):
            X_row = build_features(data, feature_cols)

        if X_row is None:
            st.error("Nepodarilo se sestavit features.")
        else:
            with st.spinner("Pocitam predikce..."):
                preds = predict(X_row, models_reg, models_clf)

            st.session_state.predictions = preds
            st.session_state.so_history  = data["so"]
            st.session_state.current_so  = float(data["so"]["SO_MW"].iloc[-1])
            st.session_state.fetch_time  = datetime.now()
            st.session_state.validation  = validate_data(st.session_state.fetch_status)
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
            <div class="label-small">Aktuální SO</div>
            <div class="big-number" style="color:{color}">{so_now:+.1f} MW</div>
            <div class="label-small">Poslední interval · {st.session_state.fetch_time.strftime('%H:%M:%S') if st.session_state.fetch_time else ''}</div>
        </div>
        """, unsafe_allow_html=True)

    # Validace zpozdenı
    if st.session_state.validation:
        bad = [v for v in st.session_state.validation if not v["ok"]]
        if bad:
            st.warning(f"⚠️ {len(bad)} zdroj(ů) překračuje maximální povolené zpoždění!")
            for v in bad:
                st.markdown(
                    f'<div class="val-warn">⚠️ <b>{v["zdroj"]}</b>: zpoždění {v["zpozdeni"]} min '
                    f'(limit {v["limit"]} min) — predikce může být méně přesná</div>',
                    unsafe_allow_html=True)
        else:
            st.markdown('<div class="val-ok">✅ Všechna data jsou v rámci povolených zpoždění</div>',
                        unsafe_allow_html=True)

    # ── Predikce ───────────────────────────────────────────────────────────────
    if st.session_state.predictions:
        st.subheader("Predikce SO")

        # Cas posledniho SO a vypocet pro jaky cas predikce plati
        last_so_time = st.session_state.so_history.index[-1]
        now_ts       = pd.Timestamp.now(tz="Europe/Prague")
        delay_min    = int((now_ts - last_so_time).total_seconds() / 60)

        # Varování pokud SO data jsou stara vice nez 1 interval
        if delay_min > 15:
            st.warning(
                f"⚠️ Poslední SO data jsou z **{last_so_time.strftime('%H:%M')}** "
                f"(zpoždění {delay_min} min). "
                f"Predikce vychází z tohoto času — výsledné časy T+1..T+4 jsou posunuté!"
            )

        # Radek s casem pro ktery predikce plati
        time_cols = st.columns(len(st.session_state.predictions))
        for i, pred in enumerate(st.session_state.predictions):
            pred_time = last_so_time + pd.Timedelta(minutes=pred["horizon"] * 15)
            with time_cols[i]:
                st.caption(f"Platí pro: **{pred_time.strftime('%H:%M')}**")

        # Karty s predikci
        cols = st.columns(len(st.session_state.predictions))
        for i, pred in enumerate(st.session_state.predictions):
            with cols[i]:
                so_val = pred["so_mw"]
                color  = pred["direction_color"]
                arrow  = "▲" if pred["direction"] == 1 else ("▼" if pred["direction"] == -1 else "●")
                proba  = pred["proba"]
                p_neg  = proba.get(-1, 0)
                p_neu  = proba.get(0,  0)
                p_pos  = proba.get(1,  0)

                st.markdown(f"""
                <div class="metric-card" style="border-left-color:{color}">
                    <div class="label-small">T+{pred['horizon']} · {pred['label']}</div>
                    <div class="big-number" style="color:{color}">{so_val:+.1f}</div>
                    <div class="label-small">MW</div>
                    <div style="color:{color}; font-size:1.1rem; margin-top:4px">{arrow} {pred['direction_label']}</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"▼ {p_neg:.0%}  ● {p_neu:.0%}  ▲ {p_pos:.0%}")

        # ── Grafy ───────────────────────────────────────────────────────────────
        if st.session_state.so_history is not None:
            df_hist = st.session_state.so_history.copy()

            # Casovy rozsah: 8h zpet, 4h dopredu
            t_start = now_ts - pd.Timedelta(hours=8)
            t_end   = now_ts + pd.Timedelta(hours=4)
            df_win  = df_hist[(df_hist.index >= t_start) & (df_hist.index <= now_ts)]

            pred_times    = [last_so_time + pd.Timedelta(minutes=h * 15) for h in HORIZONS]
            pred_vals     = [p["so_mw"] for p in st.session_state.predictions]
            pred_vals_mwh = [v * 0.25 for v in pred_vals]
            anchor_mw     = float(df_hist["SO_MW"].iloc[-1])

            # ── Graf 1: SO v MW ────────────────────────────────────────────────
            st.subheader("Graf SO [MW]")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df_win.index, y=df_win["SO_MW"],
                name="Skutečnost [MW]",
                line=dict(color="#1F4E79", width=1.5),
            ))
            fig1.add_trace(go.Scatter(
                x=[last_so_time] + pred_times,
                y=[anchor_mw] + pred_vals,
                name="Predikce [MW]",
                line=dict(color="#E87722", width=2, dash="dot"),
                mode="lines+markers",
                marker=dict(size=8),
            ))
            fig1.add_hline(y=0,   line_dash="dash", line_color="gray",    line_width=0.8)
            fig1.add_hline(y=50,  line_dash="dot",  line_color="#1E7B34", line_width=0.5)
            fig1.add_hline(y=-50, line_dash="dot",  line_color="#C00000", line_width=0.5)
            fig1.add_vrect(x0=last_so_time, x1=t_end,
                           fillcolor="rgba(232,119,34,0.05)", line_width=0,
                           annotation_text="predikce", annotation_position="top left")
            fig1.update_layout(
                height=320,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", y=1.05),
                yaxis_title="SO [MW]",
                xaxis=dict(range=[t_start, t_end]),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            fig1.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
            fig1.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
            st.plotly_chart(fig1, use_container_width=True)

            # ── Graf 2: SO v MWh ───────────────────────────────────────────────
            st.subheader("Graf SO [MWh] — přepočet na čtvrthodinu (× 0.25)")
            df_mwh = df_win.copy()
            df_mwh["SO_MWh"] = df_mwh["SO_MW"] * 0.25

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_mwh.index, y=df_mwh["SO_MWh"],
                name="Skutečnost [MWh/15min]",
                line=dict(color="#1F4E79", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(31,78,121,0.08)",
            ))
            fig2.add_trace(go.Scatter(
                x=[last_so_time] + pred_times,
                y=[anchor_mw * 0.25] + pred_vals_mwh,
                name="Predikce [MWh/15min]",
                line=dict(color="#E87722", width=2, dash="dot"),
                mode="lines+markers",
                marker=dict(size=8),
            ))
            fig2.add_hline(y=0,     line_dash="dash", line_color="gray",    line_width=0.8)
            fig2.add_hline(y=12.5,  line_dash="dot",  line_color="#1E7B34", line_width=0.5,
                           annotation_text="+50MW", annotation_position="right")
            fig2.add_hline(y=-12.5, line_dash="dot",  line_color="#C00000", line_width=0.5,
                           annotation_text="-50MW", annotation_position="right")
            fig2.add_vrect(x0=last_so_time, x1=t_end,
                           fillcolor="rgba(232,119,34,0.05)", line_width=0,
                           annotation_text="predikce", annotation_position="top left")
            fig2.update_layout(
                height=320,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", y=1.05),
                yaxis_title="SO [MWh / 15min]",
                xaxis=dict(range=[t_start, t_end]),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            fig2.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
            fig2.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
            st.plotly_chart(fig2, use_container_width=True)

        # Tabulka predikcí
        st.subheader("Detail predikcí")
        rows = []
        for p in st.session_state.predictions:
            pred_time = last_so_time + pd.Timedelta(minutes=p["horizon"] * 15)
            rows.append({
                "Platí pro":   pred_time.strftime("%H:%M"),
                "Horizont":    f"T+{p['horizon']} ({p['label']})",
                "SO [MW]":     f"{p['so_mw']:+.1f}",
                "SO [MWh]":    f"{p['so_mw']*0.25:+.2f}",
                "Směr":        p["direction_label"],
                "P(záporná)":  f"{p['proba'].get(-1,0):.1%}",
                "P(neutrální)":f"{p['proba'].get(0,0):.1%}",
                "P(kladná)":   f"{p['proba'].get(1,0):.1%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    else:
        st.info("Stiskni tlačítko **Stáhnout data a predikovat** pro zobrazení predikcí.")

# ── Panel dostupnosti dat ──────────────────────────────────────────────────────
with col_status:
    st.subheader("Stav dat")

    if st.session_state.fetch_status:
        for icon, msg in st.session_state.fetch_status:
            color = "status-ok" if icon == "✅" else "status-error"
            warn  = ""
            if "zpozdeni" in msg:
                try:
                    mins  = int(msg.split("zpozdeni")[1].split("min")[0].strip())
                    limit = None
                    for key, lim in DELAY_LIMITS.items():
                        if key in msg:
                            limit = lim
                            break
                    if limit and mins > limit:
                        color = "status-warn"
                        warn  = f" ⚠️ >{limit}min"
                except Exception:
                    pass
            short = msg[:42] + "..." if len(msg) > 42 else msg
            st.markdown(
                f'<span class="{color}">{icon}</span> <small>{short}{warn}</small>',
                unsafe_allow_html=True)
    else:
        st.caption("Data zatím nestažena.")
        st.caption("Stiskni tlačítko vlevo.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: XGBoost Regressor + Klasifikátor | Optuna tuned | T+1 MAE ~46 MW | T+4 MAE ~55 MW | Trénink: 2025-01-01 → 2026-03-01")
