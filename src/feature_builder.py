"""
src/feature_builder.py
Sestavuje feature vektor z aktualnich dat pro predikci SO.
Kazdy zdroj je vzorkovan v case T-shift aby odpovidal treninkovym datum.
"""

import numpy as np
import pandas as pd
from typing import Optional

# Shifty presne jako pri treninku (v poctu 15min intervalu)
# shift(1) = T-15min, shift(2) = T-30min, shift(4) = T-60min
SHIFT_MINUTES = {
    "so":               15,   # shift(1) - ale pouzivame lagy uvnitr
    "cb_flows":         15,   # shift(1) i shift(2) -> pouzijeme T-15min jako zaklad
    "imbalance_price":  15,   # shift(1)
    "load_rt":          30,   # shift(2)
    "generation":       30,   # shift(2)
    "generation_res":   30,   # shift(2)
    "svr":              30,   # shift(2)
    "entsoe_de_imbalance": 60, # shift(4)
    # Forecast zdroje - bez shiftu (pouzij nejnovejsi dostupnou hodnotu)
    "load_fc":          0,
    "entsoe_wind_solar_cz": 0,
    "entsoe_load_fc":   0,
    "entsoe_dap":       0,
    "entsoe_exchanges": 0,
    "entsoe_wind_solar_de": 0,
    "entsoe_de_load_fc": 0,
}


def get_value_at_shift(df: pd.DataFrame, t_ref: pd.Timestamp,
                       shift_minutes: int) -> Optional[pd.Series]:
    """
    Vraci radek dataframu nejblizsi casu t_ref - shift_minutes.
    Tolerance: +/- 8 minut (jeden interval je 15 min).
    """
    if df is None or df.empty:
        return None

    t_target = t_ref - pd.Timedelta(minutes=shift_minutes)
    tolerance = pd.Timedelta(minutes=8)

    # Najdi nejblizsi dostupny casovy razitko
    idx = df.index
    diffs = abs(idx - t_target)
    min_diff = diffs.min()

    if min_diff > tolerance + pd.Timedelta(minutes=shift_minutes * 0.5 + 15):
        # Data jsou prilis daleko od ocekavaneho casu
        return None

    closest = idx[diffs.argmin()]
    return df.loc[closest]


def build_features(data: dict, feature_cols: list) -> Optional[pd.DataFrame]:
    """
    Ze slovniku dataframu sestavi DataFrame s features pro predikci.
    Kazdy zdroj je vzorkovan ve spravnem case podle treninkovych shiftu.
    """

    if "so" not in data or data["so"].empty:
        return None

    df_so = data["so"].copy()
    df_so.index = pd.to_datetime(df_so.index).tz_convert("Europe/Prague")
    df_so = df_so.sort_index()

    # Referencni cas = posledni dostupny SO timestamp
    t_ref = df_so.index[-1]

    # Spojime vsechna data do jednoho dataframu pro vypocet lagu
    # (lagy SO potrebujeme pocitat z cele casove rady)
    df = df_so.copy()

    def safe_join(base, other):
        if other is None or other.empty:
            return base
        other = other.copy()
        other.index = pd.to_datetime(other.index).tz_convert("Europe/Prague")
        dupl = [c for c in other.columns if c in base.columns]
        if dupl:
            other = other.drop(columns=dupl)
        return base.join(other, how="left")

    for key in ["load_rt", "load_fc", "generation", "generation_res",
                "cb_flows", "svr", "imbalance_price",
                "entsoe_wind_solar_cz", "entsoe_load_fc", "entsoe_dap",
                "entsoe_exchanges", "entsoe_wind_solar_de",
                "entsoe_de_load_fc", "entsoe_de_imbalance"]:
        if key in data:
            df = safe_join(df, data[key])

    df = df.sort_index().copy()

    # ── Feature engineering ───────────────────────────────────────────────────
    # Casove features (z t_ref)
    df["hour"]       = df.index.hour
    df["minute"]     = df.index.minute
    df["weekday"]    = df.index.weekday
    df["month"]      = df.index.month
    df["quarter"]    = df.index.quarter
    df["is_weekend"] = (df.index.weekday >= 5).astype(int)
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["day_block"]   = pd.cut(df["hour"],
                               bins=[-1, 5, 9, 17, 21, 23],
                               labels=[0, 1, 2, 3, 4]).astype(float)

    # SO lags - pocitame z cele rady, shift je uz zahrnut v indexu lagu
    for lag in [1, 2, 3, 4, 8, 12, 16, 24, 48, 96, 192]:
        df[f"so_lag_{lag}"] = df["SO_MW"].shift(lag)

    so_s1 = df["SO_MW"].shift(1)
    for window in [4, 8, 16, 48, 96]:
        df[f"so_roll_mean_{window}"] = so_s1.rolling(window).mean()
        df[f"so_roll_std_{window}"]  = so_s1.rolling(window).std()
    df["so_roll_max_16"] = so_s1.rolling(16).max()
    df["so_roll_min_16"] = so_s1.rolling(16).min()

    # CB toky - shift(1) i shift(2)
    cb_rt_cols = [c for c in df.columns if c.endswith("_rt_MW") and c.startswith("cb_")]
    if cb_rt_cols:
        df["cb_netto_MW"] = df[cb_rt_cols].sum(axis=1)
        for lag in [1, 2, 4, 8]:
            df[f"cb_netto_lag{lag}"] = df["cb_netto_MW"].shift(lag)
        df["cb_netto_roll4"] = df["cb_netto_MW"].shift(1).rolling(4).mean()

    phys_cols = [c for c in df.columns if c.endswith("_phys_MW")]
    if phys_cols:
        df["cb_phys_netto_MW"] = df[phys_cols].sum(axis=1)
        for lag in [1, 2, 4]:
            df[f"cb_phys_netto_lag{lag}"] = df["cb_phys_netto_MW"].shift(lag)

    exc_cols = [c for c in df.columns if c.startswith("sched_")]
    if exc_cols:
        df["sched_netto_MW"] = df[exc_cols].sum(axis=1)
        for c in exc_cols:
            df[f"feat_{c}"] = df[c]

    # Cena odchylky - shift(1) i shift(2)
    if "imbal_price_CZK_MWh" in df.columns:
        df["imbal_price_CZK_MWh"] = df["imbal_price_CZK_MWh"].ffill()
        for lag in [1, 2, 4, 8]:
            df[f"price_lag{lag}"] = df["imbal_price_CZK_MWh"].shift(lag)
        df["price_roll_mean4"] = df["imbal_price_CZK_MWh"].shift(1).rolling(4).mean()

    # Load RT - shift(2) i shift(4)
    if "load_rt_MW" in df.columns:
        for lag in [2, 4, 8, 16]:
            df[f"load_rt_lag{lag}"] = df["load_rt_MW"].shift(lag)
        df["load_rt_roll_mean4"] = df["load_rt_MW"].shift(2).rolling(4).mean()

    # Load forecast - bez shiftu
    if "load_fc_MW" in df.columns:
        df["load_fc_t"]  = df["load_fc_MW"]
        df["load_fc_t4"] = df["load_fc_MW"].shift(-4)
    if "entsoe_load_fc_MW" in df.columns:
        df["entsoe_load_fc_t"]  = df["entsoe_load_fc_MW"]
        df["entsoe_load_fc_t4"] = df["entsoe_load_fc_MW"].shift(-4)
    if "load_fc_MW" in df.columns and "load_rt_MW" in df.columns:
        df["load_fc_error"]       = df["load_rt_MW"].shift(2) - df["load_fc_MW"]
        df["load_fc_error_roll4"] = df["load_fc_error"].rolling(4).mean()

    # Generation RT - shift(2) i shift(4)
    for col, feat in [
        ("gen_solar_MW",   "gen_solar"),
        ("gen_wind_MW",    "gen_wind"),
        ("gen_nuclear_MW", "gen_nuclear"),
        ("gen_hydro_MW",   "gen_hydro"),
        ("gen_biomass_MW", "gen_biomass"),
        ("gen_coal_MW",    "gen_coal"),
        ("gen_gas_MW",     "gen_gas"),
    ]:
        if col in df.columns:
            for lag in [2, 4, 8]:
                df[f"{feat}_lag{lag}"] = df[col].shift(lag)

    # OZE forecast CZ - bez shiftu
    for col, feat in [("entsoe_solar_fc_MW", "solar_fc_cz"),
                      ("entsoe_wind_fc_MW",  "wind_fc_cz")]:
        if col in df.columns:
            df[f"{feat}_t"]  = df[col]
            df[f"{feat}_t4"] = df[col].shift(-4)
    if "res_solar_fc_MW" in df.columns:
        df["res_solar_fc_t"] = df["res_solar_fc_MW"]
    if "res_wind_fc_MW" in df.columns:
        df["res_wind_fc_t"]  = df["res_wind_fc_MW"]
    if "gen_solar_MW" in df.columns and "entsoe_solar_fc_MW" in df.columns:
        df["solar_fc_error"] = df["gen_solar_MW"].shift(2) - df["entsoe_solar_fc_MW"]
    if "gen_wind_MW" in df.columns and "entsoe_wind_fc_MW" in df.columns:
        df["wind_fc_error_cz"] = df["gen_wind_MW"].shift(2) - df["entsoe_wind_fc_MW"]

    # DE Wind+Solar - bez shiftu
    if "de_wind_fc_total_MW" in df.columns:
        df["de_wind_fc_t"]  = df["de_wind_fc_total_MW"]
        df["de_wind_fc_t4"] = df["de_wind_fc_total_MW"].shift(-4)
    if "de_solar_fc_MW" in df.columns:
        df["de_solar_fc_t"]  = df["de_solar_fc_MW"]
        df["de_solar_fc_t4"] = df["de_solar_fc_MW"].shift(-4)
    if "de_load_fc_MW" in df.columns:
        df["de_load_fc_t"]  = df["de_load_fc_MW"]
        df["de_load_fc_t4"] = df["de_load_fc_MW"].shift(-4)

    # SVR - shift(2) i shift(4)
    if "afrr_up_MW" in df.columns and "afrr_dn_MW" in df.columns:
        svr_netto = df["afrr_up_MW"] - df["afrr_dn_MW"]
        svr_total = df["afrr_up_MW"] + df["afrr_dn_MW"]
        if "mfrr_up_MW" in df.columns:
            svr_total += df["mfrr_up_MW"]
        if "mfrr_dn_MW" in df.columns:
            svr_total += df["mfrr_dn_MW"]
        for lag in [2, 4, 8]:
            df[f"svr_netto_lag{lag}"] = svr_netto.shift(lag)
            df[f"svr_total_lag{lag}"] = svr_total.shift(lag)

    # DE Imbalance prices - shift(4) i shift(8)
    if "de_imbal_long_EUR_MWh" in df.columns:
        df["de_imbal_spread"] = (df["de_imbal_long_EUR_MWh"] -
                                 df["de_imbal_short_EUR_MWh"])
        for lag in [4, 8, 16]:
            df[f"de_imbal_long_lag{lag}"]  = df["de_imbal_long_EUR_MWh"].shift(lag)
            df[f"de_imbal_short_lag{lag}"] = df["de_imbal_short_EUR_MWh"].shift(lag)
        df["de_imbal_spread_lag4"]  = df["de_imbal_spread"].shift(4)
        df["de_imbal_spread_lag8"]  = df["de_imbal_spread"].shift(8)
        df["de_imbal_long_roll4"]   = df["de_imbal_long_EUR_MWh"].shift(4).rolling(4).mean()
        df["de_imbal_short_roll4"]  = df["de_imbal_short_EUR_MWh"].shift(4).rolling(4).mean()

    # DAP
    if "dap_EUR_MWh" in df.columns:
        df["dap_t"]     = df["dap_EUR_MWh"]
        df["dap_lag96"] = df["dap_EUR_MWh"].shift(96)

    # ── Vezmi radek odpovidajici t_ref ────────────────────────────────────────
    # Najdi nejblizsi radek k t_ref v indexu
    if t_ref not in df.index:
        diffs   = abs(df.index - t_ref)
        t_ref   = df.index[diffs.argmin()]

    # Doplnime chybejici feature sloupce jako NaN
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    last_row = df.loc[[t_ref], feature_cols]

    # ── Validace: zkontroluj klic features ze zpozdenych zdroju ──────────────
    # Lagged features by mely byt non-NaN pokud mame dostatek historie
    critical = ["so_lag_1", "so_lag_2", "so_roll_mean_4",
                 "load_rt_lag2", "gen_solar_lag2", "gen_wind_lag2"]
    nan_critical = [c for c in critical if c in last_row.columns
                    and pd.isna(last_row[c].iloc[0])]
    if nan_critical:
        import streamlit as st
        st.warning(f"⚠️ Kritické features jsou NaN: {nan_critical} — "
                   f"pravděpodobně chybí dostatek historických dat")

    return last_row
