"""
src/data_fetcher.py
Stahuje aktualni data z CEPS SOAP a ENTSO-E API.
"""

from datetime import date, datetime, timedelta
from typing import Optional
import pandas as pd
import time

CEPS_WSDL = "https://www.ceps.cz/_layouts/CepsData.asmx?WSDL"
CEPS_NS   = "https://www.ceps.cz/CepsData/StructuredData/1.0"
ENTSOE_TOKEN = "b41ce83d-7c9a-4694-91c5-d30f2ea65333"

# Kolik dni historie stahujeme pro lagy
HISTORY_DAYS = 3


def get_ceps_client():
    from zeep import Client
    return Client(wsdl=CEPS_WSDL)


def get_entsoe_client():
    from entsoe import EntsoePandasClient
    return EntsoePandasClient(api_key=ENTSOE_TOKEN)


def parse_items(result, series_map: dict) -> pd.DataFrame:
    rows  = []
    ns    = CEPS_NS
    items = result.findall(f"{{{ns}}}data/{{{ns}}}item")
    if not items:
        items = result.findall(".//{%s}item" % ns)
    for item in items:
        a        = dict(item.attrib)
        date_str = a.get("date", "")
        if not date_str:
            continue
        try:
            ts = pd.Timestamp(date_str).tz_convert("Europe/Prague")
        except Exception:
            continue
        row = {"time": ts}
        if series_map:
            for vid, col in series_map.items():
                val = a.get(vid)
                row[col] = float(val) if val is not None else None
        else:
            for k, v in a.items():
                if k.startswith("value"):
                    try:
                        row[k] = float(v)
                    except (ValueError, TypeError):
                        row[k] = v
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("time")


def fetch_ceps_series(ceps, method_name: str, params: dict,
                      series_map: dict, date_from: date, date_to: date) -> pd.DataFrame:
    try:
        fn     = getattr(ceps.service, method_name)
        result = fn(
            dateFrom=datetime(date_from.year, date_from.month, date_from.day),
            dateTo  =datetime(date_to.year,   date_to.month,   date_to.day, 23, 59, 59),
            **params
        )
        df = parse_items(result, series_map)
        if not df.empty:
            df = df[~df.index.duplicated(keep="last")].sort_index()
        return df
    except Exception as e:
        return pd.DataFrame()


def fetch_all(status_callback=None) -> dict:
    """
    Stahne vsechna potrebna data pro predikci.
    Vraci dict: {"so": df, "load_rt": df, ...}
    status: funkce(zprava: str, ok: bool)
    """

    def log(msg, ok=True):
        if status_callback:
            status_callback(msg, ok)

    now       = pd.Timestamp.now(tz="Europe/Prague")
    today     = now.date()
    date_from = today - timedelta(days=HISTORY_DAYS)
    date_to   = today

    result = {}

    # ── CEPS ──────────────────────────────────────────────────────────────────
    try:
        ceps = get_ceps_client()
        log("CEPS: spojeni OK")
    except Exception as e:
        log(f"CEPS: spojeni SELHALO - {e}", ok=False)
        return result

    # SO
    df = fetch_ceps_series(ceps, "AktualniSystemovaOdchylkaCR",
                           {"agregation": "QH", "function": "AVG"},
                           {"value1": "SO_MW"}, date_from, date_to)
    if not df.empty:
        result["so"] = df
        last = df.index.max()
        delay = int((now - last).total_seconds() / 60)
        log(f"SO: {len(df)} radku, posledni {last.strftime('%H:%M')} (zpozdeni {delay} min)")
    else:
        log("SO: zadna data", ok=False)
    time.sleep(0.3)

    # Load RT
    df = fetch_ceps_series(ceps, "Load",
                           {"agregation": "QH", "function": "AVG", "version": "RT"},
                           {"value2": "load_rt_MW"}, date_from, date_to)
    if not df.empty:
        result["load_rt"] = df
        last  = df.index.max()
        delay = int((now - last).total_seconds() / 60)
        log(f"Load RT: {len(df)} radku, posledni {last.strftime('%H:%M')} (zpozdeni {delay} min)")
    else:
        log("Load RT: zadna data", ok=False)
    time.sleep(0.3)

    # Load forecast
    df = fetch_ceps_series(ceps, "Load",
                           {"agregation": "QH", "function": "AVG", "version": "DF"},
                           {"value2": "load_fc_MW"}, date_from, date_to + timedelta(days=1))
    if not df.empty:
        result["load_fc"] = df
        log(f"Load forecast: {len(df)} radku")
    else:
        log("Load forecast: zadna data", ok=False)
    time.sleep(0.3)

    # Generation RT
    df = fetch_ceps_series(ceps, "Generation",
                           {"agregation": "QH", "function": "AVG",
                            "version": "RT", "para1": "all"},
                           {}, date_from, date_to)
    if not df.empty:
        rename = {
            "value1": "gen_nuclear_MW", "value2": "gen_hydro_MW",
            "value3": "gen_pump_MW",    "value4": "gen_coal_MW",
            "value5": "gen_gas_MW",     "value6": "gen_other_MW",
            "value7": "gen_solar_MW",   "value8": "gen_wind_MW",
            "value9": "gen_biomass_MW",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        result["generation"] = df
        last  = df.index.max()
        delay = int((now - last).total_seconds() / 60)
        log(f"Generation RT: {len(df)} radku, posledni {last.strftime('%H:%M')} (zpozdeni {delay} min)")
    else:
        log("Generation RT: zadna data", ok=False)
    time.sleep(0.3)

    # GenerationRES
    df = fetch_ceps_series(ceps, "GenerationRES",
                           {"agregation": "QH", "function": "AVG",
                            "version": "RT", "para1": "all"},
                           {}, date_from, date_to)
    if not df.empty:
        rename = {
            "value1": "res_solar_fc_MW",
            "value2": "res_wind_fc_MW",
            "value3": "res_total_fc_MW",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        result["generation_res"] = df
        log(f"GenerationRES: {len(df)} radku")
    else:
        log("GenerationRES: zadna data", ok=False)
    time.sleep(0.3)

    # CB flows
    df = fetch_ceps_series(ceps, "CrossborderPowerFlows",
                           {"agregation": "QH", "function": "AVG", "version": "RT"},
                           {}, date_from, date_to)
    if not df.empty:
        rename = {
            "value1": "cb_de_plan_MW", "value2": "cb_de_rt_MW",
            "value3": "cb_sk_plan_MW", "value4": "cb_sk_rt_MW",
            "value5": "cb_at_plan_MW", "value6": "cb_at_rt_MW",
            "value7": "cb_pl_plan_MW", "value8": "cb_pl_rt_MW",
            "value9": "cb_de_phys_MW", "value10": "cb_sk_phys_MW",
            "value11": "cb_at_phys_MW","value12": "cb_pl_phys_MW",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        result["cb_flows"] = df
        last  = df.index.max()
        delay = int((now - last).total_seconds() / 60)
        log(f"CB flows: {len(df)} radku, posledni {last.strftime('%H:%M')} (zpozdeni {delay} min)")
    else:
        log("CB flows: zadna data", ok=False)
    time.sleep(0.3)

    # SVR
    df = fetch_ceps_series(ceps, "AktivaceSVRvCR",
                           {"agregation": "QH", "function": "AVG", "param1": "all"},
                           {
                               "value1": "afrr_up_MW", "value2": "afrr_dn_MW",
                               "value3": "mfrr_up_MW", "value4": "mfrr_dn_MW",
                               "value7": "mfrr5_MW",
                           }, date_from, date_to)
    if not df.empty:
        result["svr"] = df
        log(f"SVR: {len(df)} radku")
    else:
        log("SVR: zadna data", ok=False)
    time.sleep(0.3)

    # Cena odchylky
    try:
        r = ceps.service.OdhadovanaCenaOdchylky(
            dateFrom=datetime(date_from.year, date_from.month, date_from.day),
            dateTo  =datetime(date_to.year,   date_to.month,   date_to.day, 23, 59, 59),
        )
        ns    = CEPS_NS
        items = r.findall(f"{{{ns}}}data/{{{ns}}}item")
        if not items:
            items = r.findall(".//{%s}item" % ns)
        rows = []
        for item in items:
            a        = dict(item.attrib)
            interval = a.get("value15", "")
            price    = a.get("value2",  None)
            val13    = a.get("value13", "")
            if not interval or not price or not val13:
                continue
            try:
                price_f  = float(price)
                day_ts   = pd.Timestamp(val13)
                if day_ts.tz is not None:
                    day_ts = day_ts.tz_convert("Europe/Prague").normalize()
                else:
                    day_ts = day_ts.normalize().tz_localize("Europe/Prague")
                hh, mm = interval.split("-")[0].split(":")
                ts = day_ts + pd.Timedelta(hours=int(hh), minutes=int(mm))
                rows.append({"time": ts, "imbal_price_CZK_MWh": price_f})
            except Exception:
                continue
        if rows:
            df_price = pd.DataFrame(rows).set_index("time")
            df_price = df_price[~df_price.index.duplicated(keep="last")].sort_index()
            result["imbalance_price"] = df_price
            log(f"Cena odchylky: {len(df_price)} radku")
        else:
            log("Cena odchylky: zadna data", ok=False)
    except Exception as e:
        log(f"Cena odchylky: chyba - {e}", ok=False)

    # ── ENTSO-E ───────────────────────────────────────────────────────────────
    try:
        entsoe = get_entsoe_client()
        log("ENTSO-E: spojeni OK")
    except Exception as e:
        log(f"ENTSO-E: spojeni SELHALO - {e}", ok=False)
        return result

    start = pd.Timestamp(date_from, tz="Europe/Prague")
    end   = pd.Timestamp(today + timedelta(days=2), tz="Europe/Prague")

    def resample_15min(df_in):
        if df_in is None or df_in.empty:
            return pd.DataFrame()
        df_in.index = df_in.index.tz_convert("Europe/Prague")
        idx = pd.date_range(df_in.index.min(), df_in.index.max(),
                            freq="15min", tz="Europe/Prague")
        return df_in.reindex(idx).ffill()

    # Wind+Solar CZ
    try:
        raw = entsoe.query_wind_and_solar_forecast("CZ", start=start, end=end, psr_type=None)
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = ["_".join(str(c) for c in col) for col in raw.columns]
            col_map = {}
            for c in raw.columns:
                cl = c.lower()
                if "solar" in cl or "photovoltaic" in cl:
                    col_map[c] = "entsoe_solar_fc_MW"
                elif "wind" in cl:
                    col_map[c] = "entsoe_wind_fc_MW"
            raw = raw.rename(columns=col_map)
            if "entsoe_wind_fc_MW" not in raw.columns:
                wind_cols = [c for c in raw.columns if "wind" in c.lower()]
                if wind_cols:
                    raw["entsoe_wind_fc_MW"] = raw[wind_cols].sum(axis=1)
            result["entsoe_wind_solar_cz"] = resample_15min(raw)
            log(f"ENTSO-E Wind+Solar CZ: OK")
        else:
            log("ENTSO-E Wind+Solar CZ: zadna data", ok=False)
    except Exception as e:
        log(f"ENTSO-E Wind+Solar CZ: chyba - {e}", ok=False)
    time.sleep(1.0)

    # Load forecast CZ
    try:
        raw = entsoe.query_load_forecast("CZ", start=start, end=end)
        if raw is not None and not raw.empty:
            if isinstance(raw, pd.DataFrame):
                raw = raw.iloc[:, 0]
            result["entsoe_load_fc"] = resample_15min(raw.rename("entsoe_load_fc_MW").to_frame())
            log(f"ENTSO-E Load forecast CZ: OK")
        else:
            log("ENTSO-E Load forecast CZ: zadna data", ok=False)
    except Exception as e:
        log(f"ENTSO-E Load forecast CZ: chyba - {e}", ok=False)
    time.sleep(1.0)

    # DAP
    try:
        raw = entsoe.query_day_ahead_prices("CZ", start=start, end=end)
        if raw is not None and not raw.empty:
            result["entsoe_dap"] = resample_15min(raw.rename("dap_EUR_MWh").to_frame())
            log(f"ENTSO-E DAP: OK")
        else:
            log("ENTSO-E DAP: zadna data", ok=False)
    except Exception as e:
        log(f"ENTSO-E DAP: chyba - {e}", ok=False)
    time.sleep(1.0)

    # Scheduled exchanges
    try:
        pairs = [("CZ","DE"),("DE","CZ"),("CZ","SK"),("SK","CZ"),
                 ("CZ","AT"),("AT","CZ"),("CZ","PL"),("PL","CZ")]
        frames = []
        for exp, imp in pairs:
            try:
                raw = entsoe.query_scheduled_exchanges(
                    country_code_from=exp, country_code_to=imp,
                    start=start, end=end, implicit=True)
                if raw is not None and not raw.empty:
                    frames.append(resample_15min(
                        raw.rename(f"sched_{exp}_{imp}_MW").to_frame()))
            except Exception:
                pass
            time.sleep(0.5)
        if frames:
            df_exc = pd.concat(frames, axis=1)
            result["entsoe_exchanges"] = df_exc
            log(f"ENTSO-E Scheduled exchanges: {df_exc.shape[1]} smeru")
        else:
            log("ENTSO-E Scheduled exchanges: zadna data", ok=False)
    except Exception as e:
        log(f"ENTSO-E Scheduled exchanges: chyba - {e}", ok=False)
    time.sleep(1.0)

    # DE Wind+Solar
    try:
        raw = entsoe.query_wind_and_solar_forecast("DE", start=start, end=end, psr_type=None)
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = ["_".join(str(c) for c in col) for col in raw.columns]
            col_map = {}
            wind_count = 0
            for c in raw.columns:
                cl = c.lower()
                if "solar" in cl or "photovoltaic" in cl:
                    col_map[c] = "de_solar_fc_MW"
                elif "wind" in cl:
                    wind_count += 1
                    col_map[c] = f"de_wind_fc_MW_{wind_count}"
            raw = raw.rename(columns=col_map)
            wind_cols = [c for c in raw.columns if c.startswith("de_wind_fc_MW")]
            raw["de_wind_fc_total_MW"] = raw[wind_cols].sum(axis=1)
            result["entsoe_wind_solar_de"] = resample_15min(raw)
            log(f"ENTSO-E Wind+Solar DE: OK")
        else:
            log("ENTSO-E Wind+Solar DE: zadna data", ok=False)
    except Exception as e:
        log(f"ENTSO-E Wind+Solar DE: chyba - {e}", ok=False)
    time.sleep(1.0)

    # DE Load forecast
    try:
        raw = entsoe.query_load_forecast("DE", start=start, end=end)
        if raw is not None and not raw.empty:
            if isinstance(raw, pd.DataFrame):
                raw = raw.iloc[:, 0]
            result["entsoe_de_load_fc"] = resample_15min(raw.rename("de_load_fc_MW").to_frame())
            log(f"ENTSO-E Load forecast DE: OK")
        else:
            log("ENTSO-E Load forecast DE: zadna data", ok=False)
    except Exception as e:
        log(f"ENTSO-E Load forecast DE: chyba - {e}", ok=False)
    time.sleep(1.0)

    # DE Imbalance prices
    try:
        raw = entsoe.query_imbalance_prices("DE", start=start, end=end)
        if raw is not None and not raw.empty:
            raw.index = raw.index.tz_convert("Europe/Prague")
            if "Long" in raw.columns:
                raw = raw.rename(columns={"Long": "de_imbal_long_EUR_MWh",
                                          "Short": "de_imbal_short_EUR_MWh"})
            result["entsoe_de_imbalance"] = resample_15min(raw)
            log(f"ENTSO-E DE Imbalance prices: OK")
        else:
            log("ENTSO-E DE Imbalance prices: zadna data", ok=False)
    except Exception as e:
        log(f"ENTSO-E DE Imbalance prices: chyba - {e}", ok=False)

    return result
