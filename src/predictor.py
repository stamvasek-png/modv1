"""
src/predictor.py
Nacte natrenovane modely a vraci predikce pro T+1 az T+4.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Optional

HORIZONS      = [1, 2, 3, 4]
HORIZON_LABELS = {1: "15 min", 2: "30 min", 3: "45 min", 4: "60 min"}
CLF_MAP_INV    = {0: -1, 1: 0, 2: 1}
DIR_LABELS     = {-1: "Zaporna", 0: "Neutralni", 1: "Kladna"}
DIR_COLORS     = {-1: "#C00000", 0: "#808080", 1: "#1E7B34"}
ZERO_BAND      = 50  # MW


def load_models(models_dir: str = "models"):
    """Nacte vsechny tuned modely z adresare."""
    from xgboost import XGBRegressor, XGBClassifier

    models_reg = {}
    models_clf = {}

    for h in HORIZONS:
        reg_path = os.path.join(models_dir, f"tuned_reg_h{h}.json")
        clf_path = os.path.join(models_dir, f"tuned_clf_h{h}.json")

        if os.path.exists(reg_path):
            reg = XGBRegressor()
            reg.load_model(reg_path)
            models_reg[h] = reg
        else:
            raise FileNotFoundError(f"Model nenalezen: {reg_path}")

        if os.path.exists(clf_path):
            clf = XGBClassifier()
            clf.load_model(clf_path)
            models_clf[h] = clf
        else:
            raise FileNotFoundError(f"Model nenalezen: {clf_path}")

    return models_reg, models_clf


def load_feature_cols(models_dir: str = "models") -> list:
    path = os.path.join(models_dir, "feature_cols.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"feature_cols.json nenalezen v {models_dir}")
    with open(path) as f:
        return json.load(f)


def predict(X_row: pd.DataFrame,
            models_reg: dict,
            models_clf: dict,
            current_so: Optional[float] = None) -> list:
    """
    Vraci seznam predikcí pro kazdy horizont.
    Kazda predikce je dict:
      {
        "horizon": int,
        "label": str,
        "so_mw": float,
        "direction": int (-1/0/1),
        "direction_label": str,
        "direction_color": str,
        "proba": dict {-1: float, 0: float, 1: float},
      }
    """
    results = []

    for h in HORIZONS:
        if h not in models_reg or h not in models_clf:
            continue

        # Regressor
        so_pred = float(models_reg[h].predict(X_row)[0])

        # Klasifikator - pravdepodobnosti
        proba_raw = models_clf[h].predict_proba(X_row)[0]
        proba = {
            CLF_MAP_INV[i]: float(p)
            for i, p in enumerate(proba_raw)
        }

        # Smer podle regressoru (konzistentni s ZERO_BAND)
        if so_pred > ZERO_BAND:
            direction = 1
        elif so_pred < -ZERO_BAND:
            direction = -1
        else:
            direction = 0

        results.append({
            "horizon":         h,
            "label":           HORIZON_LABELS[h],
            "so_mw":           round(so_pred, 1),
            "direction":       direction,
            "direction_label": DIR_LABELS[direction],
            "direction_color": DIR_COLORS[direction],
            "proba":           proba,
        })

    return results
