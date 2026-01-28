import pandas as pd
from io_utils import normalize_series

# ---------------------------------------------------------
# Helper: pick row-level cost if present, else fallback
# ---------------------------------------------------------
def _col_or_default(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype=float)


# ---------------------------------------------------------
# Helper: choose correct spend column based on EF unit
# ---------------------------------------------------------
def _compute_spend_base(df: pd.DataFrame, spend_unit: str, fx_rate: float = 0.85) -> pd.Series:
    """
    Decide which spend column to use and, if needed, apply FX.

    Legacy behaviour (EcoScopeAI original):
      - EF unit: kgCO2e_per_GBP
      - Data:    spend_eur
      - We convert EUR → GBP using fx_rate.

    New behaviour (EcoScopeAI-Arabia friendly):
      - If unit ends with _per_USD  and 'spend_usd' exists → use spend_usd
      - If unit ends with _per_EUR  and 'spend_eur' exists → use spend_eur
      - If unit ends with _per_SAR  and 'spend_sar' exists → use spend_sar
      - If unit ends with _per_AED  and 'spend_aed' exists → use spend_aed
      - If unit ends with _per_GBP:
            • prefer 'spend_gbp' if present
            • else convert 'spend_eur' using fx_rate (backward compatible)
      - Fallback: if only 'spend_eur' exists, use it as-is.
    """
    unit = str(spend_unit or "").lower()

    # Normalise spend columns we might use
    for col in ["spend_eur", "spend_gbp", "spend_usd", "spend_sar", "spend_aed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    # USD-based factors
    if unit.endswith("_per_usd") and "spend_usd" in df.columns:
        return df["spend_usd"]

    # EUR-based factors
    if unit.endswith("_per_eur") and "spend_eur" in df.columns:
        return df["spend_eur"]

    # SAR-based factors
    if unit.endswith("_per_sar") and "spend_sar" in df.columns:
        return df["spend_sar"]

    # AED-based factors
    if unit.endswith("_per_aed") and "spend_aed" in df.columns:
        return df["spend_aed"]

    # GBP-based factors (original EcoScopeAI logic)
    if unit.endswith("_per_gbp"):
        if "spend_gbp" in df.columns:
            return df["spend_gbp"]
        if "spend_eur" in df.columns:
            # Backward compatible: convert EUR → GBP using fx_rate
            return df["spend_eur"] * float(fx_rate or 0.85)

    # Fallbacks: keep original behaviour as much as possible
    if "spend_eur" in df.columns:
        return df["spend_eur"]
    if "spend_usd" in df.columns:
        return df["spend_usd"]

    # If nothing is available, return zeros
    return pd.Series([0.0] * len(df), index=df.index, dtype=float)


# ---------------------------------------------------------
# Core procurement + transport calculation
# ---------------------------------------------------------
def calc_core(df_pt, spend_efs, mode_ef, eur_to_gbp=0.85, spend_unit="kgCO2e_per_GBP"):
    """
    Core engine: Cat.1 (Purchased goods & services) + Cat.4/9 (Transport).

    Now supports:
      - Multiple spend currencies (EUR, GBP, USD, SAR, AED) depending on EF unit.
      - Region-specific transport keys (truck_gcc, truck_ksa, truck_uae, rail_ksa, etc.).
    """
    if df_pt is None or len(df_pt) == 0:
        return pd.DataFrame(), 0.0, 0.0, 0.0

    df = df_pt.copy()

    # numeric cleaning
    for col in ["spend_eur", "spend_gbp", "spend_usd", "spend_sar", "spend_aed",
                "distance_km", "weight_kg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    if "category" not in df.columns:
        df["category"] = ""
    if "supplier" not in df.columns:
        df["supplier"] = "Unknown"

    # --- Transport modes (normalized) ---
    if "transport_mode" not in df.columns:
        df["transport_mode"] = "none"

    tm = normalize_series(df["transport_mode"])

    # Map various synonyms & regional labels to canonical keys used in EF file
    tm = tm.replace({
        "air freight": "air",
        "airfreight": "air",
        "plane": "air",
        "air_mena": "air",

        "ocean": "sea",
        "sea freight": "sea",
        "ship": "sea",

        "road": "truck",
        "lorry": "truck",

        "truck_gcc": "truck_gcc",
        "truck_ksa": "truck_ksa",
        "truck_uae": "truck_uae",

        "railroad": "rail",
        "train": "rail",
        "rail_ksa": "rail_ksa",

        "n/a": "none",
        "": "none",
        "none": "none"
    })
    df["transport_mode"] = tm

    # --- Spend in correct base currency matching EF unit ---
    df["spend_base"] = _compute_spend_base(df, spend_unit, fx_rate=eur_to_gbp)

    # Spend EF map (DataFrame indexed by key)
    spend_map = spend_efs["ef_value"].to_dict() if isinstance(spend_efs, pd.DataFrame) else {}
    df["ef_spend"] = df["category"].map(spend_map).fillna(0.0)
    df["co2_spend_kg"] = df["spend_base"] * df["ef_spend"]

    # Transport EF
    df["weight_t"] = df.get("weight_kg", 0.0) / 1000.0
    df["tkm"] = df["weight_t"] * df.get("distance_km", 0.0)
    df["ef_mode"] = df["transport_mode"].map(mode_ef if isinstance(mode_ef, dict) else {}).fillna(0.0)
    df["co2_transport_kg"] = df["tkm"] * df["ef_mode"]

    df["co2_total_kg"] = df["co2_spend_kg"] + df["co2_transport_kg"]

    core_spend = float(df["co2_spend_kg"].sum())
    core_trans = float(df["co2_transport_kg"].sum())
    core_total = core_spend + core_trans
    return df, core_spend, core_trans, core_total


# ---------------------------------------------------------
# Waste
# ---------------------------------------------------------
def calc_waste(df_w, ef_waste, costs=None):
    """
    Returns: df, total_emissions_kg, total_cost_eur
    Costs used:
      - Waste disposal: `waste_per_kg` (€/kg)
      - Optional row override: `waste_cost_eur_per_kg`
    """
    if costs is None:
        costs = {"waste_per_kg": 0.0}
    if df_w is None or df_w.empty:
        return pd.DataFrame(), 0.0, 0.0

    w = df_w.copy()
    if "waste_tonnes" not in w.columns:
        return w.assign(ef=0, co2_waste_kg=0, cost_waste_eur=0), 0.0, 0.0

    w["waste_tonnes"] = pd.to_numeric(w["waste_tonnes"], errors="coerce").fillna(0).clip(lower=0)
    w["waste_type"] = normalize_series(w.get("waste_type", pd.Series([""] * len(w))))
    w["waste_key"] = w["waste_type"].replace({"mixed": "landfill", "paper": "recycling", "plastic": "recycling"})

    ef_map = ef_waste if isinstance(ef_waste, dict) else {}
    w["ef"] = w["waste_key"].map(ef_map).fillna(0)
    w["co2_waste_kg"] = w["waste_tonnes"] * 1000 * w["ef"]

    # cost
    cost_per_kg = _col_or_default(w, "waste_cost_eur_per_kg", costs.get("waste_per_kg", 0.0))
    w["cost_waste_eur"] = w["waste_tonnes"] * 1000 * cost_per_kg

    return w, float(w["co2_waste_kg"].sum()), float(w["cost_waste_eur"].sum())


# ---------------------------------------------------------
# Travel
# ---------------------------------------------------------
def calc_travel(df_t, ef_travel, costs=None):
    """
    Returns: df, total_emissions_kg, total_cost_eur
    Costs used:
      - Travel pkm: `travel_per_pkm` (€/pkm); row override `travel_cost_eur_per_pkm`
      - Hotel (optional): row override `hotel_cost_eur_per_night` (no default here)
    """
    if costs is None:
        costs = {"travel_per_pkm": 0.0}
    if df_t is None or df_t.empty:
        return pd.DataFrame(), 0.0, 0.0

    t = df_t.copy()
    t["passenger_km"] = pd.to_numeric(t.get("passenger_km", 0), errors="coerce").fillna(0).clip(lower=0)
    t["hotel_nights"] = pd.to_numeric(t.get("hotel_nights", 0), errors="coerce").fillna(0).clip(lower=0)
    t["travel_mode"] = normalize_series(t.get("travel_mode", pd.Series([""] * len(t))))
    ef_map = ef_travel if isinstance(ef_travel, dict) else {}

    t["travel_key"] = t["travel_mode"]
    t["ef_pkm"] = t["travel_key"].map(ef_map).fillna(0)
    ef_hotel = ef_map.get("hotel_night", 0.0)

    # emissions
    t["co2_travel_pkm_kg"] = t["passenger_km"] * t["ef_pkm"]
    t["co2_travel_hotel_kg"] = t["hotel_nights"] * ef_hotel if ef_hotel else 0
    t["co2_travel_total_kg"] = t["co2_travel_pkm_kg"] + t["co2_travel_hotel_kg"]

    # cost (pkm default/override + optional hotel cost column)
    cost_pkm = _col_or_default(t, "travel_cost_eur_per_pkm", costs.get("travel_per_pkm", 0.0))
    t["cost_travel_eur"] = t["passenger_km"] * cost_pkm
    if "hotel_cost_eur_per_night" in t.columns:
        t["cost_travel_eur"] += pd.to_numeric(t["hotel_cost_eur_per_night"], errors="coerce").fillna(0) * t["hotel_nights"]

    return t, float(t["co2_travel_total_kg"].sum()), float(t["cost_travel_eur"].sum())


# ---------------------------------------------------------
# Commute
# ---------------------------------------------------------
def calc_commute(df_c, ef_comm, costs=None):
    """
    Returns: df, total_emissions_kg, total_cost_eur
    Costs used:
      - Commute pkm: `commute_per_pkm` (€/pkm); row override `commute_cost_eur_per_pkm`
    """
    if costs is None:
        costs = {"commute_per_pkm": 0.0}
    if df_c is None or df_c.empty:
        return pd.DataFrame(), 0.0, 0.0

    c = df_c.copy()
    c["passenger_km"] = pd.to_numeric(c.get("passenger_km", 0), errors="coerce").fillna(0).clip(lower=0)
    c["mode"] = normalize_series(c.get("mode", pd.Series([""] * len(c))))
    ef_map = ef_comm if isinstance(ef_comm, dict) else {}

    c["mode_key"] = c["mode"]
    c["ef_pkm"] = c["mode_key"].map(ef_map).fillna(0)
    c["co2_commute_kg"] = c["passenger_km"] * c["ef_pkm"]

    # cost
    cost_pkm = _col_or_default(c, "commute_cost_eur_per_pkm", costs.get("commute_per_pkm", 0.0))
    c["cost_commute_eur"] = c["passenger_km"] * cost_pkm

    return c, float(c["co2_commute_kg"].sum()), float(c["cost_commute_eur"].sum())


# ---------------------------------------------------------
# Energy (Scope 2)
# ---------------------------------------------------------
def calc_energy(df_e, ef_energy, costs=None):
    """
    Returns: df, total_emissions_kg, total_cost_eur
    Costs used:
      - Electricity/steam/etc.: `per_kwh` default; row override `cost_eur_per_kwh`

    Note: For EcoScopeAI-Arabia, regional electricity keys can be:
      - meter_type = "electricity", region = "ksa"  -> energy_key "electricity_ksa"
      - meter_type = "electricity_ksa" (region empty) -> energy_key "electricity_ksa"
    """
    if costs is None:
        costs = {"per_kwh": 0.0}
    if df_e is None or df_e.empty:
        return pd.DataFrame(), 0.0, 0.0

    e = df_e.copy()
    e["kwh"] = pd.to_numeric(e.get("kwh", 0), errors="coerce").fillna(0).clip(lower=0)
    e["meter_type"] = normalize_series(e.get("meter_type", pd.Series([""] * len(e))))
    e["region"] = normalize_series(e.get("region", pd.Series([""] * len(e))))

    # energy_key: either "meter_region" or just "meter"
    e["energy_key"] = e.apply(
        lambda r: f"{r['meter_type']}_{r['region']}" if r["region"] else r["meter_type"],
        axis=1
    )

    energy_dict = {k: v["ef_value"] for k, v in (ef_energy or {}).items()}
    e["ef_kwh"] = e["energy_key"].map(energy_dict).fillna(0)

    # fallback: try using meter_type only
    miss = e["ef_kwh"].eq(0)
    if miss.any():
        e.loc[miss, "ef_kwh"] = e.loc[miss, "meter_type"].map(energy_dict).fillna(0)

    e["co2_energy_kg"] = e["kwh"] * e["ef_kwh"]

    # cost
    cost_kwh = _col_or_default(e, "cost_eur_per_kwh", costs.get("per_kwh", 0.0))
    e["cost_energy_eur"] = e["kwh"] * cost_kwh

    return e, float(e["co2_energy_kg"].sum()), float(e["cost_energy_eur"].sum())


# ---------------------------------------------------------
# CBAM
# ---------------------------------------------------------
def calc_cbam(df_cbam, ef_cbam):
    """
    Returns: df, total_embedded_tco2e
    (No € by default; CBAM cost logic depends on certificate/period pricing.)
    """
    if df_cbam is None or df_cbam.empty:
        return pd.DataFrame(), 0.0

    c = df_cbam.copy()
    c["quantity_t"] = pd.to_numeric(c.get("quantity_t", 0), errors="coerce").fillna(0).clip(lower=0)
    c["hs_code"] = normalize_series(c.get("hs_code", pd.Series([""] * len(c))))

    if "intensity_tco2e_per_t" in c.columns:
        c["intensity_tco2e_per_t"] = pd.to_numeric(c["intensity_tco2e_per_t"], errors="coerce").fillna(0).clip(lower=0)
    else:
        c["intensity_tco2e_per_t"] = c["hs_code"].map(ef_cbam if isinstance(ef_cbam, dict) else {}).fillna(0)

    c["embedded_tco2e"] = c["quantity_t"] * c["intensity_tco2e_per_t"]
    return c, float(c["embedded_tco2e"].sum())


# ---------------------------------------------------------
# Export coverage flags
# ---------------------------------------------------------
def export_gate_flags(df):
    total_spend = float(df.get("spend_base", pd.Series([0])).sum())
    mapped_spend = float(df.loc[df.get("ef_spend", 0) > 0, "spend_base"].sum())
    unmapped_cat_ratio = 1 - (mapped_spend / total_spend if total_spend > 0 else 1)

    total_tkm = float(df.get("tkm", pd.Series([0])).sum())
    mapped_tkm = float(df.loc[df.get("ef_mode", 0) > 0, "tkm"].sum())
    unmapped_mode_ratio = 1 - (mapped_tkm / total_tkm if total_tkm > 0 else 1)
    return unmapped_cat_ratio, unmapped_mode_ratio


# ---------------------------------------------------------
# Anomaly detection (IsolationForest)
# ---------------------------------------------------------
def _safe_numeric(df, cols):
    X = pd.DataFrame(index=df.index)
    for c in cols:
        X[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0)
    return X


def isolation_forest_detect(df, feature_cols, contamination=0.05, random_state=42):
    """
    Returns a copy of df with:
      - anomaly_score: decision_function (higher = more normal; we invert for ranking)
      - is_anomaly: True for predicted outliers
    If sklearn is not available, returns df with no anomalies flagged.
    """
    if df is None or len(df) == 0:
        return df

    try:
        from sklearn.ensemble import IsolationForest
    except Exception:
        out = df.copy()
        out["anomaly_score"] = 0.0
        out["is_anomaly"] = False
        return out

    X = _safe_numeric(df, feature_cols)
    if X.shape[0] < 5:  # too few rows
        out = df.copy()
        out["anomaly_score"] = 0.0
        out["is_anomaly"] = False
        return out

    contamination = max(1e-3, min(0.5, float(contamination)))

    iso = IsolationForest(
        n_estimators=200, max_samples="auto", contamination=contamination,
        random_state=random_state, n_jobs=-1, verbose=0
    )
    iso.fit(X)

    scores = iso.decision_function(X)  # higher => more normal
    preds = iso.predict(X)             # -1 outlier, 1 inlier

    out = df.copy()
    out["anomaly_score"] = -scores     # invert so higher = more anomalous
    out["is_anomaly"] = (preds == -1)
    return out


def build_anomaly_views(df_core, waste_df, travel_df, comm_df, energy_df, contamination=0.05):
    """
    Returns a dict of dataframes with anomaly flags per module.
    """
    views = {}

    if isinstance(df_core, pd.DataFrame) and not df_core.empty:
        core_feats = ["spend_base", "weight_kg", "distance_km", "tkm",
                      "co2_spend_kg", "co2_transport_kg", "co2_total_kg"]
        views["core"] = isolation_forest_detect(df_core, core_feats, contamination)

    if isinstance(waste_df, pd.DataFrame) and not waste_df.empty:
        waste_feats = ["waste_tonnes", "co2_waste_kg", "cost_waste_eur"]
        views["waste"] = isolation_forest_detect(waste_df, waste_feats, contamination)

    if isinstance(travel_df, pd.DataFrame) and not travel_df.empty:
        travel_feats = ["passenger_km", "hotel_nights", "co2_travel_pkm_kg",
                        "co2_travel_hotel_kg", "co2_travel_total_kg", "cost_travel_eur"]
        views["travel"] = isolation_forest_detect(travel_df, travel_feats, contamination)

    if isinstance(comm_df, pd.DataFrame) and not comm_df.empty:
        commute_feats = ["passenger_km", "co2_commute_kg", "cost_commute_eur"]
        views["commute"] = isolation_forest_detect(comm_df, commute_feats, contamination)

    if isinstance(energy_df, pd.DataFrame) and not energy_df.empty:
        energy_feats = ["kwh", "co2_energy_kg", "cost_energy_eur"]
        views["energy"] = isolation_forest_detect(energy_df, energy_feats, contamination)

    return views
