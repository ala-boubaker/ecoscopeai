import streamlit as st
import pandas as pd

import config
from config import (
    KEYS, APP_TITLE, APP_VERSION, DEFAULT_COSTS,
    EXPORT_UNMAPPED_THRESHOLD,
)

from io_utils import (
    load_table_any, read_multisheet, validate_ef, build_ef_structs,
    build_ef_corpora, EmbeddingEngine, normalize_series
)
from calcs import (
    calc_core, calc_waste, calc_travel, calc_commute, calc_energy, calc_cbam, export_gate_flags,
    build_anomaly_views  # NEW
)
from ui_tabs import (
    tab_coverage, tab_suppliers, tab_categories,
    tab_waste as tab_waste_ui, tab_travel as tab_travel_ui, tab_commute as tab_commute_ui,
    tab_energy as tab_energy_ui, tab_scenarios as tab_scenarios_ui, tab_reports as tab_reports_ui,
    tab_cbam as tab_cbam_ui, tab_mapping as tab_mapping_ui,
    tab_anomalies as tab_anomalies_ui,
    tab_esg_intel,
    tab_clusters_ui,
    tab_forecast_ui
)

# --- tolerant unpack helpers (handle old/new calc_* return shapes) ---
def unpack2(res, defaults=(pd.DataFrame(), 0.0)):
    """Return (df, total) whether res is (df,total) or (df,total,cost)."""
    if isinstance(res, tuple):
        return (res + defaults)[:2]
    return defaults

def unpack4(res, defaults=(pd.DataFrame(), 0.0, 0.0, 0.0)):
    """Return (df, spend_kg, trans_kg, total_kg) for calc_core variants."""
    if isinstance(res, tuple):
        return (res + defaults)[:4]
    return defaults


# ---------------- Layout / Title ----------------
st.set_page_config(page_title=f"{APP_TITLE}", layout="wide")
st.title(f"{APP_TITLE} · {APP_VERSION}")
st.caption("Upload EF + Data or use demo • KPI cards + charts • NLP Mapping • Scenarios • CSRD/CBAM exports.")

# --- CSS tweak: allow tab bar to wrap into multiple rows ---
st.markdown("""
<style>
.stTabs [role="tablist"] {
    flex-wrap: wrap;
}
</style>
""", unsafe_allow_html=True)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings & Inputs")

    # --- NEW: Region / Profile selector (EcoScopeAI vs EcoScopeAI-Arabia) ---
    region = st.selectbox(
        "Region / Profile",
        options=[config.REGION_GLOBAL_DEFRA, config.REGION_ARABIA_MENA],
        format_func=lambda r: (
            "EcoScopeAI — CSRD/CBAM (Global/DEFRA)" if r == config.REGION_GLOBAL_DEFRA
            else "EcoScopeAI-Arabia — MENA SDG 12 & 13"
        ),
        key=KEYS.get("region_sel", "sb_region_select"),
    )
    # store active region in session so other modules (ui_tabs, exports, etc.) can read it
    st.session_state["active_region"] = region

    use_demo = st.toggle("Use demo EF + data", value=True, key=KEYS.get("use_demo","sb_use_demo"))
    ef_file = st.file_uploader(
        "Upload Emission Factors (CSV/XLSX)", type=["csv","xlsx"],
        key=KEYS.get("ef_uploader","sb_ef_upload")
    )

    st.subheader("Mapping Helper (NLP)")
    use_embeddings = st.toggle("Enable semantic suggestions", value=True, key=KEYS.get("use_embeddings","use_embeddings"))
    top_k = st.number_input("Top-k suggestions", min_value=1, max_value=10, value=3, step=1, key=KEYS.get("top_k","map_top_k"))

    st.subheader("Anomaly detection")
    contamination = st.slider("Outlier rate (IsolationForest contamination)", 0.01, 0.20, 0.05, 0.01)

    currency = st.selectbox("Display currency", ["EUR","GBP"], index=0, key=KEYS.get("currency_sel","currency_sel"))
    eur_to_gbp = st.number_input(
        "EUR ➜ GBP conversion (for spend-EF per GBP)",
        min_value=0.01, max_value=10.0,
        value=DEFAULT_COSTS.get("eur_to_gbp", DEFAULT_COSTS.get("EUR_to_GBP", 0.85)),
        step=0.01, key=KEYS.get("eur_to_gbp","sb_eur2gbp"),
        help="Used only if spend emission factors are per GBP; transport/energy costs unaffected."
    )

    st.divider()
    st.caption("Default unit costs (used if rows do not provide their own cost columns).")
    c1, c2 = st.columns(2)
    with c1:
        per_kwh = st.number_input(
            "Electricity cost (€/kWh)", min_value=0.0,
            value=DEFAULT_COSTS.get("per_kwh",0.25),
            step=0.01, key=KEYS.get("cost_kwh","cost_kwh")
        )
        per_tkm = st.number_input(
            "Freight cost (€/t·km)",   min_value=0.0,
            value=DEFAULT_COSTS.get("per_tkm",0.08),
            step=0.01, key=KEYS.get("cost_tkm","cost_tkm")
        )
        waste_per_kg = st.number_input(
            "Waste cost (€/kg)",   min_value=0.0,
            value=DEFAULT_COSTS.get("waste_per_kg",0.20),
            step=0.01, key=KEYS.get("cost_waste_kg","cost_waste_kg")
        )
    with c2:
        travel_per_pkm  = st.number_input(
            "Business travel (€/pkm)", min_value=0.0,
            value=DEFAULT_COSTS.get("travel_per_pkm",0.12),
            step=0.01, key=KEYS.get("cost_travel_pkm","cost_travel_pkm")
        )
        commute_per_pkm = st.number_input(
            "Commute (€/pkm)",        min_value=0.0,
            value=DEFAULT_COSTS.get("commute_per_pkm",0.10),
            step=0.01, key=KEYS.get("cost_commute_pkm","cost_commute_pkm")
        )

    # Persist chosen costs
    st.session_state["COSTS"] = {
        "currency": currency,
        "eur_to_gbp": eur_to_gbp,
        "per_kwh": per_kwh,
        "per_tkm": per_tkm,
        "waste_per_kg": waste_per_kg,
        "travel_per_pkm": travel_per_pkm,
        "commute_per_pkm": commute_per_pkm,
    }


# ---------------- Caching wrappers ----------------
@st.cache_data(show_spinner=False)
def _cached_load_ef(path_or_file):
    return load_table_any(path_or_file)

@st.cache_data(show_spinner=False)
def _cached_load_data(path_or_file):
    return read_multisheet(path_or_file)

@st.cache_resource(show_spinner=False)
def _cached_build_engines(ef_df: pd.DataFrame):
    """Fit one TF-IDF engine per EF type; cached by EF contents."""
    if ef_df is None or ef_df.empty:
        return {}
    corpora = build_ef_corpora(ef_df)
    engines = {}
    for t, pairs in corpora.items():
        eng = EmbeddingEngine()
        eng.fit(pairs)
        engines[t] = eng
    return engines

@st.cache_data(show_spinner=False)
def _cached_anomalies(_df_core, _waste_df, _travel_df, _comm_df, _energy_df, _cont):
    # NOTE: dataframe identity/contents changes will invalidate cache automatically under Streamlit hash rules
    return build_anomaly_views(_df_core, _waste_df, _travel_df, _comm_df, _energy_df, contamination=_cont)


# ---------------- Load EF ----------------
active_region = st.session_state.get("active_region", config.ACTIVE_REGION)
demo_ef_path = config.get_setting("demo_ef_path", active_region)

efs = _cached_load_ef(demo_ef_path) if (use_demo and (ef_file is None)) else _cached_load_ef(ef_file)
if not validate_ef(efs):
    st.error("EF file missing/invalid. Needs columns: type, key, ef_value, unit. Upload or enable demo in sidebar.")
    spend_efs = pd.DataFrame(); mode_ef={}; ef_waste={}; ef_travel={}; ef_comm={}; ef_energy={}; ef_cbam={}; spend_unit="kgCO2e_per_GBP"
else:
    spend_efs, mode_ef, ef_waste, ef_travel, ef_comm, ef_energy, ef_cbam, spend_unit = build_ef_structs(efs)

# ---------------- Build NLP engines (cached) ----------------
engines = _cached_build_engines(efs) if (use_embeddings and not efs.empty) else {}

# ---------------- Load DATA ----------------
demo_data_path = config.get_setting("demo_data_path", active_region)

sheets = (
    _cached_load_data(demo_data_path)
    if (use_demo and (st.session_state.get("DATA_UP", None) is None))
    else _cached_load_data(st.session_state.get("DATA_UP"))
    if (st.session_state.get("DATA_UP") is not None)
    else _cached_load_data(demo_data_path)
    if (use_demo and (st.session_state.get("DATA_UP") is None))
    else _cached_load_data(None)
)

# When user uploads a file, keep it in session to avoid re-reading on rerun
data_file = st.file_uploader(
    "📂 Upload company data (XLSX with sheets: Procurement_Transport, Waste, Business_Travel, Employee_Commute, Energy, CBAM)",
    type=["csv","xlsx"], key=KEYS.get("data_uploader","main_data_upload")
)
if data_file is not None:
    st.session_state["DATA_UP"] = data_file
    sheets = _cached_load_data(data_file)

df_pt     = sheets.get("Procurement_Transport", pd.DataFrame())
df_waste  = sheets.get("Waste", pd.DataFrame())
df_travel = sheets.get("Business_Travel", pd.DataFrame())
df_comm   = sheets.get("Employee_Commute", pd.DataFrame())
df_energy = sheets.get("Energy", pd.DataFrame())
df_cbam   = sheets.get("CBAM", pd.DataFrame())

# Minimal column guarantees for core
if isinstance(df_pt, pd.DataFrame) and not df_pt.empty:
    df_pt = df_pt.copy()
    for c in ["supplier","category","spend_eur","weight_kg","distance_km","transport_mode"]:
        if c not in df_pt.columns:
            df_pt[c] = 0 if c in {"spend_eur","weight_kg","distance_km"} else ""
else:
    df_pt = pd.DataFrame()


# ---------------- Apply user REMAPs (from Mapping tab) ----------------
REMAP = st.session_state.get("REMAP", {"spend":{}, "transport":{}, "waste":{}, "travel":{}, "commute":{}, "energy":{}})

def apply_remaps():
    global df_pt, df_waste, df_travel, df_comm, df_energy
    # Core spend/category
    if isinstance(df_pt, pd.DataFrame) and not df_pt.empty and REMAP["spend"]:
        df_pt["category"] = df_pt["category"].astype(str).replace(REMAP["spend"])
    # Core transport mode
    if isinstance(df_pt, pd.DataFrame) and not df_pt.empty and REMAP["transport"]:
        df_pt["transport_mode"] = df_pt["transport_mode"].astype(str).replace(REMAP["transport"])
    # Waste
    if isinstance(df_waste, pd.DataFrame) and not df_waste.empty and REMAP["waste"]:
        df_waste["waste_type"] = df_waste["waste_type"].astype(str).replace(REMAP["waste"])
    # Travel
    if isinstance(df_travel, pd.DataFrame) and not df_travel.empty and REMAP["travel"]:
        df_travel["travel_mode"] = df_travel["travel_mode"].astype(str).replace(REMAP["travel"])
    # Commute
    if isinstance(df_comm, pd.DataFrame) and not df_comm.empty and REMAP["commute"]:
        df_comm["mode"] = df_comm["mode"].astype(str).replace(REMAP["commute"])
    # Energy (meter_type)
    if isinstance(df_energy, pd.DataFrame) and not df_energy.empty and REMAP["energy"]:
        mt = df_energy.get("meter_type")
        if mt is not None:
            df_energy["meter_type"] = mt.astype(str).replace(REMAP["energy"])

apply_remaps()


# ---------------- Calculations ----------------
# core
if not spend_efs.empty:
    _core = calc_core(
        df_pt, spend_efs, mode_ef, st.session_state["COSTS"]["eur_to_gbp"], spend_unit
    )
    df_core, core_spend, core_trans, core_total = unpack4(_core)
else:
    df_core, core_spend, core_trans, core_total = pd.DataFrame(), 0.0, 0.0, 0.0

# modules
if isinstance(df_waste, pd.DataFrame) and not df_waste.empty:
    waste_df, waste_total = unpack2(calc_waste(df_waste, ef_waste))
else:
    waste_df, waste_total = pd.DataFrame(), 0.0

if isinstance(df_travel, pd.DataFrame) and not df_travel.empty:
    travel_df, travel_total = unpack2(calc_travel(df_travel, ef_travel))
else:
    travel_df, travel_total = pd.DataFrame(), 0.0

if isinstance(df_comm, pd.DataFrame) and not df_comm.empty:
    comm_df, commute_total = unpack2(calc_commute(df_comm, ef_comm))
else:
    comm_df, commute_total = pd.DataFrame(), 0.0

if isinstance(df_energy, pd.DataFrame) and not df_energy.empty:
    energy_df, energy_total = unpack2(calc_energy(df_energy, ef_energy))
else:
    energy_df, energy_total = pd.DataFrame(), 0.0

# --- energy unit note ---
if ef_energy:
    try:
        energy_units = set(v["unit"] for v in ef_energy.values())
    except Exception:
        energy_units = set()
else:
    energy_units = set()

energy_unit_note = ", ".join(sorted(energy_units)) if energy_units else "kgCO2e_per_kwh (assumed)"

# cbam
if isinstance(df_cbam, pd.DataFrame) and not df_cbam.empty:
    cbam_df, cbam_total_t = unpack2(calc_cbam(df_cbam, ef_cbam), defaults=(pd.DataFrame(), 0.0))
else:
    cbam_df, cbam_total_t = pd.DataFrame(), 0.0

anomalies = _cached_anomalies(df_core, waste_df, travel_df, comm_df, energy_df, contamination)


# ---------------- Tabs ----------------
tabs = st.tabs([
    "📂 Data","🚨 Anomalies","🧭 Mapping","📊 Coverage & KPIs","🌍 Clusters","🏭 Suppliers","📦 Categories",
    "🗑️ Waste","🛫 Travel","🚇 Commute","⚡ Energy","🏷️ CBAM","🧪 Scenarios","🔮 Forecast","🧠 ESG Intel","📑 Reports"
])
(tab_data_v, tab_anom_v, tab_map_v, tab_cov_v, tab_clusters_v, tab_sup_v, tab_cat_v,
 tab_waste_v, tab_travel_v, tab_commute_v, tab_energy_v,
 tab_cbam_v, tab_scen_v, tab_forecast_v, tab_esg_v, tab_reports_v) = tabs

# Data preview
with tab_data_v:
    st.subheader("Raw Data Preview")
    if df_pt.empty and waste_df.empty and travel_df.empty and comm_df.empty and energy_df.empty and cbam_df.empty:
        st.info("No data loaded. Upload a file or toggle 'Use demo EF + data' in the sidebar.")
    else:
        st.write("**Procurement_Transport**"); st.dataframe(df_pt.head(50), use_container_width=True)
        c1,c2 = st.columns(2)
        with c1:
            st.write("**Waste**");   st.dataframe(waste_df.head(50), use_container_width=True)
            st.write("**Commute**"); st.dataframe(comm_df.head(50), use_container_width=True)
        with c2:
            st.write("**Travel**");  st.dataframe(travel_df.head(50), use_container_width=True)
            st.write("**Energy**");  st.dataframe(energy_df.head(50), use_container_width=True)
        st.write("**CBAM**"); st.dataframe(cbam_df.head(50), use_container_width=True)

tab_anomalies_ui(tab_anom_v, anomalies)

# Coverage + KPIs
extras_total = (waste_total + travel_total + commute_total + energy_total)
tab_coverage(tab_cov_v, df_core, spend_efs, mode_ef, core_spend, core_trans, core_total, extras_total)

# Clusters
tab_clusters_ui(tab_clusters_v, df_core)

# Suppliers / Categories
tab_suppliers(tab_sup_v, df_core)
tab_categories(tab_cat_v, df_core)

# Modules
tab_waste_ui(tab_waste_v, waste_df, waste_total)
tab_travel_ui(tab_travel_v, travel_df, travel_total)
tab_commute_ui(tab_commute_v, comm_df, commute_total)
tab_energy_ui(tab_energy_v, energy_df, energy_total, energy_unit_note)
tab_cbam_ui(tab_cbam_v, cbam_df, cbam_total_t)

# Mapping (NLP)
dataframes = {"core": df_core, "waste": waste_df, "travel": travel_df, "commute": comm_df, "energy": energy_df}
tab_mapping_ui(tab_map_v, engines if use_embeddings else {}, efs, dataframes, st.session_state, top_k=top_k)

# Scenarios
tab_scenarios_ui(tab_scen_v, df_core, energy_df, st.session_state)

# Export gate
if df_core is not None and not df_core.empty:
    spend_unmapped_ratio, tkm_unmapped_ratio = export_gate_flags(df_core)
else:
    spend_unmapped_ratio, tkm_unmapped_ratio = 0.0, 0.0
blocked = (spend_unmapped_ratio > EXPORT_UNMAPPED_THRESHOLD) or (tkm_unmapped_ratio > EXPORT_UNMAPPED_THRESHOLD)

# CBAM export rows (for PDF helper)
cbam_rows = []
if not cbam_df.empty:
    for _, r in cbam_df.iterrows():
        cbam_rows.append([
            r.get("hs_code",""),
            r.get("product_name",""),
            r.get("period",""),
            float(r.get("quantity_t",0.0)),
            float(r.get("intensity_tco2e_per_t",0.0)),
            float(r.get("embedded_tco2e",0.0)),
        ])
exports = {"cbam_rows": cbam_rows}

# Forecast
tab_forecast_ui(tab_forecast_v, energy_df)

# ESG Intelligence (sentiment + topics)
tab_esg_intel(tab_esg_v, st.session_state)

# Reports tab
tab_reports_ui(
    tab_reports_v, blocked, (df_core is not None and not df_core.empty),
    core_spend, core_trans,
    waste_df, waste_total, travel_df, travel_total, comm_df, commute_total, energy_df, energy_total,
    df_core, exports, spend_unmapped_ratio, tkm_unmapped_ratio
)

st.caption("© EcoScopeAI demo. EFs: spend per GBP/EUR, transport per t·km, travel per pkm, energy per kWh.")
