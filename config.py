APP_TITLE = "EcoScopeAI — CSRD + CBAM Pilot"
APP_VERSION = "v4.3"

# ------------------------------------------------------------------
# 0. REGION PROFILES (for future EcoScopeAI-Arabia support)
# ------------------------------------------------------------------
# NOTE: For now, the app still uses only the DEFRA-style demo paths
# via DEMO_EF_PATH and DEMO_DATA_PATH below, so behaviour stays
# identical. We'll start using these region settings in app.py later.

REGION_GLOBAL_DEFRA = "global_defra"
REGION_ARABIA_MENA  = "arabia_mena"

# Active region (default) – not yet used by app.py
ACTIVE_REGION = REGION_GLOBAL_DEFRA

REGION_SETTINGS = {
    REGION_GLOBAL_DEFRA: {
        "app_title": "EcoScopeAI — CSRD + CBAM Pilot",
        "currency": "EUR",
        "default_eur_to_gbp": 0.85,
        "demo_ef_path": "ef_demo.csv",
        "demo_data_path": "sample_data.xlsx",
        "language": "en",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    },
    REGION_ARABIA_MENA: {
        "app_title": "EcoScopeAI-Arabia — MENA SDG 12 & 13",
        # you can later switch to SAR/AED as base currency if you wish
        "currency": "USD",
        "default_eur_to_gbp": 0.85,
        "demo_ef_path": "ef_mena_demo.csv",      # to be created
        "demo_data_path": "sample_data_mena.xlsx",  # to be created
        "language": "ar",
        # Example multilingual model for Arabic support
        "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    },
}

def get_setting(key: str, region: str = ACTIVE_REGION):
    """Helper to fetch a setting for a given region."""
    return REGION_SETTINGS[region][key]


# ------------------------------------------------------------------
# 1. OLD GLOBAL CONSTANTS (kept for backward compatibility)
# ------------------------------------------------------------------

# Defaults
DEFAULT_EUR_TO_GBP = 0.85
EXPORT_UNMAPPED_THRESHOLD = 0.10  # 10%

# Demo file names (still used by current app.py)
DEMO_EF_PATH = "ef_demo.csv"
DEMO_DATA_PATH = "sample_data.xlsx"

# --- NLP / Embeddings (English-only now, easy to swap later) ---
USE_EMBEDDINGS_DEFAULT = True
EMBEDDING_MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"  # fast, CPU-friendly
SUGGEST_TOP_K          = 3               # top-k suggestions per unmapped value

# ------------------------------------------------------------------
# 2. Streamlit keys
# ------------------------------------------------------------------
KEYS = {
    "use_demo": "sb_use_demo",
    "ef_uploader": "sb_ef_upload",
    "data_uploader": "main_data_upload",
    "currency_sel": "sb_currency",
    "eur_to_gbp": "sb_eur2gbp",

    # NEW (for later): region selector in sidebar
    "region_sel": "sb_region_select",

    # cost inputs
    "cost_kwh": "cost_kwh",
    "cost_tkm": "cost_tkm",
    "cost_waste_kg": "cost_waste_kg",
    "cost_travel_pkm": "cost_travel_pkm",
    "cost_commute_pkm": "cost_commute_pkm",

    # NLP
    "use_embeddings": "use_embeddings",
    "top_k": "top_k",

    # Scenarios
    "sc_from": "sc_from_mode",
    "sc_to": "sc_to_mode",
    "sc_pct": "sc_shift_pct",
    "sc_name": "sc_name",
    "en_pct": "sc_en_pct",
    "en_ef_r": "sc_en_ef_renew",
    "en_ef_f": "sc_en_ef_fossil",
    "en_name": "sc_en_name",
    "sc_remove": "sc_remove_select",
}

# ------------------------------------------------------------------
# 3. Default costs used in the sidebar (kept for your app.py)
# ------------------------------------------------------------------
DEFAULT_COSTS = {
    "eur_to_gbp": DEFAULT_EUR_TO_GBP,
    "per_kwh": 0.20,
    "per_tkm": 0.05,
    "waste_per_kg": 0.12,
    "travel_per_pkm": 0.20,
    "commute_per_pkm": 0.10,
}
