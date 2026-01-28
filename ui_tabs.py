import config
import io
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, List, Tuple

from config import EXPORT_UNMAPPED_THRESHOLD, KEYS


# -------------------- Coverage & KPIs --------------------
def tab_coverage(tab, df, spend_efs, mode_ef, core_spend, core_trans, core_total, extras_total):
    with tab:
        st.subheader("Coverage & KPIs")
        if df is None or df.empty:
            st.info("Upload company data to view KPIs.")
            return

        # Coverage ratios
        total_spend = float(df.get("spend_base", pd.Series([0])).sum())
        mapped_spend = float(df.loc[df.get("ef_spend", 0) > 0, "spend_base"].sum())
        unmapped_spend = max(0.0, total_spend - mapped_spend)
        spend_unmapped_ratio = (unmapped_spend / total_spend) if total_spend > 0 else 0.0

        total_tkm = float(df.get("tkm", pd.Series([0])).sum())
        mapped_tkm = float(df.loc[df.get("ef_mode", 0) > 0, "tkm"].sum())
        unmapped_tkm = max(0.0, total_tkm - mapped_tkm)
        tkm_unmapped_ratio = (unmapped_tkm / total_tkm) if total_tkm > 0 else 0.0

        # KPI cards
        c1,c2,c3,c4 = st.columns(4)
        region = st.session_state.get("active_region", config.REGION_GLOBAL_DEFRA)
        if region == config.REGION_ARABIA_MENA:
            c1.metric("Value Chain — Purchased Goods", f"{core_spend:,.0f} kgCO₂e")
            c2.metric("Logistics Emissions (land/sea/air)", f"{core_trans:,.0f} kgCO₂e")
            c3.metric("Total Value Chain (core)", f"{core_total:,.0f} kgCO₂e")
        else:
            c1.metric("Cat.1 Purchased Goods", f"{core_spend:,.0f} kgCO₂e")
            c2.metric("Cat.4/9 Transport", f"{core_trans:,.0f} kgCO₂e")
            c3.metric("TOTAL Scope 3 (core)", f"{core_total:,.0f} kgCO₂e")
        c4.metric("Extras (modules)", f"{extras_total:,.0f} kgCO₂e")
        st.caption(f"Unmapped spend: {spend_unmapped_ratio:.1%} · Unmapped t·km: {tkm_unmapped_ratio:.1%}")

        # Stacked bar: Spend vs Transport by Category
        by_cat = (
            df.groupby("category")[["co2_spend_kg", "co2_transport_kg"]]
              .sum().reset_index().sort_values("co2_spend_kg", ascending=False)
        )
        if not by_cat.empty:
            fig = px.bar(
                by_cat,
                x="category", y=["co2_spend_kg", "co2_transport_kg"],
                barmode="stack", text_auto=True,
                title="Emissions by Category — Spend vs Transport",
                labels={"value":"kgCO₂e","category":"Category","variable":"Component"},
            )
            fig.update_layout(xaxis_tickangle=-15, yaxis_title="kgCO₂e", legend_title="Component")
            st.plotly_chart(fig, use_container_width=True)

        # Export gate
        if (spend_unmapped_ratio > EXPORT_UNMAPPED_THRESHOLD) or (tkm_unmapped_ratio > EXPORT_UNMAPPED_THRESHOLD):
            st.error("Exports blocked: reduce unmapped spend/t·km below 10% to enable downloads.")


# -------------------- Suppliers --------------------
def tab_suppliers(tab, df):
    with tab:
        st.subheader("Supplier Hotspots")
        if df is None or df.empty:
            st.info("Upload company data to view suppliers.")
            return

        sup = (
            df.groupby("supplier", dropna=False)[["spend_eur","co2_spend_kg","co2_transport_kg","co2_total_kg"]]
              .sum().reset_index().sort_values("co2_total_kg", ascending=False)
        )

        if sup.empty:
            st.info("No supplier data available.")
            return

        c1, c2 = st.columns([3,1])
        with c1:
            st.dataframe(
                sup.head(100).style.format({
                    "spend_eur":"{:,.0f}",
                    "co2_spend_kg":"{:,.0f}",
                    "co2_transport_kg":"{:,.0f}",
                    "co2_total_kg":"{:,.0f}"
                }),
                use_container_width=True
            )
        with c2:
            total_sup = float(sup["co2_total_kg"].sum())
            st.metric("Total Supplier Emissions", f"{total_sup:,.0f} kgCO₂e")

        fig1 = px.pie(
            sup.head(20),
            names="supplier", values="co2_total_kg",
            hole=0.4, title="Share by Supplier (Top 20)"
        )
        fig1.update_traces(textposition="outside", textinfo="label+percent",
                           hovertemplate="%{label}: %{percent} (%{value:,.0f} kgCO₂e)")

        fig2 = px.bar(
            sup.head(20),
            x="supplier", y="co2_total_kg", text="co2_total_kg",
            title="Totals by Supplier (Top 20)",
            labels={"co2_total_kg":"kgCO₂e","supplier":"Supplier"}
        )
        fig2.update_traces(texttemplate='%{text:,.0f}', textposition="outside")
        fig2.update_layout(xaxis_tickangle=-20, yaxis_title="kgCO₂e")
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig1, use_container_width=True)
        c2.plotly_chart(fig2, use_container_width=True)


# -------------------- Categories --------------------
def tab_categories(tab, df):
    with tab:
        st.subheader("Category Breakdown")
        if df is None or df.empty:
            st.info("Upload company data to view categories.")
            return

        by_cat = (
            df.groupby("category", dropna=False)[["co2_spend_kg","co2_transport_kg","co2_total_kg"]]
              .sum().reset_index().sort_values("co2_total_kg", ascending=False)
        )
        if by_cat.empty:
            st.info("No category data available.")
            return

        c1, c2 = st.columns([3,1])
        with c1:
            st.dataframe(
                by_cat.style.format({"co2_spend_kg":"{:,.0f}",
                                     "co2_transport_kg":"{:,.0f}",
                                     "co2_total_kg":"{:,.0f}"}),
                use_container_width=True
            )
        with c2:
            st.metric("Total Category Emissions", f"{float(by_cat['co2_total_kg'].sum()):,.0f} kgCO₂e")

        fig1 = px.pie(by_cat, names="category", values="co2_total_kg",
                      hole=0.4, title="Share of Emissions by Category")
        fig1.update_traces(textposition="outside", textinfo="label+percent",
                           hovertemplate="%{label}: %{percent} (%{value:,.0f} kgCO₂e)")

        fig2 = px.bar(by_cat, x="category", y="co2_total_kg", text="co2_total_kg",
                      title="Emissions by Category (Totals)",
                      labels={"co2_total_kg":"kgCO₂e","category":"Category"})
        fig2.update_traces(texttemplate='%{text:,.0f}', textposition="outside")
        fig2.update_layout(xaxis_tickangle=-20, yaxis_title="kgCO₂e")
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig1, use_container_width=True)
        c2.plotly_chart(fig2, use_container_width=True)


# -------------------- Waste --------------------
def tab_waste(tab, waste_df, waste_total):
    with tab:
        st.subheader("Waste (Cat.5)")
        if waste_df is None or waste_df.empty:
            st.info("No Waste sheet provided.")
            return
        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(
                waste_df.rename(columns={
                    "waste_type": "type (raw)", "waste_key": "mapped_key",
                    "waste_tonnes": "tonnes", "ef": "EF (kgCO2e/kg)", "co2_waste_kg": "CO2e (kg)"
                }),
                use_container_width=True
            )
        with c2:
            st.metric("Total Waste Emissions", f"{waste_total:,.0f} kgCO₂e")
            if "ef" in waste_df.columns and waste_df["ef"].eq(0).any():
                st.warning("Some rows have EF=0 (no mapping).")
        by_key = (waste_df.groupby("waste_key")["co2_waste_kg"]
                    .sum().reset_index().sort_values("co2_waste_kg", ascending=False))
        if not by_key.empty:
            fig1 = px.pie(by_key, names="waste_key", values="co2_waste_kg", hole=0.4, title="Share by Waste Type")
            fig1.update_traces(textposition="outside", textinfo="label+percent",
                               hovertemplate="%{label}: %{percent} (%{value:,.0f} kgCO₂e)")
            fig2 = px.bar(by_key, x="waste_key", y="co2_waste_kg", text="co2_waste_kg", title="Totals by Waste Type")
            fig2.update_traces(texttemplate='%{text:,.0f}', textposition="outside")
            col1, col2 = st.columns(2)
            col1.plotly_chart(fig1, use_container_width=True)
            col2.plotly_chart(fig2, use_container_width=True)


# -------------------- Travel --------------------
def tab_travel(tab, travel_df, travel_total):
    with tab:
        st.subheader("Business Travel (Cat.6)")
        if travel_df is None or travel_df.empty:
            st.info("No Business_Travel sheet provided.")
            return
        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(
                travel_df.rename(columns={
                    "travel_mode": "mode (raw)", "travel_key": "mapped_key",
                    "passenger_km": "pkm", "ef_pkm": "EF (kgCO2e/pkm)",
                    "co2_travel_pkm_kg": "CO2 (pkm, kg)", "co2_travel_hotel_kg": "CO2 (hotel, kg)",
                    "co2_travel_total_kg": "CO2 total (kg)"
                }),
                use_container_width=True
            )
        with c2:
            st.metric("Total Travel Emissions", f"{travel_total:,.0f} kgCO₂e")
            if "ef_pkm" in travel_df.columns and travel_df["ef_pkm"].eq(0).any():
                st.warning("Some rows have EF=0 for pkm (no mapping).")
        by_mode = (travel_df.groupby("travel_key")["co2_travel_total_kg"]
                    .sum().reset_index().sort_values("co2_travel_total_kg", ascending=False))
        if not by_mode.empty:
            fig1 = px.pie(by_mode, names="travel_key", values="co2_travel_total_kg", hole=0.4, title="Share by Travel Mode")
            fig1.update_traces(textposition="outside", textinfo="label+percent",
                               hovertemplate="%{label}: %{percent} (%{value:,.0f} kgCO₂e)")
            fig2 = px.bar(by_mode, x="travel_key", y="co2_travel_total_kg", text="co2_travel_total_kg", title="Totals by Travel Mode")
            fig2.update_traces(texttemplate='%{text:,.0f}', textposition="outside")
            col1, col2 = st.columns(2)
            col1.plotly_chart(fig1, use_container_width=True)
            col2.plotly_chart(fig2, use_container_width=True)


# -------------------- Commute --------------------
def tab_commute(tab, comm_df, commute_total):
    with tab:
        st.subheader("Employee Commute (Cat.7)")
        if comm_df is None or comm_df.empty:
            st.info("No Employee_Commute sheet provided.")
            return
        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(
                comm_df.rename(columns={
                    "mode": "mode (raw)", "mode_key": "mapped_key",
                    "passenger_km": "pkm", "ef_pkm": "EF (kgCO2e/pkm)",
                    "co2_commute_kg": "CO2 total (kg)"
                }),
                use_container_width=True
            )
        with c2:
            st.metric("Total Commute Emissions", f"{commute_total:,.0f} kgCO₂e")
            if "ef_pkm" in comm_df.columns and comm_df["ef_pkm"].eq(0).any():
                st.warning("Some rows have EF=0 for pkm (no mapping).")
        by_m = (comm_df.groupby("mode_key")["co2_commute_kg"]
                  .sum().reset_index().sort_values("co2_commute_kg", ascending=False))
        if not by_m.empty:
            fig1 = px.pie(by_m, names="mode_key", values="co2_commute_kg", hole=0.4, title="Share by Commute Mode")
            fig1.update_traces(textposition="outside", textinfo="label+percent",
                               hovertemplate="%{label}: %{percent} (%{value:,.0f} kgCO₂e)")
            fig2 = px.bar(by_m, x="mode_key", y="co2_commute_kg", text="co2_commute_kg", title="Totals by Commute Mode")
            fig2.update_traces(texttemplate='%{text:,.0f}', textposition="outside")
            col1, col2 = st.columns(2)
            col1.plotly_chart(fig1, use_container_width=True)
            col2.plotly_chart(fig2, use_container_width=True)


# -------------------- Energy --------------------
def tab_energy(tab, energy_df, energy_total, energy_unit_note):
    with tab:
        st.subheader("Energy (Scope 2)")
        if energy_df is None or energy_df.empty:
            st.info("No Energy sheet provided.")
            return

        c1, c2 = st.columns([2,1])
        with c1:
            st.dataframe(energy_df.rename(columns={
                "meter_type":"meter_type (raw)",
                "energy_key":"mapped_key",
                "kwh":"kWh",
                "ef_kwh":"EF (kgCO₂e/kWh)",
                "co2_energy_kg":"CO₂ total (kg)"
            }), use_container_width=True)
        with c2:
            st.metric("Total Energy Emissions", f"{energy_total:,.0f} kgCO₂e")
            region = st.session_state.get("active_region", config.REGION_GLOBAL_DEFRA)
            if region == config.REGION_ARABIA_MENA:
                st.caption("Using MENA grid intensity factors (Saudi / UAE baseline)")
            if energy_unit_note:
                st.caption("EF units: " + energy_unit_note)
            if energy_df["ef_kwh"].eq(0).any():
                st.warning("Some rows have EF=0 (no mapping).")

        by_type = energy_df.groupby("meter_type").agg(
            total_kwh=("kwh","sum"),
            total_co2=("co2_energy_kg","sum")
        ).reset_index().sort_values("total_co2", ascending=False)

        if not by_type.empty:
            fig1 = px.pie(by_type, names="meter_type", values="total_kwh", hole=0.4,
                          title="Energy Consumption by Meter Type")
            fig1.update_traces(textposition="outside", textinfo="label+percent",
                               hovertemplate="%{label}: %{value:,.0f} kWh")
            fig2 = px.bar(by_type, x="meter_type", y="total_co2", text="total_co2",
                          title="Emissions by Meter Type", labels={"total_co2":"kgCO₂e"})
            fig2.update_traces(texttemplate='%{text:,.0f}', textposition="outside")
            fig2.update_layout(xaxis_title="Meter Type", yaxis_title="kgCO₂e")
            col1, col2 = st.columns(2)
            col1.plotly_chart(fig1, use_container_width=True)
            col2.plotly_chart(fig2, use_container_width=True)


# -------------------- CBAM --------------------
def tab_cbam(container, cbam_df: pd.DataFrame, cbam_total_t: float):
    with container:
        st.subheader("CBAM — Product-level Summary")
        st.caption("For non-EU exporters shipping CBAM goods (e.g., HS 7208 steel) into the EU.")

        if cbam_df is None or cbam_df.empty:
            st.info("No CBAM data found. Add a 'CBAM' sheet with columns: hs_code, product_name, supplier, facility_country, period, quantity_t, [intensity_tco2e_per_t].")
            return

        show_cols = [c for c in [
            "hs_code", "product_name", "supplier", "facility_country", "period",
            "quantity_t", "intensity_tco2e_per_t", "embedded_tco2e"
        ] if c in cbam_df.columns]
        st.dataframe(cbam_df[show_cols].head(100), use_container_width=True)

        total_qty = float(cbam_df.get("quantity_t", 0).sum())
        total_emb = float(cbam_df.get("embedded_tco2e", 0).sum())
        avg_int   = (total_emb / total_qty) if total_qty > 0 else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Total quantity (t)", f"{total_qty:,.1f}")
        c2.metric("Embedded emissions (tCO₂e)", f"{total_emb:,.2f}")
        c3.metric("Avg. intensity (tCO₂e/t)", f"{avg_int:,.3f}")

        if "period" in cbam_df.columns:
            by_p = (cbam_df.groupby("period")["embedded_tco2e"]
                    .sum().reset_index().sort_values("embedded_tco2e", ascending=False))
            if not by_p.empty:
                fig = px.bar(by_p, x="period", y="embedded_tco2e", text="embedded_tco2e",
                             title="Embedded tCO₂e by Period", labels={"embedded_tco2e": "tCO₂e"})
                fig.update_traces(texttemplate='%{text:.2f}', textposition="outside")
                fig.update_layout(yaxis_title="tCO₂e")
                st.plotly_chart(fig, use_container_width=True)


# -------------------- Scenarios --------------------
def tab_scenarios(tab, df, energy_df, session_state):
    with tab:
        st.subheader("Multi-Scenario Compare (Transport & Energy)")
        if "scenarios" not in session_state:
            session_state["scenarios"] = []

        with st.expander("Transport Mode Shift", expanded=True):
            if df is None or df.empty:
                st.info("Upload Procurement_Transport to run transport scenarios.")
            else:
                modes = sorted(set(df["transport_mode"].unique()))
                c1, c2, c3 = st.columns([1, 1, 1])
                fm = c1.selectbox("Shift FROM mode", modes, key=KEYS.get("sc_from","sc_from_mode"))
                tm = c2.selectbox("Shift TO mode", modes, index=(modes.index("rail") if "rail" in modes else 0), key=KEYS.get("sc_to","sc_to_mode"))
                pct = c3.slider("Shift % (by t·km)", 0, 100, 30, step=5, key=KEYS.get("sc_pct","sc_shift_pct"))
                base = float(df["co2_transport_kg"].sum())
                ef_from = float(df.loc[df["transport_mode"] == fm, "ef_mode"].max() or 0)
                ef_to   = float(df.loc[df["transport_mode"] == tm, "ef_mode"].max() or 0)
                tkm_from = float(df.loc[df["transport_mode"] == fm, "tkm"].sum())
                scen = base - tkm_from * (pct / 100.0) * (ef_from - ef_to)
                saved = max(0.0, base - scen)
                st.caption(f"Baseline: {base:,.0f} kg | Scenario: {scen:,.0f} kg | Saved: {saved:,.0f} kg")
                sname = st.text_input("Scenario name", value=f"{fm}→{tm} {pct}%", key=KEYS.get("sc_name","sc_name"))
                if st.button("Save transport scenario"):
                    session_state["scenarios"].append({"name": sname, "type": "transport", "CO2_saved_t": saved / 1000.0, "€_saved": 0.0})

        with st.expander("Energy Mix (Renewables ↑)", expanded=True):
            if energy_df is None or energy_df.empty:
                st.info("Upload Energy sheet to run energy scenarios.")
            else:
                c1, c2, c3 = st.columns(3)
                renew_pct   = c1.slider("Renewable share (%)", 0, 100, 60, step=5, key=KEYS.get("en_pct","sc_en_pct"))
                renewable_ef= c2.number_input("Renewable EF (kg/kWh)", min_value=0.0, value=0.05, step=0.01, key=KEYS.get("en_ef_r","sc_en_ef_renew"))
                fossil_ef   = c3.number_input("Residual EF (kg/kWh)",  min_value=0.0, value=0.40, step=0.01, key=KEYS.get("en_ef_f","sc_en_ef_fossil"))
                elec = energy_df[energy_df["energy_key"].str.contains("electricity", case=False, na=False)]
                base_kwh = float(elec["kwh"].sum()) if not elec.empty else 0.0
                base_kg  = float(elec["co2_energy_kg"].sum()) if not elec.empty else 0.0
                mix_ef = (renew_pct/100.0) * renewable_ef + (1 - renew_pct/100.0) * fossil_ef
                scen_kg = base_kwh * mix_ef
                saved = max(0.0, base_kg - scen_kg)
                st.caption(f"Baseline electricity: {base_kg:,.0f} kg | Scenario: {scen_kg:,.0f} kg | Saved: {saved:,.0f} kg")
                sname_e = st.text_input("Scenario name (energy)", value=f"Renewables {renew_pct}%", key=KEYS.get("en_name","sc_en_name"))
                if st.button("Save energy scenario"):
                    session_state["scenarios"].append({"name": sname_e, "type": "energy", "CO2_saved_t": saved / 1000.0, "€_saved": 0.0})

        sc = session_state["scenarios"]
        if sc:
            st.markdown("#### Saved Scenarios")
            df_s = pd.DataFrame(sc).sort_values(["type", "CO2_saved_t"], ascending=[True, False]).reset_index(drop=True)
            st.dataframe(
                df_s[["name", "type", "CO2_saved_t", "€_saved"]]
                   .rename(columns={"name": "Scenario", "type": "Type", "CO2_saved_t": "CO₂ saved (t)", "€_saved": "€ saved"}),
                use_container_width=True
            )
            del_name = st.selectbox("Remove scenario", options=[""] + [s["name"] for s in sc], key=KEYS.get("sc_remove","sc_remove_select"))
            if del_name and st.button("Delete selected scenario"):
                session_state["scenarios"] = [s for s in sc if s["name"] != del_name]
                st.success("Deleted.")
        else:
            st.info("No scenarios saved yet.")


# -------------------- Mapping (NLP) --------------------
def _render_block(title: str, raw_values: List[str], engine, ef_type: str, state: dict, top_k: int):
    st.markdown(f"**{title}**")
    if not raw_values:
        st.caption("All values appear mapped.")
        return

    if "REMAP" not in state:
        state["REMAP"] = {"spend":{}, "transport":{}, "waste":{}, "travel":{}, "commute":{}, "energy":{}}

    for raw in raw_values:
        c1, c2 = st.columns([2, 2])
        with c1:
            st.text_input("Raw", value=raw, key=f"raw_{ef_type}_{raw}", disabled=True)
        with c2:
            options = []
            if engine is not None:
                options = engine.suggest(raw, top_k=top_k)
            label_options = [f"{k}   (score={score:.2f})" for k, score in options] or ["<no suggestion>"]
            sel = st.selectbox(
                "Suggest",
                options=label_options,
                key=f"sel_{ef_type}_{raw}"
            )
            if sel != "<no suggestion>":
                chosen = sel.split("   (score=")[0]
                if st.button(f"Apply → {chosen}", key=f"btn_{ef_type}_{raw}"):
                    state["REMAP"][ef_type][raw] = chosen
                    st.success(f"Mapped “{raw}” → “{chosen}”")

    st.caption("Changes apply on next run (or press the 'Apply & Recalc' button below).")


# -------------------- Mapping (semantic + summary like Anomalies) --------------------
def tab_mapping(tab, engines: dict, efs: pd.DataFrame, dataframes: dict, session_state, top_k: int = 3):
    """
    engines: dict[type] -> EmbeddingEngine (keys: spend, transport, waste, travel, commute, energy)
    dataframes: {"core": df_core, "waste": df_waste, "travel": df_travel, "commute": df_comm, "energy": df_energy}
    Updates session_state["REMAP"] with {type: {raw_value: selected_key}}
    """
    import pandas as pd
    import streamlit as st

    def _norm(s: pd.Series) -> pd.Series:
        from io_utils import arabic_normalize
        return s.astype(str).apply(arabic_normalize)

    # ---------- summary (like Anomalies) ----------
    def _summary_df():
        rows = []
        core = dataframes.get("core")
        if isinstance(core, pd.DataFrame) and not core.empty:
            um_cat = set(_norm(core.loc[(core.get("ef_spend", 0) == 0), "category"]))
            um_mode = set(_norm(core.loc[(core.get("ef_mode", 0) == 0), "transport_mode"]))
            rows.append({"Module": "core", "Rows": len(core), "Unmapped (unique)": len(um_cat) + len(um_mode)})
        else:
            rows.append({"Module": "core", "Rows": 0, "Unmapped (unique)": 0})

        w = dataframes.get("waste")
        rows.append({"Module": "waste",
                     "Rows": (0 if (w is None or w.empty) else len(w)),
                     "Unmapped (unique)": (0 if (w is None or w.empty or "waste_type" not in w)
                                           else len(set(_norm(w["waste_type"]))))})

        t = dataframes.get("travel")
        rows.append({"Module": "travel",
                     "Rows": (0 if (t is None or t.empty) else len(t)),
                     "Unmapped (unique)": (0 if (t is None or t.empty or "travel_mode" not in t)
                                           else len(set(_norm(t["travel_mode"]))))})

        c = dataframes.get("commute")
        rows.append({"Module": "commute",
                     "Rows": (0 if (c is None or c.empty) else len(c)),
                     "Unmapped (unique)": (0 if (c is None or c.empty or "mode" not in c)
                                           else len(set(_norm(c["mode"]))))})

        e = dataframes.get("energy")
        rows.append({"Module": "energy",
                     "Rows": (0 if (e is None or e.empty) else len(e)),
                     "Unmapped (unique)": (0 if (e is None or e.empty or "meter_type" not in e)
                                           else len(set(_norm(e["meter_type"]))))})
        return pd.DataFrame(rows)

    with tab:
        st.subheader("Mapping Helper — Semantic Suggestions")
        if not engines:
            st.info("Semantic engine is disabled or EF file is missing. Enable in the sidebar or upload EF.")
            return

        REMAP = session_state.get(
            "REMAP",
            {"spend": {}, "transport": {}, "waste": {}, "travel": {}, "commute": {}, "energy": {}}
        )

        st.markdown("##### Summary")
        st.dataframe(_summary_df(), use_container_width=True)
        st.markdown("---")

        # ---------- CORE: spend categories ----------
        core = dataframes.get("core")
        if isinstance(core, pd.DataFrame) and not core.empty:
            with st.expander("Core — Categories (spend)", expanded=False):
                unmapped_cats = sorted(set(_norm(core.loc[(core.get("ef_spend", 0) == 0), "category"])))
                if not unmapped_cats:
                    st.caption("✅ No unmapped categories.")
                else:
                    eng = engines.get("spend")
                    for raw in unmapped_cats[:200]:
                        cands = eng.suggest(raw, top_k=top_k) if eng else []
                        opts = [f"{k}  ·  sim={sim:.2f}" for k, sim in cands] or ["(no suggestion)"]
                        choice = st.selectbox("Suggest EF key", opts, key=f"map_spend_{raw}")
                        if choice != "(no suggestion)":
                            chosen = choice.split("·")[0].strip()
                            REMAP["spend"][raw] = chosen
                            st.caption(f"→ will map **{raw}** → **{chosen}**")

            # ---------- CORE: transport modes ----------
            with st.expander("Core — Transport modes", expanded=False):
                unmapped_modes = sorted(set(_norm(core.loc[(core.get("ef_mode", 0) == 0), "transport_mode"])))
                if not unmapped_modes:
                    st.caption("✅ No unmapped modes.")
                else:
                    eng = engines.get("transport")
                    for raw in unmapped_modes[:200]:
                        cands = eng.suggest(raw, top_k=top_k) if eng else []
                        opts = [f"{k}  ·  sim={sim:.2f}" for k, sim in cands] or ["(no suggestion)"]
                        choice = st.selectbox("Suggest EF key", opts, key=f"map_transport_{raw}")
                        if choice != "(no suggestion)":
                            chosen = choice.split("·")[0].strip()
                            REMAP["transport"][raw] = chosen
                            st.caption(f"→ will map **{raw}** → **{chosen}**")

        # ---------- WASTE ----------
        w = dataframes.get("waste")
        if isinstance(w, pd.DataFrame) and not w.empty and "waste_type" in w.columns:
            with st.expander("Waste — types", expanded=False):
                raw_vals = sorted(set(_norm(w["waste_type"])))
                eng = engines.get("waste")
                for raw in raw_vals[:200]:
                    cands = eng.suggest(raw, top_k=top_k) if eng else []
                    opts = [f"{k}  ·  sim={sim:.2f}" for k, sim in cands] or ["(no suggestion)"]
                    choice = st.selectbox("Suggest EF key", opts, key=f"map_waste_{raw}")
                    if choice != "(no suggestion)":
                        chosen = choice.split("·")[0].strip()
                        REMAP["waste"][raw] = chosen
                        st.caption(f"→ will map **{raw}** → **{chosen}**")

        # ---------- TRAVEL ----------
        t = dataframes.get("travel")
        if isinstance(t, pd.DataFrame) and not t.empty and "travel_mode" in t.columns:
            with st.expander("Travel — modes", expanded=False):
                raw_vals = sorted(set(_norm(t["travel_mode"])))
                eng = engines.get("travel")
                for raw in raw_vals[:200]:
                    cands = eng.suggest(raw, top_k=top_k) if eng else []
                    # example guardrail: hide hotel_night for passenger modes
                    cands = [(k, s) for (k, s) in cands if "hotel" not in k]
                    opts = [f"{k}  ·  sim={sim:.2f}" for k, sim in cands] or ["(no suggestion)"]
                    choice = st.selectbox("Suggest EF key", opts, key=f"map_travel_{raw}")
                    if choice != "(no suggestion)":
                        chosen = choice.split("·")[0].strip()
                        REMAP["travel"][raw] = chosen
                        st.caption(f"→ will map **{raw}** → **{chosen}**")

        # ---------- COMMUTE ----------
        c = dataframes.get("commute")
        if isinstance(c, pd.DataFrame) and not c.empty and "mode" in c.columns:
            with st.expander("Commute — modes", expanded=False):
                raw_vals = sorted(set(_norm(c["mode"])))
                eng = engines.get("commute")
                for raw in raw_vals[:200]:
                    cands = eng.suggest(raw, top_k=top_k) if eng else []
                    opts = [f"{k}  ·  sim={sim:.2f}" for k, sim in cands] or ["(no suggestion)"]
                    choice = st.selectbox("Suggest EF key", opts, key=f"map_commute_{raw}")
                    if choice != "(no suggestion)":
                        chosen = choice.split("·")[0].strip()
                        REMAP["commute"][raw] = chosen
                        st.caption(f"→ will map **{raw}** → **{chosen}**")

        # ---------- ENERGY ----------
        e = dataframes.get("energy")
        if isinstance(e, pd.DataFrame) and not e.empty and "meter_type" in e.columns:
            with st.expander("Energy — meter types", expanded=False):
                raw_vals = sorted(set(_norm(e["meter_type"])))
                eng = engines.get("energy")
                for raw in raw_vals[:200]:
                    cands = eng.suggest(raw, top_k=top_k) if eng else []
                    opts = [f"{k}  ·  sim={sim:.2f}" for k, sim in cands] or ["(no suggestion)"]
                    choice = st.selectbox("Suggest EF key", opts, key=f"map_energy_{raw}")
                    if choice != "(no suggestion)":
                        chosen = choice.split("·")[0].strip()
                        REMAP["energy"][raw] = chosen
                        st.caption(f"→ will map **{raw}** → **{chosen}**")

        st.markdown("---")
        col1, col2 = st.columns(2)
        if col1.button("Apply mappings & Re-calculate", use_container_width=True):
            session_state["REMAP"] = REMAP
            st.success("Mappings stored. Go back to Coverage / Reports to see updates.")
        if col2.button("Clear ALL mappings", type="secondary", use_container_width=True):
            session_state["REMAP"] = {"spend": {}, "transport": {}, "waste": {}, "travel": {}, "commute": {}, "energy": {}}
            st.warning("All mappings cleared.")

# -------------------- Reports --------------------
def tab_reports(tab, blocked, has_core, core_spend, core_trans, waste_df, waste_total,
                travel_df, travel_total, comm_df, commute_total, energy_df, energy_total,
                df, exports, spend_unmapped_ratio, tkm_unmapped_ratio):
    with tab:
        from exports import csrd_summary_rows, write_results_excel, write_cbam_pdf_simple

        st.subheader("Export Reports")
        st.caption(f"Unmapped spend: {spend_unmapped_ratio:.1%} · Unmapped t·km: {tkm_unmapped_ratio:.1%}")
        c1, c2, c3 = st.columns(3)

        enable_waste   = isinstance(waste_df, pd.DataFrame) and not waste_df.empty
        enable_travel  = isinstance(travel_df, pd.DataFrame) and not travel_df.empty
        enable_commute = isinstance(comm_df,  pd.DataFrame) and not comm_df.empty
        enable_energy  = isinstance(energy_df, pd.DataFrame) and not energy_df.empty

        with c1:
            region = st.session_state.get("active_region", config.REGION_GLOBAL_DEFRA)
            if region == config.REGION_ARABIA_MENA:
               st.markdown("**EcoScope-Arabia Export (SDG 12 & 13)**")
            else:
               st.markdown("**CSRD Export**")
            rows = csrd_summary_rows(
                       has_core, core_spend, core_trans, waste_total, travel_total, commute_total, energy_total,
                       enable_waste, enable_travel, enable_commute, enable_energy,
                       region=st.session_state.get("active_region")
                    )
            if blocked:
                st.warning("Exports blocked until coverage is ≥90%.")
            else:
                if st.button("Export CSRD Excel"):
                    by_cat = (df.groupby("category")[["co2_spend_kg", "co2_transport_kg", "co2_total_kg"]]
                                .sum().sort_values("co2_total_kg", ascending=False).reset_index()) if has_core else pd.DataFrame()
                    top_sup = (df.groupby("supplier")["co2_total_kg"]
                                .sum().sort_values(ascending=False).reset_index()) if has_core else pd.DataFrame()
                    path = write_results_excel(
                        "EcoScopeAI_results.xlsx",
                        df if has_core else pd.DataFrame(), by_cat, top_sup,
                        waste_df, travel_df, comm_df, energy_df, rows
                    )
                    with open(path, "rb") as f:
                        st.download_button("Download EcoScopeAI_results.xlsx", f, file_name=path,
                                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        with c2:
            st.markdown("**CBAM Export**")
            if exports.get("cbam_rows"):
                if blocked:
                    st.warning("Exports blocked until coverage is ≥90%.")
                else:
                    if st.button("Export CBAM PDF"):
                        out = write_cbam_pdf_simple(
                                  exports["cbam_rows"],
                                  filename="CBAM_Report.pdf",
                                  region=st.session_state.get("active_region")
                              )
                        with open(out, "rb") as f:
                            st.download_button(f"Download {out}", f, file_name=out,
                                               mime=("application/pdf" if out.endswith(".pdf") else "text/csv"))
            else:
                st.info("No CBAM data detected.")

        with c3:
            st.markdown("**Notes**")
            st.caption("• CSRD export includes CSRD_summary + detail sheets.\n• CBAM PDF shows HS Code, period, quantity, intensity, embedded tCO₂e.")
# --- Anomalies ---
def tab_anomalies(tab, anomalies_dict, default_topn=50):
    import pandas as pd
    import plotly.express as px

    with tab:
        st.subheader("🚨 Anomalies (IsolationForest)")
        if not anomalies_dict:
            st.info("No data available for anomaly detection yet.")
            return

        st.caption("Rows flagged as statistical outliers on numeric features per module. Use the contamination slider in the sidebar (or app) to tune sensitivity.")

        # Summary
        summary = []
        for name, df in anomalies_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty and "is_anomaly" in df.columns:
                summary.append({
                    "Module": name,
                    "Rows": len(df),
                    "Anomalies": int(df["is_anomaly"].sum())
                })
        if summary:
            st.dataframe(pd.DataFrame(summary), use_container_width=True)

        # Detail per module
        for name, df in anomalies_dict.items():
            if not isinstance(df, pd.DataFrame) or df.empty or "is_anomaly" not in df.columns:
                continue

            with st.expander(f"{name.capitalize()} — {int(df['is_anomaly'].sum())} anomalies", expanded=False):
                # show only anomalies sorted by anomaly_score desc
                ana = df.loc[df["is_anomaly"]].copy()
                if ana.empty:
                    st.write("No anomalies in this module.")
                    continue
                ana = ana.sort_values("anomaly_score", ascending=False)

                # Keep useful columns near front if present
                preferred = [c for c in ["supplier","category","transport_mode","travel_mode","mode",
                                         "meter_type","region","hs_code"] if c in ana.columns]
                numeric_cols = [c for c in ana.columns if c.startswith(("co2_","cost_","kwh","tkm","distance","weight","passenger_km","waste_tonnes"))]
                show_cols = list(dict.fromkeys(preferred + numeric_cols + ["anomaly_score"]))
                st.dataframe(ana[show_cols].head(default_topn), use_container_width=True)

                # Optional quick chart (top 20 by anomaly score)
                try:
                    top20 = ana.head(20)
                    fig = px.bar(top20.reset_index().rename(columns={"index":"row"}),
                                 x="row", y="anomaly_score", title=f"Top anomalies — {name}")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass
# -------------------- ESG Intelligence --------------------
def tab_esg_intel(container, session_state):
    import streamlit as st
    import pandas as pd
    from io_utils import (
        extract_text_from_pdf, split_into_chunks,
        load_sentiment_pipeline, load_zeroshot_pipeline,
        analyze_chunks_batched, aggregate_supplier_esg,
        filelike_to_bytes, file_bytes_sha1
    )

    with container:
        st.subheader("ESG Intelligence (Supplier sentiment & topics)")

        # Controls
        colA, colB, colC, colD = st.columns(4)
        fast_mode = colA.toggle("Fast mode", value=True, help="Rule-based topics (no zero-shot). Much faster.")
        max_chunks = colB.number_input("Max chunks / doc", 5, 200, 40, step=5)
        max_chars = colC.number_input("Chunk size (chars)", 300, 2000, 900, step=100)
        batch_size = colD.number_input("Batch size", 4, 64, 16, step=4)

        # Cache models once
        @st.cache_resource(show_spinner=False)
        def _load_models(fast: bool):
            sent = load_sentiment_pipeline("ProsusAI/finbert")
            zshot = None if fast else load_zeroshot_pipeline("valhalla/distilbart-mnli-12-1")
            return sent, zshot

        sent_pipe, zshot_pipe = _load_models(fast_mode)

        # Inputs
        up_files = st.file_uploader("Upload supplier ESG PDFs", type=["pdf"], accept_multiple_files=True)
        supplier_name = st.text_input("Supplier (optional; else file name is used)", "")
        pasted_text = st.text_area("…or paste ESG text", height=120)

        # Cache for analyzed docs (per file hash + settings)
        if "ESG_CACHE" not in session_state:
            session_state["ESG_CACHE"] = {}  # key -> list[ChunkResult]

        def cache_key_for_bytes(b: bytes) -> str:
            return f"{file_bytes_sha1(b)}|fast={fast_mode}|mc={max_chunks}|sz={max_chars}"

        run = st.button("Analyze ESG")
        if not run:
            st.caption("Tip: enable Fast mode for quick results. Models and results are cached.")
            return

        all_results = []

        # Pasted text
        if pasted_text.strip():
            chunks = split_into_chunks(pasted_text, max_chars=max_chars)
            chunks = chunks[:max_chunks]
            supp = supplier_name.strip() or "Pasted_Text"
            res = analyze_chunks_batched(supp, chunks, sent_pipe, zshot_pipe, fast_topics=fast_mode, batch_size=batch_size)
            all_results.extend(res)

        # PDFs
        if up_files:
            for f in up_files:
                b = filelike_to_bytes(f)
                ck = cache_key_for_bytes(b)
                if ck in session_state["ESG_CACHE"]:
                    res = session_state["ESG_CACHE"][ck]
                else:
                    text = extract_text_from_pdf(io.BytesIO(b))
                    chunks = split_into_chunks(text, max_chars=max_chars)[:max_chunks]
                    supp = supplier_name.strip() or f.name.rsplit(".",1)[0]
                    res = analyze_chunks_batched(supp, chunks, sent_pipe, zshot_pipe, fast_topics=fast_mode, batch_size=batch_size)
                    session_state["ESG_CACHE"][ck] = res  # memoize
                all_results.extend(res)

        if not all_results:
            st.info("No content found to analyze.")
            return

        # Aggregate
        agg = aggregate_supplier_esg(all_results)
        st.markdown("### Supplier ESG scores")
        st.dataframe(agg, use_container_width=True)

        # Red flags
        st.markdown("### Most negative snippets")
        df_chunks = pd.DataFrame([r.__dict__ for r in all_results])
        df_chunks["neg_rank"] = df_chunks.groupby("supplier")["sent_neg"].rank(ascending=False, method="first")
        topN = st.slider("Show top-N negative chunks per supplier", 1, 20, 5)
        red_flags = df_chunks[df_chunks["neg_rank"] <= topN].sort_values(["supplier","sent_neg"], ascending=[True,False])
        st.dataframe(red_flags[["supplier","chunk_id","sent_pos","sent_neu","sent_neg","topic_env","topic_soc","topic_gov","text"]],
                     use_container_width=True)

        session_state["ESG_CHUNKS"] = df_chunks
        session_state["ESG_SUPPLIERS"] = agg
  # --- CLUSTERS: Supplier / Product CO₂ intensity ---
def tab_clusters_ui(tab, df_core: pd.DataFrame):
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import streamlit as st

    def _build_agg(df: pd.DataFrame, by: str) -> pd.DataFrame:
        """Aggregate core rows to a chosen level and compute intensity features."""
        g = (df.groupby(by, dropna=False)
               .agg(spend=("spend_base", "sum"),
                    co2=("co2_total_kg", "sum"),
                    co2_spend=("co2_spend_kg", "sum"),
                    co2_transport=("co2_transport_kg","sum"),
                    tkm=("tkm", "sum"),
                    rows=("co2_total_kg","size"))
               .reset_index()
             )
        # Intensities (add eps to avoid division by zero)
        eps = 1e-9
        g["intensity_per_eur"] = g["co2"] / (g["spend"] + eps)
        g["intensity_per_tkm"] = g["co2_transport"] / (g["tkm"] + eps)
        # % transport share
        g["transport_share"] = (g["co2_transport"] / (g["co2"] + eps)).clip(0,1)
        return g

    @st.cache_data(show_spinner=False)
    def _run_kmeans(data: pd.DataFrame, n_clusters: int, feature_cols: tuple, random_state: int = 42):
        X = data[list(feature_cols)].fillna(0.0).to_numpy()
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
        labels = km.fit_predict(Xs)
        centers = km.cluster_centers_  # scaled space
        return labels, centers

    with tab:
        st.subheader("Clusters — CO₂ intensity groups")

        if df_core is None or df_core.empty:
            st.info("Load core data first (Procurement_Transport) so we can cluster suppliers/products.")
            return

        # Controls
        c1, c2, c3, c4 = st.columns([1.2,1.2,2,1.5])
        level = c1.selectbox("Group by", ["supplier", "category"], index=0,
                             help="Choose the entity to cluster. You can extend to product later.")
        k = c2.slider("Number of clusters (KMeans)", 2, 6, 3, help="Try 3 first (low/med/high).")
        feat_choice = c3.multiselect(
            "Features",
            ["co2", "spend", "intensity_per_eur", "intensity_per_tkm", "transport_share"],
            default=["co2", "spend", "intensity_per_eur"],
            help="Clustering is done on these standardized features."
        )
        min_rows = c4.number_input("Min. rows per entity", min_value=1, value=1, step=1,
                                   help="Filter out entities with too few rows to reduce noise.")

        # Aggregate
        agg = _build_agg(df_core, level)
        agg = agg[agg["rows"] >= min_rows].reset_index(drop=True)

        if len(agg) < k:
            st.warning(f"Not enough {level}s ({len(agg)}) for k={k}. Reduce K or load more data.")
            st.dataframe(agg, use_container_width=True)
            return

        if not feat_choice:
            st.warning("Pick at least one feature.")
            return

        # Fit KMeans (cached by data+params)
        labels, centers = _run_kmeans(agg, k, tuple(feat_choice))
        agg["cluster"] = labels

        # Friendly cluster names by mean intensity (0=lowest)
        cluster_order = (agg.groupby("cluster")["intensity_per_eur"]
                           .mean().sort_values().index.tolist())
        name_map = {cl: f"{i+1} — {'Low' if i==0 else ('High' if i==k-1 else 'Mid')} CO₂"
                    for i, cl in enumerate(cluster_order)}
        agg["cluster_name"] = agg["cluster"].map(name_map)

        # KPI row
        c5, c6, c7 = st.columns(3)
        c5.metric("Entities clustered", f"{len(agg):,}")
        c6.metric("Clusters", k)
        avg_int = float(agg["intensity_per_eur"].mean()) if len(agg) else 0.0
        c7.metric("Avg. intensity (kg/€)", f"{avg_int:,.4f}")

        # Table
        show_cols = [level, "rows", "spend", "co2", "co2_spend", "co2_transport",
                     "intensity_per_eur", "intensity_per_tkm", "transport_share", "cluster_name"]
        st.dataframe(
            agg[show_cols].sort_values(["cluster_name","co2"], ascending=[True, False])
              .style.format({"spend":"{:,.0f}", "co2":"{:,.0f}",
                             "co2_spend":"{:,.0f}","co2_transport":"{:,.0f}",
                             "intensity_per_eur":"{:.5f}", "intensity_per_tkm":"{:.5f}",
                             "transport_share":"{:.2f}"}),
            use_container_width=True
        )

        # Scatter (spend vs CO2)
        fig = px.scatter(
            agg, x="spend", y="co2", size="rows", color="cluster_name",
            hover_data=[level, "intensity_per_eur", "transport_share"],
            title=f"{level.title()} — Spend vs CO₂ (clustered)"
        )
        fig.update_layout(legend_title="Cluster")
        st.plotly_chart(fig, use_container_width=True)

        # Optional: second plot intensity vs spend
        fig2 = px.scatter(
            agg, x="spend", y="intensity_per_eur", size="rows", color="cluster_name",
            hover_data=[level, "co2", "transport_share"],
            title=f"{level.title()} — Intensity (kg/€) vs Spend"
        )
        fig2.update_layout(legend_title="Cluster")
        st.plotly_chart(fig2, use_container_width=True)

        # Download
        csv = agg.to_csv(index=False).encode("utf-8")
        st.download_button("Download clusters (CSV)", csv, file_name=f"{level}_clusters.csv", mime="text/csv")

        # Hints for actionability
        with st.expander("How to use these clusters"):
            st.markdown(
                "- **High CO₂ cluster**: target for supplier engagement, alt-materials, mode shift.\n"
                "- **Mid cluster**: quick wins (contract renewals, routing, energy mix).\n"
                "- **Low cluster**: maintain, consider for preferred supplier lists."
            )
  # --- Forecast: Energy / CO₂ ---
def tab_forecast_ui(tab, energy_df: pd.DataFrame):
    import pandas as pd
    import streamlit as st
    import plotly.express as px
    from prophet import Prophet

    with tab:
        st.subheader("🔮 Forecast Energy & CO₂")

        if energy_df is None or energy_df.empty:
            st.info("Load Energy data first.")
            return

        # Ensure date column
        if "date" not in energy_df.columns:
            st.error("Energy data needs a 'date' column (YYYY-MM or YYYY-MM-DD).")
            return

        # Aggregate monthly
        df_m = (energy_df.groupby(pd.to_datetime(energy_df["date"]).dt.to_period("M"))
                           .agg(kwh=("kwh","sum"), co2=("co2_energy_kg","sum"))
                           .reset_index())
        df_m["date"] = df_m["date"].dt.to_timestamp()

        target = st.radio("Forecast target", ["kwh","co2"], index=1, horizontal=True)

        # Prophet expects columns ds (date) and y (target)
        df_p = df_m.rename(columns={"date":"ds", target:"y"})

        # Train-test split
        horizon = st.slider("Months to forecast", 3, 24, 12)
        m = Prophet()
        m.fit(df_p)
        future = m.make_future_dataframe(periods=horizon, freq="M")
        forecast = m.predict(future)

        # Plot
        fig = px.line(forecast, x="ds", y="yhat", title=f"Forecast: {target.upper()}")
        fig.add_scatter(x=df_p["ds"], y=df_p["y"], mode="lines+markers", name="Actual")
        st.plotly_chart(fig, use_container_width=True)

        # Show forecast table
        st.dataframe(forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(horizon))

        # Download
        csv = forecast.to_csv(index=False).encode("utf-8")
        st.download_button("Download forecast (CSV)", csv, "forecast.csv", "text/csv")
