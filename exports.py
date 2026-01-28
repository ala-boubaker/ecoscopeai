import pandas as pd
from datetime import datetime

def csrd_summary_rows(has_core, core_spend, core_trans, waste_tot, travel_tot, commute_tot, energy_tot,
                      enable_waste, enable_travel, enable_commute, enable_energy,
                      region="GLOBAL_DEFRA"):
    """
    Region-aware row generator.
    - DEFRA/Global → CSRD categories
    - MENA/Arabia → SDG12/SDG13 simplified categories
    """
    rows = []
    if region == "MENA_ARABIA":
        # --- MENA SDG format ---
        if has_core:
            rows.append(("SDG 12 – Responsible Consumption (Purchased goods/services)", core_spend))
            rows.append(("SDG 13 – Transport (t·km)", core_trans))
        if enable_waste and waste_tot > 0:
            rows.append(("SDG 12 – Waste Management", waste_tot))
        if enable_travel and travel_tot > 0:
            rows.append(("SDG 13 – Business Travel", travel_tot))
        if enable_commute and commute_tot > 0:
            rows.append(("SDG 13 – Employee Commute", commute_tot))
        if enable_energy and energy_tot > 0:
            rows.append(("SDG 13 – Electricity & Energy", energy_tot))
        return rows
    # ---------------------------------------------------
    # DEFAULT: CSRD (Global/DEFRA)
    # ---------------------------------------------------
    if has_core:
        rows.append(("Scope 3 – Purchased Goods & Services (Cat.1)", core_spend))
        rows.append(("Scope 3 – Transport & Distribution (Cat.4/9)", core_trans))

    if enable_waste and waste_tot > 0:
        rows.append(("Scope 3 – Waste (Cat.5)", waste_tot))
    if enable_travel and travel_tot > 0:
        rows.append(("Scope 3 – Business Travel (Cat.6)", travel_tot))
    if enable_commute and commute_tot > 0:
        rows.append(("Scope 3 – Employee Commute (Cat.7)", commute_tot))
    if enable_energy and energy_tot > 0:
        rows.append(("Scope 2 – Energy", energy_tot))
    return rows

def write_results_excel(path, df, by_cat, top_sup, waste_df, travel_df, comm_df, energy_df, csrd_rows, region="GLOBAL_DEFRA"):
    """
    Region-aware Excel export (CSRD for Global, SDG12/13 for MENA).
    """
    summary_name = "CSRD_summary" if region == "GLOBAL_DEFRA" else "SDG_summary"
    with pd.ExcelWriter(path, engine="xlsxwriter") as w:
        # Summary sheet
        if csrd_rows:
            pd.DataFrame([{"Category":k, "kgCO2e":v} for k,v in csrd_rows]).to_excel(
                w, index=False, sheet_name=summary_name
            )
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_excel(w, index=False, sheet_name="Line_items")
        if isinstance(by_cat, pd.DataFrame) and not by_cat.empty:
            by_cat.to_excel(w, index=False, sheet_name="By_category")
        if isinstance(top_sup, pd.DataFrame) and not top_sup.empty:
            top_sup.to_excel(w, index=False, sheet_name="Top_suppliers")
        if isinstance(waste_df, pd.DataFrame) and not waste_df.empty:
            waste_df.to_excel(w, index=False, sheet_name="Waste_calc")
        if isinstance(travel_df, pd.DataFrame) and not travel_df.empty:
            travel_df.to_excel(w, index=False, sheet_name="Travel_calc")
        if isinstance(comm_df, pd.DataFrame) and not comm_df.empty:
            comm_df.to_excel(w, index=False, sheet_name="Commute_calc")
        if isinstance(energy_df, pd.DataFrame) and not energy_df.empty:
            energy_df.to_excel(w, index=False, sheet_name="Energy_calc")
    return path

def write_cbam_pdf_simple(rows, filename="CBAM_Report.pdf", region="GLOBAL_DEFRA"):
    """Very simple PDF using reportlab; falls back to CSV if reportlab missing."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
    except Exception:
        # Fallback: CSV
        pd.DataFrame(
            rows,
            columns=["HS Code","Product","Period","Quantity (t)","Intensity (tCO2e/t)","Embedded (tCO2e)"]
        ).to_csv(filename.replace(".pdf",".csv"), index=False)
        return filename.replace(".pdf",".csv")

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(filename, pagesize=A4)

    # --- Region-aware title ---
    if region == "MENA_ARABIA":
        title_text = "<b>EcoScopeAI-Arabia — CBAM تقرير انبعاثات الكربون</b>"
    else:
        # default: global / DEFRA profile
        title_text = "<b>EcoScopeAI — CBAM Report (Demo)</b>"

    elements = [
        Paragraph(title_text, styles["Title"]),
        Spacer(1, 12)
    ]

    # Table body
    data = [["HS Code","Product","Period","Quantity (t)","Intensity (tCO2e/t)","Embedded (tCO2e)"]]
    total = 0.0
    for r in rows:
        data.append([
            str(r[0]),
            str(r[1]),
            str(r[2]),
            f"{r[3]:,.2f}",
            f"{r[4]:.3f}",
            f"{r[5]:,.2f}"
        ])
        total += float(r[5])

    data.append(["TOTAL","","","", "", f"{total:,.2f}"])

    tbl = Table(data, colWidths=[70,150,80,80,90,80])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('GRID',(0,0),(-1,-1),1,colors.black),
        ('BOTTOMPADDING',(0,0),(-1,0),10),
    ]))

    elements += [
        tbl,
        Spacer(1,12),
        Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"])
    ]

    doc.build(elements)
    return filename
