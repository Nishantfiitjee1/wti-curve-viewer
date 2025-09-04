import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta

# ---------------------------- Page config and CSS ----------------------------
st.set_page_config(page_title="Futures Curve Viewer", layout="wide")
st.markdown("""
<style>
/* Dropdown width */
div[data-baseweb="select"] > div {
    width: 200px !important;
    font-size: 14px !important;
}
/* Date picker width */
div[data-baseweb="datepicker"] > div {
    width: 150px !important;
    font-size: 14px !important;
}
/* Buttons */
div.stButton > button {
    width: 100px;
    height: 30px;
    font-size: 13px;
}
/* Custom styling for placeholder text */
.placeholder-text {
    font-size: 1.5rem;
    font-weight: bold;
    color: #888;
    text-align: center;
    margin-top: 5rem;
    border: 2px dashed #ddd;
    padding: 2rem;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------- 1. CENTRAL FILE & PRODUCT CONFIGURATION ----------------------------
MASTER_EXCEL_FILE = "Futures_Data.xlsx"
NEWS_EXCEL_FILE = "Important_news_date.xlsx"

PRODUCT_CONFIG = {
    "CL": {"name": "WTI Crude Oil", "sheet": "WTI_Outright"},
    "BZ": {"name": "Brent Crude Oil", "sheet": "Brent_outright"},
    "DBI": {"name": "Dubai Crude Oil", "sheet": "Dubai_Outright"},
    "MRBN": {"name": "Murban Crude Oil", "sheet": "MURBAN_Outright"},
}


# ---------------------------- Data Loading & Utilities ----------------------------
@st.cache_data(show_spinner="Loading product data...", ttl=3600)
def load_product_data(file_path, sheet_name):
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None, engine="openpyxl")
    try:
        header_row_index = df_raw[df_raw[0] == 'Dates'].index[0] - 1
        data_start_row_index = header_row_index + 2
    except IndexError:
        st.error(f"Could not find the 'Dates' keyword in the first column of the '{sheet_name}' sheet.")
        st.stop()

    contracts = [str(x).strip() for x in df_raw.iloc[header_row_index].tolist()[1:] if pd.notna(x) and str(x).strip() != ""]
    col_names = ["Date"] + contracts
    
    df = df_raw.iloc[data_start_row_index:].copy()
    df.columns = col_names
    
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in contracts:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df, contracts

@st.cache_data(show_spinner="Loading news data...", ttl=3600)
def load_news_data(file_path):
    df_news = pd.read_excel(file_path, engine="openpyxl")
    date_col = None
    if 'Date' in df_news.columns:
        date_col = 'Date'
    elif 'Dates' in df_news.columns:
        date_col = 'Dates'
    if date_col:
        df_news.rename(columns={date_col: 'Date'}, inplace=True)
        df_news["Date"] = pd.to_datetime(df_news["Date"], errors="coerce")
        df_news = df_news.dropna(subset=["Date"])
        return df_news
    else:
        st.warning("The news file must contain a 'Date' or 'Dates' column.")
        return pd.DataFrame()

def curve_for_date(df: pd.DataFrame, contracts, d: date) -> pd.Series | None:
    row = df.loc[df["Date"].dt.date == d, contracts]
    return row.iloc[0] if not row.empty else None

def overlay_figure(contracts, curves: dict, y_label="Last Price ($)", title="Futures Curve") -> go.Figure:
    fig = go.Figure()
    for label, s in curves.items():
        fig.add_trace(go.Scatter(x=contracts, y=s.values, mode="lines+markers", name=str(label)))
    fig.update_layout(title=title, xaxis_title="Contract", yaxis_title=y_label,
                      hovermode="x unified", template="plotly_white",
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig

def filter_dates(df, selected_range):
    max_date = df["Date"].max()
    range_map = {
        "Last 1 Week": timedelta(weeks=1), "Last 2 Weeks": timedelta(weeks=2),
        "Last 1 Month": timedelta(days=30), "Last 6 Months": timedelta(days=180),
        "Last 1 Year": timedelta(days=365),
    }
    if selected_range == "Full History": return df
    min_date = max_date - range_map.get(selected_range, timedelta(0))
    return df[df["Date"] >= min_date]


# ---------------------------- Sidebar & Product Selection ----------------------------
st.sidebar.title("Global Controls")

# ✅ Replace dropdown with checkboxes
selected_symbols = []
for sym in PRODUCT_CONFIG.keys():
    checked = st.sidebar.checkbox(sym, value=(sym == "CL"))  # Default only CL selected
    if checked:
        selected_symbols.append(sym)

if not selected_symbols:
    st.warning("Please select at least one product to display.")
    st.stop()


# ---------------------------- Main App Logic ----------------------------
if not os.path.exists(MASTER_EXCEL_FILE):
    st.error(f"Master data file not found: `{MASTER_EXCEL_FILE}`.")
    st.stop()

if os.path.exists(NEWS_EXCEL_FILE):
    df_news = load_news_data(NEWS_EXCEL_FILE)
else:
    st.sidebar.warning(f"News file (`{NEWS_EXCEL_FILE}`) not found.")
    df_news = pd.DataFrame()

st.caption("Analysis of futures curves, spreads, and historical evolution.")

for selected_symbol in selected_symbols:
    selected_product_info = PRODUCT_CONFIG[selected_symbol]
    target_sheet_name_from_config = selected_product_info["sheet"]

    st.subheader(f"{selected_product_info['name']} ({selected_symbol})")

    try:
        excel_file_handler = pd.ExcelFile(MASTER_EXCEL_FILE)
        excel_sheets = excel_file_handler.sheet_names
        cleaned_target_sheet = target_sheet_name_from_config.strip().lower()
        actual_sheet_to_load = next((s for s in excel_sheets if s.strip().lower() == cleaned_target_sheet), None)
        
        if actual_sheet_to_load is None:
            st.markdown(f'<div class="placeholder-text">Sheet `{target_sheet_name_from_config}` not found for {selected_symbol}.</div>', unsafe_allow_html=True)
            continue
        
        df, contracts = load_product_data(MASTER_EXCEL_FILE, actual_sheet_to_load)

    except Exception as e:
        st.error(f"Could not read the data for {selected_symbol}. Error: {e}")
        continue

    all_dates = sorted(df["Date"].dt.date.unique().tolist(), reverse=True)
    max_d, min_d = all_dates[0], all_dates[-1]

    # Sidebar controls per product
    single_date = st.sidebar.date_input(f"{selected_symbol} - Single Date", value=max_d, min_value=min_d, max_value=max_d, key=f"date_input_{selected_symbol}")
    multi_dates = st.sidebar.multiselect(f"{selected_symbol} - Multi-Date Overlay", options=all_dates, default=[all_dates[0]], key=f"multiselect_{selected_symbol}")
    normalize = st.sidebar.checkbox(f"{selected_symbol} - Normalize", key=f"normalize_{selected_symbol}")

    work_df = df.copy()
    if normalize:
        vals = work_df[contracts].astype(float)
        work_df[contracts] = (vals - vals.mean(axis=1).values[:, None]) / vals.std(axis=1).values[:, None]

    # ✅ 3 charts in one row
    c1, c2, c3 = st.columns(3)

    # ---------------- OUTRIGHT ----------------
    with c1:
        s1 = curve_for_date(work_df, contracts, single_date)
        if s1 is not None:
            fig_outright = overlay_figure(contracts, {single_date: s1}, y_label=("Z-score" if normalize else "Last Price ($)"), title="Outright Curve")
            st.plotly_chart(fig_outright, use_container_width=True, key=f"outright_{selected_symbol}")

    # ---------------- SPREAD ----------------
    with c2:
        SPREAD_SHEET_MAP = {"CL": "Spread_CL", "BZ": "Spread_Brent", "DBI": "Spread_DBI", "MRBN": "Spread_MRBN"}
        target_spread_sheet = SPREAD_SHEET_MAP.get(selected_symbol)
        if target_spread_sheet:
            try:
                df_spreads, spread_contracts = load_product_data(MASTER_EXCEL_FILE, target_spread_sheet)
                valid_spread_curves = {}
                for d in multi_dates:
                    s = curve_for_date(df_spreads, spread_contracts, d)
                    if s is not None:
                        valid_spread_curves[d] = s
                if valid_spread_curves:
                    fig_spread_overlay = overlay_figure(spread_contracts, valid_spread_curves, y_label="Spread ($)", title="Spread Curve")
                    st.plotly_chart(fig_spread_overlay, use_container_width=True, key=f"spread_overlay_{selected_symbol}")
            except:
                st.info(f"No spread sheet for {selected_symbol}")
        else:
            st.info(f"Spread not configured for {selected_symbol}")

    # ---------------- FLY ----------------
    with c3:
        if len(contracts) >= 3:
            f1, f2, f3 = contracts[0], contracts[1], contracts[2]
            fly_values = work_df[f1] - 2 * work_df[f2] + work_df[f3]
            fig_fly = go.Figure()
            fig_fly.add_trace(go.Scatter(x=work_df["Date"], y=fly_values, mode="lines", name=f"{f1}-{f2}-{f3} Fly"))
            fig_fly.update_layout(title="Fly Curve", xaxis_title="Date", yaxis_title="Fly Value ($)", template="plotly_white")
            st.plotly_chart(fig_fly, use_container_width=True, key=f"fly_{selected_symbol}")
        else:
            st.info("Not enough contracts for fly analysis.")

with st.expander("Preview Raw Data"):
    st.dataframe(df.head(25))
