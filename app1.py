import io
import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta

# ---------------------------- Page config and CSS ----------------------------
st.set_page_config(page_title="Futures Curve Viewer", layout="wide")
st.markdown("""
<style>
/* Checkbox styling in sidebar */
.stCheckbox {
    font-size: 16px;
}
/* Date picker width */
div[data-baseweb="datepicker"] > div {
    width: 150px !important;
    font-size: 14px !important;
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

# This configuration uses your exact sheet names.
PRODUCT_CONFIG = {
    "CL": {"name": "WTI Crude Oil", "sheet": "WTI_Outright"},
    "BZ": {"name": "Brent Crude Oil", "sheet": "Brent_outright"},
    "DBI": {"name": "Dubai Crude Oil", "sheet": "Dubai_Outright"},
    "MRBN": {"name": "Murban Crude Oil", "sheet": "MURBAN_Outright"},
}

# ---------------------------- Data Loading & Utilities ----------------------------
@st.cache_data(show_spinner="Loading product data...", ttl=3600)
def load_product_data(file_path, sheet_name):
    """
    Loads and parses futures data by intelligently finding the header and data rows.
    """
    try:
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None, engine="openpyxl")
    except FileNotFoundError:
        st.error(f"The master data file '{file_path}' was not found.")
        st.stop()
    except Exception as e:
        # Check if the error is about the sheet not being found
        if "No sheet named" in str(e):
             # This is not a fatal error, just return None so the app can handle it gracefully
            return None, None
        st.error(f"An error occurred while reading the Excel file: {e}")
        st.stop()
        
    try:
        header_row_index = df_raw[df_raw[0] == 'Dates'].index[0] - 1
        data_start_row_index = header_row_index + 2
    except IndexError:
        # This is not a fatal error for spread/fly sheets, return None
        return None, None

    contracts = [str(x).strip() for x in df_raw.iloc[header_row_index].tolist()[1:] if pd.notna(x) and str(x).strip() != ""]
    df = df_raw.iloc[data_start_row_index:].copy()
    df.columns = ["Date"] + contracts
    
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in contracts:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df, contracts

@st.cache_data(show_spinner="Loading news data...", ttl=3600)
def load_news_data(file_path):
    """Loads news data and handles 'Date' vs 'Dates' column names."""
    try:
        df_news = pd.read_excel(file_path, engine="openpyxl")
        date_col = 'Date' if 'Date' in df_news.columns else 'Dates' if 'Dates' in df_news.columns else None
        if date_col:
            df_news.rename(columns={date_col: 'Date'}, inplace=True)
            df_news["Date"] = pd.to_datetime(df_news["Date"], errors="coerce")
            return df_news.dropna(subset=["Date"])
        st.warning("News file must contain a 'Date' or 'Dates' column.")
        return pd.DataFrame()
    except FileNotFoundError:
        return pd.DataFrame()


def curve_for_date(df: pd.DataFrame, contracts, d: date) -> pd.Series | None:
    row = df.loc[df["Date"].dt.date == d, contracts]
    return row.iloc[0] if not row.empty else None

def overlay_figure(contracts, curves: dict, y_label="Last Price ($)", title="Futures Curve") -> go.Figure:
    fig = go.Figure()
    for label, s in curves.items():
        fig.add_trace(go.Scatter(x=contracts, y=s.values, mode="lines+markers", name=str(label)))
    fig.update_layout(title=title, xaxis_title="Contract", yaxis_title=y_label, hovermode="x unified",
                          template="plotly_white", margin=dict(l=40, r=20, t=60, b=40))
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

st.sidebar.header("Product Selection")
selected_symbols = []
for i, (symbol, info) in enumerate(PRODUCT_CONFIG.items()):
    if st.sidebar.checkbox(info['name'], value=(i == 0), key=f"cb_{symbol}"):
        selected_symbols.append(symbol)

if not selected_symbols:
    st.sidebar.warning("Please select at least one product.")
    st.stop()

primary_symbol = selected_symbols[0]
primary_product_info = PRODUCT_CONFIG[primary_symbol]

# ---------------------------- Main App Logic ----------------------------
st.title(f"{primary_product_info['name']} Curve Viewer")

df, contracts = load_product_data(MASTER_EXCEL_FILE, primary_product_info["sheet"])
if df is None:
    st.error(f"Could not load data for {primary_product_info['name']}. Please check the sheet '{primary_product_info['sheet']}' in '{MASTER_EXCEL_FILE}'.")
    st.stop()

df_news = load_news_data(NEWS_EXCEL_FILE)

st.sidebar.header("Date Selection")
all_dates = sorted(df["Date"].dt.date.unique().tolist(), reverse=True)
max_d, min_d = all_dates[0], all_dates[-1]
multi_dates = st.sidebar.multiselect("Select Dates for Overlay", options=all_dates, default=[all_dates[0], all_dates[min(1, len(all_dates)-1)]], key="multiselect_dates")

st.sidebar.header("Display Options")
normalize = st.sidebar.checkbox("Normalize curves (z-score)", key="normalize_curves")

work_df = df.copy()
if normalize:
    vals = work_df[contracts].astype(float)
    work_df[contracts] = (vals - vals.mean(axis=1).values[:, None]) / vals.std(axis=1).values[:, None]

st.caption("Analysis of futures curves, spreads, and historical evolution.")
tab1, tab2 = st.tabs(["Curve Overlays", "Historical Analysis"])

# ========================================================================================
# TAB 1: CURVE OVERLAYS
# ========================================================================================
with tab1:
    st.header(f"Multi-Date Analysis for {primary_product_info['name']} ({primary_symbol})")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### Outright Curve Overlay")
        valid_curves = {d: s for d, s in {d: curve_for_date(work_df, contracts, d) for d in multi_dates}.items() if s is not None}
        if not valid_curves:
            st.warning("No outright data for selected dates.")
        else:
            fig_outright = overlay_figure(contracts, valid_curves, y_label=("Z-score" if normalize else "Price ($)"))
            st.plotly_chart(fig_outright, use_container_width=True, key=f"outright_chart_{primary_symbol}")

    with col2:
        st.markdown("##### Spread Curve Overlay")
        SPREAD_SHEET_MAP = {"CL": "Spread_CL", "BZ": "Spread_Brent", "DBI": "Spread_DBI", "MRBN": "Spread_MRBN"}
        target_sheet = SPREAD_SHEET_MAP.get(primary_symbol)
        if target_sheet:
            df_s, contracts_s = load_product_data(MASTER_EXCEL_FILE, target_sheet)
            if df_s is not None:
                valid_curves_s = {d: curve_for_date(df_s, contracts_s, d) for d in multi_dates if curve_for_date(df_s, contracts_s, d) is not None}
                if valid_curves_s:
                    fig_spread = overlay_figure(contracts_s, valid_curves_s, y_label="Spread ($)", title="Spread Curve")
                    st.plotly_chart(fig_spread, use_container_width=True, key=f"spread_chart_{primary_symbol}")
                else: st.warning("No spread data for dates.")
            else: st.info(f"Sheet '{target_sheet}' not found or is empty.")
        else: st.info("Not configured.")

    with col3:
        st.markdown("##### Fly Curve Overlay")
        FLY_SHEET_MAP = {"CL": "FLY_CL", "BZ": "FLY_Brent", "DBI": "FLY_DBI", "MRBN": "FLY_MRBN"}
        target_sheet = FLY_SHEET_MAP.get(primary_symbol)
        if target_sheet:
            df_f, contracts_f = load_product_data(MASTER_EXCEL_FILE, target_sheet)
            if df_f is not None:
                valid_curves_f = {d: curve_for_date(df_f, contracts_f, d) for d in multi_dates if curve_for_date(df_f, contracts_f, d) is not None}
                if valid_curves_f:
                    fig_fly = overlay_figure(contracts_f, valid_curves_f, y_label="Fly Spread ($)", title="Fly Curve")
                    st.plotly_chart(fig_fly, use_container_width=True, key=f"fly_chart_{primary_symbol}")
                else: st.warning("No fly data for dates.")
            else: st.info(f"Sheet '{target_sheet}' not found or is empty.")
        else: st.info("Not configured.")

    if len(selected_symbols) > 1:
        st.markdown("---")
        st.header("Comparative Outright Curves")
        fig_comp = go.Figure()
        for symbol in selected_symbols:
            comp_product_info = PRODUCT_CONFIG[symbol]
            df_comp, contracts_comp = load_product_data(MASTER_EXCEL_FILE, comp_product_info["sheet"])
            if df_comp is not None:
                for d in multi_dates:
                    curve = curve_for_date(df_comp, contracts_comp, d)
                    if curve is not None:
                        fig_comp.add_trace(go.Scatter(x=contracts_comp, y=curve.values, mode='lines+markers', name=f'{symbol} ({d.strftime("%b %d")})'))
        fig_comp.update_layout(title="Outright Curve Comparison for All Selected Products", yaxis_title="Price ($)", xaxis_title="Contract", hovermode="x unified", template="plotly_white")
        st.plotly_chart(fig_comp, use_container_width=True, key="comparison_chart")

# ========================================================================================
# TAB 2: HISTORICAL ANALYSIS
# ========================================================================================
with tab2:
    st.header(f"Historical Analysis for {primary_product_info['name']}")
    
    selected_range = st.selectbox("Select date range for analysis", ["Full History", "Last 1 Year", "Last 6 Months", "Last 1 Month", "Last 2 Weeks", "Last 1 Week"], index=1, key=f"range_{primary_symbol}")
    
    filtered_df = filter_dates(work_df, selected_range)
    merged_df = pd.merge(filtered_df, df_news, on="Date", how="left") if not df_news.empty else filtered_df.copy()

    sub_tab1, sub_tab2 = st.tabs(["Spread Analysis", "Fly Analysis"])

    with sub_tab1:
        st.markdown("**Compare Multiple Spreads Over Time**")
        default_spread = [f"{contracts[0]} - {contracts[1]}"] if len(contracts) > 1 else []
        spread_pairs = st.multiselect("Select contract pairs", options=[f"{c1} - {c2}" for i, c1 in enumerate(contracts) for c2 in contracts[i+1:]], default=default_spread, key=f"spread_pairs_{primary_symbol}")
        
        if spread_pairs:
            fig_spread = go.Figure()
            for pair in spread_pairs:
                c1, c2 = [x.strip() for x in pair.split("-")]
                hover_text = [f"<b>Date:</b> {d.strftime('%Y-%m-%d')}<br><b>{c1}:</b> {p1:.2f}<br><b>{c2}:</b> {p2:.2f}<br><b>Spread:</b> {s:.2f}" for d, p1, p2, s in zip(merged_df['Date'], merged_df[c1], merged_df[c2], merged_df[c1] - merged_df[c2])]
                fig_spread.add_trace(go.Scatter(x=merged_df["Date"], y=merged_df[c1] - merged_df[c2], mode="lines", name=f"{c1}-{c2}", hovertext=hover_text, hoverinfo="text"))
            
            fig_spread.update_layout(title="Historical Spread Comparison", xaxis_title="Date", yaxis_title="Price Difference ($)", template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig_spread, use_container_width=True, key=f"spread_chart_{primary_symbol}")

    with sub_tab2:
        st.markdown("**Compare Multiple Butterfly Spreads Over Time**")
        fly_type = st.radio("Fly construction method:", ["Auto (consecutive months)", "Manual selection"], index=0, horizontal=True, key=f"fly_type_{primary_symbol}")
        selected_flies = []
        if fly_type == "Manual selection":
            num_flies = st.number_input("Number of flies", 1, 5, 1, 1, key=f"num_flies_{primary_symbol}")
            for i in range(num_flies):
                cols = st.columns(3)
                f1 = cols[0].selectbox(f"Wing 1 (Fly {i+1})", contracts, 0, key=f"fly_f1_{i}_{primary_symbol}")
                f2 = cols[1].selectbox(f"Body (Fly {i+1})", contracts, 1, key=f"fly_f2_{i}_{primary_symbol}")
                f3 = cols[2].selectbox(f"Wing 2 (Fly {i+1})", contracts, 2, key=f"fly_f3_{i}_{primary_symbol}")
                selected_flies.append((f1, f2, f3))
        else: # Auto
            default_fly = [contracts[0]] if len(contracts) > 2 else []
            base_contracts = st.multiselect("Select base contracts for Auto Fly", contracts, default=default_fly, key=f"fly_base_{primary_symbol}")
            for base in base_contracts:
                idx = contracts.index(base)
                if idx + 2 < len(contracts): selected_flies.append((contracts[idx], contracts[idx+1], contracts[idx+2]))
                else: st.warning(f"Not enough consecutive contracts for '{base}' auto fly.")
        
        if selected_flies:
            fig_fly = go.Figure()
            for f1, f2, f3 in selected_flies:
                fly_values = merged_df[f1] - 2 * merged_df[f2] + merged_df[f3]
                hover_text = [f"<b>Date:</b> {d.strftime('%Y-%m-%d')}<br><b>{f1}:</b> {p1:.2f}<br><b>{f2}:</b> {p2:.2f}<br><b>{f3}:</b> {p3:.2f}<br><b>Fly Value:</b> {fv:.2f}" for d, p1, p2, p3, fv in zip(merged_df['Date'], merged_df[f1], merged_df[f2], merged_df[f3], fly_values)]
                fig_fly.add_trace(go.Scatter(x=merged_df["Date"], y=fly_values, mode="lines", name=f"Fly {f1}-{f2}-{f3}", hovertext=hover_text, hoverinfo="text"))
            
            fig_fly.update_layout(title="Historical Fly Comparison", xaxis_title="Date", yaxis_title="Price Difference ($)", template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig_fly, use_container_width=True, key=f"fly_chart_{primary_symbol}")
