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


# ---------------------------- 1. CENTRAL PRODUCT CONFIGURATION ----------------------------
# To add a new product, just add a new entry to this dictionary.
# The key (e.g., "CL") is the symbol, 'name' is the display name, and 'file' is the exact filename.
PRODUCT_CONFIG = {
    "CL": {"name": "WTI Crude Oil", "file": "WTI_Outright.xlsx"},
    "BZ": {"name": "Brent Crude Oil", "file": "Brent_Outright.xlsx"},
    "ADM": {"name": "Gasoil", "file": "ADM_Outright.xlsx"},
    "DBI": {"name": "Dubai Crude Oil", "file": "DBI_Outright.xlsx"},
    "HOU": {"name": "Houston Crude Oil", "file": "HOU_Outright.xlsx"},
    # Example for a future product:
    # "NG": {"name": "Natural Gas", "file": "NG_Outright.xlsx"},
}


# ---------------------------- Data Loading & Utilities (No major changes) ----------------------------
@st.cache_data(show_spinner="Loading product data...", ttl=3600)
def load_product_data(file_path):
    df_raw = pd.read_excel(file_path, header=None, engine="openpyxl")
    hdr0 = df_raw.iloc[0].tolist()
    contracts = [str(x).strip() for x in hdr0[1:] if pd.notna(x) and str(x).strip() != ""]
    col_names = ["Date"] + contracts
    df = df_raw.iloc[2:].copy()
    df.columns = col_names
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in contracts:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df, contracts

def curve_for_date(df: pd.DataFrame, contracts, d: date) -> pd.Series | None:
    row = df.loc[df["Date"].dt.date == d, contracts]
    if row.empty: return None
    return row.iloc[0]

def overlay_figure(contracts, curves: dict, y_label="Last Price ($)", title="Futures Curve") -> go.Figure:
    fig = go.Figure()
    for label, s in curves.items():
        fig.add_trace(go.Scatter(x=contracts, y=s.values, mode="lines+markers", name=str(label)))
    fig.update_layout(title=title, xaxis_title="Contract", yaxis_title=y_label, hovermode="x unified",
                      template="plotly_white", margin=dict(l=40, r=20, t=60, b=40))
    return fig

def filter_dates(df, selected_range):
    max_date = df["Date"].max()
    if selected_range == "Full History": return df
    elif selected_range == "Last 1 Week": min_date = max_date - timedelta(weeks=1)
    elif selected_range == "Last 2 Weeks": min_date = max_date - timedelta(weeks=2)
    elif selected_range == "Last 1 Month": min_date = max_date - timedelta(days=30)
    elif selected_range == "Last 6 Months": min_date = max_date - timedelta(days=180)
    elif selected_range == "Last 1 Year": min_date = max_date - timedelta(days=365)
    else: min_date = df["Date"].min()
    return df[df["Date"] >= min_date]

# ---------------------------- Sidebar & Product Selection ----------------------------
st.sidebar.title("Global Controls")

# Product selector
selected_symbol = st.sidebar.selectbox(
    "Select Product",
    options=list(PRODUCT_CONFIG.keys()),
    format_func=lambda symbol: PRODUCT_CONFIG[symbol]["name"],
    index=0
)
selected_product_info = PRODUCT_CONFIG[selected_symbol]
file_path = selected_product_info["file"]

# ---------------------------- Main App Logic ----------------------------
st.title(f"{selected_product_info['name']} Curve Viewer")

# --- 2. CHECK IF DATA FILE EXISTS ---
if not os.path.exists(file_path):
    # --- 3. DISPLAY PLACEHOLDER IF FILE IS MISSING ---
    st.caption("Interactive analysis of futures curves, spreads, and historical evolution.")
    st.markdown(f'<div class="placeholder-text">Data for {selected_product_info["name"]} is not yet available.<br>Work in Progress...</div>', unsafe_allow_html=True)
    st.stop() # Stop execution for this product

# --- If file exists, proceed with loading and displaying the dashboard ---
try:
    df, contracts = load_product_data(file_path)
except Exception as e:
    st.error(f"An error occurred while loading the data file `{file_path}`: {e}")
    st.stop()

# --- Sidebar Date Controls (only shown if data is loaded) ---
st.sidebar.header("Date Selection")
all_dates = sorted(df["Date"].dt.date.unique().tolist(), reverse=True)
max_d, min_d = all_dates[0], all_dates[-1]

# --- 4. USE SESSION STATE FOR UNIQUE WIDGET KEYS ---
# This prevents errors when switching between products with different date ranges/contracts
if 'single_date' not in st.session_state or st.session_state.get('product') != selected_symbol:
    st.session_state.single_date = max_d
    st.session_state.multi_dates = [all_dates[0], all_dates[min(1, len(all_dates)-1)]]
    st.session_state.product = selected_symbol

single_date = st.sidebar.date_input("Single Date", value=st.session_state.single_date, min_value=min_d, max_value=max_d, key=f"date_input_{selected_symbol}")
multi_dates = st.sidebar.multiselect("Multi-Date Overlay", options=all_dates, default=st.session_state.multi_dates, key=f"multiselect_{selected_symbol}")

st.sidebar.header("Display Options")
normalize = st.sidebar.checkbox("Normalize curves (z-score)", key=f"normalize_{selected_symbol}")
do_export = st.sidebar.checkbox("Enable CSV export", key=f"export_{selected_symbol}")

# --- Data Preparation ---
work_df = df.copy()
if normalize:
    vals = work_df[contracts].astype(float)
    work_df[contracts] = (vals - vals.mean(axis=1).values[:, None]) / vals.std(axis=1).values[:, None]

# --- Main UI Tabs ---
st.caption("Interactive analysis of futures curves, spreads, and historical evolution.")
tab1, tab2, tab3 = st.tabs(["Curve Shape Analysis", "Historical Time Series", "Curve Animation"])

with tab1:
    # (Content of Tab 1)
    st.header(f"Curve Analysis for {single_date}")
    s1 = curve_for_date(work_df, contracts, single_date)
    if s1 is None:
        st.error("No data available for the chosen date.")
    else:
        st.markdown("##### Key Curve Metrics")
        m_cols = st.columns(3)
        m_cols[0].metric(label=f"Prompt Price ({contracts[0]})", value=f"{s1.get(contracts[0], 0):.2f}")
        if len(contracts) > 1: m_cols[1].metric(label=f"M1-M2 Spread ({contracts[0]}-{contracts[1]})", value=f"{s1[contracts[0]] - s1[contracts[1]]:.2f}")
        if len(contracts) > 11: m_cols[2].metric(label=f"M1-M12 Spread ({contracts[0]}-{contracts[11]})", value=f"{s1[contracts[0]] - s1[contracts[11]]:.2f}")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Single Date Curve")
            fig_single = overlay_figure(contracts, {single_date: s1}, y_label=("Z-score" if normalize else "Last Price ($)"))
            st.plotly_chart(fig_single, use_container_width=True)
        with col2:
            st.markdown("##### Multi-Date Overlay")
            valid_curves = {d: s for d, s in {d: curve_for_date(work_df, contracts, d) for d in multi_dates}.items() if s is not None}
            if not valid_curves: st.warning("No data found for any overlay dates.")
            else:
                fig_overlay = overlay_figure(contracts, valid_curves, y_label=("Z-score" if normalize else "Last Price ($)"))
                st.plotly_chart(fig_overlay, use_container_width=True)

with tab2:
    # (Content of Tab 2)
    st.header("Spread & Fly Time Series Analysis")
    selected_range = st.selectbox("Select date range for analysis", ["Full History", "Last 1 Year", "Last 6 Months", "Last 1 Month", "Last 2 Weeks", "Last 1 Week"], index=1, key=f"range_{selected_symbol}")
    filtered_df = filter_dates(work_df, selected_range)
    sub_tab1, sub_tab2 = st.tabs(["Spread Analysis", "Fly Analysis"])

    with sub_tab1:
        st.markdown("**Compare Multiple Spreads Over Time**")
        default_spread = [f"{contracts[0]} - {contracts[1]}"] if len(contracts) > 1 else []
        spread_pairs = st.multiselect(
            "Select contract pairs",
            options=[f"{c1} - {c2}" for i, c1 in enumerate(contracts) for c2 in contracts[i+1:]],
            default=default_spread, key=f"spread_pairs_{selected_symbol}"
        )
        # ... (rest of spread logic as before)

    with sub_tab2:
        st.markdown("**Compare Multiple Butterfly Spreads Over Time**")
        fly_type = st.radio("Fly construction method:", ["Auto (consecutive months)", "Manual selection"], index=0, horizontal=True, key=f"fly_type_{selected_symbol}")
        # ... (rest of fly logic as before)

with tab3:
    # (Content of Tab 3)
    st.header("Curve Evolution Animation")
    st.info("Use the slider or the 'Play' button to animate the daily changes in the forward curve.")
    anim_df = work_df[["Date"] + contracts].copy().dropna(subset=contracts).reset_index(drop=True)
    if anim_df.empty:
        st.warning("Not enough data to create an animation.")
    else:
        # ... (rest of animation logic as before)
        pass # Placeholder for the animation figure code

with st.expander("Preview Raw Data"):
    st.dataframe(df.head(25))

