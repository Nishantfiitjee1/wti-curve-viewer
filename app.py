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
# Define the single master Excel file.
MASTER_EXCEL_FILE = "Futures_Data.xlsx"

# **FIX**: Cleaned up the config to only include products present in your Excel file.
# Ensured sheet names match your file information exactly.
PRODUCT_CONFIG = {
    "CL": {"name": "WTI Crude Oil", "sheet": "WTI_Outright"},
    "BZ": {"name": "Brent Crude Oil", "sheet": "Brent_outright"},
    "DBI": {"name": "Dubai Crude Oil", "sheet": "Dubai_Outright"},
}


# ---------------------------- Data Loading & Utilities ----------------------------
@st.cache_data(show_spinner="Loading product data...", ttl=3600)
def load_product_data(file_path, sheet_name):
    """Loads and parses futures data from a specific sheet in the master Excel file."""
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None, engine="openpyxl")
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

selected_symbol = st.sidebar.selectbox(
    "Select Product",
    options=list(PRODUCT_CONFIG.keys()),
    format_func=lambda symbol: PRODUCT_CONFIG[symbol]["name"],
    key="product_selector"
)
selected_product_info = PRODUCT_CONFIG[selected_symbol]
target_sheet_name_from_config = selected_product_info["sheet"]

# ---------------------------- Main App Logic ----------------------------
st.title(f"{selected_product_info['name']} Curve Viewer")

if not os.path.exists(MASTER_EXCEL_FILE):
    st.error(f"Master data file not found: `{MASTER_EXCEL_FILE}`. Please ensure the file is in the same directory.")
    st.stop()

# --- Robust sheet name checking (case and space insensitive) ---
try:
    excel_file_handler = pd.ExcelFile(MASTER_EXCEL_FILE)
    excel_sheets = excel_file_handler.sheet_names
    
    cleaned_target_sheet = target_sheet_name_from_config.strip().lower()
    actual_sheet_to_load = None
    
    for sheet in excel_sheets:
        if sheet.strip().lower() == cleaned_target_sheet:
            actual_sheet_to_load = sheet
            break
            
    if actual_sheet_to_load is None:
        st.caption("Analysis of futures curves, spreads, and historical evolution.")
        st.markdown(f'<div class="placeholder-text">Data for {selected_product_info["name"]} is not yet available.<br>Sheet `{target_sheet_name_from_config}` not found in the Excel file.</div>', unsafe_allow_html=True)
        st.stop()
        
except Exception as e:
    st.error(f"Could not read the master Excel file. Error: {e}")
    st.stop()

# --- If checks pass, load the data and build the dashboard ---
try:
    df, contracts = load_product_data(MASTER_EXCEL_FILE, actual_sheet_to_load)
except Exception as e:
    st.error(f"An error occurred while loading the data from sheet `{actual_sheet_to_load}`: {e}")
    st.stop()

# --- Sidebar Date Controls ---
st.sidebar.header("Date Selection")
all_dates = sorted(df["Date"].dt.date.unique().tolist(), reverse=True)
max_d, min_d = all_dates[0], all_dates[-1]

single_date = st.sidebar.date_input("Single Date", value=max_d, min_value=min_d, max_value=max_d, key=f"date_input_{selected_symbol}")
multi_dates = st.sidebar.multiselect("Multi-Date Overlay", options=all_dates, default=[all_dates[0], all_dates[min(1, len(all_dates)-1)]], key=f"multiselect_{selected_symbol}")

st.sidebar.header("Display Options")
normalize = st.sidebar.checkbox("Normalize curves (z-score)", key=f"normalize_{selected_symbol}")
do_export = st.sidebar.checkbox("Enable CSV export", key=f"export_{selected_symbol}")

# --- Data Preparation ---
work_df = df.copy()
if normalize:
    vals = work_df[contracts].astype(float)
    work_df[contracts] = (vals - vals.mean(axis=1).values[:, None]) / vals.std(axis=1).values[:, None]

# --- Main UI Tabs ---
st.caption("Analysis of futures curves, spreads, and historical evolution.")
tab1, tab2, tab3 = st.tabs(["Outright", "Spread and Fly", "Curve Animation"])

with tab1:
    st.header(f"Curve Analysis for {single_date}")
    s1 = curve_for_date(work_df, contracts, single_date)
    if s1 is None:
        st.error("No data available for the chosen date.")
    else:
        st.markdown("##### Key Curve Metrics")
        m_cols = st.columns(3)
        if len(contracts) > 0: m_cols[0].metric(label=f"Prompt Price ({contracts[0]})", value=f"{s1.get(contracts[0], 0):.2f}")
        if len(contracts) > 1: m_cols[1].metric(label=f"M1-M2 Spread ({contracts[0]}-{contracts[1]})", value=f"{s1[contracts[0]] - s1[contracts[1]]:.2f}")
        if len(contracts) > 11: m_cols[2].metric(label=f"M1-M12 Spread ({contracts[0]}-{contracts[11]})", value=f"{s1[contracts[0]] - s1[contracts[11]]:.2f}")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Single Date Curve")
            fig_single = overlay_figure(contracts, {single_date: s1}, y_label=("Z-score" if normalize else "Last Price ($)"))
            st.plotly_chart(fig_single, use_container_width=True, key=f"single_chart_{selected_symbol}")
        with col2:
            st.markdown("##### Multi-Date Overlay")
            valid_curves = {d: s for d, s in {d: curve_for_date(work_df, contracts, d) for d in multi_dates}.items() if s is not None}
            if not valid_curves: st.warning("No data found for any overlay dates.")
            else:
                fig_overlay = overlay_figure(contracts, valid_curves, y_label=("Z-score" if normalize else "Last Price ($)"))
                st.plotly_chart(fig_overlay, use_container_width=True, key=f"multi_chart_{selected_symbol}")

with tab2:
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
        if spread_pairs:
            fig_spread = go.Figure()
            stats_cols = st.columns(len(spread_pairs))
            csv_data = {"Date": filtered_df["Date"].dt.date}
            for i, pair in enumerate(spread_pairs):
                c1, c2 = [x.strip() for x in pair.split("-")]
                spread_curve = filtered_df[c1] - filtered_df[c2]
                csv_data[f"{c1}-{c2}"] = spread_curve
                fig_spread.add_trace(go.Scatter(x=filtered_df["Date"], y=spread_curve, mode="lines", name=f"{c1}-{c2}"))
                with stats_cols[i]:
                    st.metric(label=f"{c1}-{c2} (Latest)", value=f"{spread_curve.iloc[-1]:.2f}")
            st.markdown("---")
            fig_spread.update_layout(title="Historical Spread Comparison", xaxis_title="Date", yaxis_title="Price Difference ($)", template="plotly_white")
            st.plotly_chart(fig_spread, use_container_width=True, key=f"spread_chart_{selected_symbol}")
            if do_export:
                st.download_button("Download Spread CSV", pd.DataFrame(csv_data).to_csv(index=False).encode("utf-8"), file_name=f"{selected_symbol}_spreads.csv", mime="text/csv", key=f"dl_spread_{selected_symbol}")

    with sub_tab2:
        st.markdown("**Compare Multiple Butterfly Spreads Over Time**")
        fly_type = st.radio("Fly construction method:", ["Auto (consecutive months)", "Manual selection"], index=0, horizontal=True, key=f"fly_type_{selected_symbol}")
        selected_flies = []
        if fly_type == "Manual selection":
            num_flies = st.number_input("Number of flies", min_value=1, max_value=5, value=1, step=1, key=f"num_flies_{selected_symbol}")
            for i in range(num_flies):
                cols = st.columns(3)
                f1 = cols[0].selectbox(f"Wing 1 (Fly {i+1})", contracts, index=0, key=f"fly_f1_{i}_{selected_symbol}")
                f2 = cols[1].selectbox(f"Body (Fly {i+1})", contracts, index=1, key=f"fly_f2_{i}_{selected_symbol}")
                f3 = cols[2].selectbox(f"Wing 2 (Fly {i+1})", contracts, index=2, key=f"fly_f3_{i}_{selected_symbol}")
                selected_flies.append((f1, f2, f3))
        else: # Auto
            default_fly = [contracts[0]] if len(contracts) > 2 else []
            base_contracts = st.multiselect("Select base contracts for Auto Fly", contracts, default=default_fly, key=f"fly_base_{selected_symbol}")
            for base in base_contracts:
                idx = contracts.index(base)
                if idx + 2 < len(contracts): selected_flies.append((contracts[idx], contracts[idx+1], contracts[idx+2]))
                else: st.warning(f"Not enough consecutive contracts for '{base}' auto fly. Skipping.")
        
        if selected_flies:
            fig_fly = go.Figure()
            fly_stats_cols = st.columns(len(selected_flies))
            fly_csv_data = {"Date": filtered_df["Date"].dt.date}
            for i, (f1, f2, f3) in enumerate(selected_flies):
                fly_curve = filtered_df[f1] - 2 * filtered_df[f2] + filtered_df[f3]
                fly_name = f"{f1}-{f2}-{f3}"
                fly_csv_data[f"Fly_{fly_name}"] = fly_curve
                fig_fly.add_trace(go.Scatter(x=filtered_df["Date"], y=fly_curve, mode="lines", name=f"Fly {fly_name}"))
                with fly_stats_cols[i]:
                    st.metric(label=f"Fly {fly_name} (Latest)", value=f"{fly_curve.iloc[-1]:.2f}")
            st.markdown("---")
            fig_fly.update_layout(title="Historical Fly Comparison", xaxis_title="Date", yaxis_title="Price Difference ($)", template="plotly_white")
            st.plotly_chart(fig_fly, use_container_width=True, key=f"fly_chart_{selected_symbol}")
            if do_export:
                st.download_button("Download Fly CSV", pd.DataFrame(fly_csv_data).to_csv(index=False).encode("utf-8"), file_name=f"{selected_symbol}_flys.csv", mime="text/csv", key=f"dl_fly_{selected_symbol}")

with tab3:
    st.header("Curve Evolution Animation")
    st.info("Use the slider or the 'Play' button to animate the daily changes in the forward curve.")
    anim_df = work_df[["Date"] + contracts].copy().dropna(subset=contracts).reset_index(drop=True)
    if anim_df.empty:
        st.warning("Not enough data to create an animation.")
    else:
        fig_anim = go.Figure(
            data=[go.Scatter(x=contracts, y=anim_df.loc[0, contracts], mode="lines+markers")],
            layout=go.Layout(
                title="Forward Curve Evolution",
                xaxis_title="Contract", yaxis_title="Price ($)" if not normalize else "Z-score",
                template="plotly_white", margin=dict(l=40, r=20, t=60, b=40),
                updatemenus=[dict(
                    type="buttons", showactive=False, y=1.15, x=1.05, xanchor="right", yanchor="top",
                    buttons=[
                        dict(label="Play", method="animate", args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]),
                        dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                    ]
                )],
                sliders=[dict(
                    active=0, transition={"duration": 0}, currentvalue={"prefix": "Date: ", "font": {"size": 14}},
                    steps=[dict(
                        method="animate",
                        args=[[str(d.date())], {"mode": "immediate", "frame": {"duration": 100, "redraw": True}, "transition": {"duration": 50}}],
                        label=str(d.date())
                    ) for d in anim_df["Date"]]
                )]
            ),
            frames=[go.Frame(
                data=[go.Scatter(x=contracts, y=anim_df.loc[i, contracts])],
                name=str( anim_df.loc[i, "Date"].date() )
            ) for i in range(len(anim_df))]
        )
        st.plotly_chart(fig_anim, use_container_width=True, key=f"anim_chart_{selected_symbol}")

with st.expander("Preview Raw Data"):
    st.dataframe(df.head(25))
