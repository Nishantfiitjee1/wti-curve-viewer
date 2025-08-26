import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta

# ---------------------------- Page config and CSS ----------------------------
st.set_page_config(page_title="Futures Dashboard", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
    /* Main app styling for dark theme */
    body {
        background-color: #1E1E1E;
        color: #EAEAEA;
    }
    .stApp {
        background-color: #1E1E1E;
    }
    /* Control panel expander */
    .streamlit-expanderHeader {
        font-size: 1.2rem;
        font-weight: bold;
        color: #FAFAFA;
    }
    /* Section headers */
    h2 {
        color: #00A8E8;
        border-bottom: 2px solid #00A8E8;
        padding-bottom: 5px;
    }
    /* Custom styling for placeholder text */
    .placeholder-text {
        font-size: 1.5rem;
        font-weight: bold;
        color: #888;
        text-align: center;
        margin-top: 5rem;
        border: 2px dashed #444;
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
}


# ---------------------------- Data Loading & Utilities ----------------------------
@st.cache_data(show_spinner="Loading product data...", ttl=3600)
def load_product_data(file_path, sheet_name):
    """Loads and parses futures data based on the two-header structure."""
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None, engine="openpyxl")
    try:
        header_row_index = df_raw[df_raw[0] == 'Dates'].index[0] - 1
        data_start_row_index = header_row_index + 2
    except IndexError:
        st.error(f"Format error in '{sheet_name}'. Could not find 'Dates' keyword to locate headers.")
        return pd.DataFrame(), []
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
    """Loads news data and handles 'Date' vs 'Dates' column names."""
    df_news = pd.read_excel(file_path, engine="openpyxl")
    date_col = 'Date' if 'Date' in df_news.columns else 'Dates' if 'Dates' in df_news.columns else None
    if date_col:
        df_news.rename(columns={date_col: 'Date'}, inplace=True)
        df_news["Date"] = pd.to_datetime(df_news["Date"], errors="coerce")
        df_news = df_news.dropna(subset=["Date"])
        return df_news
    else:
        st.warning("News file must contain a 'Date' or 'Dates' column.")
        return pd.DataFrame()

def curve_for_date(df: pd.DataFrame, contracts, d: date) -> pd.Series | None:
    row = df.loc[df["Date"].dt.date == d, contracts]
    return row.iloc[0] if not row.empty else None

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

# ---------------------------- Chart Styling Function ----------------------------
def style_figure(fig, title):
    """Applies the custom dark theme styling to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(color='#EAEAEA')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(color='#888', gridcolor='#444'),
        yaxis=dict(color='#888', gridcolor='#444'),
        legend=dict(font=dict(color='#EAEAEA')),
        hovermode="x unified"
    )
    return fig

# ---------------------------- Main App Logic ----------------------------
st.title("🛢️ Futures Market Dashboard")

# --- Check for required files first ---
if not os.path.exists(MASTER_EXCEL_FILE):
    st.error(f"Master data file not found: `{MASTER_EXCEL_FILE}`. Please ensure it is in the same directory.")
    st.stop()

if os.path.exists(NEWS_EXCEL_FILE):
    df_news = load_news_data(NEWS_EXCEL_FILE)
else:
    st.sidebar.warning(f"News file (`{NEWS_EXCEL_FILE}`) not found. Hover data will not be available.")
    df_news = pd.DataFrame()

# ---------------------------- CONTROL PANEL ----------------------------
with st.expander("Show Control Panel", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        selected_symbol = st.selectbox(
            "Select Product",
            options=list(PRODUCT_CONFIG.keys()),
            format_func=lambda symbol: PRODUCT_CONFIG[symbol]["name"],
            key="product_selector"
        )
    
    # Load data based on product selection
    selected_product_info = PRODUCT_CONFIG[selected_symbol]
    target_sheet_name = selected_product_info["sheet"]
    df, contracts = load_product_data(MASTER_EXCEL_FILE, target_sheet_name)
    
    all_dates = sorted(df["Date"].dt.date.unique().tolist(), reverse=True)
    max_d, min_d = all_dates[0], all_dates[-1]

    with c2:
        single_date = st.date_input("Curve Date", value=max_d, min_value=min_d, max_value=max_d, key=f"date_input_{selected_symbol}")
    with c3:
        multi_dates = st.multiselect("Overlay Dates", options=all_dates, default=[all_dates[0], all_dates[min(1, len(all_dates)-1)]], key=f"multiselect_{selected_symbol}")
    with c4:
        normalize = st.checkbox("Normalize Curves (z-score)", key=f"normalize_{selected_symbol}")

# --- Data Preparation ---
work_df = df.copy()
if normalize:
    vals = work_df[contracts].astype(float)
    work_df[contracts] = (vals - vals.mean(axis=1).values[:, None]) / vals.std(axis=1).values[:, None]

# ---------------------------- CHARTING SECTION ----------------------------

st.markdown("## Outright Curve Analysis")
col1, col2 = st.columns(2)

# --- Chart 1: Single Date Curve ---
with col1:
    s1 = curve_for_date(work_df, contracts, single_date)
    if s1 is None:
        st.warning("No data for selected curve date.")
    else:
        fig_single = go.Figure()
        fig_single.add_trace(go.Scatter(x=contracts, y=s1.values, mode='lines+markers', name=str(single_date), line=dict(color='#00A8E8')))
        fig_single = style_figure(fig_single, f"Curve for {single_date}")
        st.plotly_chart(fig_single, use_container_width=True)

# --- Chart 2: Multi-Date Overlay ---
with col2:
    valid_curves = {d: curve_for_date(work_df, contracts, d) for d in multi_dates}
    valid_curves = {d: s for d, s in valid_curves.items() if s is not None}
    if not valid_curves:
        st.warning("No data for selected overlay dates.")
    else:
        fig_overlay = go.Figure()
        colors = ['#00A8E8', '#EAEAEA', '#FFA500', '#FF4500']
        for i, (d, s) in enumerate(valid_curves.items()):
            fig_overlay.add_trace(go.Scatter(x=contracts, y=s.values, mode='lines+markers', name=str(d), line=dict(color=colors[i % len(colors)])))
        fig_overlay = style_figure(fig_overlay, "Multi-Date Overlay")
        st.plotly_chart(fig_overlay, use_container_width=True)

st.markdown("---")
st.markdown("## Time Series Analysis")

# --- Time Series Controls ---
ts_c1, ts_c2, ts_c3 = st.columns([1, 2, 2])
with ts_c1:
    selected_range = st.selectbox("Select Date Range", ["Full History", "Last 1 Year", "Last 6 Months", "Last 1 Month"], index=1, key=f"range_{selected_symbol}")

filtered_df = filter_dates(work_df, selected_range)
if not df_news.empty:
    merged_df = pd.merge(filtered_df, df_news, on="Date", how="left")
else:
    merged_df = filtered_df.copy()

# --- Chart 3 & 4: Spreads ---
col3, col4 = st.columns(2)
with col3:
    st.markdown("#### Spread Analysis")
    default_spreads = [f"{contracts[0]} - {contracts[1]}"] if len(contracts) > 1 else []
    spread_pairs = st.multiselect("Select Spreads", options=[f"{c1} - {c2}" for i, c1 in enumerate(contracts) for c2 in contracts[i+1:]], default=default_spreads, key=f"spread_pairs_{selected_symbol}")
    
    if spread_pairs:
        fig_spread = go.Figure()
        for pair in spread_pairs:
            c1, c2 = [x.strip() for x in pair.split("-")]
            price_hover_text = [f"<b>Date:</b> {d.strftime('%Y-%m-%d')}<br><b>Spread:</b> {s:.2f}" for d, s in zip(merged_df['Date'], merged_df[c1] - merged_df[c2])]
            fig_spread.add_trace(go.Scatter(x=merged_df["Date"], y=merged_df[c1] - merged_df[c2], mode="lines", name=f"{c1}-{c2}", hovertext=price_hover_text, hoverinfo="text"))

            if not df_news.empty:
                news_cols = df_news.columns.drop('Date')
                news_df_in_view = merged_df.dropna(subset=news_cols, how='all')
                if not news_df_in_view.empty:
                    news_hover_text = news_df_in_view.apply(lambda row: f"<b>Date:</b> {row['Date'].strftime('%Y-%m-%d')}<br><hr>" + "<br>".join(f"<b>{col.replace('_', ' ')}:</b> {row[col]}" for col in news_cols if pd.notna(row[col])), axis=1)
                    fig_spread.add_trace(go.Scatter(x=news_df_in_view['Date'], y=news_df_in_view[c1] - news_df_in_view[c2], mode='markers', name='News Event', marker=dict(size=10, color='#FFA500', symbol='circle'), hovertext=news_hover_text, hoverinfo="text", showlegend=False))
        
        fig_spread = style_figure(fig_spread, "Historical Spreads")
        st.plotly_chart(fig_spread, use_container_width=True)

# --- Chart 5 & 6: Flies ---
with col4:
    st.markdown("#### Fly Analysis")
    default_fly = [contracts[0]] if len(contracts) > 2 else []
    base_contracts = st.multiselect("Select Base Contracts for Auto Fly", contracts, default=default_fly, key=f"fly_base_{selected_symbol}")
    
    selected_flies = []
    for base in base_contracts:
        idx = contracts.index(base)
        if idx + 2 < len(contracts):
            selected_flies.append((contracts[idx], contracts[idx+1], contracts[idx+2]))
    
    if selected_flies:
        fig_fly = go.Figure()
        for f1, f2, f3 in selected_flies:
            fly_values = merged_df[f1] - 2 * merged_df[f2] + merged_df[f3]
            price_hover_text_fly = [f"<b>Date:</b> {d.strftime('%Y-%m-%d')}<br><b>Fly Value:</b> {fv:.2f}" for d, fv in zip(merged_df['Date'], fly_values)]
            fig_fly.add_trace(go.Scatter(x=merged_df["Date"], y=fly_values, mode="lines", name=f"Fly {f1}-{f2}-{f3}", hovertext=price_hover_text_fly, hoverinfo="text"))

            if not df_news.empty:
                news_cols = df_news.columns.drop('Date')
                news_df_in_view = merged_df.dropna(subset=news_cols, how='all')
                if not news_df_in_view.empty:
                    news_hover_text = news_df_in_view.apply(lambda row: f"<b>Date:</b> {row['Date'].strftime('%Y-%m-%d')}<br><hr>" + "<br>".join(f"<b>{col.replace('_', ' ')}:</b> {row[col]}" for col in news_cols if pd.notna(row[col])), axis=1)
                    fly_values_news = news_df_in_view[f1] - 2 * news_df_in_view[f2] + news_df_in_view[f3]
                    fig_fly.add_trace(go.Scatter(x=news_df_in_view['Date'], y=fly_values_news, mode='markers', name='News Event', marker=dict(size=10, color='#FFA500', symbol='circle'), hovertext=news_hover_text, hoverinfo="text", showlegend=False))
        
        fig_fly = style_figure(fig_fly, "Historical Flies")
        st.plotly_chart(fig_fly, use_container_width=True)

# --- Raw Data Preview ---
with st.expander("Preview Raw Data"):
    st.dataframe(df.head(25))
