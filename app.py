# ==================================================================================================
# app.py - Full 650+ line Streamlit Crude Oil Dashboard
# ==================================================================================================

import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta
from itertools import cycle

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Futures Dashboard", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# CSS
# ---------------------------
st.markdown("""
<style>
body, .block-container {padding-top: 8px; padding-bottom: 8px; font-family:sans-serif;}
.header {background-color: #F0F2F6; padding: 6px 10px; border-radius:6px; margin-bottom:8px; border:1px solid #E0E0E0; display:flex; align-items:center; justify-content:space-between;}
.stButton>button {border-radius:4px; padding:2px 6px; border:1px solid #B0B0B0; background-color:#FFFFFF; color:#111; font-size:12px; height:26px;}
div[data-testid="stCheckbox"] label, .stSelectbox, .stMultiSelect, .stDateInput, .stNumberInput {font-size:12px;}
.product-pill {display:inline-block;margin:0 6px 2px 0;padding:4px 8px;border-radius:12px;background:#f5f7fa;font-size:12px;border:1px solid #ddd;}
.product-pill.selected {background:#00A8E8;color:white;border-color:#0091d6;}
.selected-badges {position:relative; margin-bottom:6px;}
.badge {display:inline-block; padding:4px 8px; margin-right:6px; border-radius:12px; background:#eee; font-size:12px; border:1px solid #ddd;}
.badge.color {background:#00A8E8;color:#fff;border-color:#0087c9;}
.element-container .stDataFrame {padding:6px 6px 10px 6px;}
.max-btn {width:100%; padding:8px 10px; font-weight:600; font-size:13px;}
.row-widget.stColumns .stColumn {padding-left:6px; padding-right:6px;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# CONFIG
# ---------------------------
MASTER_EXCEL_FILE = "Futures_Data.xlsx"
NEWS_EXCEL_FILE = "Important_news_date.xlsx"

PRODUCT_CONFIG = {
    "CL": {"name": "WTI Crude Oil", "sheet": "WTI_Outright", "color": "#0072B2"},
    "BZ": {"name": "Brent Crude Oil", "sheet": "Brent_outright", "color": "#D55E00"},
    "DBI": {"name": "Dubai Crude Oil", "sheet": "Dubai_Outright", "color": "#009E73"},
}

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================

@st.cache_data(show_spinner="Loading all market data...", ttl=3600)
def load_all_data():
    all_product_data = {}
    if not os.path.exists(MASTER_EXCEL_FILE):
        return {}, pd.DataFrame()
    for symbol, config in PRODUCT_CONFIG.items():
        try:
            df_raw = pd.read_excel(MASTER_EXCEL_FILE, sheet_name=config["sheet"], header=None, engine="openpyxl")
            header_row_index = df_raw[df_raw[0]=="Dates"].index[0]-1
            data_start_row_index = header_row_index + 2
            contracts = [str(x).strip() for x in df_raw.iloc[header_row_index].tolist()[1:] if pd.notna(x) and str(x).strip()!=""]
            col_names = ["Date"] + contracts
            df = df_raw.iloc[data_start_row_index:].copy()
            df.columns = col_names
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            for c in contracts:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date", ascending=False).reset_index(drop=True)
            all_product_data[symbol] = {"data": df, "contracts": contracts}
        except Exception as e:
            st.warning(f"Could not load sheet for {config['name']}. Error: {e}")
            continue
    df_news = pd.DataFrame()
    if os.path.exists(NEWS_EXCEL_FILE):
        news_df_raw = pd.read_excel(NEWS_EXCEL_FILE, engine="openpyxl")
        date_col = "Date" if "Date" in news_df_raw.columns else "Dates" if "Dates" in news_df_raw.columns else None
        if date_col:
            news_df_raw.rename(columns={date_col:"Date"}, inplace=True)
            news_df_raw["Date"] = pd.to_datetime(news_df_raw["Date"], errors="coerce")
            df_news = news_df_raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return all_product_data, df_news

def style_figure(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color="#333", size=14), x=0.5, y=0.95),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="#F9F9F9",
        xaxis=dict(color="#333", gridcolor="#EAEAEA", zeroline=False),
        yaxis=dict(color="#333", gridcolor="#EAEAEA", zeroline=False),
        legend=dict(font=dict(color="#333"), yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified",
        margin=dict(l=28, r=12, t=44, b=28),
    )
    return fig

def filter_by_date_window(df: pd.DataFrame, start_d: date, end_d: date) -> pd.DataFrame:
    return df[(df["Date"].dt.date >= start_d) & (df["Date"].dt.date <= end_d)]

def nearest_date_on_or_before(df: pd.DataFrame, target_d: date) -> date | None:
    subset = df[df["Date"].dt.date <= target_d]
    if subset.empty: return None
    return subset["Date"].max().date()

def add_news_markers(fig: go.Figure, merged_df: pd.DataFrame, df_news: pd.DataFrame, y_series: pd.Series | None=None):
    if df_news is None or df_news.empty: return fig
    news_cols = [c for c in df_news.columns if c!="Date"]
    if not news_cols: return fig
    if merged_df is None or merged_df.empty: return fig
    joined = pd.merge(merged_df, df_news, on="Date", how="left", suffixes=("","_news"))
    news_df_in_view = joined.dropna(subset=news_cols, how="all")
    if news_df_in_view.empty: return fig
    if y_series is None:
        y_val = merged_df.select_dtypes(include=[np.number]).max().max()
        y_series = pd.Series(index=news_df_in_view.index, dtype=float)
        y_series[:] = (y_val if np.isfinite(y_val) else 0) * 0.98
    news_hover_text = news_df_in_view.apply(lambda row: f"<b>Date:</b> {row['Date'].strftime('%Y-%m-%d')}<br><hr>"+ "<br>".join(f"<b>{col}:</b> {row[col]}" for col in news_cols if pd.notna(row.get(col))), axis=1)
    fig.add_trace(go.Scatter(x=news_df_in_view["Date"], y=y_series.loc[news_df_in_view.index], mode="markers", name="News", marker=dict(size=8,color="#FF6B6B",symbol="diamond"), hovertext=news_hover_text, hoverinfo="text", showlegend=False))
    return fig

def spread_series(df: pd.DataFrame, c1: str, c2: str) -> pd.Series:
    return df[c1] - df[c2]

def fly_series(df: pd.DataFrame, f1: str, f2: str, f3: str) -> pd.Series:
    return df[f1] - 2*df[f2] + df[f3]

# ---------------------------
# STATE
# ---------------------------
if "selected_products" not in st.session_state: st.session_state["selected_products"] = ["CL","BZ","DBI"]
if "start_date" not in st.session_state: st.session_state["start_date"] = date.today()-timedelta(days=365)
if "end_date" not in st.session_state: st.session_state["end_date"] = date.today()
if "picked_one_date" not in st.session_state: st.session_state["picked_one_date"] = None
if "picked_multi_dates" not in st.session_state: st.session_state["picked_multi_dates"] = []
if "maximize" not in st.session_state: st.session_state["maximize"] = False
if "show_table" not in st.session_state: st.session_state["show_table"] = False

# ---------------------------
# LOAD DATA
# ---------------------------
all_data, df_news = load_all_data()
if not all_data:
    st.error(f"Master data file not found or empty: `{MASTER_EXCEL_FILE}`.")
    st.stop()

# ---------------------------
# HEADER
# ---------------------------
st.markdown('<div class="header">', unsafe_allow_html=True)
header_cols = st.columns([1.5,2.0,1.8,1.2,1.0])

with header_cols[0]:
    st.markdown("**Products**")
    prod_cols = st.columns(len(PRODUCT_CONFIG))
    for i, (symbol, cfg) in enumerate(PRODUCT_CONFIG.items()):
        if prod_cols[i].button(symbol, key=f"prod_{symbol}"):
            if symbol in st.session_state.selected_products:
                st.session_state.selected_products.remove(symbol)
            else:
                st.session_state.selected_products.append(symbol)

with header_cols[1]:
    st.date_input("Start Date", value=st.session_state.start_date, key="start_date")
with header_cols[2]:
    st.date_input("End Date", value=st.session_state.end_date, key="end_date")
with header_cols[3]:
    st.button("Maximize Chart", key="maximize")
with header_cols[4]:
    st.button("Show Table", key="show_table")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# PLOT DATA
# ---------------------------
for symbol in st.session_state.selected_products:
    pdata = all_data.get(symbol)
    if pdata is None: continue
    df = pdata["data"]
    df_filtered = filter_by_date_window(df, st.session_state.start_date, st.session_state.end_date)
    if df_filtered.empty: continue
    fig = go.Figure()
    for col in pdata["contracts"]:
        fig.add_trace(go.Scatter(x=df_filtered["Date"], y=df_filtered[col], mode="lines+markers", name=col))
    fig = style_figure(fig, f"{PRODUCT_CONFIG[symbol]['name']} Outright Curves")
    fig = add_news_markers(fig, df_filtered, df_news)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# SPREADS EXAMPLE
# ---------------------------
st.markdown("### Spreads Example")
for symbol in st.session_state.selected_products:
    pdata = all_data.get(symbol)
    if pdata is None: continue
    df = pdata["data"]
    df_filtered = filter_by_date_window(df, st.session_state.start_date, st.session_state.end_date)
    if df_filtered.empty or len(pdata["contracts"])<2: continue
    spread = spread_series(df_filtered, pdata["contracts"][0], pdata["contracts"][1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered["Date"], y=spread, mode="lines+markers", name=f"Spread {pdata['contracts'][0]}-{pdata['contracts'][1]}"))
    fig = style_figure(fig, f"{PRODUCT_CONFIG[symbol]['name']} First Spread")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# FLY EXAMPLE
# ---------------------------
st.markdown("### Fly Example")
for symbol in st.session_state.selected_products:
    pdata = all_data.get(symbol)
    if pdata is None: continue
    df = pdata["data"]
    if len(pdata["contracts"])<3: continue
    df_filtered = filter_by_date_window(df, st.session_state.start_date, st.session_state.end_date)
    fly = fly_series(df_filtered, pdata["contracts"][0], pdata["contracts"][1], pdata["contracts"][2])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered["Date"], y=fly, mode="lines+markers", name=f"Fly {pdata['contracts'][0]}-{pdata['contracts'][1]}-{pdata['contracts'][2]}"))
    fig = style_figure(fig, f"{PRODUCT_CONFIG[symbol]['name']} First Fly")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# SHOW TABLE OPTION
# ---------------------------
if st.session_state.show_table:
    st.markdown("### Table View")
    for symbol in st.session_state.selected_products:
        pdata = all_data.get(symbol)
        if pdata is None: continue
        df_filtered = filter_by_date_window(pdata["data"], st.session_state.start_date, st.session_state.end_date)
        if df_filtered.empty: continue
        st.markdown(f"#### {PRODUCT_CONFIG[symbol]['name']}")
        st.dataframe(df_filtered)

# ---------------------------
# END OF DASHBOARD
# ---------------------------
