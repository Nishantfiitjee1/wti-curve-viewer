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
    .stApp {
        background-color: #121212;
        color: #EAEAEA;
    }
    /* Header styling */
    .header {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
        border: 1px solid #333;
    }
    /* Section headers */
    h2 {
        color: #00A8E8;
        border-bottom: 1px solid #444;
        padding-bottom: 10px;
        margin-top: 20px;
        font-size: 1.5rem;
    }
    /* Make Streamlit UI elements blend with dark theme */
    .stButton>button {
        background-color: #333;
        color: #EAEAEA;
        border: 1px solid #555;
    }
    .stDateInput input {
        background-color: #222;
        color: #EAEAEA;
    }
    .stMultiSelect div[data-baseweb="select"] > div {
        background-color: #222;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------- 1. CENTRAL FILE & PRODUCT CONFIGURATION ----------------------------
MASTER_EXCEL_FILE = "Futures_Data.xlsx"
NEWS_EXCEL_FILE = "Important_news_date.xlsx"

PRODUCT_CONFIG = {
    "CL": {"name": "WTI Crude Oil", "sheet": "WTI_Outright", "color": "#00A8E8"},
    "BZ": {"name": "Brent Crude Oil", "sheet": "Brent_outright", "color": "#7DFFC0"},
    "DBI": {"name": "Dubai Crude Oil", "sheet": "Dubai_Outright", "color": "#FFA500"},
}


# ---------------------------- Data Loading & Utilities ----------------------------
@st.cache_data(show_spinner="Loading data...", ttl=3600)
def load_all_data():
    """Loads all product and news data at once."""
    all_product_data = {}
    for symbol, config in PRODUCT_CONFIG.items():
        try:
            df_raw = pd.read_excel(MASTER_EXCEL_FILE, sheet_name=config["sheet"], header=None, engine="openpyxl")
            header_row_index = df_raw[df_raw[0] == 'Dates'].index[0] - 1
            data_start_row_index = header_row_index + 2
            contracts = [str(x).strip() for x in df_raw.iloc[header_row_index].tolist()[1:] if pd.notna(x) and str(x).strip() != ""]
            col_names = ["Date"] + contracts
            df = df_raw.iloc[data_start_row_index:].copy()
            df.columns = col_names
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            for c in contracts:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
            all_product_data[symbol] = {"data": df, "contracts": contracts}
        except Exception as e:
            st.warning(f"Could not load or parse sheet for {config['name']}. Error: {e}")
            continue

    df_news = pd.DataFrame()
    if os.path.exists(NEWS_EXCEL_FILE):
        news_df_raw = pd.read_excel(NEWS_EXCEL_FILE, engine="openpyxl")
        date_col = 'Date' if 'Date' in news_df_raw.columns else 'Dates' if 'Dates' in news_df_raw.columns else None
        if date_col:
            news_df_raw.rename(columns={date_col: 'Date'}, inplace=True)
            news_df_raw["Date"] = pd.to_datetime(news_df_raw["Date"], errors="coerce")
            df_news = news_df_raw.dropna(subset=["Date"])
    
    return all_product_data, df_news

# **FIX**: Added the missing curve_for_date helper function
def curve_for_date(df: pd.DataFrame, contracts, d: date) -> pd.Series | None:
    """Extracts the curve data for a single date."""
    row = df.loc[df["Date"].dt.date == d, contracts]
    return row.iloc[0] if not row.empty else None

def style_figure(fig, title):
    """Applies the custom dark theme styling to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(color='#EAEAEA', size=16), x=0.5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#1E1E1E',
        xaxis=dict(color='#888', gridcolor='#333', zeroline=False),
        yaxis=dict(color='#888', gridcolor='#333', zeroline=False),
        legend=dict(font=dict(color='#EAEAEA')),
        hovermode="x unified"
    )
    return fig

# ---------------------------- Main App Logic ----------------------------
if not os.path.exists(MASTER_EXCEL_FILE):
    st.error(f"Master data file not found: `{MASTER_EXCEL_FILE}`. Please ensure it is in the same directory.")
    st.stop()

all_data, df_news = load_all_data()

# ---------------------------- HEADER CONTROL PANEL ----------------------------
st.markdown('<div class="header">', unsafe_allow_html=True)
cols = st.columns([1, 3, 2, 1])
with cols[0]:
    st.write("**Views:**")
    st.button("Curves", use_container_width=True) # Placeholder button
    st.button("Strategy", use_container_width=True) # Placeholder button

with cols[1]:
    st.write("**Products:**")
    prod_cols = st.columns(len(PRODUCT_CONFIG))
    selected_products = []
    for i, (symbol, config) in enumerate(PRODUCT_CONFIG.items()):
        if prod_cols[i].checkbox(symbol, value=True, key=f"prod_{symbol}"):
            selected_products.append(symbol)

with cols[2]:
    st.write("**Date Range:**")
    date_cols = st.columns(2)
    start_date = date_cols[0].date_input("Start Date", value=date(2025, 1, 1), key="start_date")
    end_date = date_cols[1].date_input("End Date", value=date.today(), key="end_date")

with cols[3]:
    st.write("**Actions:**")
    if st.button("Update", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------- DYNAMIC CHART GRID ----------------------------
if not selected_products:
    st.warning("Please select at least one product to display charts.")
    st.stop()

st.markdown("## Outright Curves")
grid_cols = st.columns(len(selected_products))

for i, symbol in enumerate(selected_products):
    with grid_cols[i]:
        product_data = all_data.get(symbol)
        if not product_data:
            continue
            
        df = product_data["data"]
        contracts = product_data["contracts"]
        
        # Filter data by date range from header
        filtered_df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
        if filtered_df.empty:
            st.warning(f"No data for {symbol} in the selected date range.")
            continue

        # Use the last available date in the filtered range for the curve
        latest_date_in_range = filtered_df['Date'].max().date()
        curve_data = curve_for_date(filtered_df, contracts, latest_date_in_range)

        if curve_data is None:
            st.warning(f"No curve data for {symbol} on {latest_date_in_range}.")
            continue

        # Create the chart
        fig = go.Figure()
        
        # Add bar chart component (e.g., difference between two contracts)
        if len(contracts) > 1:
            bar_data = filtered_df[contracts[0]] - filtered_df[contracts[1]]
            fig.add_trace(go.Bar(x=filtered_df['Date'], y=bar_data, name='M1-M2 Spread', marker_color='#444'))

        # Add line chart for the curve
        fig.add_trace(go.Scatter(
            x=contracts, 
            y=curve_data.values, 
            mode='lines+markers', 
            name=f'Curve {latest_date_in_range}',
            line=dict(color=PRODUCT_CONFIG[symbol]["color"])
        ))
        
        fig = style_figure(fig, f"{PRODUCT_CONFIG[symbol]['name']} Curve")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("## Spread Analysis")
spread_grid_cols = st.columns(len(selected_products))

for i, symbol in enumerate(selected_products):
     with spread_grid_cols[i]:
        product_data = all_data.get(symbol)
        if not product_data:
            continue
        
        df = product_data["data"]
        contracts = product_data["contracts"]
        
        filtered_df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
        if filtered_df.empty:
            continue

        if len(contracts) < 2:
            st.write(f"Not enough contracts for spread analysis in {symbol}.")
            continue
            
        # Default to M1-M2 spread
        c1, c2 = contracts[0], contracts[1]
        
        merged_df = pd.merge(filtered_df, df_news, on="Date", how="left") if not df_news.empty else filtered_df

        fig_spread = go.Figure()
        
        # Main spread line
        spread_values = merged_df[c1] - merged_df[c2]
        hover_text = [f"<b>Date:</b> {d.strftime('%Y-%m-%d')}<br><b>Spread:</b> {s:.2f}" for d, s in zip(merged_df['Date'], spread_values)]
        fig_spread.add_trace(go.Scatter(x=merged_df["Date"], y=spread_values, mode="lines", name=f"{c1}-{c2}", hovertext=hover_text, hoverinfo="text", line=dict(color=PRODUCT_CONFIG[symbol]["color"])))

        # News bubbles
        if not df_news.empty:
            news_cols = df_news.columns.drop('Date')
            news_df_in_view = merged_df.dropna(subset=news_cols, how='all')
            if not news_df_in_view.empty:
                news_hover_text = news_df_in_view.apply(lambda row: f"<b>Date:</b> {row['Date'].strftime('%Y-%m-%d')}<br><hr>" + "<br>".join(f"<b>{col.replace('_', ' ')}:</b> {row[col]}" for col in news_cols if pd.notna(row[col])), axis=1)
                fig_spread.add_trace(go.Scatter(x=news_df_in_view['Date'], y=news_df_in_view[c1] - news_df_in_view[c2], mode='markers', name='News', marker=dict(size=10, color='#FFA500', symbol='circle'), hovertext=news_hover_text, hoverinfo="text", showlegend=False))

        fig_spread = style_figure(fig_spread, f"{symbol} M1-M2 Spread")
        st.plotly_chart(fig_spread, use_container_width=True)
