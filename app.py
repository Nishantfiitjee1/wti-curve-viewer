import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta

# ==================================================================================================
# PAGE CONFIGURATION AND STYLING
# ==================================================================================================
st.set_page_config(page_title="Futures Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Professional Light Theme CSS for a clean, industry-standard look
st.markdown("""
<style>
    /* Main app styling for light theme */
    .stApp {
        background-color: #FFFFFF;
        color: #1E1E1E;
    }
    /* Header styling */
    .header {
        background-color: #F0F2F6;
        padding: 8px 15px;
        border-radius: 8px;
        margin-bottom: 25px;
        border: 1px solid #E0E0E0;
    }
    /* Section headers */
    h2 {
        color: #1E1E1E;
        border-bottom: 2px solid #00A8E8;
        padding-bottom: 10px;
        margin-top: 25px;
        font-size: 1.6rem;
        font-weight: bold;
    }
    /* Custom button styling for product/view selection */
    .stButton>button {
        border-radius: 5px;
        padding: 4px 10px; /* Made buttons smaller */
        border: 1px solid #B0B0B0;
        background-color: #FFFFFF;
        color: #333;
        font-weight: 500;
        transition: all 0.2s;
        height: 32px; /* Fixed height for alignment */
    }
    .stButton>button:hover {
        border-color: #00A8E8;
        color: #00A8E8;
    }
    /* Styling for selected buttons */
    .stButton>button.selected {
        background-color: #00A8E8;
        color: white;
        border: 1px solid #00A8E8;
    }
    /* Date picker styling */
    .stDateInput {
        background-color: #FFFFFF;
        border-radius: 5px;
    }
    /* Align header elements vertically */
    .st-emotion-cache-1f8336m {
        align-items: end;
    }
</style>
""", unsafe_allow_html=True)


# ==================================================================================================
# 1. CENTRAL FILE & PRODUCT CONFIGURATION
# ==================================================================================================
MASTER_EXCEL_FILE = "Futures_Data.xlsx"
NEWS_EXCEL_FILE = "Important_news_date.xlsx"

# Configuration for each product, including sheet name and a unique color for charts
PRODUCT_CONFIG = {
    "CL": {"name": "WTI Crude Oil", "sheet": "WTI_Outright", "color": "#0072B2"},
    "BZ": {"name": "Brent Crude Oil", "sheet": "Brent_outright", "color": "#D55E00"},
    "DBI": {"name": "Dubai Crude Oil", "sheet": "Dubai_Outright", "color": "#009E73"},
}


# ==================================================================================================
# 2. DATA LOADING AND UTILITY FUNCTIONS
# ==================================================================================================
@st.cache_data(show_spinner="Loading all market data...", ttl=3600)
def load_all_data():
    """
    Loads all product data from the master Excel file and news data.
    This function runs only once and caches the result for performance.
    """
    all_product_data = {}
    if not os.path.exists(MASTER_EXCEL_FILE):
        return {}, pd.DataFrame()

    for symbol, config in PRODUCT_CONFIG.items():
        try:
            df_raw = pd.read_excel(MASTER_EXCEL_FILE, sheet_name=config["sheet"], header=None, engine="openpyxl")
            
            # Intelligently find header and data start rows based on the 'Dates' keyword
            header_row_index = df_raw[df_raw[0] == 'Dates'].index[0] - 1
            data_start_row_index = header_row_index + 2
            
            contracts = [str(x).strip() for x in df_raw.iloc[header_row_index].tolist()[1:] if pd.notna(x) and str(x).strip() != ""]
            col_names = ["Date"] + contracts
            
            df = df_raw.iloc[data_start_row_index:].copy()
            df.columns = col_names
            
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            for c in contracts:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                
            df = df.dropna(subset=["Date"]).sort_values("Date", ascending=False).reset_index(drop=True)
            all_product_data[symbol] = {"data": df, "contracts": contracts}
        except Exception as e:
            st.warning(f"Could not load or parse sheet for {config['name']}. Please check format. Error: {e}")
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

def style_figure(fig, title):
    """Applies the custom light theme styling to a Plotly figure for a professional look."""
    fig.update_layout(
        title=dict(text=title, font=dict(color='#333', size=16), x=0.5, y=0.95),
        paper_bgcolor='rgba(255,255,255,1)',
        plot_bgcolor='#F9F9F9',
        xaxis=dict(color='#333', gridcolor='#EAEAEA', zeroline=False),
        yaxis=dict(color='#333', gridcolor='#EAEAEA', zeroline=False),
        legend=dict(font=dict(color='#333'), yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified",
        margin=dict(l=50, r=20, t=60, b=50)
    )
    return fig

# ==================================================================================================
# 3. STATE MANAGEMENT INITIALIZATION
# ==================================================================================================
# Using st.session_state to keep track of user selections across reruns.
if 'selected_products' not in st.session_state:
    st.session_state['selected_products'] = ["CL", "BZ", "DBI"] # Default selection on first run
if 'start_date' not in st.session_state:
    st.session_state['start_date'] = date.today() - timedelta(days=365)
if 'end_date' not in st.session_state:
    st.session_state['end_date'] = date.today()
if 'current_view' not in st.session_state:
    st.session_state['current_view'] = "Curves" # Default view


# ==================================================================================================
# 4. MAIN APP LOGIC AND LAYOUT
# ==================================================================================================
all_data, df_news = load_all_data()

if not all_data:
    st.error(f"Master data file not found or is empty: `{MASTER_EXCEL_FILE}`. Please ensure it is in the same directory and correctly formatted.")
    st.stop()

# ---------------------------- HEADER CONTROL PANEL ----------------------------
st.markdown('<div class="header">', unsafe_allow_html=True)
header_cols = st.columns([1.5, 3, 3, 1])

# --- View Selection ---
with header_cols[0]:
    st.write("**Views**")
    view_buttons = ["Curves", "Table", "Strategy"]
    for view in view_buttons:
        if st.button(view, key=f"btn_view_{view}", use_container_width=True):
            st.session_state.current_view = view

# --- Product Selection Toggles ---
with header_cols[1]:
    st.write("**Products**")
    prod_cols = st.columns(len(PRODUCT_CONFIG))
    for i, (symbol, config) in enumerate(PRODUCT_CONFIG.items()):
        if prod_cols[i].button(symbol, key=f"btn_prod_{symbol}", use_container_width=True):
            if symbol in st.session_state.selected_products:
                if len(st.session_state.selected_products) > 1: # Prevent deselecting the last one
                    st.session_state.selected_products.remove(symbol)
            else:
                st.session_state.selected_products.append(symbol)

# --- Date Pickers ---
with header_cols[2]:
    st.write("**Date Range**")
    date_cols = st.columns(2)
    st.session_state.start_date = date_cols[0].date_input("Start Date", value=st.session_state.start_date, key="start_date_picker", label_visibility="collapsed")
    st.session_state.end_date = date_cols[1].date_input("End Date", value=st.session_state.end_date, key="end_date_picker", label_visibility="collapsed")

# --- Actions ---
with header_cols[3]:
    st.write("**Actions**")
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# This CSS trick visually highlights the selected buttons
for view in view_buttons:
    if st.session_state.current_view == view:
        st.markdown(f"<style>#root .stButton button[key='btn_view_{view}'] {{background-color: #00A8E8; color: white;}}</style>", unsafe_allow_html=True)
for symbol in PRODUCT_CONFIG:
    if symbol in st.session_state.selected_products:
        st.markdown(f"<style>#root .stButton button[key='btn_prod_{symbol}'] {{background-color: #00A8E8; color: white;}}</style>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ==================================================================================================
# 5. DYNAMIC CHART AND TABLE RENDERING
# ==================================================================================================
if not st.session_state.selected_products:
    st.warning("Please select at least one product to display.")
    st.stop()

# --- CURVES VIEW ---
if st.session_state.current_view == "Curves":
    
    # --- Section 1: Outright Curves ---
    st.markdown("## Outright Curves")
    grid_cols_outright = st.columns(len(st.session_state.selected_products))
    for i, symbol in enumerate(st.session_state.selected_products):
        with grid_cols_outright[i]:
            product_data = all_data.get(symbol)
            if not product_data: continue
            df, contracts = product_data["data"], product_data["contracts"]
            
            # Find the most recent date available within the selected end_date
            latest_date_in_range = df[df['Date'].dt.date <= st.session_state.end_date]['Date'].max().date()
            curve_data = df[df['Date'].dt.date == latest_date_in_range].iloc[0]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=contracts, y=curve_data[contracts].values, mode='lines+markers', name=str(latest_date_in_range), line=dict(color=PRODUCT_CONFIG[symbol]["color"], width=3)))
            fig = style_figure(fig, f"{PRODUCT_CONFIG[symbol]['name']} ({latest_date_in_range})")
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # --- Section 2: Spread Analysis (M1-M2) ---
    st.markdown("## Spread Analysis (M1-M2)")
    grid_cols_spread = st.columns(len(st.session_state.selected_products))
    for i, symbol in enumerate(st.session_state.selected_products):
        with grid_cols_spread[i]:
            product_data = all_data.get(symbol)
            if not product_data: continue
            df, contracts = product_data["data"], product_data["contracts"]

            if len(contracts) < 2:
                st.warning(f"Not enough contracts for {symbol} spread.")
                continue
            
            filtered_df = df[(df['Date'].dt.date >= st.session_state.start_date) & (df['Date'].dt.date <= st.session_state.end_date)]
            if filtered_df.empty: continue

            merged_df = pd.merge(filtered_df, df_news, on="Date", how="left") if not df_news.empty else filtered_df
            c1, c2 = contracts[0], contracts[1]
            
            fig_spread = go.Figure()
            spread_values = merged_df[c1] - merged_df[c2]
            hover_text = [f"<b>Date:</b> {d.strftime('%Y-%m-%d')}<br><b>Spread:</b> {s:.2f}" for d, s in zip(merged_df['Date'], spread_values)]
            fig_spread.add_trace(go.Scatter(x=merged_df["Date"], y=spread_values, mode="lines", name=f"{c1}-{c2}", hovertext=hover_text, hoverinfo="text", line=dict(color=PRODUCT_CONFIG[symbol]["color"])))

            if not df_news.empty:
                news_cols = df_news.columns.drop('Date')
                news_df_in_view = merged_df.dropna(subset=news_cols, how='all')
                if not news_df_in_view.empty:
                    news_hover_text = news_df_in_view.apply(lambda row: f"<b>Date:</b> {row['Date'].strftime('%Y-%m-%d')}<br><hr>" + "<br>".join(f"<b>{col.replace('_', ' ')}:</b> {row[col]}" for col in news_cols if pd.notna(row[col])), axis=1)
                    fig_spread.add_trace(go.Scatter(x=news_df_in_view['Date'], y=news_df_in_view[c1] - news_df_in_view[c2], mode='markers', name='News', marker=dict(size=10, color='#FF6B6B', symbol='circle'), hovertext=news_hover_text, hoverinfo="text", showlegend=False))

            fig_spread = style_figure(fig_spread, f"{symbol} M1-M2 Spread")
            st.plotly_chart(fig_spread, use_container_width=True, config={'displayModeBar': False})

    # --- Section 3: Fly Analysis (M1-M2-M3) ---
    st.markdown("## Fly Analysis (M1-M2-M3)")
    grid_cols_fly = st.columns(len(st.session_state.selected_products))
    for i, symbol in enumerate(st.session_state.selected_products):
        with grid_cols_fly[i]:
            product_data = all_data.get(symbol)
            if not product_data: continue
            df, contracts = product_data["data"], product_data["contracts"]

            if len(contracts) < 3:
                st.warning(f"Not enough contracts for {symbol} fly.")
                continue
            
            filtered_df = df[(df['Date'].dt.date >= st.session_state.start_date) & (df['Date'].dt.date <= st.session_state.end_date)]
            if filtered_df.empty: continue

            merged_df = pd.merge(filtered_df, df_news, on="Date", how="left") if not df_news.empty else filtered_df
            f1, f2, f3 = contracts[0], contracts[1], contracts[2]
            
            fig_fly = go.Figure()
            fly_values = merged_df[f1] - 2 * merged_df[f2] + merged_df[f3]
            hover_text_fly = [f"<b>Date:</b> {d.strftime('%Y-%m-%d')}<br><b>Fly:</b> {v:.2f}" for d, v in zip(merged_df['Date'], fly_values)]
            fig_fly.add_trace(go.Scatter(x=merged_df["Date"], y=fly_values, mode="lines", name=f"{f1}-{f2}-{f3}", hovertext=hover_text_fly, hoverinfo="text", line=dict(color=PRODUCT_CONFIG[symbol]["color"])))

            if not df_news.empty:
                news_cols = df_news.columns.drop('Date')
                news_df_in_view = merged_df.dropna(subset=news_cols, how='all')
                if not news_df_in_view.empty:
                    news_hover_text = news_df_in_view.apply(lambda row: f"<b>Date:</b> {row['Date'].strftime('%Y-%m-%d')}<br><hr>" + "<br>".join(f"<b>{col.replace('_', ' ')}:</b> {row[col]}" for col in news_cols if pd.notna(row[col])), axis=1)
                    fly_values_news = news_df_in_view[f1] - 2 * news_df_in_view[f2] + news_df_in_view[f3]
                    fig_fly.add_trace(go.Scatter(x=news_df_in_view['Date'], y=fly_values_news, mode='markers', name='News', marker=dict(size=10, color='#FF6B6B', symbol='circle'), hovertext=news_hover_text, hoverinfo="text", showlegend=False))

            fig_fly = style_figure(fig_fly, f"{symbol} M1-M2-M3 Fly")
            st.plotly_chart(fig_fly, use_container_width=True, config={'displayModeBar': False})

# --- TABLE VIEW ---
elif st.session_state.current_view == "Table":
    st.markdown("## Data Table")
    
    for symbol in st.session_state.selected_products:
        product_data = all_data.get(symbol)
        if not product_data: continue
        df = product_data["data"]
        
        st.markdown(f"### {PRODUCT_CONFIG[symbol]['name']}")
        filtered_df = df[(df['Date'].dt.date >= st.session_state.start_date) & (df['Date'].dt.date <= st.session_state.end_date)]
        
        if filtered_df.empty:
            st.warning(f"No data for {symbol} in the selected date range.")
        else:
            st.dataframe(filtered_df, use_container_width=True)

# --- STRATEGY VIEW (Placeholder) ---
elif st.session_state.current_view == "Strategy":
    st.markdown("## Strategy Backtesting")
    st.info("This section is under development. Strategy backtesting and analysis tools will be available here in a future version.")
