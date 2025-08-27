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

# Compact and professional light theme CSS for a high-density layout
st.markdown(
    """
<style>
    /* Main app styling and density */
    .stApp { background-color: #FFFFFF; color: #1E1E1E; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; padding-left: 1rem; padding-right: 1rem; }
    
    /* Header styling */
    .header { background-color: #F0F2F6; padding: 6px 12px; border-radius: 6px; margin-bottom: 16px; border: 1px solid #E0E0E0; }
    
    /* Section headers */
    h2 { color: #1E1E1E; border-bottom: 2px solid #00A8E8; padding-bottom: 8px; margin-top: 20px; font-size: 1.4rem; font-weight: bold; }
    h3 { font-size: 1.1rem; font-weight: bold; color: #333; }
    
    /* Compact Controls */
    .stButton>button { border-radius: 4px; padding: 2px 8px; border: 1px solid #B0B0B0; background-color: #FFFFFF; color: #333; font-weight: 500; transition: all 0.15s; height: 28px; font-size: 12px; }
    .stButton>button:hover { border-color: #00A8E8; color: #00A8E8; }
    div[data-testid="stCheckbox"] label { font-size: 13px; font-weight: 500; }
    .stMultiSelect, .stSelectbox, .stDateInput, .stNumberInput { font-size: 13px; }
    
    /* Align header elements vertically */
    .st-emotion-cache-1f8336m { align-items: end; }
</style>
""",
    unsafe_allow_html=True,
)

# ==================================================================================================
# 1. CENTRAL FILE & PRODUCT CONFIGURATION
# ==================================================================================================
MASTER_EXCEL_FILE = "Futures_Data.xlsx"
NEWS_EXCEL_FILE = "Important_news_date.xlsx"

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
    Returns a tuple: (all_product_data, df_news)
    """
    all_product_data = {}
    if not os.path.exists(MASTER_EXCEL_FILE):
        return {}, pd.DataFrame()

    for symbol, config in PRODUCT_CONFIG.items():
        try:
            df_raw = pd.read_excel(MASTER_EXCEL_FILE, sheet_name=config["sheet"], header=None, engine="openpyxl")
            header_row_index = df_raw[df_raw[0] == "Dates"].index[0] - 1
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
            st.warning(f"Could not load or parse sheet for {config['name']}. Error: {e}")
            continue

    df_news = pd.DataFrame()
    if os.path.exists(NEWS_EXCEL_FILE):
        news_df_raw = pd.read_excel(NEWS_EXCEL_FILE, engine="openpyxl")
        date_col = "Date" if "Date" in news_df_raw.columns else "Dates" if "Dates" in news_df_raw.columns else None
        if date_col:
            news_df_raw.rename(columns={date_col: "Date"}, inplace=True)
            news_df_raw["Date"] = pd.to_datetime(news_df_raw["Date"], errors="coerce")
            df_news = news_df_raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    return all_product_data, df_news


def style_figure(fig: go.Figure, title: str) -> go.Figure:
    """Applies a consistent, professional light theme to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(color="#333", size=16), x=0.5, y=0.95),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="#F9F9F9",
        xaxis=dict(color="#333", gridcolor="#EAEAEA", zeroline=False),
        yaxis=dict(color="#333", gridcolor="#EAEAEA", zeroline=False),
        legend=dict(font=dict(color="#333"), yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def filter_by_date_window(df: pd.DataFrame, start_d: date, end_d: date) -> pd.DataFrame:
    """Filters a dataframe to a given date range."""
    return df[(df["Date"].dt.date >= start_d) & (df["Date"].dt.date <= end_d)]


def nearest_date_on_or_before(df: pd.DataFrame, target_d: date) -> date | None:
    """Finds the most recent available date in the dataframe on or before a target date."""
    subset = df[df["Date"].dt.date <= target_d]
    return subset["Date"].max().date() if not subset.empty else None


def add_news_markers(fig: go.Figure, merged_df: pd.DataFrame, df_news: pd.DataFrame, y_series: pd.Series):
    """Safely adds news markers (bubbles) to a time series chart."""
    if df_news is None or df_news.empty or y_series is None or y_series.empty:
        return fig

    news_cols = [c for c in df_news.columns if c != "Date"]
    if not news_cols:
        return fig

    existing_news_cols = [col for col in news_cols if col in merged_df.columns]
    if not existing_news_cols:
        return fig
        
    news_df_in_view = merged_df.dropna(subset=existing_news_cols, how="all")
    if news_df_in_view.empty:
        return fig

    # Ensure y_series index aligns with news_df_in_view for safe lookup
    y_values_for_news = y_series.loc[news_df_in_view.index]

    news_hover_text = news_df_in_view.apply(
        lambda row: f"<b>Date:</b> {row['Date'].strftime('%Y-%m-%d')}<br><hr>"
        + "<br>".join(f"<b>{col.replace('_', ' ')}:</b> {row[col]}" for col in news_cols if pd.notna(row.get(col))),
        axis=1,
    )

    fig.add_trace(
        go.Scatter(
            x=news_df_in_view["Date"],
            y=y_values_for_news,
            mode="markers", name="News",
            marker=dict(size=8, color="#FF6B6B", symbol="circle", line=dict(width=1, color='#B22222')),
            hovertext=news_hover_text, hoverinfo="text",
            showlegend=False,
        )
    )
    return fig


def spread_series(df: pd.DataFrame, c1: str, c2: str) -> pd.Series:
    """Calculates a simple spread time series."""
    return df[c1] - df[c2]


def fly_series(df: pd.DataFrame, f1: str, f2: str, f3: str) -> pd.Series:
    """Calculates a butterfly spread time series."""
    return df[f1] - 2 * df[f2] + df[f3]

def update_selected_products():
    """Callback function to safely update the selected products list from checkboxes."""
    st.session_state["selected_products"] = [
        s for s in PRODUCT_CONFIG.keys() if st.session_state.get(f"chk_prod_{s}", False)
    ]

# ==================================================================================================
# 3. STATE MANAGEMENT INITIALIZATION
# ==================================================================================================
if "selected_products" not in st.session_state:
    st.session_state["selected_products"] = ["CL", "BZ", "DBI"]
if "start_date" not in st.session_state:
    st.session_state["start_date"] = date.today() - timedelta(days=365)
if "end_date" not in st.session_state:
    st.session_state["end_date"] = date.today()

# ==================================================================================================
# 4. MAIN APP LOGIC AND LAYOUT
# ==================================================================================================
all_data, df_news = load_all_data()

if not all_data:
    st.error(
        f"Master data file not found or is empty: `{MASTER_EXCEL_FILE}`. Please ensure it is in the same directory and correctly formatted."
    )
    st.stop()

# ---------------------------- HEADER CONTROL PANEL ----------------------------
st.markdown('<div class="header">', unsafe_allow_html=True)
header_cols = st.columns([2, 3, 1])

with header_cols[0]:
    st.write("**Products**")
    prod_cols = st.columns(len(PRODUCT_CONFIG))
    for i, (symbol, config) in enumerate(PRODUCT_CONFIG.items()):
        key = f"chk_prod_{symbol}"
        if key not in st.session_state:
            st.session_state[key] = symbol in st.session_state["selected_products"]
        
        prod_cols[i].checkbox(
            symbol,
            value=st.session_state[key],
            key=key,
            on_change=update_selected_products
        )

with header_cols[1]:
    st.write("**Date Range**")
    date_cols = st.columns(2)
    st.session_state.start_date = date_cols[0].date_input("Start Date", value=st.session_state.start_date, key="start_date_picker", label_visibility="collapsed")
    st.session_state.end_date = date_cols[1].date_input("End Date", value=st.session_state.end_date, key="end_date_picker", label_visibility="collapsed")

with header_cols[2]:
    st.write("**Actions**")
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state.selected_products:
    st.warning("Please select at least one product to display.")
    st.stop()

# ==================================================================================================
# 5. UNIFIED DASHBOARD LAYOUT
# ==================================================================================================
prods = st.session_state.selected_products
cols_per_row = min(3, max(1, len(prods)))

# --- Section 1: Outright Curves ---
st.markdown("## Outright Curves")
outright_ctl_cols = st.columns(4)
with outright_ctl_cols[0]:
    available_dates = set()
    for s in prods:
        df_s = all_data.get(s, {}).get("data")
        if df_s is not None and not df_s.empty:
            available_dates.update(df_s["Date"].dt.date.dropna().unique())
    all_dates = sorted(list(available_dates), reverse=True)
    
    overlay_dates = st.multiselect("Select Overlay Dates", options=all_dates, default=[], key="outright_overlay_dates")
with outright_ctl_cols[1]:
    normalize_curves = st.checkbox("Normalize (z)", value=False, key="outright_normalize")
with outright_ctl_cols[2]:
    show_values = st.checkbox("Show Values", value=False, key="outright_show_values")

rows = (len(prods) + cols_per_row - 1) // cols_per_row
idx = 0
for _ in range(rows):
    cols = st.columns(cols_per_row)
    for c in cols:
        if idx >= len(prods): break
        symbol = prods[idx]
        idx += 1
        product_data = all_data.get(symbol)
        if not product_data: continue
        df, contracts = product_data["data"], product_data["contracts"]
        
        dates_to_plot = []
        if overlay_dates:
            for d in overlay_dates:
                nearest = nearest_date_on_or_before(df, d)
                if nearest: dates_to_plot.append(nearest)
        else:
            latest = nearest_date_on_or_before(df, st.session_state.end_date)
            if latest: dates_to_plot.append(latest)
        
        if not dates_to_plot:
            c.warning(f"No data for {symbol} on selected date(s).")
            continue

        fig = go.Figure()
        palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]
        for i, d_plot in enumerate(dates_to_plot):
            row = df[df["Date"].dt.date == d_plot]
            if row.empty: continue
            series = row.iloc[0][contracts].astype(float)
            if normalize_curves:
                series = (series - series.mean()) / (series.std() if series.std() != 0 else 1)
            
            fig.add_trace(go.Scatter(
                x=contracts, y=series.values, mode="lines+markers" + ("+text" if show_values else ""),
                name=str(d_plot), line=dict(color=palette[i % len(palette)], width=2.5),
                text=[f"{v:.2f}" for v in series.values], textposition="top center"
            ))
        
        fig = style_figure(fig, f"{PRODUCT_CONFIG[symbol]['name']} Outright")
        fig.update_yaxes(title_text="Z-Score" if normalize_curves else "Price ($)")
        c.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# --- Section 2: Cross-Product Comparison ---
st.markdown("## Cross-Product Compare")
universe_spreads = [f"{s}: {c1}-{c2}" for s in prods for c1, c2 in zip(all_data[s]['contracts'], all_data[s]['contracts'][1:])]
universe_flies = [f"{s}: {c1}-{c2}-{c3}" for s in prods for c1, c2, c3 in zip(all_data[s]['contracts'], all_data[s]['contracts'][1:], all_data[s]['contracts'][2:])]

compare_ctl_cols = st.columns([2, 2, 1])
with compare_ctl_cols[0]:
    sel_spreads = st.multiselect("Compare Spreads", options=universe_spreads, default=universe_spreads[:2], key="compare_spreads")
with compare_ctl_cols[1]:
    sel_flies = st.multiselect("Compare Flies", options=universe_flies, default=universe_flies[:1], key="compare_flies")
with compare_ctl_cols[2]:
    normalize_ts = st.checkbox("Normalize Series (z)", value=False, key="compare_normalize")

def norm_series(s: pd.Series) -> pd.Series:
    std = s.std()
    return (s - s.mean()) / (std if std not in (0, np.nan) else 1)

compare_chart_cols = st.columns(2)
with compare_chart_cols[0]:
    if sel_spreads:
        figS = go.Figure()
        for item in sel_spreads:
            sym, pair = item.split(":")
            sym, pair = sym.strip(), pair.strip()
            c1x, c2x = pair.split("-")
            df = all_data[sym]["data"]
            sub = filter_by_date_window(df, st.session_state.start_date, st.session_state.end_date)
            if sub.empty: continue
            series = spread_series(sub, c1x, c2x)
            if normalize_ts: series = norm_series(series)
            figS.add_trace(go.Scatter(x=sub["Date"], y=series, mode="lines", name=f"{sym} {pair}", line=dict(width=2, color=PRODUCT_CONFIG[sym]['color'])))
            add_news_markers(figS, pd.merge(sub, df_news, on="Date", how="left"), df_news, series)
        
        figS = style_figure(figS, "Spread Comparison")
        figS.update_yaxes(title_text="Z-Score" if normalize_ts else "Price Diff ($)")
        st.plotly_chart(figS, use_container_width=True, config={'displayModeBar': True})

with compare_chart_cols[1]:
    if sel_flies:
        figF = go.Figure()
        for item in sel_flies:
            sym, trip = item.split(":")
            sym, trip = sym.strip(), trip.strip()
            a, b, c = trip.split("-")
            df = all_data[sym]["data"]
            sub = filter_by_date_window(df, st.session_state.start_date, st.session_state.end_date)
            if sub.empty: continue
            series = fly_series(sub, a, b, c)
            if normalize_ts: series = norm_series(series)
            figF.add_trace(go.Scatter(x=sub["Date"], y=series, mode="lines", name=f"{sym} {trip}", line=dict(width=2, color=PRODUCT_CONFIG[sym]['color'])))
            add_news_markers(figF, pd.merge(sub, df_news, on="Date", how="left"), df_news, series)
        
        figF = style_figure(figF, "Fly Comparison")
        figF.update_yaxes(title_text="Z-Score" if normalize_ts else "Price Diff ($)")
        st.plotly_chart(figF, use_container_width=True, config={'displayModeBar': True})

# --- Section 3: Data Table ---
st.markdown("## Data Table")
for symbol in prods:
    product_data = all_data.get(symbol)
    if not product_data: continue
    df = product_data["data"]
    with st.expander(f"Raw Data for {PRODUCT_CONFIG[symbol]['name']}"):
        filtered_df = filter_by_date_window(df, st.session_state.start_date, st.session_state.end_date)
        if filtered_df.empty:
            st.warning(f"No data for {symbol} in the selected date range.")
        else:
            st.dataframe(filtered_df, use_container_width=True, height=300)
