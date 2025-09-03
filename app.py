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
st.set_page_config(page_title="Futures Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM STYLING ---
# Professional styling for a denser, more trader-focused layout.
st.markdown(
    """
<style>
/* General App Density */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
/* Main Header Control Panel */
.control-header {
    background-color: #F0F2F6;
    padding: 12px 15px;
    border-radius: 8px;
    margin-bottom: 20px;
    border: 1px solid #E0E0E0;
}
/* Section headers (Products, Views, etc.) */
.control-label {
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 8px;
    color: #333;
}
/* Compact date inputs */
div[data-testid="stDateInput"] > label {
    font-size: 13px !important;
    font-weight: 500 !important;
}
div[data-testid="stDateInput"] {
    height: 60px; /* Force smaller height */
}
/* Custom checkbox pills for Products and Views */
.stCheckbox {
    padding: 0;
    margin: 0;
}
.stCheckbox label {
    display: inline-block;
    background-color: #e8eaed;
    color: #333;
    padding: 4px 10px;
    border-radius: 15px;
    margin-right: 8px;
    margin-bottom: 8px;
    border: 1px solid #ccc;
    font-size: 13px;
    transition: all 0.2s;
    cursor: pointer;
}
.stCheckbox label:hover {
    background-color: #dde1e6;
    border-color: #999;
}
/* Style for checked state */
.stCheckbox input:checked + div {
    background-color: #0072B2; /* WTI Blue */
    color: white;
    border-color: #005a8c;
}
/* Selected product badges under header */
.selected-badges {
    margin-bottom: 15px;
}
.badge {
    display: inline-block;
    padding: 4px 10px;
    margin-right: 6px;
    border-radius: 12px;
    background: #eee;
    font-size: 12px;
    border: 1px solid #ddd;
    font-weight: 500;
}
/* Reduce default column gaps for charts */
.row-widget.stColumns .stColumn {
    padding-left: 10px;
    padding-right: 10px;
}
/* Hide Streamlit default hamburger menu and footer */
#MainMenu {display: none;}
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ==================================================================================================
# 1. CENTRAL FILE & PRODUCT CONFIGURATION
# ==================================================================================================
# **CORRECTED:** Pointing to the master Excel files now.
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
    **CORRECTED:** Loads data from sheets within the master Excel files.
    """
    all_product_data = {}
    
    # Check for master futures file existence
    if not os.path.exists(MASTER_EXCEL_FILE):
        st.error(f"Master data file not found: `{MASTER_EXCEL_FILE}`. Please ensure it's in the same directory.")
        return {}, pd.DataFrame()

    for symbol, config in PRODUCT_CONFIG.items():
        try:
            # Read from the specific sheet in the master Excel file
            df_raw = pd.read_excel(MASTER_EXCEL_FILE, sheet_name=config["sheet"], header=None, engine="openpyxl")
            
            # Dynamically find header and data start (same logic as your original code)
            header_row_index = df_raw[df_raw[0] == "Dates"].index[0] - 1
            data_start_row_index = header_row_index + 2

            contracts = [
                str(x).strip() for x in df_raw.iloc[header_row_index].tolist()[1:]
                if pd.notna(x) and str(x).strip() != ""
            ]
            col_names = ["Date"] + contracts

            df = df_raw.iloc[data_start_row_index:].copy()
            df.columns = col_names
            
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            for c in contracts:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            
            df = df.dropna(subset=["Date"]).sort_values("Date", ascending=False).reset_index(drop=True)
            all_product_data[symbol] = {"data": df, "contracts": contracts}

        except Exception as e:
            st.error(f"Could not load/parse sheet '{config['sheet']}' for {config['name']}. Error: {e}")
            continue

    df_news = pd.DataFrame()
    if os.path.exists(NEWS_EXCEL_FILE):
        try:
            news_df_raw = pd.read_excel(NEWS_EXCEL_FILE, engine="openpyxl")
            date_col = "Dates" if "Dates" in news_df_raw.columns else "Date"
            news_df_raw.rename(columns={date_col: "Date"}, inplace=True)
            news_df_raw["Date"] = pd.to_datetime(news_df_raw["Date"], errors="coerce")
            df_news = news_df_raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        except Exception as e:
            st.warning(f"Could not load news file '{NEWS_EXCEL_FILE}'. Error: {e}")

    return all_product_data, df_news


def style_figure(fig: go.Figure, title: str) -> go.Figure:
    """Applies a consistent style to all Plotly figures."""
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
    """Filters a DataFrame to a given date range."""
    return df[(df["Date"].dt.date >= start_d) & (df["Date"].dt.date <= end_d)]

def nearest_date_on_or_before(df: pd.DataFrame, target_d: date) -> date | None:
    """Finds the latest available date in the DataFrame on or before a target date."""
    subset = df[df["Date"].dt.date <= target_d]
    return None if subset.empty else subset["Date"].max().date()

def add_news_markers(fig: go.Figure, df_in_view: pd.DataFrame, df_news: pd.DataFrame, y_series: pd.Series | None = None):
    """Adds news event markers to a Plotly figure."""
    if df_news is None or df_news.empty or df_in_view is None or df_in_view.empty:
        return fig

    merged_data = pd.merge(df_in_view, df_news, on="Date", how="inner")
    if merged_data.empty:
        return fig

    news_cols = [c for c in df_news.columns if c not in ["Date", "Link"]]
    news_hover_text = merged_data.apply(
        lambda row: f"<b>{row['Date'].strftime('%Y-%m-%d')}</b><br><hr>" +
                    "<br>".join(f"<b>{col}:</b> {row[col]}" for col in news_cols if pd.notna(row.get(col))),
        axis=1
    )

    y_values = y_series.loc[merged_data.index] if y_series is not None and not y_series.empty else None

    fig.add_trace(
        go.Scatter(
            x=merged_data["Date"],
            y=y_values,
            mode="markers",
            name="News",
            marker=dict(size=9, color="#FF6B6B", symbol="diamond-dot", line=dict(width=1, color='black')),
            hovertext=news_hover_text,
            hoverinfo="text",
            showlegend=True,
        )
    )
    return fig

# Calculation Functions
def spread_series(df: pd.DataFrame, c1: str, c2: str) -> pd.Series:
    return df[c1] - df[c2]

def fly_series(df: pd.DataFrame, f1: str, f2: str, f3: str) -> pd.Series:
    return df[f1] - 2 * df[f2] + df[f3]

# ==================================================================================================
# 3. STATE MANAGEMENT INITIALIZATION
# ==================================================================================================
if "selected_products" not in st.session_state:
    st.session_state.selected_products = ["CL", "BZ"]
if "selected_views" not in st.session_state:
    st.session_state.selected_views = ["Outright", "Spreads"]
if "start_date" not in st.session_state:
    st.session_state.start_date = date.today() - timedelta(days=365)
if "end_date" not in st.session_state:
    st.session_state.end_date = date.today()
if "picked_one_date" not in st.session_state:
    st.session_state.picked_one_date = None
if "picked_multi_dates" not in st.session_state:
    st.session_state.picked_multi_dates = []
if "show_table" not in st.session_state:
    st.session_state.show_table = False

# ==================================================================================================
# 4. LOAD DATA
# ==================================================================================================
all_data, df_news = load_all_data()

if not all_data:
    st.error("Master data files not found or failed to load. The application cannot continue.")
    st.stop()

# ==================================================================================================
# 5. HEADER CONTROL PANEL
# ==================================================================================================
st.markdown('<div class="control-header">', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns([1.5, 1.5, 2, 2, 1.2])

# --- Products Selection ---
with c1:
    st.markdown('<p class="control-label">Products</p>', unsafe_allow_html=True)
    for symbol in PRODUCT_CONFIG:
        st.checkbox(PRODUCT_CONFIG[symbol]['name'], value=(symbol in st.session_state.selected_products), key=f"prod_{symbol}")
    
    st.session_state.selected_products = [s for s in PRODUCT_CONFIG if st.session_state[f"prod_{s}"]]

# --- Views Selection ---
with c2:
    st.markdown('<p class="control-label">Views</p>', unsafe_allow_html=True)
    VIEWS = ["Outright", "Spreads", "Time Series"]
    for view in VIEWS:
        st.checkbox(view, value=(view in st.session_state.selected_views), key=f"view_{view}")

    st.session_state.selected_views = [v for v in VIEWS if st.session_state[f"view_{v}"]]

# --- Date Selection ---
with c3:
    st.markdown('<p class="control-label">Curve Date</p>', unsafe_allow_html=True)
    
    # Set a sensible default for the single date picker
    latest_available_date = nearest_date_on_or_before(all_data[list(all_data.keys())[0]]["data"], date.today())
    
    st.date_input("Select Date", value=st.session_state.picked_one_date or latest_available_date or date.today(), key="picked_one_date")

    # The multiselect for overlay is now directly tied to session state
    if st.checkbox("Overlay Dates", value=bool(st.session_state.picked_multi_dates), key="overlay_toggle"):
        all_available_dates = sorted(list(set(all_data[list(all_data.keys())[0]]["data"]["Date"].dt.date)), reverse=True)
        st.multiselect(
            "Select dates to overlay",
            options=all_available_dates,
            default=st.session_state.picked_multi_dates,
            key="picked_multi_dates"
        )
    else:
        # If checkbox is unchecked, clear the multi-date selection
        if st.session_state.overlay_toggle is False:
             st.session_state.picked_multi_dates = []

# --- Time Series Range ---
with c4:
    st.markdown('<p class="control-label">Time Series Range</p>', unsafe_allow_html=True)
    st.date_input("Start", value=st.session_state.start_date, key="start_date")
    st.date_input("End", value=st.session_state.end_date, key="end_date")

# --- Tools ---
with c5:
    st.markdown('<p class="control-label">Tools</p>', unsafe_allow_html=True)
    st.button("Show/Hide Table", on_click=lambda: st.session_state.update(show_table=not st.session_state.show_table))

st.markdown("</div>", unsafe_allow_html=True)

# --- Stop if no products are selected ---
if not st.session_state.selected_products:
    st.warning("Please select at least one product to begin analysis.")
    st.stop()

# --- Show selected product badges ---
selected_badges_html = "<div class='selected-badges'>"
for sym in st.session_state.selected_products:
    color = PRODUCT_CONFIG[sym]['color']
    selected_badges_html += f"<span class='badge' style='border-left: 5px solid {color};'>{PRODUCT_CONFIG[sym]['name']}</span>"
selected_badges_html += "</div>"
st.markdown(selected_badges_html, unsafe_allow_html=True)

# ==================================================================================================
# 6. DYNAMIC CHARTING AREA
# ==================================================================================================
prods = st.session_state.selected_products
cols_per_row = min(3, max(1, len(prods)))
single_picked = st.session_state.get("picked_one_date", None)
multi_dates = st.session_state.get("picked_multi_dates", [])

# --- OUTRIGHT CURVES ---
if "Outright" in st.session_state.selected_views:
    st.markdown("## Outright Curves", help="Shows the price of all contracts for a given day.")
    norm_out, show_vals_out = st.columns(2)
    normalize_curves = norm_out.checkbox("Normalize curves (z-score)", key="normalize_outright")
    show_values = show_vals_out.checkbox("Show point values", key="show_values_outright")

    rows = (len(prods) + cols_per_row - 1) // cols_per_row
    for i in range(rows):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i * cols_per_row + j
            if idx >= len(prods): break
            
            with cols[j]:
                symbol = prods[idx]
                product_data = all_data.get(symbol)
                if not product_data: continue
                df, contracts = product_data["data"], product_data["contracts"]

                sel_dates = [nearest_date_on_or_before(df, d) for d in multi_dates] if multi_dates else [nearest_date_on_or_before(df, single_picked)]
                sel_dates = [d for d in sel_dates if d]

                if not sel_dates:
                    st.warning(f"No data available for {symbol} on the selected date(s).")
                    continue

                fig = go.Figure()
                for d_plot in sel_dates:
                    row = df[df["Date"].dt.date == d_plot]
                    if row.empty: continue
                    
                    s = row.iloc[0][contracts].astype(float)
                    if normalize_curves:
                        s = (s - s.mean()) / (s.std() if s.std() != 0 else 1)
                    
                    fig.add_trace(go.Scatter(
                        x=contracts, y=s.values,
                        mode="lines+markers" + ("+text" if show_values else ""),
                        name=str(d_plot),
                        line=dict(color=PRODUCT_CONFIG[symbol]["color"], width=2.5),
                        text=[f"{val:.2f}" for val in s.values] if show_values else None,
                        textposition="top center"
                    ))
                
                title = f"{PRODUCT_CONFIG[symbol]['name']} Outright"
                style_figure(fig, title).update_yaxes(title_text="Z-score" if normalize_curves else "Price ($)")
                st.plotly_chart(fig, use_container_width=True)

# --- SPREAD CURVES (NEW SECTION) ---
if "Spreads" in st.session_state.selected_views:
    st.markdown("## Spread Curves", help="Shows the spread between consecutive contracts (e.g., M1-M2, M2-M3) for a given day.")
    norm_spread, show_vals_spread = st.columns(2)
    normalize_spreads = norm_spread.checkbox("Normalize curves (z-score)", key="normalize_spreads")
    show_values_spreads = show_vals_spread.checkbox("Show point values", key="show_values_spreads")
    
    rows = (len(prods) + cols_per_row - 1) // cols_per_row
    for i in range(rows):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i * cols_per_row + j
            if idx >= len(prods): break
            
            with cols[j]:
                symbol = prods[idx]
                product_data = all_data.get(symbol)
                if not product_data or len(product_data["contracts"]) < 2: continue
                df, contracts = product_data["data"], product_data["contracts"]

                sel_dates = [nearest_date_on_or_before(df, d) for d in multi_dates] if multi_dates else [nearest_date_on_or_before(df, single_picked)]
                sel_dates = [d for d in sel_dates if d]

                if not sel_dates:
                    st.warning(f"No data for {symbol} on selected date(s).")
                    continue

                fig = go.Figure()
                for d_plot in sel_dates:
                    row = df[df["Date"].dt.date == d_plot]
                    if row.empty: continue
                    
                    # Calculate spread values and labels
                    spread_labels = [f"M{i+1}-M{i+2}" for i in range(len(contracts) - 1)]
                    spread_values = [row.iloc[0][contracts[i]] - row.iloc[0][contracts[i+1]] for i in range(len(contracts) - 1)]
                    s = pd.Series(spread_values, index=spread_labels)

                    if normalize_spreads:
                        s = (s - s.mean()) / (s.std() if s.std() != 0 else 1)
                    
                    fig.add_trace(go.Scatter(
                        x=s.index, y=s.values,
                        mode="lines+markers" + ("+text" if show_values_spreads else ""),
                        name=str(d_plot),
                        line=dict(color=PRODUCT_CONFIG[symbol]["color"], width=2.5),
                        text=[f"{val:.2f}" for val in s.values] if show_values_spreads else None,
                        textposition="top center"
                    ))
                
                title = f"{PRODUCT_CONFIG[symbol]['name']} Spread Curve"
                style_figure(fig, title).update_yaxes(title_text="Z-score" if normalize_spreads else "Price Diff ($)")
                st.plotly_chart(fig, use_container_width=True)

# --- TIME SERIES ANALYSIS ---
if "Time Series" in st.session_state.selected_views:
    st.markdown("## Time Series Analysis", help="Analyze spreads and flies over the selected date range.")
    
    # --- Quick Spreads & Flies per product ---
    st.markdown("### Per-Product Analysis")
    qcols = st.columns(2)
    with qcols[0]:
        st.markdown("#### Spreads")
        for symbol in prods:
            df, contracts = all_data[symbol]["data"], all_data[symbol]["contracts"]
            if len(contracts) < 2: continue
            
            default_pairs = [f"{contracts[i]}-{contracts[i+1]}" for i in range(min(3, len(contracts)-1))]
            choices = st.multiselect(f"{symbol} spreads", options=[f"{c1}-{c2}" for i, c1 in enumerate(contracts) for c2 in contracts[i+1:]], default=default_pairs, key=f"spread_multi_{symbol}")
            
            if choices:
                sub = filter_by_date_window(df, st.session_state.start_date, st.session_state.end_date)
                if sub.empty: continue
                
                fig = go.Figure()
                series_for_news = pd.Series(dtype=float)
                for pair in choices:
                    a, b = [p.strip() for p in pair.split("-")]
                    series = spread_series(sub, a, b)
                    series_for_news = series
                    fig.add_trace(go.Scatter(x=sub["Date"], y=series, mode="lines", name=f"{symbol} {a}-{b}", line=dict(color=PRODUCT_CONFIG[symbol]["color"], width=1.6)))
                
                add_news_markers(fig, sub, df_news, series_for_news)
                style_figure(fig, f"{symbol} Spreads").update_yaxes(title_text="Price Diff ($)")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with qcols[1]:
        st.markdown("#### Flies")
        for symbol in prods:
            df, contracts = all_data[symbol]["data"], all_data[symbol]["contracts"]
            if len(contracts) < 3: continue

            default_flies = [f"{contracts[i]}-{contracts[i+1]}-{contracts[i+2]}" for i in range(min(2, len(contracts)-2))]
            choices = st.multiselect(f"{symbol} flies", options=[f"{a}-{b}-{c}" for i,a in enumerate(contracts) for j,b in enumerate(contracts[i+1:],start=i+1) for c in contracts[j+1:]], default=default_flies, key=f"fly_multi_{symbol}")
            
            if choices:
                sub = filter_by_date_window(df, st.session_state.start_date, st.session_state.end_date)
                if sub.empty: continue
                
                fig = go.Figure()
                series_for_news = pd.Series(dtype=float)
                for item in choices:
                    f1, f2, f3 = [p.strip() for p in item.split("-")]
                    series = fly_series(sub, f1, f2, f3)
                    series_for_news = series
                    fig.add_trace(go.Scatter(x=sub["Date"], y=series, mode="lines", name=f"{symbol} {f1}-{f2}-{f3}", line=dict(color=PRODUCT_CONFIG[symbol]["color"], width=1.6)))
                
                add_news_markers(fig, sub, df_news, series_for_news)
                style_figure(fig, f"{symbol} Flies").update_yaxes(title_text="Price Diff ($)")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # --- Cross-product Compare area ---
    st.markdown("### Cross-Product Comparison")
    universe_spreads = [f"{s}: {c1}-{c2}" for s in prods for i,c1 in enumerate(all_data[s]["contracts"]) for c2 in all_data[s]["contracts"][i+1:]]
    
    sel_spreads = st.multiselect("Select spreads to compare", options=universe_spreads, default=universe_spreads[:2] if len(universe_spreads)>1 else universe_spreads, key="sel_spreads_cross")
    normalize_ts = st.checkbox("Normalize selected series (z-score)", value=False, key="normalize_cross")

    def norm_series(s: pd.Series) -> pd.Series:
        std = s.std()
        return (s - s.mean()) / (std if std != 0 and pd.notna(std) else 1)

    if sel_spreads:
        figS = go.Figure()
        last_sub_df = pd.DataFrame()
        for item in sel_spreads:
            try:
                sym, pair = item.split(":")
                sym, pair = sym.strip(), pair.strip()
                cA, cB = [p.strip() for p in pair.split("-")]
            except ValueError:
                continue # Skip malformed items

            df = all_data[sym]["data"]
            sub = filter_by_date_window(df, st.session_state.start_date, st.session_state.end_date)
            if sub.empty: continue
            last_sub_df = sub
            
            series = spread_series(sub, cA, cB)
            if normalize_ts: series = norm_series(series)
            
            figS.add_trace(go.Scatter(x=sub["Date"], y=series, mode="lines", name=f"{sym} {cA}-{cB}", line=dict(width=2)))
        
        add_news_markers(figS, last_sub_df, df_news)
        style_figure(figS, "Cross Product Spread Comparison").update_yaxes(title_text="Z-score" if normalize_ts else "Price Diff ($)")
        st.plotly_chart(figS, use_container_width=True)


# ==================================================================================================
# 7. DATA TABLE (TOGGLEABLE)
# ==================================================================================================
if st.session_state["show_table"]:
    st.markdown("## Data Tables")
    for symbol in prods:
        product_data = all_data.get(symbol)
        if not product_data: continue
        df = product_data["data"]
        st.markdown(f"### {PRODUCT_CONFIG[symbol]['name']}")
        filtered_df = filter_by_date_window(df, st.session_state.start_date, st.session_state.end_date)
        if filtered_df.empty:
            st.caption(f"No data for {symbol} in the selected range.")
        else:
            st.dataframe(filtered_df, use_container_width=True)
