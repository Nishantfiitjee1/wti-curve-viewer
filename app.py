import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta
import numpy as np

# ==================================================================================================
# PAGE CONFIGURATION AND STYLING
# ==================================================================================================
st.set_page_config(page_title="Futures Curve Analyzer", layout="wide", initial_sidebar_state="expanded")

# Advanced CSS for a compact, professional UI inspired by the reference image
st.markdown("""
<style>
    /* Remove Streamlit's default padding */
    .block-container {
        padding-top: 1rem; padding-bottom: 0rem; padding-left: 2.5rem; padding-right: 2.5rem;
    }
    /* Main control header */
    .control-header {
        background-color: #F0F2F6; padding: 10px 15px; border-radius: 8px;
        margin-bottom: 1rem; border: 1px solid #E0E0E0;
    }
    /* Section titles within header */
    .control-label {
        font-weight: 600; font-size: 0.9rem; color: #333; margin-bottom: 5px;
    }
    /* Custom pill-style checkboxes */
    div[data-testid="stCheckbox"] > label {
        display: inline-block; background-color: #e8eaed; color: #444;
        padding: 4px 12px; border-radius: 16px; margin-right: 8px; margin-bottom: 8px;
        border: 1px solid #ccc; font-size: 0.8rem; transition: all 0.2s; cursor: pointer;
        font-weight: 500;
    }
    div[data-testid="stCheckbox"] > label:hover {
        background-color: #d8dcdf; border-color: #999;
    }
    div[data-testid="stCheckbox"] input:checked + div {
        background-color: #0072B2; color: white; border-color: #005a8c;
    }
    /* Shrink date input widgets */
    div[data-testid="stDateInput"] {
        padding-bottom: 0;
    }
    div[data-testid="stDateInput"] > label {
        font-size: 0.9rem !important; font-weight: 600 !important; color: #333;
    }
    /* Tightly packed columns */
    .row-widget.stColumns {
        gap: 12px;
    }
    /* Chart titles */
    h2 {
        font-size: 1.3rem;
        border-bottom: 2px solid #eee;
        padding-bottom: 5px;
        margin-top: 1rem;
    }
    /* Hide default Streamlit elements */
    #MainMenu, footer { display: none; }
</style>
""", unsafe_allow_html=True)

# ==================================================================================================
# 1. CENTRAL FILE & PRODUCT CONFIGURATION
# ==================================================================================================
MASTER_EXCEL_FILE = "Futures_Data.xlsx"
NEWS_EXCEL_FILE = "Important_news_date.xlsx"

PRODUCT_CONFIG = {
    "CL": {"name": "WTI Crude", "sheet": "WTI_Outright", "color": "#0072B2"},
    "BZ": {"name": "Brent Crude", "sheet": "Brent_outright", "color": "#D55E00"},
    "DBI": {"name": "Dubai Crude", "sheet": "Dubai_Outright", "color": "#009E73"},
}

# ==================================================================================================
# 2. DATA LOADING AND UTILITY FUNCTIONS
# ==================================================================================================
@st.cache_data(show_spinner="Loading all market data...", ttl=3600)
def load_all_data():
    all_product_data = {}
    if not os.path.exists(MASTER_EXCEL_FILE):
        st.error(f"FATAL: Master data file not found: `{MASTER_EXCEL_FILE}`.")
        return {}, pd.DataFrame(), []

    all_dates = set()
    for symbol, config in PRODUCT_CONFIG.items():
        try:
            df_raw = pd.read_excel(MASTER_EXCEL_FILE, sheet_name=config["sheet"], header=None, engine="openpyxl")
            header_row_index = df_raw[df_raw[0] == "Dates"].index[0] - 1
            data_start_row_index = header_row_index + 2
            contracts = [str(x).strip() for x in df_raw.iloc[header_row_index].tolist()[1:] if pd.notna(x) and str(x).strip()]
            col_names = ["Date"] + contracts
            df = df_raw.iloc[data_start_row_index:].copy()
            df.columns = col_names
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            for c in contracts:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date", ascending=False).reset_index(drop=True)
            all_product_data[symbol] = {"data": df, "contracts": contracts}
            all_dates.update(df["Date"].dt.date.unique())
        except Exception as e:
            st.warning(f"Could not load sheet '{config['sheet']}' for {config['name']}. Error: {e}")

    df_news = pd.DataFrame()
    if os.path.exists(NEWS_EXCEL_FILE):
        try:
            news_df_raw = pd.read_excel(NEWS_EXCEL_FILE, engine="openpyxl")
            date_col = "Dates" if "Dates" in news_df_raw.columns else "Date"
            news_df_raw.rename(columns={date_col: "Date"}, inplace=True)
            news_df_raw["Date"] = pd.to_datetime(news_df_raw["Date"], errors="coerce")
            df_news = news_df_raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        except Exception as e:
            st.warning(f"Could not load news file. Error: {e}")

    return all_product_data, df_news, sorted(list(all_dates), reverse=True)

def style_figure(fig, title):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14), x=0.5, y=0.95),
        paper_bgcolor="white", plot_bgcolor="#F9F9F9",
        xaxis=dict(gridcolor="#EAEAEA"), yaxis=dict(gridcolor="#EAEAEA"),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified", margin=dict(l=40, r=20, t=50, b=40)
    )
    return fig

def nearest_date_on_or_before(df, target_d):
    subset = df[df["Date"].dt.date <= target_d]
    return None if subset.empty else subset["Date"].max().date()

def add_news_markers(fig, df_view, df_news):
    if df_news is None or df_news.empty or df_view is None or df_view.empty:
        return
    merged = pd.merge(df_view, df_news, on="Date", how="inner")
    if merged.empty:
        return

    news_hover = merged.apply(
        lambda r: f"<b>{r['Date'].strftime('%Y-%m-%d')}</b><br><hr>" +
                  "<br>".join(f"<b>{c}:</b> {r[c]}" for c in df_news.columns if c != 'Date' and pd.notna(r.get(c))), axis=1)

    fig.add_trace(go.Scatter(
        x=merged["Date"], y=merged.select_dtypes(include=np.number).mean(axis=1) * 0.95,
        mode="markers", name="News",
        marker=dict(size=8, color="#FF6B6B", symbol="diamond", line=dict(width=1, color='black')),
        hovertext=news_hover, hoverinfo="text", showlegend=False
    ))

def spread_series(df, c1, c2): return df[c1] - df[c2]
def fly_series(df, f1, f2, f3): return df[f1] - 2 * df[f2] + df[f3]
def norm_series(s):
    std = s.std()
    return (s - s.mean()) / (std if std != 0 and pd.notna(std) else 1)

# ==================================================================================================
# 3. STATE MANAGEMENT
# ==================================================================================================
def init_state():
    defaults = {
        "selected_products": ["CL", "BZ"],
        "selected_views": ["Outright Curves", "Spread Curves"],
        "start_date": date.today() - timedelta(days=365),
        "end_date": date.today(),
        "picked_one_date": None,
        "picked_multi_dates": [],
        "overlay_enabled": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ==================================================================================================
# 4. MAIN APP EXECUTION
# ==================================================================================================
all_data, df_news, all_available_dates = load_all_data()
init_state()

if not all_data:
    st.stop()

# --- HEADER / CONTROLS ---
with st.container():
    st.markdown('<div class="control-header">', unsafe_allow_html=True)
    # --- Row 1: Products & Views ---
    c1, c2 = st.columns([1.8, 2.2])
    with c1:
        st.markdown('<p class="control-label">Products</p>', unsafe_allow_html=True)
        prod_cols = st.columns(len(PRODUCT_CONFIG))
        for i, (symbol, config) in enumerate(PRODUCT_CONFIG.items()):
            prod_cols[i].checkbox(config['name'], value=(symbol in st.session_state.selected_products), key=f"prod_{symbol}")
        st.session_state.selected_products = [s for s, c in PRODUCT_CONFIG.items() if st.session_state[f"prod_{s}"]]

    with c2:
        st.markdown('<p class="control-label">Views</p>', unsafe_allow_html=True)
        VIEWS = ["Outright Curves", "Spread Curves", "Per-Product Time Series", "Cross-Product Compare"]
        view_cols = st.columns(len(VIEWS))
        for i, view in enumerate(VIEWS):
            view_cols[i].checkbox(view, value=(view in st.session_state.selected_views), key=f"view_{view}")
        st.session_state.selected_views = [v for v in VIEWS if st.session_state[f"view_{v}"]]

    st.markdown("---") # Visual separator

    # --- Row 2: Dates & Range ---
    c3, c4, c5 = st.columns([1, 1.8, 1.2])
    with c3:
        latest_date = st.session_state.picked_one_date or all_available_dates[0]
        st.date_input("Curve Date", value=latest_date, key="picked_one_date", min_value=all_available_dates[-1], max_value=all_available_dates[0])
        st.checkbox("Overlay Dates", key="overlay_enabled")

    with c4:
        if st.session_state.overlay_enabled:
            st.multiselect("Select dates to overlay", options=all_available_dates, key="picked_multi_dates")
        else:
            st.session_state.picked_multi_dates = []

    with c5:
        st.date_input("Time Series Start", key="start_date")
        st.date_input("Time Series End", key="end_date")

    st.markdown("</div>", unsafe_allow_html=True)

# --- Stop if no products are selected ---
if not st.session_state.selected_products:
    st.info("Please select at least one product to begin analysis.")
    st.stop()

# --- CHARTING SECTIONS ---
prods = st.session_state.selected_products
cols_per_row = min(3, max(1, len(prods)))
dates_to_plot = st.session_state.picked_multi_dates if st.session_state.overlay_enabled and st.session_state.picked_multi_dates else [st.session_state.picked_one_date]

# --- View 1: Outright Curves ---
if "Outright Curves" in st.session_state.selected_views:
    st.header("Outright Curves")
    for symbol in prods:
        product_data = all_data.get(symbol)
        if not product_data: continue
        df, contracts = product_data["data"], product_data["contracts"]
        fig = go.Figure()
        for d in dates_to_plot:
            d_use = nearest_date_on_or_before(df, d)
            if not d_use: continue
            row = df[df["Date"].dt.date == d_use].iloc[0]
            s = row[contracts].astype(float)
            fig.add_trace(go.Scatter(x=contracts, y=s.values, mode="lines+markers", name=str(d_use), line=dict(color=PRODUCT_CONFIG[symbol]['color'])))
        style_figure(fig, f"{PRODUCT_CONFIG[symbol]['name']} Outright")
        st.plotly_chart(fig, use_container_width=True)

# --- View 2: Spread Curves ---
if "Spread Curves" in st.session_state.selected_views:
    st.header("Spread Curves")
    for symbol in prods:
        product_data = all_data.get(symbol)
        if not product_data or len(product_data["contracts"]) < 2: continue
        df, contracts = product_data["data"], product_data["contracts"]
        fig = go.Figure()
        for d in dates_to_plot:
            d_use = nearest_date_on_or_before(df, d)
            if not d_use: continue
            row = df[df["Date"].dt.date == d_use].iloc[0]
            spread_labels = [f"M{i+1}-M{i+2}" for i in range(len(contracts) - 1)]
            spread_values = [row[contracts[i]] - row[contracts[i+1]] for i in range(len(contracts) - 1)]
            s = pd.Series(spread_values, index=spread_labels)
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers", name=str(d_use), line=dict(color=PRODUCT_CONFIG[symbol]['color'])))
        style_figure(fig, f"{PRODUCT_CONFIG[symbol]['name']} Spread Curve").update_yaxes(title_text="Price Diff ($)")
        st.plotly_chart(fig, use_container_width=True)

# --- View 3: Per-Product Time Series ---
if "Per-Product Time Series" in st.session_state.selected_views:
    st.header("Per-Product Time Series Analysis")
    sc1, sc2 = st.columns(2)
    for symbol in prods:
        df, contracts = all_data[symbol]["data"], all_data[symbol]["contracts"]
        sub_df = df[(df["Date"].dt.date >= st.session_state.start_date) & (df["Date"].dt.date <= st.session_state.end_date)]
        if sub_df.empty: continue

        with sc1: # Spreads
            if len(contracts) < 2: continue
            st.subheader(f"{PRODUCT_CONFIG[symbol]['name']} Spreads")
            default_spreads = [f"{contracts[i]}-{contracts[i+1]}" for i in range(min(3, len(contracts) - 1))]
            spread_choices = st.multiselect("", options=[f"{c1}-{c2}" for i, c1 in enumerate(contracts) for c2 in contracts[i+1:]], default=default_spreads, key=f"spread_multi_{symbol}")
            if spread_choices:
                fig = go.Figure()
                for pair in spread_choices:
                    c1, c2 = [p.strip() for p in pair.split('-')]
                    series = spread_series(sub_df, c1, c2)
                    fig.add_trace(go.Scatter(x=sub_df["Date"], y=series, mode="lines", name=f"{c1}-{c2}"))
                add_news_markers(fig, sub_df, df_news)
                style_figure(fig, "").update_layout(showlegend=True).update_yaxes(title_text="Price Diff ($)")
                st.plotly_chart(fig, use_container_width=True, height=400)

        with sc2: # Flies
            if len(contracts) < 3: continue
            st.subheader(f"{PRODUCT_CONFIG[symbol]['name']} Flies")
            default_flies = [f"{contracts[i]}-{contracts[i+1]}-{contracts[i+2]}" for i in range(min(2, len(contracts) - 2))]
            fly_choices = st.multiselect("", options=[f"{a}-{b}-{c}" for i,a in enumerate(contracts) for j,b in enumerate(contracts[i+1:],start=i+1) for c in contracts[j+1:]], default=default_flies, key=f"fly_multi_{symbol}")
            if fly_choices:
                fig = go.Figure()
                for item in fly_choices:
                    f1, f2, f3 = [p.strip() for p in item.split('-')]
                    series = fly_series(sub_df, f1, f2, f3)
                    fig.add_trace(go.Scatter(x=sub_df["Date"], y=series, mode="lines", name=f"{f1}-{f2}-{f3}"))
                add_news_markers(fig, sub_df, df_news)
                style_figure(fig, "").update_layout(showlegend=True).update_yaxes(title_text="Price Diff ($)")
                st.plotly_chart(fig, use_container_width=True, height=400)

# --- View 4: Cross-Product Compare ---
if "Cross-Product Compare" in st.session_state.selected_views:
    st.header("Cross-Product Comparison")
    universe_spreads = [f"{s}: {c1}-{c2}" for s in prods for i,c1 in enumerate(all_data[s]["contracts"]) for c2 in all_data[s]["contracts"][i+1:]]
    universe_flies = [f"{s}: {a}-{b}-{c}" for s in prods for i,a in enumerate(all_data[s]["contracts"]) for j,b in enumerate(all_data[s]["contracts"][i+1:],start=i+1) for c in all_data[s]["contracts"][j+1:]]
    
    normalize_ts = st.checkbox("Normalize all series (z-score)", key="normalize_cross")

    st.subheader("Compare Spreads")
    sel_spreads = st.multiselect("Select spreads", options=universe_spreads, default=universe_spreads[:2], key="sel_spreads_cross")
    if sel_spreads:
        figS = go.Figure()
        for item in sel_spreads:
            sym, pair = item.split(":")
            df = all_data[sym.strip()]["data"]
            sub = df[(df["Date"].dt.date >= st.session_state.start_date) & (df["Date"].dt.date <= st.session_state.end_date)]
            if sub.empty: continue
            cA, cB = [p.strip() for p in pair.split("-")]
            series = spread_series(sub, cA, cB)
            if normalize_ts: series = norm_series(series)
            figS.add_trace(go.Scatter(x=sub["Date"], y=series, mode="lines", name=f"{sym} {pair.strip()}", line=dict(width=2)))
        add_news_markers(figS, sub, df_news)
        style_figure(figS, "").update_yaxes(title_text="Z-score" if normalize_ts else "Price Diff ($)")
        st.plotly_chart(figS, use_container_width=True)

    st.subheader("Compare Flies")
    sel_flies = st.multiselect("Select flies", options=universe_flies, default=universe_flies[:1], key="sel_flies_cross")
    if sel_flies:
        figF = go.Figure()
        for item in sel_flies:
            sym, trip = item.split(":")
            df = all_data[sym.strip()]["data"]
            sub = df[(df["Date"].dt.date >= st.session_state.start_date) & (df["Date"].dt.date <= st.session_state.end_date)]
            if sub.empty: continue
            a, b, c = [p.strip() for p in trip.split("-")]
            series = fly_series(sub, a, b, c)
            if normalize_ts: series = norm_series(series)
            figF.add_trace(go.Scatter(x=sub["Date"], y=series, mode="lines", name=f"{sym} {trip.strip()}", line=dict(width=2)))
        add_news_markers(figF, sub, df_news)
        style_figure(figF, "").update_yaxes(title_text="Z-score" if normalize_ts else "Price Diff ($)")
        st.plotly_chart(figF, use_container_width=True)

