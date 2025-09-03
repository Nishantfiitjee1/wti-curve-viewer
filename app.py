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

# Compact professional styling and denser layout to use screen pixels efficiently
st.markdown(
    """
<style>
/* App density & fonts */
body, .block-container { padding-top: 8px; padding-bottom: 8px; }
/* Header */
.header { background-color: #F0F2F6; padding: 6px 10px; border-radius: 6px; margin-bottom: 15px; border: 1px solid #E0E0E0; }
/* Small compact controls */
.stButton>button { border-radius: 4px; padding: 2px 6px; border: 1px solid #B0B0B0; background-color: #FFFFFF; color: #111; font-size: 12px; height: 26px; }
div[data-testid="stCheckbox"] label, .stSelectbox, .stMultiSelect, .stDateInput, .stNumberInput { font-size: 12px; }
/* Product pills inline and dense */
.product-pill {
  display:inline-block;
  margin:0 6px 2px 0;
  padding:4px 8px;
  border-radius:12px;
  background:#f5f7fa;
  font-size:12px;
  border:1px solid #ddd;
}
.product-pill.selected { background:#00A8E8; color:white; border-color:#0091d6; }
/* Selected names overlay (small badge group) */
.selected-badges { position: relative; margin-bottom: 6px; }
.badge { display: inline-block; padding: 4px 8px; margin-right:6px; border-radius:12px; background:#eee; font-size:12px; border:1px solid #ddd; }
.badge.color { background: #00A8E8; color: #fff; border-color:#0087c9; }
/* Tight dataframe area */
.element-container .stDataFrame { padding: 6px 6px 10px 6px; }
/* Full-width "Maximize" button appearance */
.max-btn { width:100%; padding:8px 10px; font-weight:600; font-size:13px; }
/* Reduce default column gaps */
.row-widget.stColumns .stColumn { padding-left:6px; padding-right:6px; }
/* Hide Streamlit default footer (optional) */
/* footer {display:none;} */
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
    all_product_data = {}
    if not os.path.exists(MASTER_EXCEL_FILE):
        return {}, pd.DataFrame()
    for symbol, config in PRODUCT_CONFIG.items():
        try:
            df_raw = pd.read_excel(MASTER_EXCEL_FILE, sheet_name=config["sheet"], header=None, engine="openpyxl")
            header_row_index = df_raw[df_raw[0] == "Dates"].index[0] - 1
            data_start_row_index = header_row_index + 2

            contracts = [
                str(x).strip()
                for x in df_raw.iloc[header_row_index].tolist()[1:]
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
            st.warning(f"Could not load/parse sheet for {config['name']}. Error: {e}")
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
    if subset.empty:
        return None
    return subset["Date"].max().date()


def add_news_markers(fig: go.Figure, merged_df: pd.DataFrame, df_news: pd.DataFrame, y_series: pd.Series | None = None):
    """
    Safe news markers add. y_series optional — if provided align markers to series values.
    """
    if df_news is None or df_news.empty:
        return fig
    news_cols = [c for c in df_news.columns if c != "Date"]
    if not news_cols:
        return fig

    # If merged_df does not have news cols, attempt to merge on Date and dropna safely
    if merged_df is None or merged_df.empty:
        return fig

    # if merged_df already contains news columns
    if set(news_cols).issubset(set(merged_df.columns)):
        news_df_in_view = merged_df.dropna(subset=news_cols, how="all")
    else:
        joined = pd.merge(merged_df, df_news, on="Date", how="left", suffixes=("", "_news"))
        # find which news cols exist now
        existing_news = [c for c in news_cols if c in joined.columns]
        if not existing_news:
            return fig
        news_df_in_view = joined.dropna(subset=existing_news, how="all")

    if news_df_in_view.empty:
        return fig

    if y_series is None:
        # place markers slightly above maximum of plotted data if possible
        try:
            y_val = merged_df.select_dtypes(include=[np.number]).max().max()
            y_series = pd.Series(index=news_df_in_view.index, dtype=float)
            y_series[:] = (y_val if np.isfinite(y_val) else 0) * 0.98
        except Exception:
            y_series = pd.Series(index=news_df_in_view.index, dtype=float)
            y_series[:] = np.nan

    news_hover_text = news_df_in_view.apply(
        lambda row: f"<b>Date:</b> {row['Date'].strftime('%Y-%m-%d')}<br><hr>"
        + "<br>".join(f"<b>{col.replace('_', ' ')}:</b> {row[col]}" for col in news_cols if pd.notna(row.get(col))),
        axis=1,
    )

    fig.add_trace(
        go.Scatter(
            x=news_df_in_view["Date"],
            y=y_series.loc[news_df_in_view.index] if y_series is not None else None,
            mode="markers",
            name="News",
            marker=dict(size=8, color="#FF6B6B", symbol="diamond"),
            hovertext=news_hover_text,
            hoverinfo="text",
            showlegend=False,
        )
    )
    return fig


def spread_series(df: pd.DataFrame, c1: str, c2: str) -> pd.Series:
    return df[c1] - df[c2]


def fly_series(df: pd.DataFrame, f1: str, f2: str, f3: str) -> pd.Series:
    return df[f1] - 2 * df[f2] + df[f3]


# ==================================================================================================
# 3. STATE MANAGEMENT INITIALIZATION
# ==================================================================================================
if "selected_products" not in st.session_state:
    st.session_state["selected_products"] = ["CL", "BZ", "DBI"]
if "start_date" not in st.session_state:
    st.session_state["start_date"] = date.today() - timedelta(days=365)
if "end_date" not in st.session_state:
    st.session_state["end_date"] = date.today()
if "picked_one_date" not in st.session_state:
    st.session_state["picked_one_date"] = None
if "picked_multi_dates" not in st.session_state:
    st.session_state["picked_multi_dates"] = []
if "maximize" not in st.session_state:
    st.session_state["maximize"] = False
if "show_table" not in st.session_state:
    st.session_state["show_table"] = False

# ==================================================================================================
# 4. LOAD DATA
# ==================================================================================================
all_data, df_news = load_all_data()

if not all_data:
    st.error(f"Master data file not found or empty: `{MASTER_EXCEL_FILE}`.")
    st.stop()

# ---------------------------- HEADER (no views) ----------------------------
st.markdown('<div class="header">', unsafe_allow_html=True)
header_cols = st.columns([1.2, 1.8, 1.6, 1.2, 1.0])

# ------------------------
# Products
# ------------------------
with header_cols[0]:
    st.markdown("**Products**")

    # Inline product pills (compact checkboxes side-by-side)
    prod_cols = st.columns(len(PRODUCT_CONFIG))
    for i, (symbol, cfg) in enumerate(PRODUCT_CONFIG.items()):
        key = f"chk_prod_{symbol}"
        # initialize if not in session_state
        if key not in st.session_state:
            st.session_state[key] = symbol in st.session_state.get("selected_products", [])

        # checkbox manages its own session_state
        prod_cols[i].checkbox(symbol, value=st.session_state[key], key=key)

    # update selected product list
    st.session_state["selected_products"] = [
        s for s in PRODUCT_CONFIG.keys()
        if st.session_state.get(f"chk_prod_{s}", False)
    ]

# ------------------------
# Date
# ------------------------
with header_cols[1]:
    st.markdown("**Date**")

    default_one_date = (
        st.session_state.get("picked_one_date")
        or nearest_date_on_or_before(
            all_data[list(all_data.keys())[0]]["data"],
            st.session_state.get("end_date")
        )
        or date.today()
    )

    st.date_input(
        "Date",
        value=default_one_date,
        key="picked_one_date"
    )

    overlay = st.checkbox(
        "Overlay dates",
        value=bool(st.session_state.get("picked_multi_dates")),
        key="overlay_toggle"
    )

    if overlay:
        available_dates = set()
        for s in st.session_state["selected_products"]:
            df_s = all_data.get(s, {}).get("data")
            if df_s is None or df_s.empty:
                continue
            available_dates |= set(df_s["Date"].dt.date.dropna().unique())

        all_dates = sorted(available_dates, reverse=True)

        st.multiselect(
            "Overlay",
            options=all_dates,
            default=st.session_state.get("picked_multi_dates", []),
            key="picked_multi_dates",
            help="Pick multiple dates to overlay on curves"
        )
    else:
        st.session_state["picked_multi_dates"] = []

# ------------------------
# Range
# ------------------------
with header_cols[2]:
    st.markdown("**Range**")

    st.date_input(
        "Start",
        value=st.session_state.get("start_date", date.today()),
        key="start_date"
    )
    st.date_input(
        "End",
        value=st.session_state.get("end_date", date.today()),
        key="end_date"
    )

# ------------------------
# Tools
# ------------------------
with header_cols[3]:
    st.markdown("**Tools**")

    if st.button("Maximize Charts", key="max_btn"):
        st.session_state["maximize"] = not st.session_state.get("maximize", False)

    if st.button("Show Table", key="table_btn"):
        st.session_state["show_table"] = not st.session_state.get("show_table", False)

# ------------------------
# Actions
# ------------------------
with header_cols[4]:
    st.markdown("**Actions**")

    if st.button("Refresh Data", key="refresh_btn"):
        st.cache_data.clear()
        st.experimental_rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ------------------------
# If no products selected, stop
# ------------------------
if not st.session_state["selected_products"]:
    st.warning("Select at least one product.")
    st.stop()

# ------------------------
# Show selected product badges
# ------------------------
selected_badges_html = "<div class='selected-badges'>"
for sym in st.session_state["selected_products"]:
    selected_badges_html += (
        f"<span class='badge' style='border-left:4px solid {PRODUCT_CONFIG[sym]['color']};'>{sym}</span>"
    )
selected_badges_html += "</div>"

st.markdown(selected_badges_html, unsafe_allow_html=True)


# ==================================================================================================
# 5. MAIN PAGE — merged Curves, Compare, Workspace (all visible)
# ==================================================================================================
# Layout behavior: if maximize True, render charts in single column full width; otherwise use 2 or 3 column grid optimally
prods = st.session_state["selected_products"]
if st.session_state["maximize"]:
    cols_per_row = 1
else:
    cols_per_row = min(3, max(1, len(prods)))

# ------------------ Outright Curves (per product) ------------------
st.markdown("## Outright Curves")
normalize_curves = st.checkbox("Normalize curves (z-score)", value=False, key="normalize_all")
show_values = st.checkbox("Show point values", value=False, key="show_values_all")

# Determine which dates to plot per product
# If overlay multi dates provided -> use those; else use single picked date or latest
multi_dates = st.session_state.get("picked_multi_dates", []) or []
single_picked = st.session_state.get("picked_one_date", None)

rows = (len(prods) + cols_per_row - 1) // cols_per_row
idx = 0
for _ in range(rows):
    cols = st.columns(cols_per_row)
    for c in cols:
        if idx >= len(prods):
            break
        symbol = prods[idx]
        idx += 1
        product_data = all_data.get(symbol)
        if not product_data:
            continue
        df, contracts = product_data["data"], product_data["contracts"]

        # choose dates
        if multi_dates:
            sel_dates = [nearest_date_on_or_before(df, d) for d in multi_dates]
            sel_dates = [d for d in sel_dates if d is not None]
        else:
            if single_picked:
                d_use = nearest_date_on_or_before(df, single_picked)
            else:
                d_use = nearest_date_on_or_before(df, st.session_state["end_date"])
            sel_dates = [d_use] if d_use else []

        if not sel_dates:
            c.warning(f"No curve for selected date(s) in {symbol}.")
            continue

        fig = go.Figure()
        for d_plot in sel_dates:
            row = df[df["Date"].dt.date == d_plot]
            if row.empty:
                continue
            s = row.iloc[0][contracts].astype(float)
            if normalize_curves:
                s = (s - s.mean()) / (s.std() if s.std() != 0 else 1)
            fig.add_trace(
                go.Scatter(
                    x=contracts,
                    y=s.values,
                    mode="lines+markers" + ("+text" if show_values else ""),
                    name=str(d_plot),
                    line=dict(color=PRODUCT_CONFIG[symbol]["color"], width=2.5),
                    text=[f"{val:.2f}" for val in s.values] if show_values else None,
                    textposition="top center" if show_values else None,
                )
            )

        fig = style_figure(fig, f"{PRODUCT_CONFIG[symbol]['name']} Outright")
        fig.update_yaxes(title_text="Z-score" if normalize_curves else "Price ($)")
        c.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

# ------------------ Quick Spreads & Flies per product (compact area) ------------------
st.markdown("## Quick Spreads & Flies (per product)")
qcols = st.columns(2)
with qcols[0]:
    for symbol in prods:
        df, contracts = all_data[symbol]["data"], all_data[symbol]["contracts"]
        if len(contracts) < 2:
            st.caption(f"{symbol}: not enough contracts for spreads")
            continue
        # default top 3 consecutive pairs
        default_pairs = [f"{contracts[i]} - {contracts[i+1]}" for i in range(min(3, len(contracts)-1))]
        choices = st.multiselect(f"{symbol} spreads", options=[f"{c1} - {c2}" for i, c1 in enumerate(contracts) for c2 in contracts[i+1:]], default=default_pairs, key=f"spread_multi_{symbol}")
        if choices:
            sub = filter_by_date_window(df, st.session_state["start_date"], st.session_state["end_date"])
            if sub.empty:
                st.caption(f"{symbol}: no data in range")
            else:
                fig = go.Figure()
                for pair in choices:
                    a, b = [p.strip() for p in pair.split("-")]
                    series = spread_series(sub, a, b)
                    fig.add_trace(go.Scatter(x=sub["Date"], y=series, mode="lines", name=f"{symbol} {a}-{b}", line=dict(color=PRODUCT_CONFIG[symbol]["color"], width=1.6)))
                # align markers at series mean for news
                mean_series = sub.select_dtypes(include=[np.number]).mean(axis=1) if not sub.empty else None
                add_news_markers(fig, sub, df_news, y_series=series if not series.empty else None)
                fig = style_figure(fig, f"{symbol} Spreads")
                fig.update_yaxes(title_text="Price Diff ($)")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with qcols[1]:
    for symbol in prods:
        df, contracts = all_data[symbol]["data"], all_data[symbol]["contracts"]
        if len(contracts) < 3:
            st.caption(f"{symbol}: not enough contracts for flies")
            continue
        default_flies = [f"{contracts[i]} - {contracts[i+1]} - {contracts[i+2]}" for i in range(min(2, len(contracts)-2))]
        choices = st.multiselect(f"{symbol} flies", options=[f"{a} - {b} - {c}" for i,a in enumerate(contracts) for j,b in enumerate(contracts[i+1:], start=i+1) for c in contracts[j+1:]], default=default_flies, key=f"fly_multi_{symbol}")
        if choices:
            sub = filter_by_date_window(df, st.session_state["start_date"], st.session_state["end_date"])
            if sub.empty:
                st.caption(f"{symbol}: no data in range")
            else:
                fig = go.Figure()
                for item in choices:
                    f1, f2, f3 = [p.strip() for p in item.split("-")]
                    series = fly_series(sub, f1, f2, f3)
                    fig.add_trace(go.Scatter(x=sub["Date"], y=series, mode="lines", name=f"{symbol} {f1}-{f2}-{f3}", line=dict(color=PRODUCT_CONFIG[symbol]["color"], width=1.6)))
                add_news_markers(fig, sub, df_news, y_series=series if not series.empty else None)
                fig = style_figure(fig, f"{symbol} Flies")
                fig.update_yaxes(title_text="Price Diff ($)")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ------------------ Cross-product Compare area ------------------
st.markdown("## Cross-Product Compare (multi-product time series)")
# Build universe
universe_spreads = []
universe_flies = []
for symbol in prods:
    contracts = all_data[symbol]["contracts"]
    universe_spreads += [f"{symbol}: {c1}-{c2}" for i,c1 in enumerate(contracts) for c2 in contracts[i+1:]]
    universe_flies += [f"{symbol}: {a}-{b}-{c}" for i,a in enumerate(contracts) for j,b in enumerate(contracts[i+1:], start=i+1) for c in contracts[j+1:]]

c1, c2 = st.columns(2)
with c1:
    sel_spreads = st.multiselect("Select spreads", options=universe_spreads, default=universe_spreads[:4] if universe_spreads else [], key="sel_spreads")
with c2:
    sel_flies = st.multiselect("Select flies", options=universe_flies, default=universe_flies[:3] if universe_flies else [], key="sel_flies")

normalize_ts = st.checkbox("Normalize selected series (z per series)", value=False, key="normalize_sel")

def norm_series(s: pd.Series) -> pd.Series:
    std = s.std()
    return (s - s.mean()) / (std if std not in (0, np.nan) else 1)

if sel_spreads:
    figS = go.Figure()
    last_sub = pd.DataFrame()
    for item in sel_spreads:
        try:
            sym, pair = item.split(":")
        except ValueError:
            continue
        sym = sym.strip()
        cA, cB = [p.strip() for p in pair.split("-")]
        df = all_data[sym]["data"]
        sub = filter_by_date_window(df, st.session_state["start_date"], st.session_state["end_date"])
        last_sub = sub
        if sub.empty:
            continue
        series = spread_series(sub, cA, cB)
        if normalize_ts:
            series = norm_series(series)
        figS.add_trace(go.Scatter(x=sub["Date"], y=series, mode="lines", name=f"{sym} {cA}-{cB}", line=dict(width=2)))
    add_news_markers(figS, last_sub, df_news, y_series=series if 'series' in locals() else None)
    figS = style_figure(figS, "Selected Spreads – Cross Product")
    figS.update_yaxes(title_text="Z" if normalize_ts else "Price Diff ($)")
    st.plotly_chart(figS, use_container_width=True, config={"displayModeBar": True})

if sel_flies:
    figF = go.Figure()
    last_sub = pd.DataFrame()
    for item in sel_flies:
        try:
            sym, trip = item.split(":")
        except ValueError:
            continue
        sym = sym.strip()
        a, b, c = [p.strip() for p in trip.split("-")]
        df = all_data[sym]["data"]
        sub = filter_by_date_window(df, st.session_state["start_date"], st.session_state["end_date"])
        last_sub = sub
        if sub.empty:
            continue
        series = fly_series(sub, a, b, c)
        if normalize_ts:
            series = norm_series(series)
        figF.add_trace(go.Scatter(x=sub["Date"], y=series, mode="lines", name=f"{sym} {a}-{b}-{c}", line=dict(width=2)))
    add_news_markers(figF, last_sub, df_news, y_series=series if 'series' in locals() else None)
    figF = style_figure(figF, "Selected Flies – Cross Product")
    figF.update_yaxes(title_text="Z" if normalize_ts else "Price Diff ($)")
    st.plotly_chart(figF, use_container_width=True, config={"displayModeBar": True})

# ------------------ Workspace consolidated charts (summary area) ------------------
st.markdown("## Workspace Summary")
# Latest Outrights aggregated
st.markdown("### Latest Outrights by product")
cols_per_row = cols_per_row
idx = 0
rows = (len(prods) + cols_per_row - 1) // cols_per_row
for _ in range(rows):
    cols = st.columns(cols_per_row)
    for c in cols:
        if idx >= len(prods): break
        symbol = prods[idx]; idx += 1
        df, contracts = all_data[symbol]["data"], all_data[symbol]["contracts"]
        d_use = nearest_date_on_or_before(df, st.session_state["end_date"])
        if d_use is None:
            c.caption(f"{symbol}: no date")
            continue
        row = df[df["Date"].dt.date == d_use]
        if row.empty:
            c.caption(f"{symbol}: no row")
            continue
        s = row.iloc[0][contracts].astype(float)
        fig = go.Figure(go.Scatter(x=contracts, y=s.values, mode="lines+markers", line=dict(color=PRODUCT_CONFIG[symbol]["color"], width=2)))
        style_figure(fig, f"{symbol} Outright ({d_use})")
        c.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# M1-M2 combined chart
st.markdown("### M1-M2 Combined (all selected products)")
fig_comb = go.Figure()
any_added = False
last_sub = pd.DataFrame()
for symbol in prods:
    df, contracts = all_data[symbol]["data"], all_data[symbol]["contracts"]
    if len(contracts) < 2: continue
    sub = filter_by_date_window(df, st.session_state["start_date"], st.session_state["end_date"])
    last_sub = sub
    if sub.empty: continue
    series = spread_series(sub, contracts[0], contracts[1])
    fig_comb.add_trace(go.Scatter(x=sub["Date"], y=series, mode="lines", name=f"{symbol} {contracts[0]}-{contracts[1]}", line=dict(color=PRODUCT_CONFIG[symbol]["color"], width=1.8)))
    any_added = True
if any_added:
    add_news_markers(fig_comb, last_sub, df_news, y_series=series if 'series' in locals() else None)
    fig_comb = style_figure(fig_comb, "M1-M2 – All Selected Products")
    st.plotly_chart(fig_comb, use_container_width=True, config={"displayModeBar": True})
else:
    st.info("No M1-M2 spreads found for selection.")

# M1-M2-M3 combined chart
st.markdown("### M1-M2-M3 Combined (all selected products)")
fig_comb_f = go.Figure()
any_added = False
last_sub = pd.DataFrame()
for symbol in prods:
    df, contracts = all_data[symbol]["data"], all_data[symbol]["contracts"]
    if len(contracts) < 3: continue
    sub = filter_by_date_window(df, st.session_state["start_date"], st.session_state["end_date"])
    last_sub = sub
    if sub.empty: continue
    series = fly_series(sub, contracts[0], contracts[1], contracts[2])
    fig_comb_f.add_trace(go.Scatter(x=sub["Date"], y=series, mode="lines", name=f"{symbol} {contracts[0]}-{contracts[1]}-{contracts[2]}", line=dict(color=PRODUCT_CONFIG[symbol]["color"], width=1.8)))
    any_added = True
if any_added:
    add_news_markers(fig_comb_f, last_sub, df_news, y_series=series if 'series' in locals() else None)
    fig_comb_f = style_figure(fig_comb_f, "M1-M2-M3 – All Selected Products")
    st.plotly_chart(fig_comb_f, use_container_width=True, config={"displayModeBar": True})
else:
    st.info("No flies found for selection.")

# ------------------ Table toggle (on demand) ------------------
if st.session_state["show_table"]:
    st.markdown("## Data Tables")
    for symbol in prods:
        product_data = all_data.get(symbol)
        if not product_data: continue
        df = product_data["data"]
        st.markdown(f"### {PRODUCT_CONFIG[symbol]['name']}")
        filtered_df = filter_by_date_window(df, st.session_state["start_date"], st.session_state["end_date"])
        if filtered_df.empty:
            st.caption(f"No data for {symbol} in the selected range.")
        else:
            st.dataframe(filtered_df, use_container_width=True)

# ==================================================================================================
# END
# ==================================================================================================

