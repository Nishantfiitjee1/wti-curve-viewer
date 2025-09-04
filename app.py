# app.py
# Futures Dashboard (Streamlit) – compact UI, overlay fix, view toggles, robust loaders
# ------------------------------------------------------------------------------------
# What’s inside:
#  - Compact header (Products • Views • Date • Range)
#  - Outright Curves (date overlays)
#  - Spread Curves over chain for a day (M1-M2, M2-M3, ...) with overlays + compare
#  - Spreads & Flies time-series (selectable across products) with optional normalization
#  - News markers (safe, optional)
#  - Robust Excel loading (safe header detection, flexible parsing)
#  - Removed: Maximize button, Refresh, Workspace Summary duplication
#
# Files expected beside this app.py in your GitHub repo:
#   - Futures_Data.xlsx
#   - Important_news_date.xlsx   (optional)
#   - requirements.txt

import os
from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =========================================
# 0) PAGE SETUP + COMPACT CSS
# =========================================
st.set_page_config(page_title="Futures Dashboard", layout="wide", initial_sidebar_state="collapsed")

COMPACT_CSS = """
<style>
/* Tight app paddings */
.block-container { padding-top: 10px !important; padding-bottom: 10px !important; }

/* Header bar */
.header {
  background-color: #F7F8FA;
  padding: 8px 10px;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
  margin-bottom: 10px;
}

/* Group labels */
.group-title {
  font-weight: 600;
  font-size: 13px;
  margin-bottom: 6px;
}

/* Small form controls */
div[data-baseweb="input"] input, .stDateInput input, .stMultiSelect, .stSelectbox, .stNumberInput, .stTextInput>div>div>input {
  font-size: 12px !important;
  padding: 4px 6px !important;
  min-height: 28px !important;
}
.stDateInput, .stMultiSelect, .stSelectbox { min-height: 32px !important; }

/* Checkbox row pills */
.pill-group { display: flex; flex-wrap: wrap; gap: 6px; }
.pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 8px;
  border-radius: 12px;
  border: 1px solid #E5E7EB;
  background: #FBFDFF;
  font-size: 12px;
  user-select: none;
}
.pill input { transform: scale(1.0); }

/* Small caption + badges */
.small-note { font-size: 11px; color: #6B7280; margin-top: 2px; }

/* Plotly container tweaks */
.element-container .stPlotlyChart {
  padding: 6px 6px 10px 6px !important;
}

/* Reduce column gaps */
.row-widget.stColumns, .stColumns {
  gap: 8px !important;
}
</style>
"""
st.markdown(COMPACT_CSS, unsafe_allow_html=True)

# =========================================
# 1) CONFIG
# =========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_EXCEL_FILE = os.path.join(BASE_DIR, "Futures_Data.xlsx")
NEWS_EXCEL_FILE = os.path.join(BASE_DIR, "Important_news_date.xlsx")

PRODUCT_CONFIG = {
    "CL": {"name": "WTI Crude Oil",  "sheet": "WTI_Outright",  "color": "#0072B2"},
    "BZ": {"name": "Brent Crude Oil","sheet": "Brent_outright","color": "#D55E00"},
    "DBI":{"name": "Dubai Crude Oil","sheet": "Dubai_Outright","color": "#009E73"},
}

# =========================================
# 2) UTILITIES
# =========================================
def _safe_contract_list(row_vals):
    """Return cleaned contract list from header row values (skip empty/date labels)."""
    labels = []
    for x in row_vals[1:]:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            continue
        s = str(x).strip()
        if not s:
            continue
        sl = s.lower()
        if sl in ("date","dates","dates."):
            continue
        labels.append(s)
    return labels

def nearest_date_on_or_before(df: pd.DataFrame, target_d: date) -> date | None:
    if df is None or df.empty or "Date" not in df.columns:
        return None
    subset = df[df["Date"].dt.date <= target_d]
    if subset.empty:
        return None
    return subset["Date"].max().date()

def filter_by_date_window(df: pd.DataFrame, start_d: date, end_d: date) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    m = (df["Date"].dt.date >= start_d) & (df["Date"].dt.date <= end_d)
    return df.loc[m].copy()

def spread_series(df: pd.DataFrame, c1: str, c2: str) -> pd.Series:
    return pd.to_numeric(df.get(c1), errors="coerce") - pd.to_numeric(df.get(c2), errors="coerce")

def fly_series(df: pd.DataFrame, a: str, b: str, c: str) -> pd.Series:
    return pd.to_numeric(df.get(a), errors="coerce") - 2 * pd.to_numeric(df.get(b), errors="coerce") + pd.to_numeric(df.get(c), errors="coerce")

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std()
    return (s - s.mean()) / (std if std not in (0, np.nan) else 1)

def style_figure(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color="#111827", size=14), x=0.5, y=0.95),
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
        margin=dict(l=28, r=12, t=46, b=36),
        xaxis=dict(color="#374151", gridcolor="#E5E7EB", zeroline=False),
        yaxis=dict(color="#374151", gridcolor="#E5E7EB", zeroline=False),
        hovermode="x unified",
        legend=dict(font=dict(size=11), y=0.99, x=0.01),
    )
    return fig

def add_news_markers(fig: go.Figure, merged_df: pd.DataFrame, df_news: pd.DataFrame, y_series: pd.Series | None = None):
    """Add safe markers for news. If y_series is None, place markers near top of y-range."""
    try:
        if df_news is None or df_news.empty or merged_df is None or merged_df.empty:
            return fig
        news_cols = [c for c in df_news.columns if c != "Date"]
        if not news_cols:
            return fig

        # Left join on Date to discover which rows have news
        joined = pd.merge(
            merged_df[["Date"]].copy(),
            df_news,
            on="Date",
            how="left",
            suffixes=("", "_news"),
        )
        existing_news = [c for c in news_cols if c in joined.columns]
        if not existing_news:
            return fig

        news_df_in_view = joined.dropna(subset=existing_news, how="all")
        if news_df_in_view.empty:
            return fig

        if y_series is None or y_series.empty:
            # set a flat y ~ 98% of current maximum numeric value in merged_df
            y_max = None
            try:
                y_max = merged_df.select_dtypes(include=[np.number]).max().max()
            except Exception:
                y_max = None
            plateau = (y_max if y_max and np.isfinite(y_max) else 1.0) * 0.98
            y_values = [plateau] * len(news_df_in_view)
        else:
            y_values = y_series.reindex(news_df_in_view.index).fillna(method="ffill").fillna(method="bfill").tolist()

        # Build hover text
        def _row_html(r):
            chunks = [f"<b>Date:</b> {pd.to_datetime(r['Date']).strftime('%Y-%m-%d')}"]
            for c in existing_news:
                val = r.get(c)
                if pd.notna(val):
                    chunks.append(f"<b>{c.replace('_',' ')}:</b> {val}")
            return "<br>".join(chunks)

        hover_texts = [ _row_html(r) for _, r in news_df_in_view.iterrows() ]

        fig.add_trace(
            go.Scatter(
                x=news_df_in_view["Date"],
                y=y_values,
                mode="markers",
                name="News",
                marker=dict(size=8, symbol="diamond"),
                hovertext=hover_texts,
                hoverinfo="text",
                showlegend=False,
            )
        )
        return fig
    except Exception:
        # Never block charts on news error
        return fig

# =========================================
# 3) DATA LOADING
# =========================================
@st.cache_data(show_spinner="Loading data ...", ttl=3600)
def load_all_data():
    """Load product curves and news with robust header parsing."""
    all_product_data: dict[str, dict] = {}

    if not os.path.exists(MASTER_EXCEL_FILE):
        return {}, pd.DataFrame()

    for symbol, cfg in PRODUCT_CONFIG.items():
        try:
            df_raw = pd.read_excel(
                MASTER_EXCEL_FILE,
                sheet_name=cfg["sheet"],
                header=None,
                engine="openpyxl",
            )
            if df_raw is None or df_raw.empty:
                raise ValueError("Sheet is empty or not found")

            # detect header row
            header_row_index = None
            for i in range(min(10, df_raw.shape[0])):
                row_list = [str(x).strip().lower() for x in df_raw.iloc[i] if pd.notna(x)]
                if any(x in ("date", "dates", "dates.") for x in row_list):
                    header_row_index = i
                    break
            if header_row_index is None:
                header_row_index = 0

            data_start_row_index = header_row_index + 1

            # column names
            header_vals = [str(x).strip() for x in df_raw.iloc[header_row_index] if pd.notna(x)]
            if not header_vals:
                raise ValueError("No header row detected")

            # first column must be Date
            col_names = ["Date"] + header_vals[1:]

            df = df_raw.iloc[data_start_row_index:].copy()

            # align
            valid_cols = min(len(col_names), df.shape[1])
            df = df.iloc[:, :valid_cols]
            df.columns = col_names[:valid_cols]

            # types
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            for c in df.columns:
                if c != "Date":
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            df = df.dropna(subset=["Date"]).sort_values("Date", ascending=False).reset_index(drop=True)

            all_product_data[symbol] = {"data": df, "contracts": [c for c in df.columns if c != "Date"]}

        except Exception as e:
            st.warning(f"Could not load/parse sheet for {cfg['name']} (sheet='{cfg['sheet']}'). Error: {e}")

    # NEWS file
    df_news = pd.DataFrame()
    if os.path.exists(NEWS_EXCEL_FILE):
        try:
            df_news = pd.read_excel(NEWS_EXCEL_FILE, engine="openpyxl")
            df_news.columns = [str(c).strip() for c in df_news.columns]
            if "Date" in df_news.columns or "Dates" in df_news.columns:
                date_col = "Date" if "Date" in df_news.columns else "Dates"
                df_news = df_news.rename(columns={date_col: "Date"})
                df_news["Date"] = pd.to_datetime(df_news["Date"], errors="coerce")
                df_news = df_news.dropna(subset=["Date"]).reset_index(drop=True)
        except Exception as e:
            st.warning(f"Could not load/parse news file. Error: {e}")
            df_news = pd.DataFrame()

    return all_product_data, df_news




# =========================================
# 4) STATE DEFAULTS
# =========================================
if "selected_products" not in st.session_state:
    st.session_state["selected_products"] = list(PRODUCT_CONFIG.keys())  # default all

if "start_date" not in st.session_state:
    st.session_state["start_date"] = date.today() - timedelta(days=365)

if "end_date" not in st.session_state:
    st.session_state["end_date"] = date.today()

if "picked_one_date" not in st.session_state:
    st.session_state["picked_one_date"] = None

if "picked_multi_dates" not in st.session_state:
    st.session_state["picked_multi_dates"] = []

if "view_outright" not in st.session_state:
    st.session_state["view_outright"] = True
if "view_spread_curve" not in st.session_state:
    st.session_state["view_spread_curve"] = True
if "view_spreads_ts" not in st.session_state:
    st.session_state["view_spreads_ts"] = False
if "view_flies_ts" not in st.session_state:
    st.session_state["view_flies_ts"] = False


# =========================================
# 5) LOAD DATA
# =========================================
all_data, df_news = load_all_data()

if not all_data:
    st.error(
        "Master data file not found or empty. "
        "Make sure **Futures_Data.xlsx** is present in the same GitHub repo folder as this app.py."
    )
    st.stop()

# =========================================
# 6) HEADER BAR (COMPACT)
# =========================================
st.markdown('<div class="header">', unsafe_allow_html=True)

left, mid1, mid2, right = st.columns([1.4, 1.6, 1.8, 1.2])

with left:
    st.markdown('<div class="group-title">Products</div>', unsafe_allow_html=True)
    # pill-like inline checkboxes
    cols = st.columns(len(PRODUCT_CONFIG))
    for i, (symbol, cfg) in enumerate(PRODUCT_CONFIG.items()):
        key = f"prod_{symbol}"
        if key not in st.session_state:
            st.session_state[key] = symbol in st.session_state["selected_products"]
        tick = cols[i].checkbox(symbol, value=st.session_state[key], key=key)
    # update selection
    st.session_state["selected_products"] = [s for s in PRODUCT_CONFIG.keys() if st.session_state.get(f"prod_{s}", False)]
    if not st.session_state["selected_products"]:
        st.info("Select at least one product to proceed.")

with mid1:
    st.markdown('<div class="group-title">Views</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    st.session_state["view_outright"]      = c1.checkbox("Outright", value=st.session_state["view_outright"], key="v_out")
    st.session_state["view_spread_curve"]  = c2.checkbox("Spread Curve", value=st.session_state["view_spread_curve"], key="v_sc")
    st.session_state["view_spreads_ts"]    = c3.checkbox("Spreads TS", value=st.session_state["view_spreads_ts"], key="v_s_ts")
    st.session_state["view_flies_ts"]      = c4.checkbox("Flies TS", value=st.session_state["view_flies_ts"], key="v_f_ts")
    st.caption("Pick which sections render. All updates apply immediately.")

with mid2:
    st.markdown('<div class="group-title">Date</div>', unsafe_allow_html=True)
    # Single "focus" date
    # default chosen as the last available <= end_date for the first selected product
    default_focus = None
    if st.session_state["selected_products"]:
        first_sym = st.session_state["selected_products"][0]
        df0 = all_data[first_sym]["data"]
        default_focus = nearest_date_on_or_before(df0, st.session_state["end_date"])
    if default_focus is None:
        default_focus = date.today()
    st.session_state["picked_one_date"] = st.date_input("Date", value=st.session_state.get("picked_one_date") or default_focus, key="one_date")
    # Overlay toggle + pick multiple dates (these will render instantly—Streamlit reruns on change)
    overlay = st.checkbox("Overlay dates", value=bool(st.session_state.get("picked_multi_dates")), key="overlay_toggle")
    if overlay:
    available = set()
    for sym in st.session_state["selected_products"]:
        df_s = all_data.get(sym, {}).get("data")
        if df_s is not None and not df_s.empty:
            available |= set(df_s["Date"].dt.date.dropna().unique())
    all_dates_sorted = sorted(available, reverse=True)

    chosen = st.multiselect("Overlay", options=all_dates_sorted,
                            default=st.session_state.get("picked_multi_dates", []),
                            key="multi_dates")
    if chosen != st.session_state.get("picked_multi_dates"):
        st.session_state["picked_multi_dates"] = chosen
        st.experimental_rerun()

with right:
    st.markdown('<div class="group-title">Range</div>', unsafe_allow_html=True)
    st.session_state["start_date"] = st.date_input("Start", value=st.session_state["start_date"], key="start")
    st.session_state["end_date"]   = st.date_input("End",   value=st.session_state["end_date"],   key="end")

st.markdown("</div>", unsafe_allow_html=True)


# =========================================
# 7) OUTRIGHT CURVES (per product, date overlays)
# =========================================
if st.session_state["view_outright"] and st.session_state["selected_products"]:
    st.subheader("Outright Curves")
    opt1, opt2 = st.columns([1,1])
    normalize_curves = opt1.checkbox("Normalize (z per curve)", value=False, key="outright_norm")
    show_values     = opt2.checkbox("Show point values", value=False, key="outright_vals")

    prods = st.session_state["selected_products"]
    cols_per_row = min(3, max(1, len(prods)))
    rows = (len(prods) + cols_per_row - 1) // cols_per_row

    # Selected dates to plot
    if st.session_state["picked_multi_dates"]:
        selected_dates = st.session_state["picked_multi_dates"]
    else:
        # Use focus date
        selected_dates = [st.session_state["picked_one_date"]]

    idx = 0
    for _ in range(rows):
        cols = st.columns(cols_per_row)
        for c in cols:
            if idx >= len(prods): break
            sym = prods[idx]; idx += 1
            pdata = all_data.get(sym)
            if not pdata: 
                c.warning(f"{sym}: no data.")
                continue
            df, contracts = pdata["data"], pdata["contracts"]
            if df.empty or not contracts:
                c.warning(f"{sym}: empty.")
                continue

            # find actual available dates for each requested date
            plot_dates = []
            for d in selected_dates:
                d_use = nearest_date_on_or_before(df, d)
                if d_use: plot_dates.append(d_use)
            plot_dates = sorted(set(plot_dates))  # unique

            if not plot_dates:
                c.info(f"{sym}: no curves for selected date(s).")
                continue

            fig = go.Figure()
            for d_plot in plot_dates:
                row = df[df["Date"].dt.date == d_plot]
                if row.empty: 
                    continue
                series = row.iloc[0][contracts].astype(float)
                if normalize_curves:
                    series = zscore(series)
                fig.add_trace(
                    go.Scatter(
                        x=contracts,
                        y=series.values,
                        mode="lines+markers" + ("+text" if show_values else ""),
                        name=str(d_plot),
                        line=dict(color=PRODUCT_CONFIG[sym]["color"], width=2.4),
                        text=[f"{v:.2f}" for v in series.values] if show_values else None,
                        textposition="top center" if show_values else None,
                    )
                )
            fig = style_figure(fig, f"{PRODUCT_CONFIG[sym]['name']} Outright")
            fig.update_yaxes(title_text="Z-score" if normalize_curves else "Price ($)")
            c.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})


# =========================================
# 8) SPREAD CURVE (full chain M1-M2, M2-M3, ...)
#     with date overlays + date-wise compare
# =========================================
if st.session_state["view_spread_curve"] and st.session_state["selected_products"]:
    st.subheader("Spread Curve (chain: M1-M2, M2-M3, …)")

    left_opts, right_opts = st.columns([1,1])
    sc_show_vals = left_opts.checkbox("Show values", value=False, key="sc_vals")
    sc_normalize = right_opts.checkbox("Normalize (z per curve)", value=False, key="sc_norm")

    prods = st.session_state["selected_products"]
    cols_per_row = min(3, max(1, len(prods)))
    rows = (len(prods) + cols_per_row - 1) // cols_per_row

    # Determine dates
    if st.session_state["picked_multi_dates"]:
        sc_dates = st.session_state["picked_multi_dates"]
    else:
        sc_dates = [st.session_state["picked_one_date"]]

    idx = 0
    for _ in range(rows):
        cols = st.columns(cols_per_row)
        for c in cols:
            if idx >= len(prods): break
            sym = prods[idx]; idx += 1
            pdata = all_data.get(sym)
            if not pdata: 
                c.warning(f"{sym}: no data.")
                continue
            df, contracts = pdata["data"], pdata["contracts"]
            if len(contracts) < 2:
                c.warning(f"{sym}: not enough contracts for spreads.")
                continue

            # create chain pairs (M1-M2, M2-M3, ...)
            chain_pairs = []
            for i in range(len(contracts)-1):
                chain_pairs.append((contracts[i], contracts[i+1]))

            fig = go.Figure()
            # plot one spread-curve per selected date
            used_dates = []
            for d in sc_dates:
                d_use = nearest_date_on_or_before(df, d)
                if not d_use: 
                    continue
                used_dates.append(d_use)
                row = df[df["Date"].dt.date == d_use]
                if row.empty:
                    continue

                # Build Y over chain pairs
                y_vals = []
                x_labels = []
                for (a, b) in chain_pairs:
                    if a in row.columns and b in row.columns:
                        val = float(row.iloc[0][a]) - float(row.iloc[0][b])
                        y_vals.append(val)
                        x_labels.append(f"{a}-{b}")

                s = pd.Series(y_vals, index=x_labels, dtype=float)
                if sc_normalize:
                    s = zscore(s)

                fig.add_trace(
                    go.Scatter(
                        x=s.index.tolist(),
                        y=s.values.tolist(),
                        mode="lines+markers" + ("+text" if sc_show_vals else ""),
                        name=str(d_use),
                        line=dict(color=PRODUCT_CONFIG[sym]["color"], width=2.4),
                        text=[f"{v:.2f}" for v in s.values] if sc_show_vals else None,
                        textposition="top center" if sc_show_vals else None,
                    )
                )

            fig = style_figure(fig, f"{PRODUCT_CONFIG[sym]['name']} – Spread Curve")
            fig.update_yaxes(title_text="Z" if sc_normalize else "Price Diff ($)")
            c.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})


# =========================================
# 9) SPREADS TIME SERIES (cross-product)
# =========================================
if st.session_state["view_spreads_ts"] and st.session_state["selected_products"]:
    st.subheader("Spreads – Time Series (Cross Product)")

    # Build choices
    universe_spreads = []
    for sym in st.session_state["selected_products"]:
        contracts = all_data[sym]["contracts"]
        for i, c1 in enumerate(contracts):
            for c2 in contracts[i+1:]:
                universe_spreads.append(f"{sym}: {c1}-{c2}")

    cA, cB = st.columns(2)
    sel_spreads = cA.multiselect("Select spreads", options=universe_spreads, default=universe_spreads[:4] if universe_spreads else [])
    normalize_ts = cB.checkbox("Normalize (z per series)", value=False, key="spread_ts_norm")

    if sel_spreads:
        figS = go.Figure()
        last_sub = pd.DataFrame()
        for item in sel_spreads:
            try:
                sym, pair = item.split(":")
                sym = sym.strip()
                c1, c2 = [p.strip() for p in pair.split("-")]
            except Exception:
                continue
            df = all_data[sym]["data"]
            sub = filter_by_date_window(df, st.session_state["start_date"], st.session_state["end_date"])
            last_sub = sub
            if sub.empty:
                continue
            series = spread_series(sub, c1, c2)
            if normalize_ts:
                series = zscore(series)
            figS.add_trace(
                go.Scatter(
                    x=sub["Date"],
                    y=series,
                    mode="lines",
                    name=f"{sym} {c1}-{c2}",
                    line=dict(width=2),
                )
            )
        add_news_markers(figS, last_sub, df_news, y_series=None)
        figS = style_figure(figS, "Selected Spreads – Cross Product")
        figS.update_yaxes(title_text="Z" if normalize_ts else "Price Diff ($)")
        st.plotly_chart(figS, use_container_width=True, config={"displayModeBar": True})
    else:
        st.info("Pick at least one spread.")


# =========================================
# 10) FLIES TIME SERIES (cross-product)
# =========================================
if st.session_state["view_flies_ts"] and st.session_state["selected_products"]:
    st.subheader("Flies – Time Series (Cross Product)")

    # Build choices
    universe_flies = []
    for sym in st.session_state["selected_products"]:
        contracts = all_data[sym]["contracts"]
        for i, a in enumerate(contracts):
            for j, b in enumerate(contracts[i+1:], start=i+1):
                for c in contracts[j+1:]:
                    universe_flies.append(f"{sym}: {a}-{b}-{c}")

    cA, cB = st.columns(2)
    sel_flies = cA.multiselect("Select flies", options=universe_flies, default=universe_flies[:3] if universe_flies else [])
    normalize_ts_f = cB.checkbox("Normalize (z per series)", value=False, key="flies_ts_norm")

    if sel_flies:
        figF = go.Figure()
        last_sub = pd.DataFrame()
        for item in sel_flies:
            try:
                sym, trip = item.split(":")
                sym = sym.strip()
                a, b, c = [p.strip() for p in trip.split("-")]
            except Exception:
                continue
            df = all_data[sym]["data"]
            sub = filter_by_date_window(df, st.session_state["start_date"], st.session_state["end_date"])
            last_sub = sub
            if sub.empty:
                continue
            series = fly_series(sub, a, b, c)
            if normalize_ts_f:
                series = zscore(series)
            figF.add_trace(
                go.Scatter(
                    x=sub["Date"],
                    y=series,
                    mode="lines",
                    name=f"{sym} {a}-{b}-{c}",
                    line=dict(width=2),
                )
            )
        add_news_markers(figF, last_sub, df_news, y_series=None)
        figF = style_figure(figF, "Selected Flies – Cross Product")
        figF.update_yaxes(title_text="Z" if normalize_ts_f else "Price Diff ($)")
        st.plotly_chart(figF, use_container_width=True, config={"displayModeBar": True})
    else:
        st.info("Pick at least one fly.")

# =========================================
# 11) FOOTER NOTE
# =========================================
st.caption("Tip: Toggle Views in the header to show/hide sections instantly. Overlay multiple dates to compare curves without extra clicks.")


