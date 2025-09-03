import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta

# ==================================================================================================
# Compact Futures Dashboard — Updated UI & behavior
# - Reduced header density and input sizes
# - Removed "Maximize Charts" and problematic "Refresh Data" rerun
# - Fixed overlay multi-date behavior using explicit on_change callbacks
# - Added view checkboxes: Outright / Spreads / Flies
# - Added per-product Spread curve (consecutive-contract spreads plotted across contracts)
# - Removed Workspace Summary
# ==================================================================================================

# -----------------------------
# PAGE CONFIG & TIGHT STYLING
# -----------------------------
st.set_page_config(page_title="Futures Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
/* Tighter spacing for header */
body, .block-container { padding-top: 8px; padding-bottom: 8px; }
.header { background-color: #F8F9FB; padding: 8px; border-radius: 6px; margin-bottom: 10px; border: 1px solid #EDEFF2; }
.row-widget.stColumns .stColumn { padding-left:6px; padding-right:6px; }
/* Small compact inputs */
div[data-baseweb="dateinput"] { max-width: 220px; }
.stDateInput, .stNumberInput, .stSelectbox, .stMultiSelect { font-size: 12px; }
/* product pills */
.product-pill { display:inline-block; margin:0 6px 2px 0; padding:4px 8px; border-radius:12px; background:#f5f7fa; font-size:12px; border:1px solid #ddd; }
.product-pill.selected { background:#00A8E8; color:white; border-color:#0091d6; }
/* view pills */
.view-pill { display:inline-block; margin:0 6px 2px 0; padding:4px 8px; border-radius:8px; background:#ffffff; font-size:12px; border:1px solid #ddd; }
.view-pill.checked { background:#333; color:white; border-color:#222; }
/* reduce space around charts */
[data-testid="stPlotlyChart"] { padding: 6px 4px 12px 4px; }
</style>
""",
    unsafe_allow_html=True,
)

# ==================================================================================================
# CONFIG
# ==================================================================================================
MASTER_EXCEL_FILE = "Futures_Data.xlsx"
NEWS_EXCEL_FILE = "Important_news_date.xlsx"

PRODUCT_CONFIG = {
    "CL": {"name": "WTI Crude Oil", "sheet": "WTI_Outright", "color": "#0072B2"},
    "BZ": {"name": "Brent Crude Oil", "sheet": "Brent_outright", "color": "#D55E00"},
    "DBI": {"name": "Dubai Crude Oil", "sheet": "Dubai_Outright", "color": "#009E73"},
}

# ==================================================================================================
# DATA LOADING
# ==================================================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def load_all_data():
    all_product_data = {}
    if not os.path.exists(MASTER_EXCEL_FILE):
        return {}, pd.DataFrame()

    # ---------------------------
    # Load Product Sheets
    # ---------------------------
    for symbol, config in PRODUCT_CONFIG.items():
        try:
            df_raw = pd.read_excel(
                MASTER_EXCEL_FILE,
                sheet_name=config["sheet"],
                header=None,
                engine="openpyxl"
            )

            # find header row
            header_row_index = None
            for i in range(0, min(10, df_raw.shape[0])):
                if any(str(x).strip().lower() in ("date", "dates", "dates.") 
                       for x in df_raw.iloc[i].tolist() if pd.notna(x)):
                    header_row_index = i
                    break
            if header_row_index is None:
                header_row_index = 0

            data_start_row_index = header_row_index + 1
            row_vals = [str(x).strip() for x in df_raw.iloc[header_row_index].tolist()]

            # build column names
            contracts = [x for x in row_vals[1:] if x and x.lower() not in ("date", "dates")]
            col_names = ["Date"] + contracts

            df = df_raw.iloc[data_start_row_index:].copy()
            if df.shape[1] < len(col_names):
                df = pd.concat([df, pd.DataFrame(columns=list(range(len(col_names) - df.shape[1])))], axis=1)

            df.columns = col_names[: df.shape[1]]
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            for c in contracts:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            # Drop rows with invalid Date only
            df = df.dropna(subset=["Date"]).sort_values("Date", ascending=False).reset_index(drop=True)

            all_product_data[symbol] = {"data": df, "contracts": contracts}
        except Exception as e:
            st.warning(f"Could not load/parse sheet for {config['name']}. Error: {e}")
            continue

    # ---------------------------
    # Load News File
    # ---------------------------
    df_news = pd.DataFrame()
    if os.path.exists(NEWS_EXCEL_FILE):
        try:
            news_df_raw = pd.read_excel(NEWS_EXCEL_FILE, engine="openpyxl")

            # detect Date column
            date_col = None
            for candidate in ["Date", "Dates"]:
                if candidate in news_df_raw.columns:
                    date_col = candidate
                    break

            if date_col:
                news_df_raw.rename(columns={date_col: "Date"}, inplace=True)
                news_df_raw["Date"] = pd.to_datetime(news_df_raw["Date"], errors="coerce")

                # Keep all valid dates, even if other cols are empty
                df_news = news_df_raw[news_df_raw["Date"].notna()].copy()

                # Sort by date ascending (so future rows stay at bottom naturally)
                df_news = df_news.sort_values("Date").reset_index(drop=True)

        except Exception as e:
            st.warning(f"Could not load/parse news file. Error: {e}")
            df_news = pd.DataFrame()

    return all_product_data, df_news


# ==================================================================================================
# UTILS
# ==================================================================================================

def style_figure(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color="#333", size=12), x=0.5, y=0.95),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="#F9F9F9",
        xaxis=dict(color="#333", gridcolor="#EAEAEA", zeroline=False),
        yaxis=dict(color="#333", gridcolor="#EAEAEA", zeroline=False),
        legend=dict(font=dict(color="#333"), yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified",
        margin=dict(l=28, r=12, t=36, b=28),
    )
    return fig


def filter_by_date_window(df: pd.DataFrame, start_d: date, end_d: date) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df[(df["Date"].dt.date >= start_d) & (df["Date"].dt.date <= end_d)]


def nearest_date_on_or_before(df: pd.DataFrame, target_d: date) -> date | None:
    if df is None or df.empty or target_d is None:
        return None
    subset = df[df["Date"].dt.date <= target_d]
    if subset.empty:
        return None
    return subset["Date"].max().date()


def add_news_markers(fig: go.Figure, merged_df: pd.DataFrame, df_news: pd.DataFrame, y_series: pd.Series | None = None):
    if df_news is None or df_news.empty:
        return fig
    if merged_df is None or merged_df.empty:
        return fig
    news_cols = [c for c in df_news.columns if c != "Date"]
    if not news_cols:
        return fig
    joined = pd.merge(merged_df, df_news, on="Date", how="left", suffixes=("", "_news"))
    existing_news = [c for c in news_cols if c in joined.columns]
    if not existing_news:
        return fig
    news_df_in_view = joined.dropna(subset=existing_news, how="all")
    if news_df_in_view.empty:
        return fig
    if y_series is None:
        try:
            y_val = merged_df.select_dtypes(include=[np.number]).max().max()
            y_series = pd.Series(index=news_df_in_view.index, dtype=float)
            y_series[:] = (y_val if np.isfinite(y_val) else 0) * 0.98
        except Exception:
            y_series = pd.Series(index=news_df_in_view.index, dtype=float)
            y_series[:] = np.nan
    news_hover_text = news_df_in_view.apply(
        lambda row: f"<b>Date:</b> {row['Date'].strftime('%Y-%m-%d')}<br><hr>" + "<br>".join(
            f"<b>{col.replace('_', ' ')}:</b> {row[col]}" for col in existing_news if pd.notna(row.get(col))
        ),
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
# STATE INIT
# ==================================================================================================
if "selected_products" not in st.session_state:
    st.session_state["selected_products"] = list(PRODUCT_CONFIG.keys())
if "start_date" not in st.session_state:
    st.session_state["start_date"] = date.today() - timedelta(days=365)
if "end_date" not in st.session_state:
    st.session_state["end_date"] = date.today()
if "picked_one_date" not in st.session_state:
    st.session_state["picked_one_date"] = None
if "picked_multi_dates" not in st.session_state:
    st.session_state["picked_multi_dates"] = []
if "overlay_toggle" not in st.session_state:
    st.session_state["overlay_toggle"] = False
if "show_table" not in st.session_state:
    st.session_state["show_table"] = False
# view options: Outright / Spreads / Flies
if "view_outright" not in st.session_state:
    st.session_state["view_outright"] = True
if "view_spread" not in st.session_state:
    st.session_state["view_spread"] = False
if "view_fly" not in st.session_state:
    st.session_state["view_fly"] = False

# explicit callback functions to ensure immediate UI reaction
def _on_toggle_overlay():
    # when overlay toggled on, keep current picked_multi_dates or initialize to nearest available
    st.session_state["overlay_toggle"] = st.session_state.get("overlay_toggle", False)

def _on_multi_dates_change():
    # ensure picked_one_date cleared when multi selected
    if st.session_state.get("picked_multi_dates"):
        st.session_state["picked_one_date"] = None

# ==================================================================================================
# LOAD DATA
# ==================================================================================================
all_data, df_news = load_all_data()
if not all_data:
    st.error(f"Master data file not found or empty: `{MASTER_EXCEL_FILE}`.")
    st.stop()

# ---------------------------- HEADER ----------------------------
st.markdown('<div class="header">', unsafe_allow_html=True)
cols = st.columns([1.2, 1.8, 1.4, 1.0])

# Products pills (compact)
with cols[0]:
    st.markdown("**Products**")
    prod_cols = st.columns(len(PRODUCT_CONFIG))
    for i, (symbol, cfg) in enumerate(PRODUCT_CONFIG.items()):
        key = f"chk_prod_{symbol}"
        if key not in st.session_state:
            st.session_state[key] = symbol in st.session_state.get("selected_products", [])
        checked = prod_cols[i].checkbox(symbol, value=st.session_state[key], key=key)
    # update selected list
    st.session_state["selected_products"] = [s for s in PRODUCT_CONFIG.keys() if st.session_state.get(f"chk_prod_{s}", False)]

# Date and Overlay controls (compact)
with cols[1]:
    st.markdown("**Date**")
    # date input (single)
    st.date_input("Pick date", value=st.session_state.get("picked_one_date") or st.session_state.get("end_date"), key="picked_one_date")
    # overlay toggle and multi-selection
    st.checkbox("Overlay dates (compare)", value=st.session_state.get("overlay_toggle", False), key="overlay_toggle", on_change=_on_toggle_overlay)
    if st.session_state.get("overlay_toggle"):
        # gather available dates from selected products
        available_dates = set()
        for s in st.session_state["selected_products"]:
            df_s = all_data.get(s, {}).get("data")
            if df_s is None or df_s.empty:
                continue
            available_dates |= set(df_s["Date"].dt.date.dropna().unique())
        all_dates = sorted(available_dates, reverse=True)
        st.multiselect("Compare dates", options=all_dates, default=st.session_state.get("picked_multi_dates", []), key="picked_multi_dates", on_change=_on_multi_dates_change)
    else:
        st.session_state["picked_multi_dates"] = []

# Range (compact)
with cols[2]:
    st.markdown("**Range**")
    st.date_input("Start", value=st.session_state.get("start_date"), key="start_date")
    st.date_input("End", value=st.session_state.get("end_date"), key="end_date")

# Tools & Actions simplified
with cols[3]:
    st.markdown("**Tools**")
    if st.button("Reload cache", key="reload_cache"):
        # safer: clear cached data so load_all_data will fetch fresh on next run
        st.cache_data.clear()
        st.success("Cache cleared — data will reload on next interaction.")

st.markdown('</div>', unsafe_allow_html=True)

if not st.session_state["selected_products"]:
    st.warning("Select at least one product.")
    st.stop()

# Show selected badges
badges_html = "<div style='margin-bottom:8px;'>"
for sym in st.session_state["selected_products"]:
    badges_html += f"<span class='product-pill' style='border-left:4px solid {PRODUCT_CONFIG[sym]['color']};'>{sym}</span>"
badges_html += "</div>"
st.markdown(badges_html, unsafe_allow_html=True)

# View option pills (Outright / Spread / Fly)
st.markdown("**View**")
view_cols = st.columns(3)
view_cols[0].checkbox("Outright curves", value=st.session_state["view_outright"], key="view_outright")
view_cols[1].checkbox("Spreads (consecutive)", value=st.session_state["view_spread"], key="view_spread")
view_cols[2].checkbox("Flies", value=st.session_state["view_fly"], key="view_fly")

# ==================================================================================================
# MAIN DISPLAY — LOOP PRODUCTS
# ==================================================================================================
prods = st.session_state["selected_products"]

# find which dates to use for plotting
multi_dates = st.session_state.get("picked_multi_dates", []) or []
single_picked = st.session_state.get("picked_one_date", None)

# Helper to build list of sel_dates per product (ensures nearest on or before)
def get_sel_dates_for(df):
    if multi_dates:
        sel_dates = [nearest_date_on_or_before(df, d) for d in multi_dates]
        sel_dates = [d for d in sel_dates if d is not None]
    else:
        if single_picked:
            d_use = nearest_date_on_or_before(df, single_picked)
        else:
            d_use = nearest_date_on_or_before(df, st.session_state["end_date"])
        sel_dates = [d_use] if d_use else []
    return sel_dates

# ------------------ Outright curves ------------------
if st.session_state.get("view_outright"):
    st.markdown("## Outright Curves")
    normalize_curves = st.checkbox("Normalize curves (z-score)", value=False, key="normalize_all")
    show_values = st.checkbox("Show point values", value=False, key="show_values_all")

    cols_per_row = min(3, max(1, len(prods)))
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
            sel_dates = get_sel_dates_for(df)
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
            c.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

# ------------------ Spreads curves (per product) ------------------
# Spreads here means consecutive-contract spreads across contract indices for a single date
if st.session_state.get("view_spread"):
    st.markdown("## Spread Curves (consecutive contracts)")
    compare_mode = st.checkbox("Compare dates (overlay)", value=False, key="compare_spread")
    cols_per_row = min(3, max(1, len(prods)))
    rows = (len(prods) + cols_per_row - 1) // cols_per_row
    idx = 0
    for _ in range(rows):
        cols = st.columns(cols_per_row)
        for c in cols:
            if idx >= len(prods):
                break
            symbol = prods[idx]; idx += 1
            product_data = all_data.get(symbol)
            if not product_data: continue
            df, contracts = product_data["data"], product_data["contracts"]
            if len(contracts) < 2:
                c.caption(f"{symbol}: not enough contracts for spreads")
                continue
            # choose dates
            if compare_mode:
                # allow overlay multi-dates
                sel_dates = get_sel_dates_for(df) if multi_dates or single_picked else []
            else:
                # single date only
                use_date = get_sel_dates_for(df)
                sel_dates = [use_date[0]] if use_date else []
            if not sel_dates:
                c.warning(f"No spread for selected date(s) in {symbol}.")
                continue
            fig = go.Figure()
            for d_plot in sel_dates:
                row = df[df["Date"].dt.date == d_plot]
                if row.empty: continue
                # build consecutive spreads
                vals = row.iloc[0][contracts].astype(float).values
                spreads = [vals[i] - vals[i + 1] for i in range(len(vals) - 1)]
                xlabels = [f"{contracts[i]}-{contracts[i+1]}" for i in range(len(contracts) - 1)]
                fig.add_trace(
                    go.Scatter(x=xlabels, y=spreads, mode="lines+markers", name=str(d_plot), line=dict(color=PRODUCT_CONFIG[symbol]["color"], width=2))
                )
            fig = style_figure(fig, f"{PRODUCT_CONFIG[symbol]['name']} Spreads")
            fig.update_yaxes(title_text="Price Diff ($)")
            c.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

# ------------------ Quick Spreads & Flies (per product) compact area ------------------
st.markdown("## Quick Spreads & Flies (per product)")
qcols = st.columns(2)
with qcols[0]:
    for symbol in prods:
        df, contracts = all_data[symbol]["data"], all_data[symbol]["contracts"]
        if len(contracts) < 2:
            st.caption(f"{symbol}: not enough contracts for spreads")
            continue
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
                mean_series = sub.select_dtypes(include=[np.number]).mean(axis=1) if not sub.empty else None
                add_news_markers(fig, sub, df_news, y_series=mean_series if mean_series is not None else None)
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
                mean_series = sub.select_dtypes(include=[np.number]).mean(axis=1) if not sub.empty else None
                add_news_markers(fig, sub, df_news, y_series=mean_series if mean_series is not None else None)
                fig = style_figure(fig, f"{symbol} Flies")
                fig.update_yaxes(title_text="Price Diff ($)")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ------------------ Table toggle ------------------
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


