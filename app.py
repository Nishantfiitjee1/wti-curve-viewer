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

# Compact styling (buttons smaller, denser spacing)
st.markdown(
    """
<style>
    .stApp { background-color: #FFFFFF; color: #1E1E1E; }
    .header { background-color: #F0F2F6; padding: 6px 10px; border-radius: 6px; margin-bottom: 12px; border: 1px solid #E0E0E0; }
    h2 { color: #1E1E1E; border-bottom: 2px solid #00A8E8; padding-bottom: 8px; margin-top: 16px; font-size: 1.3rem; font-weight: bold; }
    .stButton>button { border-radius: 4px; padding: 2px 6px; border: 1px solid #B0B0B0; background-color: #FFFFFF; color: #333; font-weight: 500; transition: all 0.12s; height: 28px; font-size: 12px; }
    .stButton>button:hover { border-color: #00A8E8; color: #00A8E8; }
    .stDateInput { background-color: #FFFFFF; border-radius: 4px; }
    /* Make checkboxes more compact */
    div[data-testid="stCheckbox"] label { font-size: 12px; }
    /* make multiselect/inputs compact */
    .stMultiSelect, .stSelectbox, .stDateInput { font-size: 12px; }
    /* tighten dataframe padding */
    .element-container .stDataFrame { padding: 6px 6px 10px 6px; }
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
            st.warning(f"Could not load or parse sheet for {config['name']}. Please check format. Error: {e}")
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
        plot_bgcolor="#0F0F0F",  # keep plot bg dark if you prefer; change to '#F9F9F9' for light
        xaxis=dict(color="#DDD", gridcolor="#333", zeroline=False),
        yaxis=dict(color="#DDD", gridcolor="#333", zeroline=False),
        legend=dict(font=dict(color="#DDD"), yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
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
    Adds news markers to fig. Safe checks:
     - If df_news is empty -> nothing added.
     - If df_news has no non-Date columns -> nothing added.
     - If merged_df doesn't contain the news columns, skip dropna and attempt to match by Date.
    """
    if df_news is None or df_news.empty:
        return fig

    # identify news columns besides 'Date'
    news_cols = [c for c in df_news.columns if c != "Date"]
    if not news_cols:
        return fig

    # if merged_df doesn't have news_cols, try to align on Date only:
    if not set(news_cols).issubset(set(merged_df.columns)):
        # fallback: try to join on Date from df_news to merged_df to get rows that have any news info
        joined = pd.merge(merged_df, df_news, on="Date", how="left", suffixes=("", "_news"))
        # if still no news columns, exit
        if not any(col in joined.columns for col in news_cols):
            return fig
        news_df_in_view = joined.dropna(subset=news_cols, how="all")
    else:
        news_df_in_view = merged_df.dropna(subset=news_cols, how="all")

    if news_df_in_view.empty:
        return fig

    if y_series is None:
        # set y positions as NaN so markers draw at baseline; callers can pass specific y_series
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
            marker=dict(size=8, color="#FF6B6B", symbol="circle"),
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
if "current_view" not in st.session_state:
    st.session_state["current_view"] = "Curves"
if "outright_dates" not in st.session_state:
    st.session_state["outright_dates"] = {}

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
header_cols = st.columns([1.6, 3, 2.6, 1.2, 0.8])

# --- View Selection ---
with header_cols[0]:
    st.write("**Views**")
    view_buttons = ["Curves", "Compare", "Table", "Workspace"]
    for view in view_buttons:
        if st.button(view, key=f"btn_view_{view}", use_container_width=True):
            st.session_state.current_view = view

# --- Product Selection (compact checkboxes that persist) ---
with header_cols[1]:
    st.write("**Products**")
    # show compact checkboxes in a row (wrap if many)
    prod_cols = st.columns(len(PRODUCT_CONFIG))
    for i, (symbol, config) in enumerate(PRODUCT_CONFIG.items()):
        key = f"chk_prod_{symbol}"
        # initialize state if missing
        if key not in st.session_state:
            st.session_state[key] = symbol in st.session_state["selected_products"]
        checked = prod_cols[i].checkbox(symbol, value=st.session_state[key], key=key)
        st.session_state[key] = checked

    # update selected_products from the checkboxes
    st.session_state["selected_products"] = [s for s in PRODUCT_CONFIG.keys() if st.session_state.get(f"chk_prod_{s}", False)]

# --- Date Pickers ---
with header_cols[2]:
    st.write("**Date Range**")
    date_cols = st.columns(2)
    st.session_state.start_date = date_cols[0].date_input(
        "Start Date", value=st.session_state.start_date, key="start_date_picker", label_visibility="collapsed"
    )
    st.session_state.end_date = date_cols[1].date_input(
        "End Date", value=st.session_state.end_date, key="end_date_picker", label_visibility="collapsed"
    )

# --- Actions ---
with header_cols[3]:
    st.write("**Actions**")
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# visually highlight current view (still using CSS injection for button keys)
for view in view_buttons:
    if st.session_state.current_view == view:
        st.markdown(
            f"<style>#root .stButton button[key='btn_view_{view}'] {{background-color: #00A8E8; color: white;}}</style>",
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state.selected_products:
    st.warning("Please select at least one product to display.")
    st.stop()

# ==================================================================================================
# 5A. CURVES VIEW – Outright per-date overlays + per-product pickers
# ==================================================================================================
if st.session_state.current_view == "Curves":
    st.markdown("## Outright Curves – Select Dates & Compare")

    # Controls for outright dates
    ctl_cols = st.columns([2.2, 2.2, 1, 1])
    with ctl_cols[0]:
        global_dates_mode = st.radio(
            "Date selection mode",
            ["Latest per product", "Pick one date", "Pick multiple dates"],
            index=0,
            horizontal=True,
        )

    # build aggregated list of available dates (safe: use .dt.date)
    available_dates = set()
    for s in st.session_state.selected_products:
        df_s = all_data.get(s, {}).get("data")
        if df_s is None or df_s.empty:
            continue
        # df_s["Date"].dt.date already yields python date objects or numpy dates — no .date() call needed
        available_dates |= set(df_s["Date"].dt.date.dropna().unique())
    all_dates = sorted(available_dates, reverse=True)

    with ctl_cols[1]:
        if global_dates_mode == "Pick one date":
            if all_dates:
                picked_one_date = st.date_input("Outright date", value=all_dates[0], key="picked_one_date")
            else:
                picked_one_date = st.date_input("Outright date", value=date.today(), key="picked_one_date")
        elif global_dates_mode == "Pick multiple dates":
            picked_multi_dates = st.multiselect(
                "Outright dates (overlay)", options=all_dates, default=all_dates[:3] if len(all_dates) >= 3 else all_dates, key="picked_multi_dates"
            )

    with ctl_cols[2]:
        normalize_curves = st.checkbox("Normalize (z)", value=False)
    with ctl_cols[3]:
        show_values = st.checkbox("Show values", value=False)

    # Render per-product charts in an auto grid (up to 3 columns)
    prods = st.session_state.selected_products
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

            # prepare dates to plot
            if global_dates_mode == "Latest per product":
                d_use = nearest_date_on_or_before(df, st.session_state.end_date)
                sel_dates = [d_use] if d_use else []
            elif global_dates_mode == "Pick one date":
                d_use = nearest_date_on_or_before(df, picked_one_date)
                sel_dates = [d_use] if d_use else []
            else:  # multi
                sel_dates = [nearest_date_on_or_before(df, d) for d in picked_multi_dates] if picked_multi_dates else []
                sel_dates = [d for d in sel_dates if d is not None]

            if not sel_dates:
                c.warning(f"No curve for selected date(s) in {symbol}.")
                continue

            # build figure
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
                        mode="lines+markers",
                        name=str(d_plot),
                        line=dict(color=PRODUCT_CONFIG[symbol]["color"], width=2.5),
                        text=[f"{val:.2f}" for val in s.values] if show_values else None,
                        textposition="top center" if show_values else None,
                    )
                )
            ylab = "Z-score" if normalize_curves else "Price ($)"
            fig = style_figure(fig, f"{PRODUCT_CONFIG[symbol]['name']} Outright")
            fig.update_yaxes(title_text=ylab)
            c.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ============================ SPREAD & FLY in Curves view ============================
    st.markdown("## Quick Spreads & Flies (per product)")

    q1, q2 = st.columns(2)
    with q1:
        st.markdown("**Spreads**")
        for symbol in st.session_state.selected_products:
            df, contracts = all_data[symbol]["data"], all_data[symbol]["contracts"]
            if len(contracts) < 2:
                st.warning(f"{symbol}: not enough contracts for spreads")
                continue
            pairs = [f"{contracts[i]} - {contracts[i+1]}" for i in range(min(3, len(contracts)-1))]
            choices = st.multiselect(
                f"{symbol} pairs",
                options=[f"{c1} - {c2}" for i, c1 in enumerate(contracts) for c2 in contracts[i + 1 :]],
                default=pairs,
                key=f"spread_pairs_{symbol}",
            )
            if choices:
                sub = filter_by_date_window(df, st.session_state.start_date, st.session_state.end_date)
                if sub.empty:
                    st.warning(f"{symbol}: no data in range")
                else:
                    merged = pd.merge(sub, df_news, on="Date", how="left") if not df_news.empty else sub
                    fig = go.Figure()
                    for pair in choices:
                        c1n, c2n = [x.strip() for x in pair.split("-")]
                        series = spread_series(merged, c1n, c2n)
                        fig.add_trace(
                            go.Scatter(
                                x=merged["Date"],
                                y=series,
                                mode="lines",
                                name=f"{symbol} {c1n}-{c2n}",
                                hovertext=[
                                    f"<b>Date:</b> {d.strftime('%Y-%m-%d')}<br><b>{c1n}-{c2n}:</b> {v:.2f}"
                                    for d, v in zip(merged["Date"], series)
                                ],
                                hoverinfo="text",
                                line=dict(width=2),
                            )
                        )
                    add_news_markers(fig, merged, df_news)
                    fig = style_figure(fig, f"{symbol} Spreads")
                    fig.update_yaxes(title_text="Price Diff ($)")
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with q2:
        st.markdown("**Flies**")
        for symbol in st.session_state.selected_products:
            df, contracts = all_data[symbol]["data"], all_data[symbol]["contracts"]
            if len(contracts) < 3:
                st.warning(f"{symbol}: not enough contracts for flies")
                continue
            default_flies = [f"{contracts[i]} - {contracts[i+1]} - {contracts[i+2]}" for i in range(min(2, len(contracts)-2))]
            choices = st.multiselect(
                f"{symbol} flies",
                options=[
                    f"{a} - {b} - {c}"
                    for i, a in enumerate(contracts)
                    for j, b in enumerate(contracts[i + 1 :], start=i + 1)
                    for c in contracts[j + 1 :]
                ],
                default=default_flies,
                key=f"fly_sets_{symbol}",
            )
            if choices:
                sub = filter_by_date_window(df, st.session_state.start_date, st.session_state.end_date)
                if sub.empty:
                    st.warning(f"{symbol}: no data in range")
                else:
                    merged = pd.merge(sub, df_news, on="Date", how="left") if not df_news.empty else sub
                    fig = go.Figure()
                    for item in choices:
                        f1, f2, f3 = [x.strip() for x in item.split("-")]
                        series = fly_series(merged, f1, f2, f3)
                        fig.add_trace(
                            go.Scatter(
                                x=merged["Date"],
                                y=series,
                                mode="lines",
                                name=f"{symbol} {f1}-{f2}-{f3}",
                                hovertext=[
                                    f"<b>Date:</b> {d.strftime('%Y-%m-%d')}<br><b>Fly:</b> {v:.2f}"
                                    for d, v in zip(merged["Date"], series)
                                ],
                                hoverinfo="text",
                                line=dict(width=2),
                            )
                        )
                    add_news_markers(fig, merged, df_news)
                    fig = style_figure(fig, f"{symbol} Flies")
                    fig.update_yaxes(title_text="Price Diff ($)")
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ==================================================================================================
# 5B. COMPARE VIEW – Cross-product multi-compare (spreads & flies on one canvas)
# ==================================================================================================
elif st.session_state.current_view == "Compare":
    st.markdown("## Cross-Product Compare – Spreads & Flies")

    universe_spreads = []
    universe_flies = []
    for symbol in st.session_state.selected_products:
        contracts = all_data[symbol]["contracts"]
        universe_spreads += [
            f"{symbol}: {c1}-{c2}"
            for i, c1 in enumerate(contracts)
            for c2 in contracts[i + 1 :]
        ]
        universe_flies += [
            f"{symbol}: {a}-{b}-{c}"
            for i, a in enumerate(contracts)
            for j, b in enumerate(contracts[i + 1 :], start=i + 1)
            for c in contracts[j + 1 :]
        ]

    c1, c2 = st.columns(2)
    with c1:
        sel_spreads = st.multiselect(
            "Select spreads to compare",
            options=universe_spreads,
            default=universe_spreads[:4] if universe_spreads else [],
            key="sel_spreads"
        )
    with c2:
        sel_flies = st.multiselect(
            "Select flies to compare",
            options=universe_flies,
            default=universe_flies[:3] if universe_flies else [],
            key="sel_flies"
        )

    normalize_ts = st.checkbox("Normalize time series (z per series)", value=False)

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
            c1x, c2x = [p.strip() for p in pair.split("-")]
            df = all_data[sym]["data"]
            sub = filter_by_date_window(df, st.session_state.start_date, st.session_state.end_date)
            last_sub = sub
            if sub.empty:
                continue
            series = spread_series(sub, c1x, c2x)
            if normalize_ts:
                series = norm_series(series)
            figS.add_trace(
                go.Scatter(x=sub["Date"], y=series, mode="lines", name=f"{sym} {c1x}-{c2x}", line=dict(width=2))
            )
        add_news_markers(figS, last_sub if not last_sub.empty else pd.DataFrame(), df_news)
        figS = style_figure(figS, "Selected Spreads – Cross Product")
        figS.update_yaxes(title_text=("Z" if normalize_ts else "Price Diff ($)"))
        st.plotly_chart(figS, use_container_width=True, config={"displayModeBar": False})

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
            sub = filter_by_date_window(df, st.session_state.start_date, st.session_state.end_date)
            last_sub = sub
            if sub.empty:
                continue
            series = fly_series(sub, a, b, c)
            if normalize_ts:
                series = norm_series(series)
            figF.add_trace(
                go.Scatter(x=sub["Date"], y=series, mode="lines", name=f"{sym} {a}-{b}-{c}", line=dict(width=2))
            )
        add_news_markers(figF, last_sub if not last_sub.empty else pd.DataFrame(), df_news)
        figF = style_figure(figF, "Selected Flies – Cross Product")
        figF.update_yaxes(title_text=("Z" if normalize_ts else "Price Diff ($)"))
        st.plotly_chart(figF, use_container_width=True, config={"displayModeBar": False})

# ==================================================================================================
# 5C. TABLE VIEW – Simple tables per product for chosen range
# ==================================================================================================
elif st.session_state.current_view == "Table":
    st.markdown("## Data Table")
    for symbol in st.session_state.selected_products:
        product_data = all_data.get(symbol)
        if not product_data:
            continue
        df = product_data["data"]
        st.markdown(f"### {PRODUCT_CONFIG[symbol]['name']}")
        filtered_df = filter_by_date_window(df, st.session_state.start_date, st.session_state.end_date)
        if filtered_df.empty:
            st.warning(f"No data for {symbol} in the selected date range.")
        else:
            st.dataframe(filtered_df, use_container_width=True)

# ==================================================================================================
# 5D. WORKSPACE VIEW – See many things at once (efficient screen use)
# ==================================================================================================
elif st.session_state.current_view == "Workspace":
    st.markdown("## Trader Workspace – Outrights, Spreads, Flies at a glance")

    prods = st.session_state.selected_products
    cols_per_row = min(3, max(1, len(prods)))  # auto-adjust columns
    # Row 1: Latest Outrights for each product (auto date <= end_date)
    st.markdown("### Latest Outrights (by product)")
    idx = 0
    rows = (len(prods) + cols_per_row - 1) // cols_per_row
    for _ in range(rows):
        cols = st.columns(cols_per_row)
        for c in cols:
            if idx >= len(prods):
                break
            symbol = prods[idx]
            idx += 1
            df, contracts = all_data[symbol]["data"], all_data[symbol]["contracts"]
            d_use = nearest_date_on_or_before(df, st.session_state.end_date)
            if d_use is None:
                c.warning(f"{symbol}: no date available")
                continue
            row = df[df["Date"].dt.date == d_use]
            if row.empty:
                c.warning(f"{symbol}: no row for {d_use}")
                continue
            s = row.iloc[0][contracts].astype(float)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=contracts, y=s.values, mode="lines+markers", name=str(d_use), line=dict(color=PRODUCT_CONFIG[symbol]["color"], width=2.5)
                )
            )
            fig = style_figure(fig, f"{symbol} Outright ({d_use})")
            fig.update_yaxes(title_text="Price ($)")
            c.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Row 2: M1-M2 spreads for all products on ONE chart
    st.markdown("### M1-M2 Spreads – Cross Product")
    figS = go.Figure()
    any_added = False
    last_sub = pd.DataFrame()
    for symbol in prods:
        df, contracts = all_data[symbol]["data"], all_data[symbol]["contracts"]
        if len(contracts) < 2:
            continue
        sub = filter_by_date_window(df, st.session_state.start_date, st.session_state.end_date)
        last_sub = sub
        if sub.empty:
            continue
        series = spread_series(sub, contracts[0], contracts[1])
        figS.add_trace(
            go.Scatter(x=sub["Date"], y=series, mode="lines", name=f"{symbol} {contracts[0]}-{contracts[1]}")
        )
        any_added = True
    if any_added:
        add_news_markers(figS, last_sub if not last_sub.empty else pd.DataFrame(), df_news)
        figS = style_figure(figS, "M1-M2 – All Selected Products")
        figS.update_yaxes(title_text="Price Diff ($)")
        st.plotly_chart(figS, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No spreads available for the selection.")

    # Row 3: M1-M2-M3 flies for all products on ONE chart
    st.markdown("### M1-M2-M3 Flies – Cross Product")
    figF = go.Figure()
    any_added = False
    last_sub = pd.DataFrame()
    for symbol in prods:
        df, contracts = all_data[symbol]["data"], all_data[symbol]["contracts"]
        if len(contracts) < 3:
            continue
        sub = filter_by_date_window(df, st.session_state.start_date, st.session_state.end_date)
        last_sub = sub
        if sub.empty:
            continue
        series = fly_series(sub, contracts[0], contracts[1], contracts[2])
        figF.add_trace(
            go.Scatter(x=sub["Date"], y=series, mode="lines", name=f"{symbol} {contracts[0]}-{contracts[1]}-{contracts[2]}")
        )
        any_added = True
    if any_added:
        add_news_markers(figF, last_sub if not last_sub.empty else pd.DataFrame(), df_news)
        figF = style_figure(figF, "M1-M2-M3 – All Selected Products")
        figF.update_yaxes(title_text="Price Diff ($)")
        st.plotly_chart(figF, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No flies available for the selection.")
