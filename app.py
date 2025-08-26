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
st.markdown(
    """
<style>
    /* Main app styling for light theme */
    .stApp { background-color: #FFFFFF; color: #1E1E1E; }
    /* Header styling */
    .header { background-color: #F0F2F6; padding: 8px 15px; border-radius: 8px; margin-bottom: 25px; border: 1px solid #E0E0E0; }
    /* Section headers */
    h2 { color: #1E1E1E; border-bottom: 2px solid #00A8E8; padding-bottom: 10px; margin-top: 25px; font-size: 1.6rem; font-weight: bold; }
    /* Custom button styling for product/view selection */
    .stButton>button { border-radius: 5px; padding: 4px 10px; border: 1px solid #B0B0B0; background-color: #FFFFFF; color: #333; font-weight: 500; transition: all 0.2s; height: 32px; }
    .stButton>button:hover { border-color: #00A8E8; color: #00A8E8; }
    /* Styling for selected buttons */
    .stButton>button.selected { background-color: #00A8E8; color: white; border: 1px solid #00A8E8; }
    /* Date picker styling */
    .stDateInput { background-color: #FFFFFF; border-radius: 5px; }
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
    Returns: (all_product_data: dict, df_news: DataFrame)
    all_product_data[symbol] = {"data": df (Date desc), "contracts": [M1, M2, ...]}
    """
    all_product_data = {}
    if not os.path.exists(MASTER_EXCEL_FILE):
        return {}, pd.DataFrame()

    for symbol, config in PRODUCT_CONFIG.items():
        try:
            df_raw = pd.read_excel(
                MASTER_EXCEL_FILE, sheet_name=config["sheet"], header=None, engine="openpyxl"
            )
            # find header and data start rows based on the 'Dates' keyword in col 0
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
            df_news = news_df_raw.dropna(subset=["Date"]).sort_values("Date")
    return all_product_data, df_news


def style_figure(fig: go.Figure, title: str) -> go.Figure:
    """Applies the custom light theme styling to a Plotly figure for a professional look."""
    fig.update_layout(
        title=dict(text=title, font=dict(color="#333", size=16), x=0.5, y=0.95),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="#F9F9F9",
        xaxis=dict(color="#333", gridcolor="#EAEAEA", zeroline=False),
        yaxis=dict(color="#333", gridcolor="#EAEAEA", zeroline=False),
        legend=dict(font=dict(color="#333"), yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified",
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


def filter_by_date_window(df: pd.DataFrame, start_d: date, end_d: date) -> pd.DataFrame:
    return df[(df["Date"].dt.date >= start_d) & (df["Date"].dt.date <= end_d)]


def nearest_date_on_or_before(df: pd.DataFrame, target_d: date) -> date | None:
    """Return the nearest available date (<= target) present in df['Date'] or None if none."""
    subset = df[df["Date"].dt.date <= target_d]
    if subset.empty:
        return None
    return subset["Date"].max().date()


def add_news_markers(fig: go.Figure, merged_df: pd.DataFrame, df_news: pd.DataFrame, y_series: pd.Series):
    if df_news.empty:
        return fig

    news_cols = df_news.columns.drop("Date")
    # Ensure news columns exist in the merged dataframe before trying to drop NaNs
    existing_news_cols = [col for col in news_cols if col in merged_df.columns]
    if not existing_news_cols:
        return fig

    news_df_in_view = merged_df.dropna(subset=existing_news_cols, how="all")
    if news_df_in_view.empty:
        return fig

    news_hover_text = news_df_in_view.apply(
        lambda row: f"<b>Date:</b> {row['Date'].strftime('%Y-%m-%d')}<br><hr>"
        + "<br>".join(
            f"<b>{col.replace('_', ' ')}:</b> {row[col]}" for col in news_cols if pd.notna(row[col])
        ),
        axis=1,
    )
    fig.add_trace(
        go.Scatter(
            x=news_df_in_view["Date"],
            y=y_series.loc[news_df_in_view.index],
            mode="markers",
            name="News",
            marker=dict(size=9, color="#FF6B6B", symbol="circle"),
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
if "layout_cols" not in st.session_state:
    st.session_state["layout_cols"] = 3

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
header_cols = st.columns([1.5, 3, 3, 1.2, 0.8])

# --- View Selection ---
with header_cols[0]:
    st.write("**Views**")
    view_buttons = ["Curves", "Compare", "Table", "Workspace"]
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
                if len(st.session_state.selected_products) > 1:
                    st.session_state.selected_products.remove(symbol)
            else:
                st.session_state.selected_products.append(symbol)

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

# --- Layout & Actions ---
with header_cols[3]:
    st.write("**Layout**")
    st.session_state.layout_cols = st.slider(
        "Charts per row", min_value=1, max_value=4, value=st.session_state.layout_cols, key="cols_slider"
    )

with header_cols[4]:
    st.write("**Actions**")
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# highlight selected buttons
for view in view_buttons:
    if st.session_state.current_view == view:
        st.markdown(
            f"<style>#root .stButton button[key='btn_view_{view}'] {{background-color: #00A8E8; color: white;}}</style>",
            unsafe_allow_html=True,
        )
for symbol in PRODUCT_CONFIG:
    if symbol in st.session_state.selected_products:
        st.markdown(
            f"<style>#root .stButton button[key='btn_prod_{symbol}'] {{background-color: #00A8E8; color: white;}}</style>",
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
    ctl_cols = st.columns([2, 2, 1, 1])
    with ctl_cols[0]:
        global_dates_mode = st.radio(
            "Date selection mode",
            ["Latest per product", "Pick one date", "Pick multiple dates"],
            index=0,
            horizontal=True,
        )
    with ctl_cols[1]:
        if global_dates_mode == "Pick one date":
            all_dates = sorted(
                set(
                    d.date()
                    for s in st.session_state.selected_products
                    for d in all_data[s]["data"]["Date"].dt.date.unique()
                ),
                reverse=True,
            )
            picked_one_date = st.date_input("Outright date", value=all_dates[0] if all_dates else date.today())
        elif global_dates_mode == "Pick multiple dates":
            all_dates = sorted(
                set(
                    d.date()
                    for s in st.session_state.selected_products
                    for d in all_data[s]["data"]["Date"].dt.date.unique()
                ),
                reverse=True,
            )
            picked_multi_dates = st.multiselect(
                "Outright dates (overlay)", options=all_dates, default=all_dates[:3] if len(all_dates) >= 3 else all_dates
            )
    with ctl_cols[2]:
        normalize_curves = st.checkbox("Normalize (z)" , value=False)
    with ctl_cols[3]:
        show_values = st.checkbox("Show values", value=False)

    # Render per-product charts in a grid
    prods = st.session_state.selected_products
    rows = (len(prods) + st.session_state.layout_cols - 1) // st.session_state.layout_cols
    idx = 0
    for _ in range(rows):
        cols = st.columns(st.session_state.layout_cols)
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
                sel_dates = [nearest_date_on_or_before(df, d) for d in picked_multi_dates]
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
                        mode="lines+markers" + ("+text" if show_values else ""),
                        name=str(d_plot),
                        line=dict(color=PRODUCT_CONFIG[symbol]["color"], width=3),
                        text=[f"{val:.2f}" for val in s.values],
                        textposition="top center",
                    )
                )
            ylab = "Z-score" if normalize_curves else "Price ($)"
            fig = style_figure(fig, f"{PRODUCT_CONFIG[symbol]['name']} Outright")
            fig.update_yaxes(title_text=ylab)
            c.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    
# ==================================================================================================
# 5B. COMPARE VIEW – Cross-product multi-compare (spreads & flies on one canvas)
# ==================================================================================================
elif st.session_state.current_view == "Compare":
    st.markdown("## Cross-Product Compare – Spreads & Flies")

    # Build selectable universe of series across products
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
            default=[s for s in universe_spreads if ": CL1 Comdty-CL2 Comdty" in s or ": CO1 Comdty-CO2 Comdty" in s][:4],
        )
    with c2:
        sel_flies = st.multiselect(
            "Select flies to compare",
            options=universe_flies,
            default=[f for f in universe_flies if ": CL1 Comdty-CL2 Comdty-CL3 Comdty" in f][:3],
        )

    normalize_ts = st.checkbox("Normalize time series (z per series)", value=False)

    def norm_series(s: pd.Series) -> pd.Series:
        std = s.std()
        return (s - s.mean()) / (std if std not in (0, np.nan) else 1)

    # One combined chart for spreads
    if sel_spreads:
        figS = go.Figure()
        for item in sel_spreads:
            sym, pair = item.split(":")
            sym = sym.strip()
            c1x, c2x = [p.strip() for p in pair.split("-")]
            df = all_data[sym]["data"]
            sub = filter_by_date_window(df, st.session_state.start_date, st.session_state.end_date)
            if sub.empty:
                continue
            series = spread_series(sub, c1x, c2x)
            if normalize_ts:
                series = norm_series(series)
            figS.add_trace(
                go.Scatter(x=sub["Date"], y=series, mode="lines", name=f"{sym} {c1x}-{c2x}", line=dict(width=2))
            )
        add_news_markers(figS, sub if not sub.empty else pd.DataFrame(), df_news, series)
        figS = style_figure(figS, "Selected Spreads – Cross Product")
        figS.update_yaxes(title_text=("Z-Score" if normalize_ts else "Price Diff ($)"))
        st.plotly_chart(figS, use_container_width=True, config={"displayModeBar": False})

    # One combined chart for flies
    if sel_flies:
        figF = go.Figure()
        for item in sel_flies:
            sym, trip = item.split(":")
            sym = sym.strip()
            a, b, c = [p.strip() for p in trip.split("-")]
            df = all_data[sym]["data"]
            sub = filter_by_date_window(df, st.session_state.start_date, st.session_state.end_date)
            if sub.empty:
                continue
            series = fly_series(sub, a, b, c)
            if normalize_ts:
                series = norm_series(series)
            figF.add_trace(
                go.Scatter(x=sub["Date"], y=series, mode="lines", name=f"{sym} {a}-{b}-{c}", line=dict(width=2))
            )
        add_news_markers(figF, sub if not sub.empty else pd.DataFrame(), df_news, series)
        figF = style_figure(figF, "Selected Flies – Cross Product")
        figF.update_yaxes(title_text=("Z-Score" if normalize_ts else "Price Diff ($)"))
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
# 5D. WORKSPACE VIEW – High-density overview
# ==================================================================================================
elif st.session_state.current_view == "Workspace":
    st.markdown("## Trader Workspace – Outrights, Spreads, Flies at a glance")

    prods = st.session_state.selected_products
    cols_per_row = st.session_state.layout_cols

    # Row 1: Latest Outrights
    st.markdown("### Latest Outrights (by product)")
    idx = 0
    rows = (len(prods) + cols_per_row - 1) // cols_per_row
    for _ in range(rows):
        cols = st.columns(cols_per_row)
        for c in cols:
            if idx >= len(prods): break
            symbol = prods[idx]
            idx += 1
            df, contracts = all_data[symbol]["data"], all_data[symbol]["contracts"]
            d_use = nearest_date_on_or_before(df, st.session_state.end_date)
            if d_use is None:
                c.warning(f"{symbol}: no date available")
                continue
            row = df[df["Date"].dt.date == d_use]
            if row.empty: continue
            s = row.iloc[0][contracts].astype(float)
            fig = go.Figure(go.Scatter(x=contracts, y=s.values, mode="lines+markers", name=str(d_use), line=dict(color=PRODUCT_CONFIG[symbol]["color"], width=3)))
            fig = style_figure(fig, f"{symbol} Outright ({d_use})")
            c.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Row 2: M1-M2 spreads on ONE chart
    st.markdown("### M1-M2 Spreads – Cross Product")
    figS = go.Figure()
    any_added = False
    for symbol in prods:
        df, contracts = all_data[symbol]["data"], all_data[symbol]["contracts"]
        if len(contracts) < 2: continue
        sub = filter_by_date_window(df, st.session_state.start_date, st.session_state.end_date)
        if sub.empty: continue
        series = spread_series(sub, contracts[0], contracts[1])
        figS.add_trace(go.Scatter(x=sub["Date"], y=series, mode="lines", name=f"{symbol} {contracts[0]}-{contracts[1]}", line=dict(color=PRODUCT_CONFIG[symbol]["color"])))
        any_added = True
    if any_added:
        add_news_markers(figS, sub, df_news, series)
        figS = style_figure(figS, "M1-M2 – All Selected Products")
        st.plotly_chart(figS, use_container_width=True, config={"displayModeBar": False})

    # Row 3: M1-M2-M3 flies on ONE chart
    st.markdown("### M1-M2-M3 Flies – Cross Product")
    figF = go.Figure()
    any_added = False
    for symbol in prods:
        df, contracts = all_data[symbol]["data"], all_data[symbol]["contracts"]
        if len(contracts) < 3: continue
        sub = filter_by_date_window(df, st.session_state.start_date, st.session_state.end_date)
        if sub.empty: continue
        series = fly_series(sub, contracts[0], contracts[1], contracts[2])
        figF.add_trace(go.Scatter(x=sub["Date"], y=series, mode="lines", name=f"{symbol} {contracts[0]}-{contracts[1]}-{contracts[2]}", line=dict(color=PRODUCT_CONFIG[symbol]["color"])))
        any_added = True
    if any_added:
        add_news_markers(figF, sub, df_news, series)
        figF = style_figure(figF, "M1-M2-M3 – All Selected Products")
        st.plotly_chart(figF, use_container_width=True, config={"displayModeBar": False})
