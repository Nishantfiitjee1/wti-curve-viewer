
import io
import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =====================================================================================
# PAGE CONFIG
# =====================================================================================
st.set_page_config(
    page_title="Fly Curve Comparator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------- THEME & STYLES -----------------------------------
PRIMARY = "#4F46E5"   # indigo-600
ACCENT = "#22C55E"    # green-500
BG = "#0B1220"        # dark bg (optional)
CARD_BG = "#0F172A"   # slate-900
TEXT = "#E5E7EB"      # slate-200
MUTED = "#94A3B8"     # slate-400
GRID = "#1F2937"      # grid lines

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, #0b1220 0%, #0e1526 70%);
        color: {TEXT};
        font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
    }}
    .title {{
        font-size: 28px;
        font-weight: 800;
        color: {TEXT};
        letter-spacing: 0.3px;
    }}
    .subtitle {{
        color: {MUTED};
        font-size: 14px;
    }}
    .pill {{
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(79,70,229,0.15);
        color: {TEXT};
        border: 1px solid rgba(79,70,229,0.35);
        font-size: 12px;
        margin-right: 8px;
    }}
    .card {{
        background: {CARD_BG};
        border: 1px solid #1f2937;
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }}
    .metric {{
        font-size: 22px;
        font-weight: 800;
        color: {TEXT};
    }}
    .muted {{
        color: {MUTED};
        font-size: 12px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================================================
# HELPERS
# =====================================================================================

DATE_CANDIDATES = [
    "date", "trade_date", "dt", "timestamp", "time", "day", "as_of_date"
]
CLOSE_CANDIDATES = [
    "close", "settlement", "settle", "last", "price", "px_close", "closing_price", "closeprice"
]

def normalize_col(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(col).strip().lower())

def find_col(columns, candidates):
    norm_map = {normalize_col(c): c for c in columns}
    for cand in candidates:
        if cand in norm_map:
            return norm_map[cand]
    # fuzzy fallback: contains candidate substring
    for c in columns:
        nc = normalize_col(c)
        if any(cand in nc for cand in candidates):
            return c
    return None

def coerce_dates(date_series: pd.Series, sheet_year_hint: int | None = None) -> pd.Series:
    """
    Convert a column to datetimes. Handles strings like '4/21' by injecting a year.
    If year missing and no hint, default to most frequent inferred year or current year.
    """
    s = date_series.copy()

    # if already datetime-like
    if np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s, errors="coerce")

    # Try direct parse first
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

    # If we got enough successes, return
    if dt.notna().mean() > 0.5:
        return dt

    # Otherwise handle yearless entries like "4/21" or "04/21"
    def add_year(val):
        if pd.isna(val):
            return np.nan
        txt = str(val).strip()
        if not txt:
            return np.nan
        # if already has a year (e.g., 2024/04/21 or 04/21/24)
        if re.search(r"\b\d{4}\b", txt) or re.search(r"\d+/\d+/\d+", txt):
            try:
                return pd.to_datetime(txt, errors="coerce")
            except Exception:
                return np.nan
        # yearless like 4/21 or 04-21
        txt = re.sub(r"[.\-]", "/", txt)
        parts = txt.split("/")
        if len(parts) >= 2:
            m, d = parts[0], parts[1]
            year = sheet_year_hint or datetime.now().year
            try:
                return datetime(int(year), int(m), int(d))
            except Exception:
                return np.nan
        return np.nan

    # Guess year hint from any successful direct parse
    if sheet_year_hint is None and dt.notna().any():
        sheet_year_hint = int(dt.dropna().dt.year.mode().iloc[0])

    filled = s.apply(add_year)
    return pd.to_datetime(filled, errors="coerce")


def infer_sheet_year(sheet_name: str, fallback: int | None = None) -> int | None:
    """
    Try to parse a 2-digit or 4-digit year from sheet name like 'CL_25_Fly' or 'CL_2024'.
    Returns a likely calendar year, translating '25'->2025, '16'->2016, etc.
    """
    m4 = re.search(r"(20\d{2})", sheet_name)
    if m4:
        return int(m4.group(1))
    m2 = re.search(r"(?<!\d)(\d{2})(?!\d)", sheet_name)  # better: last 2-digit group
    if m2:
        yy = int(m2.group(1))
        # heuristic window 2010..2035
        if 0 <= yy <= 35:
            return 2000 + yy
        # coarse fallback
        return 2000 + yy
    return fallback


def align_to_synthetic_year(df: pd.DataFrame, date_col: str, start_from_first=True) -> pd.DataFrame:
    """
    Map actual dates to a synthetic timeline to compare curves across different years.
    - If start_from_first=True: day 0 = first observed date per sheet.
    - Output columns: ['synthetic_date', 'md_label', 'actual_date', 'Close']
    where md_label is 'MM/DD' string for axis ticks.
    """
    dfx = df.copy().dropna(subset=[date_col, "Close"])
    dfx = dfx.sort_values(date_col).reset_index(drop=True)
    if dfx.empty:
        dfx["synthetic_date"] = pd.NaT
        dfx["md_label"] = np.nan
        dfx["actual_date"] = np.nan
        return dfx

    first_date = dfx[date_col].iloc[0]
    base_year = 2000
    base = datetime(base_year, first_date.month, first_date.day)

    # ensure strictly increasing timeline
    syn_dates = []
    last_dt = first_date
    offset_days = 0
    for dt_ in dfx[date_col]:
        # if time went backwards due to year wrap, compute delta from last
        delta = (dt_ - last_dt).days
        if delta < 0:
            # assume wrap over new year; add absolute delta but keep cumulative
            delta = abs(delta)
        offset_days += delta
        syn_dates.append(base + timedelta(days=offset_days))
        last_dt = dt_

    dfx["synthetic_date"] = syn_dates
    dfx["md_label"] = dfx["synthetic_date"].dt.strftime("%m/%d")
    dfx["actual_date"] = dfx[date_col]
    return dfx


@st.cache_data(show_spinner=False)
def load_excel(file_like) -> dict:
    xls = pd.ExcelFile(file_like)
    out = {}
    for sheet in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name=sheet)
        if raw.empty:
            continue
        cols = list(raw.columns)
        date_col = find_col([normalize_col(c) for c in cols], DATE_CANDIDATES)
        close_col = find_col([normalize_col(c) for c in cols], CLOSE_CANDIDATES)

        # map back to real column name
        # we need original column whose normalized matches the found normalized
        def original_from_norm(norm_name):
            for c in cols:
                if normalize_col(c) == norm_name:
                    return c
            return None

        date_col_real = original_from_norm(date_col) if date_col else None
        close_col_real = original_from_norm(close_col) if close_col else None

        # if still not found, try loose guess: first datetime-like & first numeric-like
        if date_col_real is None:
            # choose a column that looks like date by name or dtype
            for c in cols:
                nc = normalize_col(c)
                if any(k in nc for k in DATE_CANDIDATES):
                    date_col_real = c
                    break
            if date_col_real is None:
                # pick first col and hope
                date_col_real = cols[0]

        if close_col_real is None:
            # pick by name similarity or numeric dtype
            for c in cols:
                nc = normalize_col(c)
                if any(k in nc for k in CLOSE_CANDIDATES):
                    close_col_real = c
                    break
            if close_col_real is None:
                # fallback: pick rightmost numeric column
                numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(raw[c])]
                close_col_real = numeric_cols[-1] if numeric_cols else cols[-1]

        # build cleaned df
        df = raw[[date_col_real, close_col_real]].copy()
        df.columns = ["Date", "Close"]

        # coerce dates
        hint_year = infer_sheet_year(sheet, None)
        df["Date"] = coerce_dates(df["Date"], hint_year)
        df = df.dropna(subset=["Date", "Close"])
        if df.empty:
            continue

        # ensure numeric close
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"])

        out[sheet] = df.sort_values("Date")
    return out


def rebase_series(y: pd.Series, mode: str) -> pd.Series:
    """
    mode: 'none' | 'first=0' | 'pct_from_first'
    """
    if y.empty:
        return y
    first = y.iloc[0]
    if pd.isna(first):
        return y
    if mode == "first=0":
        return y - first
    if mode == "pct_from_first":
        return (y / first - 1.0) * 100.0
    return y


# =====================================================================================
# SIDEBAR CONTROLS
# =====================================================================================

with st.sidebar:
    st.markdown('<div class="title">⚙️ Controls</div>', unsafe_allow_html=True)
    st.write("")

    built_in_path = "FLY_CHART.xlsx"
    st.session_state.setdefault("data_source", "built-in")
    source = st.radio("Data Source", ["built-in", "upload"], horizontal=True)
    st.session_state["data_source"] = source

    uploaded = None
    if source == "upload":
        uploaded = st.file_uploader(
            "Upload Excel (.xlsx) with Date & Close columns (any order)",
            type=["xlsx", "xls"], accept_multiple_files=False
        )

    # rebase mode
    rebase = st.selectbox(
        "Compare as",
        options=["Absolute Close", "Change from first", "% Change from first"],
        index=0
    )
    rebase_mode = {"Absolute Close": "none", "Change from first": "first=0", "% Change from first": "pct_from_first"}[rebase]

    # x-axis style
    x_mode = st.selectbox("X-axis", ["Synthetic (start→next months)", "Actual Calendar"], index=0)

    # smoothing
    smooth_win = st.slider("Rolling Smoothing (days)", min_value=1, max_value=15, value=1, step=1)

    # selection
    st.write("")
    st.markdown("**Overlay Options**")
    show_legend = st.toggle("Show Legend", True)
    show_markers = st.toggle("Show Markers", False)


# =====================================================================================
# LOAD DATA
# =====================================================================================

file_to_load = uploaded if uploaded is not None else built_in_path

try:
    sheets = load_excel(file_to_load)
except Exception as e:
    st.error(f"Failed to read Excel: {e}")
    st.stop()

if not sheets:
    st.warning("No valid sheets found with (Date, Close). Please check your file.")
    st.stop()

sheet_names = list(sheets.keys())

# =====================================================================================
# HEADER
# =====================================================================================
left, right = st.columns([0.68, 0.32])
with left:
    st.markdown('<div class="title">🪁 Fly Curve Comparator</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Overlay and compare Close curves across sheets — regardless of month/year. Auto-detects Date & Close. </div>', unsafe_allow_html=True)
    st.markdown('<span class="pill">Excel → Curves</span><span class="pill">Auto Date Parsing</span><span class="pill">Synthetic Year Alignment</span><span class="pill">Normalization</span>', unsafe_allow_html=True)

with right:
    # quick metrics
    nrows_total = sum(len(df) for df in sheets.values())
    first_dates = [df["Date"].min() for df in sheets.values() if not df.empty]
    last_dates = [df["Date"].max() for df in sheets.values() if not df.empty]
    mind = min(first_dates) if first_dates else None
    maxd = max(last_dates) if last_dates else None

    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric">{len(sheets)}</div><div class="muted">Sheets</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric">{nrows_total}</div><div class="muted">Total Rows</div>', unsafe_allow_html=True)
    if mind and maxd:
        span = (maxd - mind).days
        c3.markdown(f'<div class="metric">{span}d</div><div class="muted">Date Span</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =====================================================================================
# SHEET PICKER & MODE
# =====================================================================================
selector_col, focus_col = st.columns([0.65, 0.35])
with selector_col:
    pick = st.multiselect("Select sheets to overlay", options=sheet_names, default=sheet_names)
with focus_col:
    focus = st.selectbox("Focus sheet (thicker line)", options=["(none)"] + sheet_names, index=0)


# =====================================================================================
# PREPARE DATA FOR PLOT
# =====================================================================================
plot_rows = []

for name in pick:
    df = sheets[name].copy().sort_values("Date")

    # optional smoothing
    if smooth_win and smooth_win > 1:
        df["Close"] = df["Close"].rolling(smooth_win, min_periods=1).mean()

    if x_mode.startswith("Synthetic"):
        dfx = align_to_synthetic_year(df, "Date")
        x_vals = dfx["synthetic_date"]
        x_show = dfx["md_label"]
        actual = dfx["actual_date"]
    else:
        dfx = df.copy()
        x_vals = dfx["Date"]
        x_show = dfx["Date"].dt.strftime("%Y-%m-%d")
        actual = dfx["Date"]

    y_vals = rebase_series(dfx["Close"], rebase_mode)

    plot_rows.append(pd.DataFrame({
        "sheet": name,
        "x": x_vals,
        "x_label": x_show,
        "y": y_vals,
        "actual_date": actual
    }))

if not plot_rows:
    st.warning("Nothing to plot. Check your selections.")
    st.stop()

plot_df = pd.concat(plot_rows, ignore_index=True)


# =====================================================================================
# PLOT
# =====================================================================================
fig = go.Figure()

for name in pick:
    sub = plot_df[plot_df["sheet"] == name]
    if sub.empty:
        continue
    line_width = 3.5 if focus == name else 2.0
    fig.add_trace(go.Scatter(
        x=sub["x"],
        y=sub["y"],
        mode="lines+markers" if show_markers else "lines",
        name=name,
        line=dict(width=line_width),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"  # sheet
            "X: %{customdata[1]}<br>"
            "Actual Date: %{customdata[2]}<br>"
            "Y: %{y:.5f}<extra></extra>"
        ),
        customdata=np.stack([sub["sheet"], sub["x_label"], sub["actual_date"].dt.strftime("%Y-%m-%d")], axis=-1)
    ))

y_title = {
    "none": "Close",
    "first=0": "Change from First (abs)",
    "pct_from_first": "% Change from First"
}[rebase_mode]

x_title = "Synthetic Month/Day (aligned by first date)" if x_mode.startswith("Synthetic") else "Calendar Date"

fig.update_layout(
    height=620,
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0, bgcolor="rgba(0,0,0,0)") if show_legend else dict(visible=False),
    xaxis=dict(
        title=x_title,
        showgrid=True, gridcolor="rgba(255,255,255,0.08)",
        zeroline=False
    ),
    yaxis=dict(
        title=y_title,
        showgrid=True, gridcolor="rgba(255,255,255,0.08)",
        zeroline=False
    ),
    title=dict(
        text="Fly Curves — Close Overlay",
        x=0.02, xanchor="left"
    )
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================================================
# DATA PREVIEW & EXPORT
# =====================================================================================
st.markdown("### 📄 Data Preview")
tabs = st.tabs([f"{nm} ({len(sheets[nm])} rows)" for nm in sheet_names])
for t, nm in zip(tabs, sheet_names):
    with t:
        st.dataframe(sheets[nm].assign(Date=sheets[nm]["Date"].dt.strftime("%Y-%m-%d")))

st.write("")
colA, colB = st.columns([0.5, 0.5])

with colA:
    st.download_button(
        "⬇️ Download overlay data (CSV)",
        data=plot_df.assign(
            x=plot_df["x"].astype(str),
            actual_date=plot_df["actual_date"].dt.strftime("%Y-%m-%d")
        ).to_csv(index=False).encode("utf-8"),
        file_name="fly_curves_overlay.csv",
        mime="text/csv"
    )

with colB:
    if uploaded is None:
        st.info("Using built-in Excel. To try your own, switch to 'upload' in the sidebar.")
    else:
        st.success("Custom Excel loaded.")
