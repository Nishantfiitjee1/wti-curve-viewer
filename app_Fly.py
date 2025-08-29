# app.py
import re
from datetime import datetime, timedelta
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Fly Curve Comparator (Trading)", layout="wide")
APP_TITLE = "🪁 Fly Curve Comparator — Trading Analysis"

# ---------------------------
# Helpers: column detection & date coercion
# ---------------------------
DATE_CANDIDATES = ["date", "trade_date", "dt", "timestamp", "time", "day", "as_of_date"]
CLOSE_CANDIDATES = ["close", "settlement", "settle", "last", "price", "px_close", "closing_price", "closeprice"]


def normalize_colname(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def find_best_column(cols, candidates):
    """
    Return original column name best matching candidates via normalize_colname
    """
    norm_map = {normalize_colname(c): c for c in cols}
    # direct
    for cand in candidates:
        if cand in norm_map:
            return norm_map[cand]
    # fuzzy: any candidate substring in normalized name
    for c in cols:
        n = normalize_colname(c)
        if any(cand in n for cand in candidates):
            return c
    return None


def infer_year_from_sheetname(sheet_name: str):
    """Try to infer a year (4-digit or 2-digit) from sheet name like 'CL_25_Fly' or 'CL_2024'"""
    m4 = re.search(r"(20\d{2})", sheet_name)
    if m4:
        return int(m4.group(1))
    m2 = re.search(r"(?<!\d)(\d{2})(?!\d)", sheet_name)
    if m2:
        yy = int(m2.group(1))
        # heuristics: 00-35 => 2000-2035
        if 0 <= yy <= 35:
            return 2000 + yy
        return 2000 + yy
    return None


def coerce_dates(series: pd.Series, sheet_year_hint: int | None = None) -> pd.Series:
    """
    Robust date coercion: accepts full dates or month/day-only strings like '4/21'.
    If year missing, uses sheet_year_hint or tries to infer from parsed dates, or falls back to current year.
    """
    s = series.copy()
    # quick parse
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if dt.notna().mean() > 0.5:
        return dt

    # if many are NaT, try handling yearless forms
    def parse_one(val):
        if pd.isna(val):
            return pd.NaT
        txt = str(val).strip()
        if not txt:
            return pd.NaT
        # if has 4-digit year
        if re.search(r"\b\d{4}\b", txt) or re.search(r"\d+/\d+/\d+", txt):
            try:
                return pd.to_datetime(txt, errors="coerce")
            except Exception:
                return pd.NaT
        # unify separators
        txt2 = re.sub(r"[.\-]", "/", txt)
        parts = txt2.split("/")
        if len(parts) >= 2:
            m = parts[0].zfill(2)
            d = parts[1].zfill(2)
            year = sheet_year_hint or datetime.now().year
            try:
                return datetime(year, int(m), int(d))
            except Exception:
                return pd.NaT
        return pd.NaT

    parsed = s.apply(parse_one)
    return pd.to_datetime(parsed, errors="coerce")


# ---------------------------
# Load Excel -> standard sheets
# ---------------------------
@st.cache_data(show_spinner=False)
def load_excel_file(file_like) -> dict:
    """
    Reads Excel file-like and returns dict: {sheet_name: DataFrame(Date: datetime, Close: float, MM-DD: str)}
    Auto-detects date & close columns (case/position-insensitive), coerces dates robustly.
    """
    xls = pd.ExcelFile(file_like)
    out = {}
    for sheet in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name=sheet)
        if raw.empty:
            continue
        cols = list(raw.columns)

        # find by name (normalized)
        date_col = find_best_column(cols, DATE_CANDIDATES)
        close_col = find_best_column(cols, CLOSE_CANDIDATES)

        # fallback heuristics
        if date_col is None:
            # try first datetime-like or first column that converts well
            best = None
            max_dt = -1
            for c in cols:
                try:
                    parsed = pd.to_datetime(raw[c], errors="coerce")
                    n_ok = parsed.notna().sum()
                    if n_ok > max_dt:
                        max_dt = n_ok
                        best = c
                except Exception:
                    pass
            date_col = best

        if close_col is None:
            # choose numeric-looking column (most numeric values)
            numeric_scores = {c: pd.to_numeric(raw[c], errors="coerce").notna().sum() for c in cols}
            sorted_scores = sorted(numeric_scores.items(), key=lambda x: x[1], reverse=True)
            close_col = sorted_scores[0][0] if sorted_scores else cols[-1]

        # if still missing, skip sheet
        if date_col is None or close_col is None:
            continue

        df = raw[[date_col, close_col]].copy()
        df.columns = ["Date", "Close"]

        # infer sheet-year other
        hint = infer_year_from_sheetname(sheet)
        df["Date"] = coerce_dates(df["Date"], hint)
        df = df.dropna(subset=["Date", "Close"])
        if df.empty:
            continue

        # ensure Close numeric
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)

        # Add MM-DD label for axis (but keep Date for sorting)
        df["MM-DD"] = df["Date"].dt.strftime("%m-%d")

        out[sheet] = df

    return out


# ---------------------------
# Synthetic alignment & rebase utilities
# ---------------------------
def align_to_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given df with Date & Close, map to synthetic timeline where first date of df becomes base at year 2000,
    and subsequent days are placed cumulatively to preserve original ordering and day gaps.
    Returns df with 'synthetic_date' and 'md_label' (MM/DD)
    """
    dfx = df.copy().dropna(subset=["Date", "Close"])
    dfx = dfx.sort_values("Date").reset_index(drop=True)
    if dfx.empty:
        return dfx

    first = dfx["Date"].iloc[0]
    base_year = 2000
    base = datetime(base_year, first.month, first.day)

    syn_dates = []
    last = first
    cum_days = 0
    for dt in dfx["Date"]:
        delta = (dt - last).days
        if delta < 0:
            # year wrap or disorder; treat as positive
            delta = abs(delta)
        cum_days += delta
        syn_dates.append(base + timedelta(days=cum_days))
        last = dt

    dfx["synthetic_date"] = pd.to_datetime(syn_dates)
    dfx["md_label"] = dfx["synthetic_date"].dt.strftime("%m-%d")
    return dfx


def rebase(series: pd.Series, mode: str):
    """mode: 'none'|'first0'|'pct'|'rebase100'"""
    if series.empty:
        return series
    first = series.iloc[0]
    if pd.isna(first) or first == 0:
        return series
    if mode == "first0":
        return series - first
    if mode == "pct":
        return (series / first - 1.0) * 100
    if mode == "rebase100":
        return (series / first) * 100
    return series


# ---------------------------
# Plotting utilities
# ---------------------------
def build_overlay_figure(sheet_dfs: dict, picks: list, x_mode: str, rebase_mode: str, ma_windows: list,
                         smooth_win: int, show_markers: bool, focus_sheet: str | None, show_legend: bool):
    fig = go.Figure()
    for name in picks:
        df = sheet_dfs.get(name)
        if df is None or df.empty:
            continue

        if smooth_win > 1:
            df_plot = df.copy()
            df_plot["Close"] = df_plot["Close"].rolling(smooth_win, min_periods=1).mean()
        else:
            df_plot = df.copy()

        if x_mode == "synthetic":
            dfx = align_to_synthetic(df_plot)
            x_vals = dfx["synthetic_date"]
            x_label = dfx["md_label"]
            hover_x = dfx["Date"].dt.strftime("%Y-%m-%d")
        else:
            dfx = df_plot.copy()
            x_vals = dfx["Date"]
            x_label = dfx["Date"].dt.strftime("%Y-%m-%d")
            hover_x = x_label

        y_vals = rebase(dfx["Close"], rebase_mode)

        # add primary
        lw = 4 if focus_sheet == name else 2
        mode = "lines+markers" if show_markers else "lines"
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode=mode,
            name=name,
            line=dict(width=lw),
            hovertemplate="<b>%{text}</b><br>X: %{customdata[0]}<br>Actual: %{customdata[1]}<br>Y: %{y:.5f}<extra></extra>",
            text=[name] * len(dfx),
            customdata=np.stack([x_label, hover_x], axis=-1)
        ))

        # add moving averages (if any)
        for window in ma_windows:
            if window and window > 1:
                ma = dfx["Close"].rolling(window, min_periods=1).mean()
                y_ma = rebase(ma, rebase_mode)
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_ma,
                    mode="lines",
                    name=f"{name} MA{window}",
                    line=dict(width=1.5, dash="dash")
                ))

    # layout
    y_title = {
        "none": "Close",
        "first0": "Change from first (abs)",
        "pct": "% change from first",
        "rebase100": "Rebased (first=100)"
    }.get(rebase_mode, "Close")

    x_title = "Synthetic MM-DD (aligned)" if x_mode == "synthetic" else "Calendar Date (YYYY-MM-DD)"
    fig.update_layout(
        template="plotly_dark",
        height=620,
        title="Fly Curves — Overlay",
        xaxis=dict(title=x_title, showgrid=True),
        yaxis=dict(title=y_title, showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0, bgcolor="rgba(0,0,0,0)") if show_legend else dict(visible=False),
        margin=dict(l=10, r=10, t=60, b=10)
    )
    return fig


def build_spread_figure(sheet_dfs: dict, sheet_a: str, sheet_b: str, ma_window: int | None, x_mode: str, smooth_win: int):
    """
    Build a figure for spread = A - B (interpolates based on Date index)
    We will merge on actual Date via an outer join and compute difference.
    """
    a = sheet_dfs.get(sheet_a)
    b = sheet_dfs.get(sheet_b)
    if a is None or b is None:
        return None, None

    # prepare copies and smoothing
    A = a.copy().sort_values("Date").reset_index(drop=True)
    B = b.copy().sort_values("Date").reset_index(drop=True)
    if smooth_win > 1:
        A["Close"] = A["Close"].rolling(smooth_win, min_periods=1).mean()
        B["Close"] = B["Close"].rolling(smooth_win, min_periods=1).mean()

    merged = pd.merge(A[["Date", "Close"]].rename(columns={"Close": f"Close_{sheet_a}"}),
                      B[["Date", "Close"]].rename(columns={"Close": f"Close_{sheet_b}"}),
                      on="Date", how="outer").sort_values("Date").reset_index(drop=True)

    # forward/back fill to align - or leave gaps? Using linear interpolation for fairness
    merged[f"Close_{sheet_a}"] = merged[f"Close_{sheet_a}"].interpolate().ffill().bfill()
    merged[f"Close_{sheet_b}"] = merged[f"Close_{sheet_b}"].interpolate().ffill().bfill()

    merged["Spread"] = merged[f"Close_{sheet_a}"] - merged[f"Close_{sheet_b}"]

    if ma_window and ma_window > 1:
        merged[f"MA{ma_window}_Spread"] = merged["Spread"].rolling(ma_window, min_periods=1).mean()

    # x values
    if x_mode == "synthetic":
        # create synthetic aligned axis from first date of merged
        merged = merged.dropna(subset=["Date"])
        syn = align_to_synthetic(merged.rename(columns={"Date": "Date", "Spread": "Close"}))
        x_vals = syn["synthetic_date"]
        hover_x = syn["actual_date"].dt.strftime("%Y-%m-%d")
        y_vals = syn["Close"]
        merged_for_plot = pd.DataFrame({"x": x_vals, "y": y_vals, "actual": hover_x})
        if ma_window:
            merged_for_plot["ma"] = syn["Close"].rolling(ma_window, min_periods=1).mean()
    else:
        x_vals = merged["Date"]
        hover_x = merged["Date"].dt.strftime("%Y-%m-%d")
        y_vals = merged["Spread"]
        merged_for_plot = pd.DataFrame({"x": x_vals, "y": y_vals, "actual": hover_x})
        if ma_window:
            merged_for_plot["ma"] = merged["Spread"].rolling(ma_window, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged_for_plot["x"], y=merged_for_plot["y"], mode="lines", name=f"{sheet_a} - {sheet_b}"))
    if ma_window:
        fig.add_trace(go.Scatter(x=merged_for_plot["x"], y=merged_for_plot["ma"], mode="lines", name=f"MA{ma_window} of Spread", line=dict(dash="dash")))
    fig.update_layout(title=f"Spread: {sheet_a} - {sheet_b}", template="plotly_dark", height=520,
                      xaxis_title="Synthetic MM-DD" if x_mode == "synthetic" else "Calendar Date", yaxis_title="Spread")
    return fig, merged


def seasonality_chart(sheet_dfs: dict, sheet_name: str, group_by_months: bool = True):
    """
    Build a seasonality chart: aggregate the same MM-DD or month across different years.
    - If group_by_months=True => show average per month across years (Apr->Mar properly aligned)
    - Else show average per MM-DD (may be many unique days)
    """
    df = sheet_dfs.get(sheet_name)
    if df is None or df.empty:
        return None

    d = df.copy()
    d["year"] = d["Date"].dt.year
    d["mmdd"] = d["Date"].dt.strftime("%m-%d")
    d["month"] = d["Date"].dt.month

    # Align Apr-Mar -> treat months <4 as next year for seasonal roll
    d.loc[d["month"] < 4, "year_roll"] = d.loc[d["month"] < 4, "year"] + 1
    d.loc[d["month"] >= 4, "year_roll"] = d.loc[d["month"] >= 4, "year"]
    d["season_year"] = d["year_roll"].astype(int)

    if group_by_months:
        # compute mean per season month (Apr -> Mar)
        # create an ordered month list starting at Apr
        order = list(range(4, 13)) + list(range(1, 4))
        agg = d.groupby([d["season_year"], d["month"]])["Close"].mean().reset_index()
        # pivot: columns season_year; index month
        pivot = agg.pivot(index="month", columns="season_year", values="Close").reindex(order)
        # labels
        x_labels = [datetime(2000, m, 1).strftime("%b") for m in pivot.index]
        fig = go.Figure()
        for col in pivot.columns:
            fig.add_trace(go.Scatter(x=x_labels, y=pivot[col].values, mode="lines+markers", name=str(col)))
        fig.update_layout(title=f"Seasonality by Month — {sheet_name} (Apr→Mar)", template="plotly_dark", height=520,
                          xaxis_title="Season Month (Apr→Mar)", yaxis_title="Avg Close")
        return fig
    else:
        # group by mmdd — may be many points; show average across season years
        agg = d.groupby(["mmdd", "season_year"])["Close"].mean().reset_index()
        # pivot with mmdd as index sorted by synthetic alignment starting from earliest observed mmdd
        # We'll sort mmdd by calendar order starting Apr
        mm = sorted(agg["mmdd"].unique(), key=lambda x: (int(x.split("-")[0]), int(x.split("-")[1])))
        pivot = agg.pivot(index="mmdd", columns="season_year", values="Close").reindex(mm)
        fig = go.Figure()
        for col in pivot.columns:
            fig.add_trace(go.Scatter(x=pivot.index, y=pivot[col].values, mode="lines", name=str(col)))
        fig.update_layout(title=f"Seasonality by MM-DD — {sheet_name}", template="plotly_dark", height=520,
                          xaxis_title="MM-DD", yaxis_title="Avg Close")
        return fig


# ---------------------------
# Stats & utility
# ---------------------------
def compute_basic_stats(sheet_dfs: dict, picks: list):
    rows = []
    for name in picks:
        df = sheet_dfs.get(name)
        if df is None or df.empty:
            continue
        vals = df["Close"]
        rows.append({
            "Sheet": name,
            "Start Date": df["Date"].min().strftime("%Y-%m-%d"),
            "End Date": df["Date"].max().strftime("%Y-%m-%d"),
            "Start": vals.iloc[0],
            "End": vals.iloc[-1],
            "Mean": vals.mean(),
            "StdDev": vals.std(),
            "Min": vals.min(),
            "Max": vals.max(),
            "Pct Change": (vals.iloc[-1] / vals.iloc[0] - 1.0) * 100.0 if vals.iloc[0] != 0 else np.nan,
        })
    return pd.DataFrame(rows).set_index("Sheet")


# ---------------------------
# App UI
# ---------------------------
st.title(APP_TITLE)
st.markdown("Upload your Excel (or use built-in). App auto-detects Date & Close columns and provides trading analysis tools.")

# Sidebar controls
with st.sidebar:
    st.header("Controls")

    data_source = st.radio("Data Source", options=["built-in", "upload"], index=0, help="Built-in loads provided FLY_CHART.xlsx in /mnt/data; upload allows user file")

    uploaded = None
    if data_source == "upload":
        uploaded = st.file_uploader("Upload Excel (.xlsx) with sheets", type=["xlsx", "xls"], accept_multiple_files=False)

    # plot options
    st.subheader("Display")
    x_mode = st.selectbox("X-axis style", ["synthetic", "calendar"], format_func=lambda v: "Synthetic (aligned to first date)" if v == "synthetic" else "Calendar (YYYY-MM-DD)")
    rebase = st.selectbox("Y-axis transform", options=["none", "first0", "pct", "rebase100"], index=0,
                          format_func=lambda v: {"none": "Absolute Close", "first0": "Change from first (abs)", "pct": "% Change from first", "rebase100": "Rebase: first=100"}[v])
    smooth_win = st.slider("Smoothing (rolling window days)", min_value=1, max_value=15, value=1, step=1)
    ma_choices = st.multiselect("Overlay Moving Averages (days)", options=[5, 10, 20, 50], default=[10])

    st.subheader("Overlay / Focus")
    show_legend = st.checkbox("Show Legend", value=True)
    show_markers = st.checkbox("Show Markers", value=False)
    focus_sheet = st.selectbox("Focus sheet (thicker)", options=["(none)"], index=0)

    st.subheader("Spread Analysis")
    enable_spread = st.checkbox("Enable spread calculation (A - B)", value=False)
    spread_sheet_a = st.selectbox("Spread A (left)", options=["(none)"], index=0)
    spread_sheet_b = st.selectbox("Spread B (right)", options=["(none)"], index=0)
    spread_ma = st.selectbox("Spread MA window", options=[None, 5, 10, 20], index=0)

    st.subheader("Seasonality")
    enable_seasonality = st.checkbox("Show seasonality charts", value=False)
    season_by_month = st.checkbox("Seasonality aggregated by month (Apr→Mar)", value=True)

    st.markdown("---")
    st.caption("Pro tip: Use 'synthetic' X-axis to compare curves starting at different months (e.g., Apr vs Jun) on a common MM-DD timeline.")


# ---------------------------
# Load data (built-in or uploaded)
# ---------------------------
# Default built-in path (should be in /mnt/data in colab environment)
builtin_path = "FLY_CHART.xlsx"
sheet_dfs = {}
try:
    if data_source == "upload" and uploaded is not None:
        sheet_dfs = load_excel_file(uploaded)
    else:
        # load built-in
        with open(builtin_path, "rb") as fh:
            sheet_dfs = load_excel_file(BytesIO(fh.read()))
except FileNotFoundError:
    st.error("Built-in FLY_CHART.xlsx not found at /mnt/data/FLY_CHART.xlsx. Upload your file or ensure the built-in file exists.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load Excel: {e}")
    st.stop()

if not sheet_dfs:
    st.warning("No sheets detected or no valid Date/Close columns found. Please check your file.")
    st.stop()

sheet_names = list(sheet_dfs.keys())

# Update spread selectors choices (dependent on loaded sheets)
# (we can't update the sidebar selectboxes after creation, so we instead create small selects below if spread enabled)
# Main selection area
st.markdown("### Select sheets to compare")
default_picks = sheet_names if len(sheet_names) <= 6 else sheet_names[:6]
picks = st.multiselect("Pick sheets (overlay)", options=sheet_names, default=default_picks)

if focus_sheet == "(none)":
    focus_sheet = None

# show quick metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Sheets loaded", len(sheet_names))
with col2:
    nrows_total = sum(len(df) for df in sheet_dfs.values())
    st.metric("Total rows", nrows_total)
with col3:
    all_first = [df["Date"].min() for df in sheet_dfs.values()]
    all_last = [df["Date"].max() for df in sheet_dfs.values()]
    if all_first and all_last:
        span_days = (max(all_last) - min(all_first)).days
        st.metric("Date span (days)", span_days)

# ---------------------------
# Main overlay plot
# ---------------------------
if not picks:
    st.info("Choose at least one sheet to plot from the multiselect above.")
else:
    overlay_fig = build_overlay_figure(sheet_dfs, picks, x_mode, rebase, ma_choices, smooth_win, show_markers, focus_sheet, show_legend)
    st.plotly_chart(overlay_fig, use_container_width=True)

    # Stats
    st.subheader("📊 Summary Statistics")
    stats_df = compute_basic_stats(sheet_dfs, picks)
    st.dataframe(stats_df.style.format({
        "Start": "{:.6f}", "End": "{:.6f}", "Mean": "{:.6f}", "StdDev": "{:.6f}", "Min": "{:.6f}", "Max": "{:.6f}", "Pct Change": "{:.2f}%"
    }))

    # download overlay data (concatenate combined plotting frames)
    export_rows = []
    for name in picks:
        df = sheet_dfs[name].copy()
        df_to_export = df[["Date", "MM-DD", "Close"]].copy()
        df_to_export["Sheet"] = name
        export_rows.append(df_to_export)
    export_df = pd.concat(export_rows, ignore_index=True)
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download overlay CSV (Date, MM-DD, Close, Sheet)", csv_bytes, file_name="fly_overlay_export.csv", mime="text/csv")

# ---------------------------
# Spread calculation section
# ---------------------------
if enable_spread:
    st.markdown("---")
    st.subheader("⚖️ Spread Analysis (A - B)")
    # pick A and B here reliably
    sp_cols = st.columns(3)
    with sp_cols[0]:
        sp_a = st.selectbox("Spread A (left)", options=["(none)"] + sheet_names, index=0)
    with sp_cols[1]:
        sp_b = st.selectbox("Spread B (right)", options=["(none)"] + sheet_names, index=0)
    with sp_cols[2]:
        sp_ma = st.selectbox("MA window (Spread)", options=[None, 5, 10, 20], index=0)

    if sp_a == "(none)" or sp_b == "(none)":
        st.info("Select both A and B sheets to compute spread.")
    elif sp_a == sp_b:
        st.warning("Pick two different sheets for spread calculation.")
    else:
        fig_spread, merged = build_spread_figure(sheet_dfs, sp_a, sp_b, sp_ma, x_mode, smooth_win)
        if fig_spread is None:
            st.error("Could not compute spread for the chosen sheets.")
        else:
            st.plotly_chart(fig_spread, use_container_width=True)
            st.markdown("**Spread data preview (first 10 rows)**")
            st.dataframe(merged.head(10).assign(Date=lambda d: d["Date"].dt.strftime("%Y-%m-%d")))
            csv_spread = merged.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download spread CSV", csv_spread, file_name=f"spread_{sp_a}_minus_{sp_b}.csv", mime="text/csv")

# ---------------------------
# Seasonality analysis
# ---------------------------
if enable_seasonality:
    st.markdown("---")
    st.subheader("🌀 Seasonality Analysis")
    sheet_for_season = st.selectbox("Select sheet for seasonality", options=sheet_names, index=0)
    group_by_months = season_by_month

    fig_seas = seasonality_chart(sheet_dfs, sheet_for_season, group_by_months)
    if fig_seas is None:
        st.warning("No data for seasonality on selected sheet.")
    else:
        st.plotly_chart(fig_seas, use_container_width=True)

# ---------------------------
# Footer / Help
# ---------------------------
st.markdown("---")
st.markdown("#### Notes & Tips")
st.markdown("""
- The app keeps the full `Date` (YYYY-MM-DD) internally for correct chronological ordering while allowing `MM-DD` display to compare across start months (April→March etc.).
- Use **Synthetic** X-axis to align different start months on the same MM-DD timeline.
- **Rebase 100** is useful to compare relative performance from initial level.
- Spread calculations perform an outer merge on Date and use interpolation to align values before computing A - B.
- If your incoming data uses `MM-DD` only (no year), the app tries to infer the year from the sheet name (e.g., `CL_25_Fly` → 2025) or defaults to current year.
""")

st.caption("If you want additional trading features (autodetect breakouts, signal generation, or pair-trade suggestions based on historical correlation/cointegration), tell me and I will add those modules next.")
