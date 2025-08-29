import re
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from io import BytesIO
from datetime import datetime

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fly Curve Comparator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Data Loading and Processing Utilities
# -----------------------------------------------------------------------------

def find_target_column(columns: list[str], candidates: list[str]) -> str | None:
    """
    Finds the best matching column name from a prioritized list of candidates.
    The logic is as follows:
    1.  It first checks for an exact (case-insensitive) match for each candidate in order.
    2.  If no exact match is found, it then checks for any column that CONTAINS
        a candidate keyword (e.g., finding "fly" in a column named "fly_price").

    Args:
        columns: The list of column names in the DataFrame.
        candidates: A prioritized list of keywords to search for (e.g., ["close", "fly"]).

    Returns:
        The best matching column name or None if no suitable column is found.
    """
    normalized_cols_map = {col.lower().replace("_", "").replace(" ", ""): col for col in columns}

    # Priority 1: Check for an exact match from the candidates list (in order).
    for cand in candidates:
        if cand in normalized_cols_map:
            return normalized_cols_map[cand]

    # Priority 2: If no exact match, check for a partial match (e.g., 'fly' in 'fly_value').
    for cand in candidates:
        for norm_col, orig_col in normalized_cols_map.items():
            if cand in norm_col:
                return orig_col

    return None

def infer_year_from_sheetname(sheet_name: str) -> int | None:
    """
    Infers the year from the sheet name (e.g., 'CL_25_Fly' -> 2025).
    """
    match = re.search(r'(20\d{2})|_(\d{2})', sheet_name)
    if match:
        year_part = match.group(1) or match.group(2)
        year = int(year_part)
        return 2000 + year if year < 100 else year
    return None

def process_sheet(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame | None:
    """
    Processes a single sheet to standardize it for plotting.
    """
    # Define prioritized lists of keywords for column detection.
    date_candidates = ["date", "time", "day"]
    price_candidates = ["close", "fly", "settle", "price"]

    date_col = find_target_column(df.columns, date_candidates)
    close_col = find_target_column(df.columns, price_candidates)

    if not date_col or not close_col:
        return None

    processed_df = df[[date_col, close_col]].copy()
    processed_df.columns = ["Date", "Close"]

    processed_df.dropna(subset=["Date", "Close"], inplace=True)
    processed_df["Close"] = pd.to_numeric(processed_df["Close"], errors='coerce')
    processed_df.dropna(subset=["Close"], inplace=True)

    if processed_df.empty:
        return None

    # Handle various date formats (including MM/DD strings).
    try:
        # Attempt direct conversion first, which is fast for standard formats.
        processed_df["Date"] = pd.to_datetime(processed_df["Date"])
    except (ValueError, TypeError):
        # Fallback for non-standard formats like 'MM/DD'.
        sheet_year = infer_year_from_sheetname(sheet_name) or datetime.now().year
        def parse_custom_date(d):
            try:
                dt = pd.to_datetime(d, infer_datetime_format=True)
                if dt.year == datetime.now().year and sheet_year != datetime.now().year:
                    return dt.replace(year=sheet_year)
                return dt
            except (ValueError, TypeError):
                return pd.NaT
        processed_df["Date"] = processed_df["Date"].apply(parse_custom_date)

    processed_df.dropna(subset=["Date"], inplace=True)
    processed_df.sort_values("Date", inplace=True, ignore_index=True)
    
    if processed_df.empty:
        return None

    # Smartly handle seasonal year rollovers (e.g., Feb to Jan).
    start_month = processed_df["Date"].iloc[0].month
    rollover_mask = processed_df["Date"].dt.month < start_month
    processed_df.loc[rollover_mask, "Date"] = processed_df.loc[rollover_mask, "Date"].apply(
        lambda d: d.replace(year=d.year + 1)
    )
    processed_df.sort_values("Date", inplace=True, ignore_index=True)

    # Calculate the X-axis value: "Months from Start"
    start_date = processed_df["Date"].iloc[0]
    processed_df["Months_from_Start"] = (processed_df["Date"] - start_date).dt.days / 30.44

    return processed_df


@st.cache_data(show_spinner="Loading and processing Excel file...")
def load_and_process_excel(file_source) -> dict[str, pd.DataFrame]:
    """
    Loads an Excel file and processes each sheet.
    """
    try:
        xls = pd.ExcelFile(file_source)
        processed_sheets = {}
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            processed_df = process_sheet(df, sheet_name)
            if processed_df is not None and not processed_df.empty:
                processed_sheets[sheet_name] = processed_df
        return processed_sheets
    except Exception as e:
        st.error(f"An error occurred while reading the Excel file: {e}")
        return {}

# -----------------------------------------------------------------------------
# Plotting Utilities
# -----------------------------------------------------------------------------

def create_comparison_chart(data: dict[str, pd.DataFrame], selected_sheets: list[str], ma_windows: list[int], focus_sheet: str | None):
    """
    Creates the main Plotly chart for comparing fly curves.
    """
    fig = go.Figure()
    for name in selected_sheets:
        df = data.get(name)
        if df is None: continue
        line_width, opacity = (4, 1.0) if name == focus_sheet else (2, 0.8)
        fig.add_trace(go.Scatter(
            x=df["Months_from_Start"], y=df["Close"], mode='lines', name=name,
            line=dict(width=line_width), opacity=opacity,
            hovertemplate=f"<b>{name}</b><br>Date: %{{customdata|%Y-%m-%d}}<br>Months: %{{x:.2f}}<br>Price: %{{y:.4f}}<extra></extra>",
            customdata=df["Date"]
        ))
        for window in ma_windows:
            if window > 1:
                ma_series = df["Close"].rolling(window=window, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=df["Months_from_Start"], y=ma_series, mode='lines', name=f"{name} {window}-day MA",
                    line=dict(width=1.5, dash='dash'), opacity=0.7, visible='legendonly'
                ))
    fig.update_layout(
        height=600, title="Fly Curve Comparison", xaxis_title="Months from Start Date",
        yaxis_title="Close Price", template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------

st.title("📈 Trading Fly Curve Comparator")
st.markdown("This tool visualizes and compares the price evolution of different 'fly' contracts over their lifecycle. Upload your Excel file or use the built-in sample data to get started.")

with st.sidebar:
    st.header("⚙️ Controls")
    source_option = st.radio("Select Data Source", ("Use Built-in Sample", "Upload Your Excel File"))
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"]) if source_option == "Upload Your Excel File" else None

    all_sheets_data = {}
    if source_option == "Use Built-in Sample":
        try:
            all_sheets_data = load_and_process_excel("FLY_CHART.xlsx")
        except FileNotFoundError:
            st.error("Built-in 'FLY_CHART.xlsx' not found. Please upload a file.")
    elif uploaded_file:
        all_sheets_data = load_and_process_excel(uploaded_file)

    if all_sheets_data:
        sheet_names = list(all_sheets_data.keys())
        st.markdown("---")
        st.header("📊 Chart Options")
        selected_sheets = st.multiselect("Select Sheets to Plot", sheet_names, default=sheet_names[:min(len(sheet_names), 5)])
        focus_sheet = st.selectbox("Highlight a Curve", [None] + selected_sheets, format_func=lambda x: "None" if x is None else x)
        ma_windows = st.multiselect("Add Moving Averages (days)", [5, 10, 20, 50, 100], default=[])
    else:
        st.info("Please select or upload an Excel file to begin analysis.")

if not all_sheets_data:
    st.warning("No data loaded. Please select a data source from the sidebar.")
elif not selected_sheets:
    st.info("👈 Please select at least one sheet from the sidebar to display the chart.")
else:
    st.header("Curve Comparison Chart")
    chart = create_comparison_chart(all_sheets_data, selected_sheets, ma_windows, focus_sheet)
    st.plotly_chart(chart, use_container_width=True)

    with st.expander("Show Data Summary and Export Options"):
        st.subheader("Summary Statistics")
        summary_rows = [
            {
                "Sheet": name, "Start Date": df["Date"].min().strftime('%Y-%m-%d'),
                "End Date": df["Date"].max().strftime('%Y-%m-%d'),
                "Duration (Days)": (df["Date"].max() - df["Date"].min()).days,
                "Start Price": df["Close"].iloc[0], "End Price": df["Close"].iloc[-1],
                "Min Price": df["Close"].min(), "Max Price": df["Close"].max(),
                "Avg Price": df["Close"].mean(),
            }
            for name in selected_sheets if (df := all_sheets_data.get(name)) is not None
        ]
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows).set_index("Sheet")
            st.dataframe(summary_df.style.format("{:,.4f}", subset=["Start Price", "End Price", "Min Price", "Max Price", "Avg Price"]))

        st.subheader("Export Processed Data")
        export_dfs = []
        for name in selected_sheets:
            if (df := all_sheets_data.get(name)) is not None:
                df_copy = df.copy()
                df_copy['Sheet'] = name
                export_dfs.append(df_copy)
        
        if export_dfs:
            full_export_df = pd.concat(export_dfs, ignore_index=True)
            csv_data = full_export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Data as CSV", data=csv_data,
                file_name="fly_curve_data_export.csv", mime="text/csv"
            )
