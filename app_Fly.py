import re
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from io import BytesIO
from datetime import datetime
import numpy as np

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Advanced Fly Curve Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Data Loading and Processing Engine
# -----------------------------------------------------------------------------

def find_target_column(columns: list[str], candidates: list[str]) -> str | None:
    """
    Finds the best matching column name from a prioritized list of candidates.
    It checks for exact matches first, then for partial matches (e.g., 'fly' in 'fly_price').
    """
    normalized_cols_map = {col.lower().replace("_", "").replace(" ", ""): col for col in columns}
    for cand in candidates:
        if cand in normalized_cols_map:
            return normalized_cols_map[cand]
    for cand in candidates:
        for norm_col, orig_col in normalized_cols_map.items():
            if cand in norm_col:
                return orig_col
    return None

def infer_year_from_sheetname(sheet_name: str) -> int:
    """Infers the year from the sheet name (e.g., 'CL_25_Fly' -> 2025), defaulting to the current year."""
    match = re.search(r'(20\d{2})|_(\d{2})', sheet_name)
    if match:
        year_part = match.group(1) or match.group(2)
        year = int(year_part)
        return 2000 + year if year < 100 else year
    return datetime.now().year

def process_sheet(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame | None:
    """
    Robustly processes a single sheet to standardize it for plotting and analysis.
    This version includes a definitive fix for the 'day is out of range' error.
    """
    date_candidates = ["date", "time", "day"]
    price_candidates = ["close", "fly", "settle", "price"]

    date_col = find_target_column(df.columns, date_candidates)
    close_col = find_target_column(df.columns, price_candidates)
    if not date_col or not close_col: return None

    processed_df = df[[date_col, close_col]].copy()
    processed_df.columns = ["Date", "Close"]
    processed_df.dropna(subset=["Date", "Close"], inplace=True)
    processed_df["Close"] = pd.to_numeric(processed_df["Close"], errors='coerce')
    processed_df.dropna(subset=["Close"], inplace=True)
    if processed_df.empty: return None

    # --- Bulletproof Date Parsing Engine ---
    # Attempt standard date conversion first.
    parsed_dates = pd.to_datetime(processed_df['Date'], errors='coerce')

    # If standard parsing fails for most rows, activate the robust MM/DD parser.
    if parsed_dates.isna().mean() > 0.5:
        sheet_year = infer_year_from_sheetname(sheet_name)
        
        def parse_md_format(val):
            try:
                # Standardize separators and extract parts
                txt = str(val).strip().replace('-', '/')
                parts = txt.split('/')
                month, day = int(parts[0]), int(parts[1])
                # CRITICAL FIX: Use a known leap year (2000) for initial parsing.
                # This ensures '2/29' is always accepted at this stage.
                return datetime(2000, month, day)
            except Exception:
                return pd.NaT

        # Apply the safe parser
        parsed_dates = processed_df['Date'].apply(parse_md_format)
        processed_df['Date'] = parsed_dates.dropna()
        if processed_df.empty: return None

        # Safely apply the correct year, gracefully handling Feb 29 in non-leap years.
        def apply_correct_year(date):
            try:
                return date.replace(year=sheet_year)
            except ValueError:  # This triggers on dates like 29th Feb in a non-leap year
                if date.month == 2 and date.day == 29:
                    return date.replace(year=sheet_year, day=28) # Fallback to 28th
                raise
        processed_df['Date'] = processed_df['Date'].apply(apply_correct_year)
    else:
        processed_df['Date'] = parsed_dates
    
    processed_df.dropna(subset=["Date"], inplace=True)
    processed_df.sort_values("Date", inplace=True, ignore_index=True)
    if processed_df.empty: return None

    # --- Seasonal Year Rollover Logic ---
    start_month = processed_df["Date"].iloc[0].month
    rollover_mask = processed_df["Date"].dt.month < start_month
    processed_df.loc[rollover_mask, "Date"] = processed_df.loc[rollover_mask, "Date"].apply(
        lambda d: d.replace(year=d.year + 1)
    )
    processed_df.sort_values("Date", inplace=True, ignore_index=True)

    # --- Final Calculations ---
    start_date = processed_df["Date"].iloc[0]
    processed_df["Months_from_Start"] = (processed_df["Date"] - start_date).dt.days / 30.44
    processed_df['Daily_Return'] = processed_df['Close'].pct_change()

    return processed_df

@st.cache_data(show_spinner="Loading and processing Excel file...")
def load_and_process_excel(file_source) -> dict[str, pd.DataFrame]:
    """Loads an Excel file and processes each sheet."""
    try:
        xls = pd.ExcelFile(file_source)
        processed_sheets = {
            sheet_name: process_sheet(pd.read_excel(xls, sheet_name=sheet_name), sheet_name)
            for sheet_name in xls.sheet_names
        }
        return {k: v for k, v in processed_sheets.items() if v is not None and not v.empty}
    except Exception as e:
        st.error(f"An error occurred while reading the Excel file: {e}")
        return {}

# -----------------------------------------------------------------------------
# Analysis and Plotting Utilities
# -----------------------------------------------------------------------------

def create_comparison_chart(data: dict, selected_sheets: list):
    """Creates the main Plotly chart for comparing fly curves."""
    fig = go.Figure()
    for name in selected_sheets:
        df = data.get(name)
        if df is None: continue
        fig.add_trace(go.Scatter(
            x=df["Months_from_Start"], y=df["Close"], mode='lines', name=name,
            hovertemplate=f"<b>{name}</b><br>Date: %{{customdata|%Y-%m-%d}}<br>Months: %{{x:.2f}}<br>Price: %{{y:.4f}}<extra></extra>",
            customdata=df["Date"]
        ))
    fig.update_layout(height=600, title="Fly Curve Comparison", xaxis_title="Months from Start Date", yaxis_title="Close Price", template="plotly_dark")
    return fig

def calculate_trading_stats(data: dict, selected_sheets: list) -> pd.DataFrame:
    """Calculates advanced performance and volatility statistics."""
    stats = []
    for name in selected_sheets:
        df = data.get(name)
        if df is None or len(df) < 2: continue
        
        annualized_vol = df['Daily_Return'].std() * np.sqrt(252)
        cum_returns = (1 + df['Daily_Return']).cumprod()
        peak = cum_returns.expanding(min_periods=1).max()
        drawdown = (cum_returns/peak) - 1
        max_drawdown = drawdown.min()
        sharpe_ratio = (df['Daily_Return'].mean() / df['Daily_Return'].std()) * np.sqrt(252) if df['Daily_Return'].std() != 0 else 0

        stats.append({
            "Sheet": name,
            "Annualized Volatility": f"{annualized_vol:.2%}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Avg Price": f"{df['Close'].mean():.4f}",
            "Min Price": f"{df['Close'].min():.4f}",
            "Max Price": f"{df['Close'].max():.4f}"
        })
    return pd.DataFrame(stats).set_index("Sheet")

# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------

st.title("📈 Advanced Trading Fly Curve Analyzer")
st.markdown("An intelligent tool to analyze and compare fly contracts. It automatically handles various data formats and provides key trading insights.")

with st.sidebar:
    st.header("⚙️ Controls")
    source_option = st.radio(
        "Select Data Source",
        ("Use Built-in Sample", "Upload Your Excel File"),
        index=0, help="The built-in sample is always available."
    )
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"]) if source_option == "Upload Your Excel File" else None

all_sheets_data = {}
# Load built-in data by default
if source_option == "Use Built-in Sample":
    try:
        all_sheets_data = load_and_process_excel("FLY_CHART.xlsx")
    except FileNotFoundError:
        st.sidebar.error("Built-in 'FLY_CHART.xlsx' not found. Please ensure it's in the same directory.")

# If user uploads a file, it replaces the built-in data
if uploaded_file:
    all_sheets_data = load_and_process_excel(uploaded_file)

if not all_sheets_data:
    st.warning("No data loaded. Please select or upload a valid Excel file to begin.")
else:
    sheet_names = list(all_sheets_data.keys())
    with st.sidebar:
        st.markdown("---")
        st.header("📊 Analysis Options")
        selected_sheets = st.multiselect("Select Sheets to Analyze", sheet_names, default=sheet_names[:min(len(sheet_names), 5)])

    if not selected_sheets:
        st.info("👈 Please select at least one sheet from the sidebar to display the analysis.")
    else:
        tab1, tab2 = st.tabs(["📈 Curve Comparison", "📊 Statistics & Volatility"])
        
        with tab1:
            st.header("Price Curve Comparison")
            chart = create_comparison_chart(all_sheets_data, selected_sheets)
            st.plotly_chart(chart, use_container_width=True)
        
        with tab2:
            st.header("Performance & Volatility Metrics")
            stats_df = calculate_trading_stats(all_sheets_data, selected_sheets)
            st.dataframe(stats_df, use_container_width=True)
