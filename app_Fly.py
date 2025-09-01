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
#
# These functions are responsible for loading the Excel file, finding the
# correct columns, and processing the dates to create a standardized format
# for plotting.
# -----------------------------------------------------------------------------

def find_target_column(columns: list[str], candidates: list[str]) -> str | None:
    """
    Finds the best matching column name from a list of candidates.
    It checks for exact matches first, then for partial matches.

    Args:
        columns: The list of column names in the DataFrame.
        candidates: A list of possible names for the target column (e.g., ["date", "timestamp"]).

    Returns:
        The best matching column name or None if no suitable column is found.
    """
    normalized_cols = {col.lower().replace("_", "").replace(" ", ""): col for col in columns}
    for cand in candidates:
        if cand in normalized_cols:
            return normalized_cols[cand]
    for norm_col, orig_col in normalized_cols.items():
        if any(cand in norm_col for cand in candidates):
            return orig_col
    return None

def infer_year_from_sheetname(sheet_name: str) -> int | None:
    """
    Infers the year from the sheet name (e.g., 'CL_25_Fly' -> 2025).
    This is crucial for handling dates that only have month and day.
    """
    match = re.search(r'(20\d{2})|_(\d{2})', sheet_name)
    if match:
        year_part = match.group(1) or match.group(2)
        year = int(year_part)
        if year < 100:
            return 2000 + year
        return year
    return None

def process_sheet(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame | None:
    """
    Processes a single sheet (DataFrame) to standardize it for plotting.
    - Finds Date and Close columns.
    - Converts dates to a proper datetime format, inferring the year.
    - Calculates 'Months_from_Start' for the x-axis.
    """
    # 1. Find the Date and Close columns using candidate names.
    date_col = find_target_column(df.columns, ["date", "time", "day"])
    close_col = find_target_column(df.columns, ["close", "settle", "price"])

    if not date_col or not close_col:
        return None

    # 2. Create a clean DataFrame with just the essential columns.
    processed_df = df[[date_col, close_col]].copy()
    processed_df.columns = ["Date", "Close"]

    # 3. Handle missing values and ensure 'Close' is numeric.
    processed_df.dropna(subset=["Date", "Close"], inplace=True)
    processed_df["Close"] = pd.to_numeric(processed_df["Close"], errors='coerce')
    processed_df.dropna(subset=["Close"], inplace=True)

    if processed_df.empty:
        return None

    # 4. Convert the 'Date' column to datetime objects.
    sheet_year = infer_year_from_sheetname(sheet_name) or datetime.now().year
    
    def parse_date(d):
        try:
            # Assumes format like '4/21' or '2024-04-21'
            dt = pd.to_datetime(d, infer_datetime_format=True)
            if dt.year == datetime.now().year and sheet_year != datetime.now().year:
                 # If pandas defaults to current year, replace with inferred year
                 return dt.replace(year=sheet_year)
            return dt
        except (ValueError, TypeError):
             # Handle cases like 'MM/DD' strings without a year
            try:
                parts = str(d).split('/')
                month, day = int(parts[0]), int(parts[1])
                # A new season starts in April. If month is before April, it belongs to the next calendar year.
                year_offset = 1 if month < 4 else 0
                return datetime(sheet_year + year_offset, month, day)
            except Exception:
                return pd.NaT

    processed_df["Date"] = processed_df["Date"].apply(parse_date)
    processed_df.dropna(subset=["Date"], inplace=True)
    processed_df.sort_values("Date", inplace=True)
    
    if processed_df.empty:
        return None

    # 5. Calculate the X-axis value: "Months from Start"
    start_date = processed_df["Date"].iloc[0]
    processed_df["Months_from_Start"] = (processed_df["Date"] - start_date).dt.days / 30.44

    return processed_df

@st.cache_data(show_spinner="Loading and processing Excel file...")
def load_and_process_excel(file_source) -> dict[str, pd.DataFrame]:
    """
    Loads an Excel file and processes each sheet.

    Args:
        file_source: A file-like object (from upload) or a file path (for built-in).

    Returns:
        A dictionary where keys are sheet names and values are processed DataFrames.
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
        if df is None:
            continue

        # Determine line style for focused vs. other sheets
        line_width = 4 if name == focus_sheet else 2
        opacity = 1.0 if name == focus_sheet else 0.8

        # Add the primary closing price curve
        fig.add_trace(go.Scatter(
            x=df["Months_from_Start"],
            y=df["Close"],
            mode='lines',
            name=name,
            line=dict(width=line_width),
            opacity=opacity,
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Date: %{customdata|%Y-%m-%d}<br>"
                "Months: %{x:.2f}<br>"
                "Close: %{y:.4f}<extra></extra>"
            ),
            customdata=df["Date"]
        ))

        # Add Moving Averages for the primary curve
        for window in ma_windows:
            if window > 1:
                ma_series = df["Close"].rolling(window=window, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=df["Months_from_Start"],
                    y=ma_series,
                    mode='lines',
                    name=f"{name} {window}-day MA",
                    line=dict(width=1.5, dash='dash'),
                    opacity=0.7,
                    visible='legendonly' # Initially hide MAs to keep the chart clean
                ))
    
    # --- Figure Layout and Styling ---
    fig.update_layout(
        height=600,
        title="Fly Curve Comparison",
        xaxis_title="Months from Start Date",
        yaxis_title="Close Price",
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_monthly_comparison_chart(data, selected_sheets, selected_month):
    import calendar

    fig = go.Figure()

    # Convert month name to number (1=Jan, 12=Dec)
    month_number = list(calendar.month_name).index(selected_month)

    for sheet_name in selected_sheets:
        df = data[sheet_name].copy()

        # Ensure Date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Filter only rows for selected month
        df_month = df[df["Date"].dt.month == month_number]

        if df_month.empty:
            continue

        # X-axis = day of month
        df_month["Day"] = df_month["Date"].dt.day

        fig.add_trace(go.Scatter(
            x=df_month["Day"],
            y=df_month["Close"],
            mode="lines+markers",
            name=sheet_name,
            hovertemplate=(
                f"<b>{sheet_name}</b><br>"
                "Day %{x}<br>"
                "Close: %{y:.2f}<extra></extra>"
            )
        ))

    fig.update_layout(
        title=f"Monthly Comparison – {selected_month}",
        xaxis_title="Day of Month",
        yaxis_title="Close Price",
        template="plotly_dark",
        height=600,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig

# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------

# --- Title and Introduction ---
st.title("Trading Fly Curve Comparator")
st.markdown("This tool visualizes and compares the price evolution of different 'fly' contracts over their lifecycle. Upload your Excel file or use the built-in sample data to get started.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")

    # 1. Data Source Selection
    source_option = st.radio(
        "Select Data Source",
        ("Use Built-in Sample", "Upload Your Excel File"),
        help="The built-in sample contains historical data. Or, you can upload your own .xlsx file."
    )

    uploaded_file = None
    if source_option == "Upload Your Excel File":
        uploaded_file = st.file_uploader(
            "Choose an Excel file",
            type=["xlsx", "xls"]
        )

    # 2. Load and process the data based on the selected source.
    all_sheets_data = {}
    if source_option == "Use Built-in Sample":
        try:
            # This path assumes the script is run in an environment where the file exists.
            all_sheets_data = load_and_process_excel("FLY_CHART.xlsx")
        except FileNotFoundError:
            st.error("The built-in 'FLY_CHART.xlsx' was not found. Please upload a file instead.")
    elif uploaded_file:
        all_sheets_data = load_and_process_excel(uploaded_file)

    # If data is loaded, display the rest of the controls.
    if all_sheets_data:
        sheet_names = list(all_sheets_data.keys())
        st.markdown("---")
        st.header("📊 Chart Options")

        # 3. Sheet Selection for Plotting
        selected_sheets = st.multiselect(
            "Select Sheets to Plot",
            options=sheet_names,
            default=sheet_names[:min(len(sheet_names), 5)],
            help="Choose which contracts you want to compare on the chart."
        )

        # 4. Focus and MA Controls
        focus_sheet = st.selectbox(
            "Highlight a Curve",
            options=[None] + selected_sheets,
            format_func=lambda x: "None" if x is None else x,
            help="Select a curve to make it thicker and more prominent."
        )

        ma_windows = st.multiselect(
            "Add Moving Averages (days)",
            options=[5, 10, 20, 50, 100],
            default=[],
            help="Select moving average windows. These will be added but hidden by default; click the legend to show them."
        )
    else:
        st.info("Please upload an Excel file to begin analysis.")

# --- Main Page Content ---
if not all_sheets_data:
    st.warning("No data loaded. Please select a data source from the sidebar.")
elif not selected_sheets:
    st.info("ℹ️ Please select at least one sheet to visualize.")
else:
    # Toggle between views
    view_mode = st.radio(
        "Choose view mode",
        ["Seasonal (Months_from_Start)", "Monthly Comparison"],
        horizontal=True,
    )

    if view_mode == "Seasonal (Months_from_Start)":
        st.subheader("📈 Seasonal Chart")
        fig = create_comparison_chart(data, selected_sheets, ma_windows, focus_sheet)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.subheader("📊 Monthly Comparison Chart")
        # Month selector
        month_options = list(calendar.month_name)[1:]  # Jan–Dec
        selected_month = st.selectbox("Select Month", month_options, index=0)

        fig = create_monthly_comparison_chart(data, selected_sheets, selected_month)
        st.plotly_chart(fig, use_container_width=True)


    # --- Data Summary Section ---
    with st.expander("Show Data Summary and Export Options"):
        st.subheader("Summary Statistics")
        summary_rows = []
        for name in selected_sheets:
            df = all_sheets_data.get(name)
            if df is not None:
                summary_rows.append({
                    "Sheet": name,
                    "Start Date": df["Date"].min().strftime('%Y-%m-%d'),
                    "End Date": df["Date"].max().strftime('%Y-%m-%d'),
                    "Duration (Days)": (df["Date"].max() - df["Date"].min()).days,
                    "Start Price": df["Close"].iloc[0],
                    "End Price": df["Close"].iloc[-1],
                    "Min Price": df["Close"].min(),
                    "Max Price": df["Close"].max(),
                    "Avg Price": df["Close"].mean(),
                })
        
        summary_df = pd.DataFrame(summary_rows).set_index("Sheet")
        st.dataframe(summary_df.style.format("{:,.4f}", subset=["Start Price", "End Price", "Min Price", "Max Price", "Avg Price"]))

        # --- Data Export ---
        st.subheader("Export Processed Data")
        export_dfs = []
        for name in selected_sheets:
            df = all_sheets_data[name].copy()
            df['Sheet'] = name
            export_dfs.append(df)
        
        if export_dfs:
            full_export_df = pd.concat(export_dfs, ignore_index=True)
            csv_data = full_export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Data as CSV",
                data=csv_data,
                file_name="fly_curve_data_export.csv",
                mime="text/csv",
            )
