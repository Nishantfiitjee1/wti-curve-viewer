import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from io import BytesIO
from datetime import datetime

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Advanced Fly Curve Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Data Loading and Processing Utilities
# -----------------------------------------------------------------------------

def find_target_column(columns: list[str], candidates: list[str]) -> str | None:
    """Finds the best matching column name from a prioritized list of candidates."""
    normalized_cols_map = {col.lower().replace("_", "").replace(" ", ""): col for col in columns}
    for cand in candidates:
        if cand in normalized_cols_map:
            return normalized_cols_map[cand]
    for cand in candidates:
        for norm_col, orig_col in normalized_cols_map.items():
            if cand in norm_col:
                return orig_col
    return None

def infer_year_from_sheetname(sheet_name: str) -> int | None:
    """Infers the year from the sheet name (e.g., 'CL_25_Fly' -> 2025)."""
    match = re.search(r'(20\d{2})|_(\d{2})', sheet_name)
    if match:
        year_part = match.group(1) or match.group(2)
        year = int(year_part)
        return 2000 + year if year < 100 else year
    return datetime.now().year

def robust_date_parser(series: pd.Series, year_hint: int) -> pd.Series:
    """A powerful date parser that handles full dates, MM/DD, and mixed formats."""
    # Attempt direct conversion first. 'coerce' turns failures into NaT (Not a Time).
    dates = pd.to_datetime(series, errors='coerce')
    
    # For any dates that failed conversion, try a manual regex approach for MM/DD or MM-DD.
    failed_indices = dates[dates.isna()].index
    for idx in failed_indices:
        val = str(series.loc[idx])
        match = re.match(r'(\d{1,2})[/\-](\d{1,2})', val)
        if match:
            try:
                month, day = int(match.group(1)), int(match.group(2))
                dates.loc[idx] = datetime(year_hint, month, day)
            except ValueError:
                continue # Skip if date is invalid, e.g., month > 12
    return dates

def process_sheet(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame | None:
    """Processes a single sheet to standardize it for plotting and analysis."""
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

    if processed_df.empty: return None

    # Use the robust date parser with a year hint from the sheet name.
    year_hint = infer_year_from_sheetname(sheet_name)
    processed_df["Date"] = robust_date_parser(processed_df["Date"], year_hint)
    processed_df.dropna(subset=["Date"], inplace=True)
    processed_df.sort_values("Date", inplace=True, ignore_index=True)
    
    if processed_df.empty: return None

    # Smartly handle seasonal year rollovers.
    start_month = processed_df["Date"].iloc[0].month
    rollover_mask = processed_df["Date"].dt.month < start_month
    processed_df.loc[rollover_mask, "Date"] = processed_df.loc[rollover_mask, "Date"].apply(lambda d: d.replace(year=d.year + 1))
    processed_df.sort_values("Date", inplace=True, ignore_index=True)

    start_date = processed_df["Date"].iloc[0]
    processed_df["Months_from_Start"] = (processed_df["Date"] - start_date).dt.days / 30.44
    return processed_df

@st.cache_data(show_spinner="Analyzing Excel file...")
def load_and_process_excel(file_source) -> dict[str, pd.DataFrame]:
    """Loads an Excel file and processes each sheet."""
    try:
        xls = pd.ExcelFile(file_source)
        processed_sheets = {
            sheet_name: df
            for sheet_name in xls.sheet_names
            if (df := process_sheet(pd.read_excel(xls, sheet_name=sheet_name), sheet_name)) is not None and not df.empty
        }
        return processed_sheets
    except Exception as e:
        st.error(f"An error occurred while reading the Excel file: {e}")
        return {}

# -----------------------------------------------------------------------------
# Analysis and Plotting
# -----------------------------------------------------------------------------

def create_main_chart(data, selected_sheets, ma_windows, focus_sheet):
    """Creates the main Plotly chart for comparing fly curves."""
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
                ma = df["Close"].rolling(window=window, min_periods=1).mean()
                fig.add_trace(go.Scatter(x=df["Months_from_Start"], y=ma, mode='lines', name=f"{name} {window}d MA", line=dict(width=1.5, dash='dash'), opacity=0.7, visible='legendonly'))
    fig.update_layout(height=600, title="Fly Curve Comparison", xaxis_title="Months from Start Date", yaxis_title="Close Price", template="plotly_dark", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=20, r=20, t=50, b=20))
    return fig

def calculate_performance_metrics(data, selected_sheets):
    """Calculates volatility, returns, and max drawdown."""
    metrics = []
    for name in selected_sheets:
        df = data.get(name)
        if df is None or len(df) < 2: continue
        daily_returns = df['Close'].pct_change().dropna()
        if daily_returns.empty: continue
        
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
        
        cumulative_returns = (1 + daily_returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()
        
        metrics.append({"Sheet": name, "Annualized Volatility": annualized_volatility, "Total Return %": total_return * 100, "Max Drawdown %": max_drawdown * 100})
    return pd.DataFrame(metrics).set_index("Sheet")

def create_correlation_heatmap(data, selected_sheets):
    """Creates a heatmap of the correlation between selected curves."""
    aligned_dfs = []
    for name in selected_sheets:
        df = data.get(name)
        if df is not None:
            # Use 'Months_from_Start' as the index for alignment
            aligned_df = df.set_index('Months_from_Start')['Close'].rename(name)
            aligned_dfs.append(aligned_df)
    
    if len(aligned_dfs) < 2:
        return None

    combined_df = pd.concat(aligned_dfs, axis=1).interpolate(method='linear', limit_direction='both')
    corr_matrix = combined_df.corr()
    
    fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', range_color=[-1, 1], title="Correlation Matrix of Fly Curves")
    fig.update_layout(template="plotly_dark")
    return fig

def create_historical_comparison_plot(data, selected_sheets, focus_sheet):
    """Plots a focus curve against the average of the other selected curves."""
    if focus_sheet is None or len(selected_sheets) < 2:
        return None

    focus_df = data[focus_sheet].set_index('Months_from_Start')['Close'].rename(focus_sheet)
    
    other_dfs = []
    for name in selected_sheets:
        if name != focus_sheet:
            other_dfs.append(data[name].set_index('Months_from_Start')['Close'])
    
    if not other_dfs: return None

    combined_others = pd.concat(other_dfs, axis=1).interpolate(method='linear', limit_direction='both')
    average_curve = combined_others.mean(axis=1).rename('Historical Average')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=focus_df.index, y=focus_df, mode='lines', name=focus_sheet, line=dict(width=3, color='cyan')))
    fig.add_trace(go.Scatter(x=average_curve.index, y=average_curve, mode='lines', name='Historical Average', line=dict(width=2, dash='dash', color='orange')))
    fig.update_layout(title=f"Comparison: {focus_sheet} vs. Historical Average", xaxis_title="Months from Start Date", yaxis_title="Close Price", template="plotly_dark")
    return fig

# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------

st.title("📈 Advanced Trading Fly Curve Analyzer")
st.markdown("An interactive tool to visualize, compare, and analyze fly contract curves from Excel data.")

with st.sidebar:
    st.header("⚙️ Controls")
    source_option = st.radio("Select Data Source", ("Use Built-in Sample", "Upload Your Excel File"), index=0)
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"]) if source_option == "Upload Your Excel File" else None

all_sheets_data = {}
if source_option == "Use Built-in Sample":
    try:
        all_sheets_data = load_and_process_excel("FLY_CHART.xlsx")
    except FileNotFoundError:
        st.error("Built-in 'FLY_CHART.xlsx' not found. Please upload a file to proceed.")
elif uploaded_file:
    all_sheets_data = load_and_process_excel(uploaded_file)

if not all_sheets_data:
    st.warning("No data loaded. Please select a valid data source from the sidebar.")
else:
    with st.sidebar:
        sheet_names = list(all_sheets_data.keys())
        st.markdown("---")
        st.header("📊 Chart Options")
        selected_sheets = st.multiselect("Select Sheets to Plot", sheet_names, default=sheet_names[:min(len(sheet_names), 6)])
        focus_sheet = st.selectbox("Highlight a Curve", [None] + selected_sheets, format_func=lambda x: "None" if x is None else x)
        ma_windows = st.multiselect("Add Moving Averages (days)", [5, 10, 20, 50, 100], default=[])

    if not selected_sheets:
        st.info("👈 Please select at least one sheet from the sidebar to display the chart.")
    else:
        st.header("Curve Comparison Chart")
        main_chart = create_main_chart(all_sheets_data, selected_sheets, ma_windows, focus_sheet)
        st.plotly_chart(main_chart, use_container_width=True)

        st.header("Advanced Analysis")
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance & Volatility", "🔗 Correlation Matrix", "📈 Historical Comparison", "📋 Raw Data"])

        with tab1:
            st.markdown("Key performance indicators for each selected contract.")
            perf_df = calculate_performance_metrics(all_sheets_data, selected_sheets)
            st.dataframe(perf_df.style.format({
                "Annualized Volatility": "{:.2%}",
                "Total Return %": "{:.2f}%",
                "Max Drawdown %": "{:.2f}%"
            }))

        with tab2:
            st.markdown("Correlation shows how different curves move in relation to each other. A value of **1** means they move perfectly together; **-1** means they move in opposite directions.")
            corr_fig = create_correlation_heatmap(all_sheets_data, selected_sheets)
            if corr_fig:
                st.plotly_chart(corr_fig, use_container_width=True)
            else:
                st.warning("Select at least two sheets to calculate correlation.")

        with tab3:
            st.markdown("Select a curve from the sidebar's 'Highlight a Curve' dropdown to compare it against the average of all other selected curves.")
            if focus_sheet:
                hist_fig = create_historical_comparison_plot(all_sheets_data, selected_sheets, focus_sheet)
                if hist_fig:
                    st.plotly_chart(hist_fig, use_container_width=True)
                else:
                    st.warning("Select more than one sheet to create a historical comparison.")
            else:
                st.info("👈 Please select a curve to highlight from the sidebar to use this feature.")
        
        with tab4:
             with st.expander("Show Data Summary and Export Options"):
                st.subheader("Summary Statistics")
                summary_rows = [
                    {
                        "Sheet": name, "Start Date": df["Date"].min().strftime('%Y-%m-%d'),
                        "End Date": df["Date"].max().strftime('%Y-%m-%d'),
                        "Duration (Days)": (df["Date"].max() - df["Date"].min()).days,
                        "Start Price": df["Close"].iloc[0], "End Price": df["Close"].iloc[-1],
                        "Min Price": df["Close"].min(), "Max Price": df["Close"].max(),
                    }
                    for name in selected_sheets if (df := all_sheets_data.get(name)) is not None
                ]
                if summary_rows:
                    st.dataframe(pd.DataFrame(summary_rows).set_index("Sheet"))

                st.subheader("Export Processed Data")
                export_dfs = [df.assign(Sheet=name) for name in selected_sheets if (df := all_sheets_data.get(name)) is not None]
                if export_dfs:
                    csv_data = pd.concat(export_dfs, ignore_index=True).to_csv(index=False).encode('utf-8')
                    st.download_button(label="⬇️ Download Data as CSV", data=csv_data, file_name="fly_curve_export.csv", mime="text/csv")
