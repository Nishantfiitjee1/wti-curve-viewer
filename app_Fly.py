import re
import calendar
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fly Curve Comparator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Advanced UI & Dark Mode Styling ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    :root {
        --color-bg-primary: #121212;
        --color-bg-secondary: #1E1E1E;
        --color-sidebar: #151515;
        --color-primary-accent: #00A3FF;
        --color-secondary-accent: #8A2BE2;
        --color-text-primary: #EAEAEA;
        --color-text-secondary: #B0B0B0;
        --color-border: #333333;
    }

    /* General App Styling */
    body {
        font-family: 'Roboto', sans-serif;
        color: var(--color-text-primary);
    }
    
    .main {
        background-color: var(--color-bg-primary);
        border-radius: 10px;
        padding: 2rem;
    }

    /* Sidebar Styling */
    .stSidebar {
        background-color: var(--color-sidebar);
        border-right: 1px solid var(--color-border);
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--color-text-primary);
        font-weight: 700;
    }
    
    h1 {
        color: var(--color-primary-accent);
        text-align: center;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--color-primary-accent);
    }

    /* Card Styling */
    .card {
        background-color: var(--color-bg-secondary);
        border-radius: 10px;
        padding: 25px;
        margin-top: 20px;
        box-shadow: 0 4px 12px 0 rgba(0,0,0,0.2);
        border: 1px solid var(--color-border);
    }
    
    .stRadio > label {
        background-color: var(--color-bg-secondary);
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 5px;
    }

    /* Custom Dividers */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, var(--color-border), transparent);
        margin: 2rem 0;
    }

    /* Customizing Streamlit Widgets */
    .stButton > button {
        background: linear-gradient(45deg, var(--color-primary-accent), var(--color-secondary-accent));
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
    }
    .stButton > button:hover {
        box-shadow: 0 0 15px var(--color-primary-accent);
        transform: translateY(-2px);
    }
    
    .stFileUploader label {
        font-size: 1.1rem;
        color: var(--color-primary-accent);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# Data Loading and Processing Utilities (UNCHANGED)
# -----------------------------------------------------------------------------

def find_target_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized_cols = {col.lower().replace("_", "").replace(" ", ""): col for col in columns}
    for cand in candidates:
        if cand in normalized_cols:
            return normalized_cols[cand]
    for norm_col, orig_col in normalized_cols.items():
        if any(cand in norm_col for cand in candidates):
            return orig_col
    return None

def infer_year_from_sheetname(sheet_name: str) -> int | None:
    match = re.search(r'(20\d{2})|_(\d{2})', sheet_name)
    if match:
        year_part = match.group(1) or match.group(2)
        year = int(year_part)
        if year < 100:
            return 2000 + year
        return year
    return None

def process_sheet(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame | None:
    date_col = find_target_column(df.columns, ["date", "time", "day", "timestamp", "datetime"])
    close_col = find_target_column(df.columns, ["close", "settle", "price", "last", "mid"])
    if not date_col or not close_col:
        return None

    processed_df = df[[date_col, close_col]].copy()
    processed_df.columns = ["Date", "Close"]

    processed_df.dropna(subset=["Date", "Close"], inplace=True)
    processed_df["Close"] = pd.to_numeric(processed_df["Close"], errors='coerce')
    processed_df.dropna(subset=["Close"], inplace=True)

    if processed_df.empty:
        return None

    sheet_year = infer_year_from_sheetname(sheet_name) or datetime.now().year

    def parse_date(d):
        try:
            dt = pd.to_datetime(d, infer_datetime_format=True, dayfirst=False)
            if dt.year == datetime.now().year and sheet_year != datetime.now().year:
                return dt.replace(year=sheet_year)
            return dt
        except Exception:
            try:
                parts = str(d).strip().split('/')
                if len(parts) >= 2:
                    month, day = int(parts[0]), int(parts[1])
                    year_offset = 1 if month < 4 else 0
                    return datetime(sheet_year + year_offset, month, day)
            except Exception:
                return pd.NaT
        return pd.NaT

    processed_df["Date"] = processed_df["Date"].apply(parse_date)
    processed_df.dropna(subset=["Date"], inplace=True)
    processed_df.sort_values("Date", inplace=True)

    if processed_df.empty:
        return None

    start_date = processed_df["Date"].iloc[0]
    processed_df["Months_from_Start"] = (processed_df["Date"] - start_date).dt.days / 30.44
    processed_df = processed_df.reset_index(drop=True)
    return processed_df

@st.cache_data(show_spinner="Loading and processing Excel file...")
def load_and_process_excel(file_source) -> dict[str, pd.DataFrame]:
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
# Plotting Utilities (UNCHANGED)
# -----------------------------------------------------------------------------

def create_comparison_chart(data, selected_sheets, ma_windows, focus_sheet):
    fig = go.Figure()
    for name in selected_sheets:
        df = data.get(name)
        if df is None:
            continue
        line_width = 4 if name == focus_sheet else 2
        opacity = 1.0 if name == focus_sheet else 0.8
        fig.add_trace(go.Scatter(
            x=df["Months_from_Start"], y=df["Close"],
            mode='lines', name=name,
            line=dict(width=line_width), opacity=opacity,
            hovertemplate=(f"<b>{name}</b><br>"
                           "Date: %{customdata|%Y-%m-%d}<br>"
                           "Months: %{x:.2f}<br>"
                           "Close: %{y:.4f}<extra></extra>"),
            customdata=df["Date"]
        ))
        for window in ma_windows:
            if window > 1:
                ma_series = df["Close"].rolling(window=window, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=df["Months_from_Start"], y=ma_series,
                    mode='lines', name=f"{name} {window}-day MA",
                    line=dict(width=1.5, dash='dash'), opacity=0.7,
                    visible='legendonly'
                ))
    fig.update_layout(
        height=600, title="Fly Curve Comparison",
        xaxis_title="Months from Start Date", yaxis_title="Close Price",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def create_monthly_comparison_chart(data, selected_sheets, selected_month):
    fig = go.Figure()
    month_number = list(calendar.month_name).index(selected_month)
    for sheet_name in selected_sheets:
        df = data.get(sheet_name)
        if df is None: continue
        df_month = df[df["Date"].dt.month == month_number].copy()
        if df_month.empty: continue
        df_month["Day"] = df_month["Date"].dt.day
        df_month.sort_values("Day", inplace=True)
        fig.add_trace(go.Scatter(
            x=df_month["Day"], y=df_month["Close"], mode="lines+markers",
            name=sheet_name, customdata=df_month["Date"],
            hovertemplate=(f"<b>{sheet_name}</b><br>"
                           "Date: %{customdata|%Y-%m-%d}<br>"
                           "Day %{x}<br>"
                           "Close: %{y:.4f}<extra></extra>")
        ))
    fig.update_layout(
        title=f"Monthly Comparison – {selected_month}",
        xaxis_title="Day of Month", yaxis_title="Close Price",
        template="plotly_dark", height=600,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=40, t=60, b=40), xaxis=dict(tickmode="linear")
    )
    return fig

def create_weekly_comparison_chart(data, selected_sheets):
    fig = go.Figure()
    for sheet_name in selected_sheets:
        df = data.get(sheet_name)
        if df is None: continue
        df["Week"] = df["Date"].dt.isocalendar().week
        weekly_df = df.groupby("Week", as_index=False).mean(numeric_only=True)
        if weekly_df.empty: continue
        fig.add_trace(go.Scatter(
            x=weekly_df["Week"], y=weekly_df["Close"], mode="lines+markers",
            name=sheet_name,
            hovertemplate=(f"<b>{sheet_name}</b><br>"
                           "Week: %{x}<br>"
                           "Close: %{y:.4f}<extra></extra>")
        ))
    fig.update_layout(
        title="Weekly Comparison (Week 1–52)",
        xaxis_title="Week Number", yaxis_title="Average Close Price",
        template="plotly_dark", height=600,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=40, t=60, b=40), xaxis=dict(tickmode="linear")
    )
    return fig

# -----------------------------------------------------------------------------
# Main Application UI (Minor changes for styling)
# -----------------------------------------------------------------------------
st.title("📊 Trading Fly Curve Comparator")

selected_sheets, ma_windows, focus_sheet, all_sheets_data = [], [], None, {}

with st.sidebar:
    st.header("⚙️ Controls")
    source_option = st.radio("Select Data Source",
        ("Use Built-in Sample", "Upload Your Excel File"))
    uploaded_file = None
    if source_option == "Upload Your Excel File":
        uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
    if source_option == "Use Built-in Sample":
        try: all_sheets_data = load_and_process_excel("FLY_CHART.xlsx")
        except FileNotFoundError: st.error("Built-in sample not found."); all_sheets_data = {}
    elif uploaded_file:
        all_sheets_data = load_and_process_excel(uploaded_file)

    if all_sheets_data:
        sheet_names = list(all_sheets_data.keys())
        st.markdown("<hr>", unsafe_allow_html=True) # Custom Divider
        st.header("📊 Chart Options")
        selected_sheets = st.multiselect("Select Sheets to Plot",
            options=sheet_names, default=sheet_names[:min(len(sheet_names), 5)])
        focus_sheet = st.selectbox("Highlight a Curve", [None] + selected_sheets,
            format_func=lambda x: "None" if x is None else x)
        ma_windows = st.multiselect("Add Moving Averages (days)",
            options=[5, 10, 20, 50, 100], default=[])

if not all_sheets_data:
    st.warning("No data loaded. Please upload a file or use the built-in sample.")
elif not selected_sheets:
    st.info("ℹ️ Please select at least one sheet from the sidebar to display a chart.")
else:
    # Wrap main content in a card for better visual separation
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    view_mode = st.radio("Choose view mode",
        ["Seasonal", "Monthly", "Weekly"], horizontal=True)
    
    if view_mode == "Seasonal":
        st.subheader("📈 Seasonal Chart")
        st.plotly_chart(create_comparison_chart(all_sheets_data, selected_sheets, ma_windows, focus_sheet), use_container_width=True)
    elif view_mode == "Monthly":
        st.subheader("🗓️ Monthly Comparison")
        selected_month = st.selectbox("Select Month", list(calendar.month_name)[1:])
        st.plotly_chart(create_monthly_comparison_chart(all_sheets_data, selected_sheets, selected_month), use_container_width=True)
    else:
        st.subheader("📅 Weekly Comparison")
        st.plotly_chart(create_weekly_comparison_chart(all_sheets_data, selected_sheets), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
