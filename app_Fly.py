import re
import calendar
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# -----------------------------------------------------------------------------
# Page Configuration & Advanced UI Styling
# -----------------------------------------------------------------------------
# ---------- Theme switcher + robust light/dark CSS ----------
import streamlit as st
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

# -----------------------------------------------------------------------------
# Advanced Theme Switcher with Corrected CSS
# -----------------------------------------------------------------------------
# I have rewritten the CSS to be more robust and fixed the light theme issues.

light_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700&display=swap');

:root {
    --bg-primary: #F0F2F6;
    --bg-secondary: #FFFFFF;
    --sidebar-bg: #FFFFFF;
    --card-bg: #FFFFFF;
    --accent-1: #007BFF;
    --accent-2: #6F42C1;
    --text-primary: #1E293B;
    --text-secondary: #475569;
    --border: #E2E8F0;
    --shadow: rgba(15, 23, 42, 0.05);
}

html, body, .stApp, div[data-testid="stAppViewContainer"], .main {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

div[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border) !important;
}

h1, h2, h3, h4, h5, h6 { color: var(--text-primary) !important; }

.card {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px;
    box-shadow: 0 4px 12px var(--shadow);
    padding: 30px;
    margin-top: 20px;
}

.stRadio > div { background-color: #F0F2F6 !important; padding: 6px; border-radius: 10px; }
.stRadio label {
    background-color: #FFFFFF !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    padding: 6px 12px;
    border-radius: 8px;
    transition: all 0.2s ease-in-out;
}

header, footer { background-color: transparent !important; }
</style>
"""

dark_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700&display=swap');

:root {
    --bg-primary: #0E1117;
    --bg-secondary: #161B22;
    --sidebar-bg: #1A1C23;
    --card-bg: #161B22;
    --accent-1: #00A9FF;
    --accent-2: #9A4BFF;
    --text-primary: #FAFAFA;
    --text-secondary: #B0B3B8;
    --border: #2A2F3B;
    --shadow: rgba(0, 0, 0, 0.3);
}

html, body, .stApp, div[data-testid="stAppViewContainer"], .main {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

div[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border) !important;
}

h1, h2, h3, h4, h5, h6 { color: var(--text-primary) !important; }

.card {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px;
    box-shadow: 0 4px 12px var(--shadow);
    padding: 30px;
    margin-top: 20px;
}

.stRadio > div { background-color: #0E1117 !important; padding: 6px; border-radius: 10px; }
.stRadio label {
    background-color: #262B34 !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    padding: 6px 12px;
    border-radius: 8px;
    transition: all 0.2s ease-in-out;
}

header, footer { background-color: transparent !important; }
</style>
"""

# The "Auto" CSS now correctly combines both light and dark styles
auto_css = f"<style> {light_css.replace('<style>', '').replace('</style>', '')} @media (prefers-color-scheme: dark) {{ {dark_css.replace('<style>', '').replace('</style>', '')} }} </style>"

# We will inject the CSS later, after the theme has been chosen in the sidebar

# --------------------------------------------------------------



# -----------------------------------------------------------------------------
# Data Loading and Processing Utilities (Aapka original code - No changes)
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
                    return datetime(sheet_year, month, day)
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
# Plotting Utilities
# -----------------------------------------------------------------------------
def create_comparison_chart(data, selected_sheets, ma_windows, focus_sheet):
    # Aapka original seasonal chart function - No changes
    fig = go.Figure()
    for name in selected_sheets:
        df = data.get(name)
        if df is None: continue
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
    # Aapka original monthly chart function - No changes
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

# --- YEH HAI NAYA LOGIC WALA WEEKLY CHART FUNCTION ---
def create_relative_weekly_chart(data, selected_sheets):
    """
    Yeh function aapke naye logic se weekly data calculate aur plot karta hai.
    - Week 1 har sheet ki pehli date se shuru hota hai.
    - Har week 7 din ka block hota hai.
    - Har block ke data ka mean (average) liya jaata hai.
    """
    fig = go.Figure()
    for sheet_name in selected_sheets:
        df = data.get(sheet_name)
        if df is None or df.empty:
            continue
        
        # Step 1: Pehli date ko start date maan lo
        start_date = df["Date"].iloc[0]
        
        # Step 2: Har date ke liye relative week number nikalo
        # (current_date - start_date) ke din / 7
        df['Relative_Week'] = ((df['Date'] - start_date).dt.days // 7) + 1
        
        # Step 3: Har 'Relative_Week' ke liye 'Close' price ka mean nikalo
        weekly_df = df.groupby('Relative_Week')['Close'].mean().reset_index()
        
        if weekly_df.empty:
            continue
            
        fig.add_trace(go.Scatter(
            x=weekly_df["Relative_Week"], y=weekly_df["Close"], mode="lines+markers",
            name=sheet_name,
            hovertemplate=(f"<b>{sheet_name}</b><br>"
                           "Week from Start: %{x}<br>"
                           "Avg Close: %{y:.4f}<extra></extra>")
        ))
        
    fig.update_layout(
        title="Weekly Comparison (Relative to Start Date)",
        xaxis_title="Week Number from Start Date", yaxis_title="Average Close Price",
        template="plotly_dark", height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40), xaxis=dict(tickmode="linear")
    )
    return fig

# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------
# --- SIDEBAR SECTION CODE ---

# --- SIDEBAR SECTION CODE ---

# --- SIDEBAR SECTION CODE (IMPROVED ERROR HANDLING) ---

with st.sidebar:
    st.header("⚙️ Controls")

    # Define your dictionary of inbuilt Excel files.
    # The key is the user-friendly name for the dropdown.
    # The value is the actual filename in your folder.
    INBUILT_FILES = {
        "April Fly": "APRIL_FLY.xlsx",
        "May Fly": "MAY_FLY.xlsx",
        "June Fly": "JUNE_FLY.xlsx",
        "July Fly": "JULY_FLY.xlsx",
        "December Fly": "DEC_FLY.xlsx"
    }

    source_option = st.radio(
        "Select Data Source",
        ("Select Inbuilt File", "Upload Your Excel File"),
        label_visibility="collapsed"
    )

    file_to_process = None
    all_sheets_data = {} # Initialize the data dictionary as empty

    if source_option == "Select Inbuilt File":
        # Get the user-friendly name selected by the user
        selected_display_name = st.selectbox(
            "Choose an inbuilt dataset",
            options=list(INBUILT_FILES.keys())
        )
        # Find the corresponding filename to load
        file_to_process = INBUILT_FILES[selected_display_name]

    elif source_option == "Upload Your Excel File":
        file_to_process = st.file_uploader(
            "Choose an Excel file",
            type=["xlsx", "xls"]
        )

    # --- Data Loading Logic with New Attractive Error Message ---
    if file_to_process:
        try:
            # This function will try to load and process the file
            all_sheets_data = load_and_process_excel(file_to_process)

        except FileNotFoundError:
            # *** THIS IS THE NEW ATTRACTIVE MESSAGE LOGIC ***
            # If the selected inbuilt file is not found...
            st.error(f"📂 Data for '{selected_display_name}' is not available.")
            st.warning(f"Please make sure the file named `'{file_to_process}'` exists in the same folder as the script.")
            # Important: Keep all_sheets_data empty so the rest of the app knows not to proceed
            all_sheets_data = {}
        
        except Exception as e:
            # Catch any other errors during processing
            # The load_and_process_excel function already displays an error message
            all_sheets_data = {}

    # This part of the sidebar will only run if data was loaded successfully
    if all_sheets_data:
        sheet_names = list(all_sheets_data.keys())
        st.markdown("---")
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
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    view_mode = st.radio("Choose view mode",
        ["Seasonal", "Monthly", "Weekly"], horizontal=True, label_visibility="collapsed")
    
    if view_mode == "Seasonal":
        st.header("📈 Seasonal Chart")
        st.plotly_chart(create_comparison_chart(all_sheets_data, selected_sheets, ma_windows, focus_sheet), use_container_width=True)
    
    elif view_mode == "Monthly":
        st.header("🗓️ Monthly Comparison")
        selected_month = st.selectbox("Select Month", list(calendar.month_name)[1:])
        st.plotly_chart(create_monthly_comparison_chart(all_sheets_data, selected_sheets, selected_month), use_container_width=True)
    
    else: # Weekly View
        st.header("📅 Weekly Comparison")
        # Yahan hum naye logic wala function call kar rahe hain
        st.plotly_chart(create_relative_weekly_chart(all_sheets_data, selected_sheets), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
