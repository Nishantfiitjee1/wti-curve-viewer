import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO

# ========================================================================================
# Utility Functions
# ========================================================================================
def load_excel(file) -> dict:
    """Load Excel file and return dictionary of sheet DataFrames with standardized Date+Close."""
    xls = pd.ExcelFile(file)
    sheets = {}
    for sheet in xls.sheet_names:
        df = pd.read_excel(file, sheet_name=sheet)
        
        # Auto-detect date column
        date_col = None
        for c in df.columns:
            if "date" in c.lower():
                date_col = c
                break
        if not date_col:
            continue

        # Auto-detect close/price column
        close_col = None
        for c in df.columns:
            if any(x in c.lower() for x in ["close", "price", "settlement"]):
                close_col = c
                break
        if not close_col:
            continue

        # Parse date
        df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["Date"])

        # Extract only Date & Close
        df = df[["Date", close_col]].rename(columns={close_col: "Close"})
        df = df.sort_values("Date").reset_index(drop=True)

        # Handle Apr–Mar cycle (rollover)
        df["Month"] = df["Date"].dt.month
        df.loc[df["Month"] < 4, "Date"] = df.loc[df["Month"] < 4, "Date"] + pd.DateOffset(years=1)
        df = df.sort_values("Date").reset_index(drop=True)

        sheets[sheet] = df
    return sheets


def plot_curves(sheets, selected_sheets, normalize=False, ma_window=None):
    fig = go.Figure()

    for sheet in selected_sheets:
        df = sheets[sheet].copy()

        # Normalization
        if normalize:
            df["Close"] = (df["Close"] / df["Close"].iloc[0]) * 100

        # Moving Average
        if ma_window:
            df[f"MA{ma_window}"] = df["Close"].rolling(ma_window).mean()

        # Plot main curve
        fig.add_trace(go.Scatter(
            x=df["Date"].dt.strftime("%m-%d"),
            y=df["Close"],
            mode="lines",
            name=f"{sheet}"
        ))

        # Plot MA if selected
        if ma_window:
            fig.add_trace(go.Scatter(
                x=df["Date"].dt.strftime("%m-%d"),
                y=df[f"MA{ma_window}"],
                mode="lines",
                name=f"{sheet} MA{ma_window}",
                line=dict(dash="dot")
            ))

    fig.update_layout(
        title="FLY Curve Comparison",
        xaxis_title="Date (MM-DD)",
        yaxis_title="Close" if not normalize else "Normalized Close (Rebased to 100)",
        template="plotly_dark",
        hovermode="x unified"
    )
    return fig


def compute_stats(sheets, selected_sheets):
    stats = {}
    for sheet in selected_sheets:
        df = sheets[sheet]
        stats[sheet] = {
            "Start": df["Close"].iloc[0],
            "End": df["Close"].iloc[-1],
            "Mean": df["Close"].mean(),
            "Min": df["Close"].min(),
            "Max": df["Close"].max(),
            "Volatility (Std)": df["Close"].std(),
        }
    return pd.DataFrame(stats).T

# ========================================================================================
# Streamlit App
# ========================================================================================
st.set_page_config(page_title="FLY Curve Dashboard", layout="wide")
st.title("📈 FLY Curve Comparison Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your FLY_CHART.xlsx", type=["xlsx"])
if uploaded_file:
    sheets = load_excel(uploaded_file)
else:
    st.info("Using default FLY_CHART.xlsx from repository")
    with open("FLY_CHART.xlsx", "rb") as f:
        sheets = load_excel(f)

# Sidebar Controls
st.sidebar.header("Controls")
selected_sheets = st.sidebar.multiselect("Select Curves to Compare", list(sheets.keys()), default=list(sheets.keys())[:2])
normalize = st.sidebar.checkbox("Normalize (Rebase to 100)")
ma_window = st.sidebar.selectbox("Moving Average Window", [None, 5, 10, 20])

# Plot curves
if selected_sheets:
    fig = plot_curves(sheets, selected_sheets, normalize, ma_window)
    st.plotly_chart(fig, use_container_width=True)

    # Show stats
    stats_df = compute_stats(sheets, selected_sheets)
    st.subheader("📊 Curve Statistics")
    st.dataframe(stats_df)

    # Download option
    st.download_button(
        label="Download Selected Data (CSV)",
        data=stats_df.to_csv().encode("utf-8"),
        file_name="fly_curve_stats.csv",
        mime="text/csv"
    )
else:
    st.warning("Please select at least one curve to compare.")
