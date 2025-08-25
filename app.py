import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta

# ---------------------------- Page config and CSS ----------------------------
st.set_page_config(page_title="Curve Viewer", layout="wide")
st.markdown("""
<style>
/* Dropdown width */
div[data-baseweb="select"] > div {
    width: 200px !important;
    font-size: 14px !important;
}

/* Date picker width */
div[data-baseweb="datepicker"] > div {
    width: 150px !important;
    font-size: 14px !important;
}

/* Buttons */
div.stButton > button {
    width: 100px;
    height: 30px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------- Data loader (No changes) ----------------------------
@st.cache_data(show_spinner=True, ttl=0)
def load_wti(file_or_path):
    df_raw = pd.read_excel(file_or_path, header=None, engine="openpyxl")
    hdr0 = df_raw.iloc[0].tolist()
    contracts = [str(x).strip() for x in hdr0[1:] if pd.notna(x) and str(x).strip() != ""]
    col_names = ["Date"] + contracts
    df = df_raw.iloc[2:].copy()
    df.columns = col_names
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in contracts:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df, contracts

# ---------------------------- Utility functions (No changes) ----------------------------
def curve_for_date(df: pd.DataFrame, contracts, d: date) -> pd.Series | None:
    row = df.loc[df["Date"].dt.date == d, contracts]
    if row.empty:
        return None
    return row.iloc[0]

def overlay_figure(contracts, curves: dict, y_label="Last Price ($)", title="WTI Curve") -> go.Figure:
    fig = go.Figure()
    for label, s in curves.items():
        fig.add_trace(go.Scatter(x=contracts, y=s.values, mode="lines+markers", name=str(label)))
    fig.update_layout(
        title=title,
        xaxis_title="Contract",
        yaxis_title=y_label,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig

def filter_dates(df, selected_range):
    max_date = df["Date"].max()
    if selected_range == "Full History":
        return df
    elif selected_range == "Last 1 Week":
        min_date = max_date - timedelta(weeks=1)
    elif selected_range == "Last 2 Weeks":
        min_date = max_date - timedelta(weeks=2)
    elif selected_range == "Last 1 Month":
        min_date = max_date - timedelta(days=30)
    elif selected_range == "Last 6 Months":
        min_date = max_date - timedelta(days=180)
    elif selected_range == "Last 1 Year":
        min_date = max_date - timedelta(days=365)
    else:
        min_date = df["Date"].min()
    return df[df["Date"] >= min_date]

# ---------------------------- Load Excel ----------------------------
FILE_PATH = "WTI_Outright.xlsx"
try:
    df, contracts = load_wti(FILE_PATH)
except Exception as e:
    st.error(f"Failed to read Excel file: {e}")
    st.stop()

# ---------------------------- Sidebar: Global Controls ----------------------------
st.sidebar.title("🛠️ Curve Controls")

all_dates = sorted(df["Date"].dt.date.unique().tolist(), reverse=True)
max_d = all_dates[0]
min_d = all_dates[-1]

st.sidebar.header("Date Selection")
single_date = st.sidebar.date_input(
    "Single date", value=max_d, min_value=min_d, max_value=max_d
)

multi_dates = st.sidebar.multiselect(
    "Multi-date overlay",
    options=all_dates,
    default=[all_dates[0], all_dates[1]],
    help="Select multiple dates to compare their curves side-by-side."
)

st.sidebar.header("Global Settings")
normalize = st.sidebar.checkbox("Normalize curves (z-score)")
do_export = st.sidebar.checkbox("Enable CSV export")

# ---------------------------- Main title ----------------------------
st.title("WTI Outright Curve Viewer")
st.caption("An interactive tool for analyzing futures curves, spreads, and historical evolution.")

# ---------------------------- Normalize data (No changes) ----------------------------
work_df = df.copy()
if normalize:
    vals = work_df[contracts].astype(float)
    work_df[contracts] = (vals - vals.mean(axis=1).values[:, None]) / vals.std(axis=1).values[:, None]

# ---------------------------- UI Structure: Main tabs for different analysis types ----------------------------
tab_overview, tab_timeseries, tab_evolution = st.tabs(["📈 Curve Overview", "📊 Time Series Analysis", "🎥 Curve Evolution"])

# ---------- TAB 1: CURVE OVERVIEW ----------
with tab_overview:
    col1, col2 = st.columns(2)

    # Section A: Single-date curve
    with col1:
        st.subheader(f"Curve for: {single_date}")
        s1 = curve_for_date(work_df, contracts, single_date)
        if s1 is None:
            st.error("No data for the chosen date.")
        else:
            fig_single = overlay_figure(contracts, {single_date: s1}, y_label=("Z-score" if normalize else "Last Price ($)"))
            st.plotly_chart(fig_single, use_container_width=True)

    # Section B: Multi-date overlay
    with col2:
        st.subheader("Multi-Date Overlay")
        curves = {d: curve_for_date(work_df, contracts, d) for d in multi_dates}
        valid_curves = {d: s for d, s in curves.items() if s is not None}
        
        if not valid_curves:
            st.warning("No data found for any of the selected dates.")
        else:
            fig_overlay = overlay_figure(contracts, valid_curves, y_label=("Z-score" if normalize else "Last Price ($)"))
            st.plotly_chart(fig_overlay, use_container_width=True)
            
            missing_dates = [d for d, s in curves.items() if s is None]
            if missing_dates:
                st.info(f"Skipped dates with no data: {missing_dates}")


# ---------- TAB 2: TIME SERIES ANALYSIS (SPREAD & FLY) ----------
with tab_timeseries:
    st.subheader("Spread & Fly Time Series")
    
    # Shared date range selector for this tab
    date_range_options = ["Full History", "Last 1 Year", "Last 6 Months", "Last 1 Month", "Last 2 Weeks", "Last 1 Week"]
    selected_range = st.selectbox(
        "Select date range for time series plots",
        options=date_range_options,
        index=0,
        key="timeseries_range"
    )
    filtered_df = filter_dates(work_df, selected_range)

    sub_tab1, sub_tab2 = st.tabs(["Spread Analysis", "Fly Analysis"])

    # ---------- Sub-Tab 1: Spread ----------
    with sub_tab1:
        st.markdown("**Compare Multiple Spreads Over Time**")
        spread_pairs = st.multiselect(
            "Select contract pairs for spread (e.g., CLF4-CLG4)",
            options=[f"{c1} - {c2}" for i, c1 in enumerate(contracts) for c2 in contracts[i+1:]],
            default=[f"{contracts[0]} - {contracts[1]}"],
            key="spread_pairs"
        )
        
        fig_spread = go.Figure()
        for pair in spread_pairs:
            c1, c2 = [x.strip() for x in pair.split("-")]
            spread_curve = filtered_df[c1] - filtered_df[c2]
            fig_spread.add_trace(go.Scatter(
                x=filtered_df["Date"], y=spread_curve, mode="lines", name=f"{c1}-{c2}"
            ))
        fig_spread.update_layout(
            title="Historical Spread Comparison",
            xaxis_title="Date", yaxis_title="Price Difference ($)", template="plotly_white", margin=dict(l=40, r=20, t=60, b=40)
        )
        st.plotly_chart(fig_spread, use_container_width=True)
        
        if do_export and spread_pairs:
            csv_data = {"Date": filtered_df["Date"].dt.date}
            for pair in spread_pairs:
                c1, c2 = [x.strip() for x in pair.split("-")]
                csv_data[f"{c1}-{c2}"] = filtered_df[c1] - filtered_df[c2]
            st.download_button("Download Spread CSV", pd.DataFrame(csv_data).to_csv(index=False).encode("utf-8"), file_name="spread_comparison.csv", mime="text/csv")

    # ---------- Sub-Tab 2: Fly ----------
    with sub_tab2:
        st.markdown("**Compare Multiple Butterfly Spreads Over Time**")
        fly_type = st.radio("Choose Fly type:", ["Auto (consecutive months)", "Manual selection"], index=0, key="fly_type", horizontal=True)
        
        selected_flies = []
        if fly_type == "Manual selection":
            num_flies = st.number_input("Number of flies to compare", min_value=1, max_value=5, value=1, step=1)
            for i in range(num_flies):
                st.markdown(f"--- \n**Fly {i+1}**")
                cols = st.columns(3)
                f1 = cols[0].selectbox(f"Contract 1 (wing)", contracts, index=0, key=f"fly_f1_{i}")
                f2 = cols[1].selectbox(f"Contract 2 (body)", contracts, index=1, key=f"fly_f2_{i}")
                f3 = cols[2].selectbox(f"Contract 3 (wing)", contracts, index=2, key=f"fly_f3_{i}")
                selected_flies.append((f1, f2, f3))
        else: # Auto
            base_contracts = st.multiselect("Select base contracts for Auto Fly", contracts, default=[contracts[0]], key="fly_base")
            for base in base_contracts:
                idx = contracts.index(base)
                if idx + 2 < len(contracts):
                    f1, f2, f3 = contracts[idx], contracts[idx+1], contracts[idx+2]
                    selected_flies.append((f1, f2, f3))
                    st.info(f"Auto Fly added: {f1} – (2 * {f2}) + {f3}")
                else:
                    st.warning(f"Not enough consecutive contracts for '{base}' auto fly. Skipping.")

        fig_fly = go.Figure()
        for f1, f2, f3 in selected_flies:
            fly_curve = filtered_df[f1] - 2 * filtered_df[f2] + filtered_df[f3]
            fig_fly.add_trace(go.Scatter(
                x=filtered_df["Date"], y=fly_curve, mode="lines", name=f"Fly {f1}-{f2}-{f3}"
            ))
        fig_fly.update_layout(
            title="Historical Fly Comparison",
            xaxis_title="Date", yaxis_title="Price Difference ($)", template="plotly_white", margin=dict(l=40, r=20, t=60, b=40)
        )
        st.plotly_chart(fig_fly, use_container_width=True)

        if do_export and selected_flies:
            fly_csv_data = {"Date": filtered_df["Date"].dt.date}
            for f1, f2, f3 in selected_flies:
                fly_csv_data[f"Fly_{f1}-{f2}-{f3}"] = filtered_df[f1] - 2*filtered_df[f2] + filtered_df[f3]
            st.download_button("Download Fly CSV", pd.DataFrame(fly_csv_data).to_csv(index=False).encode("utf-8"), file_name="fly_comparison.csv", mime="text/csv")


# ---------- TAB 3: CURVE EVOLUTION (ANIMATION) ----------
with tab_evolution:
    st.subheader("Historical Curve Evolution")
    st.info("Use the slider or the 'Play' button to animate how the forward curve has changed over time.")
    
    anim_df = work_df[["Date"] + contracts].copy().dropna(subset=contracts).reset_index(drop=True)

    fig_anim = go.Figure(
        data=[go.Scatter(x=contracts, y=anim_df.loc[0, contracts], mode="lines+markers")],
        layout=go.Layout(
            title="Forward Curve Evolution",
            xaxis_title="Contract",
            yaxis_title="Price ($)" if not normalize else "Z-score",
            template="plotly_white",
            margin=dict(l=40, r=20, t=60, b=40),
            updatemenus=[dict(
                type="buttons", showactive=False, y=1.15, x=1.05, xanchor="right", yanchor="top",
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                ]
            )],
            sliders=[dict(
                active=0,
                transition={"duration": 0},
                currentvalue={"prefix": "Date: ", "font": {"size": 14}},
                steps=[dict(
                    method="animate",
                    args=[[str(d.date())], {"mode": "immediate", "frame": {"duration": 100, "redraw": True}, "transition": {"duration": 50}}],
                    label=str(d.date())
                ) for d in anim_df["Date"]]
            )]
        ),
        frames=[go.Frame(
            data=[go.Scatter(x=contracts, y=anim_df.loc[i, contracts])],
            name=str(anim_df.loc[i, "Date"].date())
        ) for i in range(len(anim_df))]
    )
    st.plotly_chart(fig_anim, use_container_width=True)


# ---------------------------- Preview parsed data (No change) ----------------------------
with st.expander("Preview Raw Data (first 25 rows)"):
    st.dataframe(df.head(25))
