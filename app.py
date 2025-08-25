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
st.sidebar.title("Global Controls")

all_dates = sorted(df["Date"].dt.date.unique().tolist(), reverse=True)
max_d = all_dates[0]
min_d = all_dates[-1]

st.sidebar.header("Date Selection")
single_date = st.sidebar.date_input(
    "Single Date", value=max_d, min_value=min_d, max_value=max_d
)

multi_dates = st.sidebar.multiselect(
    "Multi-Date Overlay",
    options=all_dates,
    default=[all_dates[0], all_dates[1]],
    help="Select multiple dates to compare their curves side-by-side."
)

st.sidebar.header("Display Options")
normalize = st.sidebar.checkbox("Normalize curves (z-score)")
do_export = st.sidebar.checkbox("Enable CSV export")

# ---------------------------- Main title ----------------------------
st.title("Curve")
st.caption("Analysis of futures curves, spreads, and historical evolution.")

# ---------------------------- Data Preparation ----------------------------
work_df = df.copy()
if normalize:
    vals = work_df[contracts].astype(float)
    work_df[contracts] = (vals - vals.mean(axis=1).values[:, None]) / vals.std(axis=1).values[:, None]

# ---------------------------- UI Structure: Main tabs ----------------------------
tab1, tab2, tab3 = st.tabs(["Outright", "Spread and Fly", "Curve Animation"])

# ---------- TAB 1: CURVE SHAPE ANALYSIS ----------
with tab1:
    st.header(f"Curve Analysis for {single_date}")
    
    s1 = curve_for_date(work_df, contracts, single_date)
    
    if s1 is None:
        st.error("No data available for the chosen date.")
    else:
        # --- Key Metrics ---
        st.markdown("##### Key Curve Metrics")
        m_cols = st.columns(3)
        m_cols[0].metric(label=f"Prompt Price ({contracts[0]})", value=f"{s1[contracts[0]]:.2f}")
        
        m1m2_spread = s1[contracts[0]] - s1[contracts[1]] if len(contracts) > 1 else "N/A"
        m_cols[1].metric(label=f"M1-M2 Spread ({contracts[0]}-{contracts[1]})", value=f"{m1m2_spread:.2f}")
        
        st.markdown("---")
        
        # --- Plots ---
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Single Date Curve")
            fig_single = overlay_figure(contracts, {single_date: s1}, y_label=("Z-score" if normalize else "Last Price ($)"))
            st.plotly_chart(fig_single, use_container_width=True)

        with col2:
            st.markdown("##### Multi-Date Overlay")
            curves = {d: curve_for_date(work_df, contracts, d) for d in multi_dates}
            valid_curves = {d: s for d, s in curves.items() if s is not None}
            if not valid_curves:
                st.warning("No data found for any of the selected overlay dates.")
            else:
                fig_overlay = overlay_figure(contracts, valid_curves, y_label=("Z-score" if normalize else "Last Price ($)"))
                st.plotly_chart(fig_overlay, use_container_width=True)


# ---------- TAB 2: HISTORICAL TIME SERIES ----------
with tab2:
    st.header("Spread & Fly Time Series Analysis")
    
    date_range_options = ["Full History", "Last 1 Year", "Last 6 Months", "Last 1 Month", "Last 2 Weeks", "Last 1 Week"]
    selected_range = st.selectbox("Select date range for analysis", options=date_range_options, index=1)
    filtered_df = filter_dates(work_df, selected_range)

    sub_tab1, sub_tab2 = st.tabs(["Spread Analysis", "Fly Analysis"])

    with sub_tab1:
        st.markdown("**Compare Multiple Spreads Over Time**")
        spread_pairs = st.multiselect(
            "Select contract pairs for spread analysis",
            options=[f"{c1} - {c2}" for i, c1 in enumerate(contracts) for c2 in contracts[i+1:]],
            default=[f"{contracts[0]} - {contracts[1]}"], key="spread_pairs"
        )
        
        if spread_pairs:
            fig_spread = go.Figure()
            stats_cols = st.columns(len(spread_pairs))
            csv_data = {"Date": filtered_df["Date"].dt.date}

            for i, pair in enumerate(spread_pairs):
                c1, c2 = [x.strip() for x in pair.split("-")]
                spread_curve = filtered_df[c1] - filtered_df[c2]
                csv_data[f"{c1}-{c2}"] = spread_curve
                fig_spread.add_trace(go.Scatter(x=filtered_df["Date"], y=spread_curve, mode="lines", name=f"{c1}-{c2}"))
                
                with stats_cols[i]:
                    st.metric(label=f"{c1}-{c2} (Latest)", value=f"{spread_curve.iloc[-1]:.2f}")
            
            st.markdown("---")
            fig_spread.update_layout(title="Historical Spread Comparison", xaxis_title="Date", yaxis_title="Price Difference ($)", template="plotly_white")
            st.plotly_chart(fig_spread, use_container_width=True)

            if do_export:
                st.download_button("Download Spread CSV", pd.DataFrame(csv_data).to_csv(index=False).encode("utf-8"), file_name="spread_comparison.csv", mime="text/csv")

    with sub_tab2:
        st.markdown("**Compare Multiple Butterfly Spreads Over Time**")
        fly_type = st.radio("Choose Fly construction method:", ["Auto (consecutive months)", "Manual selection"], index=0, horizontal=True)
        
        selected_flies = []
        if fly_type == "Manual selection":
            num_flies = st.number_input("Number of flies to compare", min_value=1, max_value=5, value=1, step=1)
            for i in range(num_flies):
                st.markdown(f"**Fly {i+1}**")
                cols = st.columns(3)
                f1 = cols[0].selectbox(f"Contract 1 (wing)", contracts, index=0, key=f"fly_f1_{i}")
                f2 = cols[1].selectbox(f"Contract 2 (body)", contracts, index=1, key=f"fly_f2_{i}")
                f3 = cols[2].selectbox(f"Contract 3 (wing)", contracts, index=2, key=f"fly_f3_{i}")
                selected_flies.append((f1, f2, f3))
        else: # Auto
            base_contracts = st.multiselect("Select base contracts for Auto Fly", contracts, default=[contracts[0]])
            for base in base_contracts:
                idx = contracts.index(base)
                if idx + 2 < len(contracts):
                    selected_flies.append((contracts[idx], contracts[idx+1], contracts[idx+2]))
                else:
                    st.warning(f"Not enough consecutive contracts for '{base}' auto fly. Skipping.")

        if selected_flies:
            fig_fly = go.Figure()
            fly_stats_cols = st.columns(len(selected_flies))
            fly_csv_data = {"Date": filtered_df["Date"].dt.date}

            for i, (f1, f2, f3) in enumerate(selected_flies):
                fly_curve = filtered_df[f1] - 2 * filtered_df[f2] + filtered_df[f3]
                fly_name = f"{f1}-{f2}-{f3}"
                fly_csv_data[f"Fly_{fly_name}"] = fly_curve
                fig_fly.add_trace(go.Scatter(x=filtered_df["Date"], y=fly_curve, mode="lines", name=f"Fly {fly_name}"))
                
                with fly_stats_cols[i]:
                    st.metric(label=f"Fly {fly_name} (Latest)", value=f"{fly_curve.iloc[-1]:.2f}")

            st.markdown("---")
            fig_fly.update_layout(title="Historical Fly Comparison", xaxis_title="Date", yaxis_title="Price Difference ($)", template="plotly_white")
            st.plotly_chart(fig_fly, use_container_width=True)

            if do_export:
                st.download_button("Download Fly CSV", pd.DataFrame(fly_csv_data).to_csv(index=False).encode("utf-8"), file_name="fly_comparison.csv", mime="text/csv")


# ---------- TAB 3: CURVE ANIMATION ----------
with tab3:
    st.header("Curve Evolution Animation")
    st.info("Use the slider or the 'Play' button to animate the daily changes in the forward curve.")
    
    anim_df = work_df[["Date"] + contracts].copy().dropna(subset=contracts).reset_index(drop=True)

    fig_anim = go.Figure(
        data=[go.Scatter(x=contracts, y=anim_df.loc[0, contracts], mode="lines+markers")],
        layout=go.Layout(
            title="Forward Curve Evolution",
            xaxis_title="Contract", yaxis_title="Price ($)" if not normalize else "Z-score",
            template="plotly_white", margin=dict(l=40, r=20, t=60, b=40),
            updatemenus=[dict(
                type="buttons", showactive=False, y=1.15, x=1.05, xanchor="right", yanchor="top",
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                ]
            )],
            sliders=[dict(
                active=0, transition={"duration": 0}, currentvalue={"prefix": "Date: ", "font": {"size": 14}},
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

# ---------------------------- Data Preview ----------------------------
with st.expander("Preview Raw Data"):
    st.dataframe(df.head(25))
