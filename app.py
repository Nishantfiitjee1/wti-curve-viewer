import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta

# ---------------------------- Page config and CSS ----------------------------
st.set_page_config(page_title="Curve viewer", layout="wide")
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

# ---------------------------- Data loader ----------------------------
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

# ---------------------------- Utility functions ----------------------------
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

# ---------------------------- Sidebar: Dates ----------------------------
st.sidebar.header("Select dates")
all_dates = sorted(df["Date"].dt.date.unique().tolist(), reverse=True)
max_d = all_dates[0]
min_d = all_dates[-1]

single_date = st.sidebar.date_input(
    "Single date (calendar)", value=max_d, min_value=min_d, max_value=max_d
)

multi_dates = st.sidebar.multiselect(
    "Multi-date overlay selection",
    options=all_dates,
    default=[all_dates[0], all_dates[1]],
    help="Add/remove dates for overlay comparison."
)

st.sidebar.header("Options")
normalize = st.sidebar.checkbox("Normalize each curve (z-score)")
st.sidebar.header("Export")
do_export = st.sidebar.checkbox("Enable CSV export")

# ---------------------------- Main title ----------------------------
st.title("WTI Outright Curve Viewer")
st.caption("Single-date curve • Multi-date overlay • Spread & Fly • Curve Evolution")

# ---------------------------- Normalize data ----------------------------
work_df = df.copy()
if normalize:
    vals = work_df[contracts].astype(float)
    work_df[contracts] = (vals - vals.mean(axis=1).values[:, None]) / vals.std(axis=1).values[:, None]

# ---------------------------- Section A: Single-date curve ----------------------------
st.subheader(f"Single-date curve: {single_date}")
s1 = curve_for_date(work_df, contracts, single_date)
if s1 is None:
    st.error("No data for the chosen date.")
else:
    fig_single = overlay_figure(contracts, {single_date: s1}, y_label=("Z-score" if normalize else "Last Price ($)"))
    st.plotly_chart(fig_single, use_container_width=True)

# ---------------------------- Section B: Multi-date overlay ----------------------------
st.subheader("Multi-date overlay")
curves = {}
missing = []

for d in multi_dates:
    s = curve_for_date(work_df, contracts, d)
    if s is None:
        missing.append(d)
    else:
        curves[d] = s

if missing:
    st.info(f"Some selected dates had no data and were skipped: {missing}")

if len(curves) > 1:
    fig_overlay = overlay_figure(contracts, curves, y_label=("Z-score" if normalize else "Last Price ($)"))
    st.plotly_chart(fig_overlay, use_container_width=True)

# ---------------------------- Section C: Spread & Fly Tabs ----------------------------
st.subheader("Spread & Fly Analysis")
tab1, tab2 = st.tabs(["Spread", "Fly"])

date_range_options = ["Full History", "Last 1 Week", "Last 2 Weeks", "Last 1 Month", "Last 6 Months", "Last 1 Year"]

# ---------- Tab 1: Spread ----------
with tab1:
    st.markdown("**Compare Multiple Spreads**")
    selected_range_spread = st.selectbox("Select date range to view", options=date_range_options, index=0, key="spread_range")
    filtered_df = filter_dates(work_df, selected_range_spread)
    
    spread_pairs = st.multiselect(
        "Select contract pairs for spread (format: Contract1 - Contract2)", 
        options=[f"{c1} - {c2}" for i, c1 in enumerate(contracts) for c2 in contracts[i+1:]],
        default=[f"{contracts[0]} - {contracts[1]}"],
        key="spread_pairs"
    )
    
    fig_spread = go.Figure()
    for pair in spread_pairs:
        c1, c2 = [x.strip() for x in pair.split("-")]
        spread_curve = filtered_df[c1] - filtered_df[c2]
        fig_spread.add_trace(go.Scatter(
            x=filtered_df["Date"], y=spread_curve, mode="lines+markers", name=f"{c1}-{c2}"
        ))
    fig_spread.update_layout(
        title="Spread Comparison",
        xaxis_title="Date",
        yaxis_title="Price Difference ($)",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig_spread, use_container_width=True)
    
    if do_export and spread_pairs:
        csv_spread = pd.DataFrame({f"{c1}-{c2}": filtered_df[c1]-filtered_df[c2] for pair in spread_pairs for c1, c2 in [pair.split("-")]}).assign(Date=filtered_df["Date"])
        st.download_button("Download Spread CSV", csv_spread.to_csv(index=False).encode("utf-8"), file_name="spread_comparison.csv", mime="text/csv")

# ---------- Tab 2: Fly ----------
with tab2:
    st.markdown("**Compare Multiple Fly Curves**")
    fly_type = st.radio("Choose Fly type:", ["Manual selection", "Auto month fly"], index=1, key="fly_type")
    selected_range_fly = st.selectbox("Select date range to view", options=date_range_options, index=0, key="fly_range")
    filtered_df_fly = filter_dates(work_df, selected_range_fly)
    
    selected_flies = []
    if fly_type == "Manual selection":
        num_flies = st.number_input("Number of flies to compare", min_value=1, max_value=5, value=1, step=1)
        for i in range(num_flies):
            st.markdown(f"**Fly {i+1}**")
            f1 = st.selectbox(f"Contract 1 (wing) - Fly {i+1}", contracts, index=0, key=f"fly_f1_{i}")
            f2 = st.selectbox(f"Contract 2 (center) - Fly {i+1}", contracts, index=1, key=f"fly_f2_{i}")
            f3 = st.selectbox(f"Contract 3 (wing) - Fly {i+1}", contracts, index=2, key=f"fly_f3_{i}")
            selected_flies.append((f1, f2, f3))
    else:
        month_base = st.selectbox("Select base contract for Auto Fly", contracts, key="fly_base")
        idx = contracts.index(month_base)
        if idx+2 < len(contracts):
            selected_flies.append((contracts[idx], contracts[idx+1], contracts[idx+2]))
            st.info(f"Auto Fly selected: {contracts[idx]} - 2×{contracts[idx+1]} + {contracts[idx+2]}")
        else:
            selected_flies.append((contracts[-3], contracts[-2], contracts[-1]))
            st.warning("Not enough consecutive contracts for auto fly. Using last three.")
    
    fig_fly = go.Figure()
    for f1, f2, f3 in selected_flies:
        fly_curve = filtered_df_fly[f1] - 2*filtered_df_fly[f2] + filtered_df_fly[f3]
        fig_fly.add_trace(go.Scatter(
            x=filtered_df_fly["Date"], y=fly_curve, mode="lines+markers", name=f"Fly {f1}-{f2}-{f3}"
        ))
    fig_fly.update_layout(
        title="Fly Curve Comparison",
        xaxis_title="Date",
        yaxis_title="Price Difference ($)",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig_fly, use_container_width=True)
    
    if do_export and selected_flies:
        fly_csv_data = {"Date": filtered_df_fly["Date"]}
        for f1, f2, f3 in selected_flies:
            fly_csv_data[f"Fly {f1}-{f2}-{f3}"] = filtered_df_fly[f1] - 2*filtered_df_fly[f2] + filtered_df_fly[f3]
        st.download_button("Download Fly CSV", pd.DataFrame(fly_csv_data).to_csv(index=False).encode("utf-8"), file_name="fly_comparison.csv", mime="text/csv")

# ---------------------------- Section D: Curve Evolution Animation ----------------------------
st.subheader("Historical Curve Evolution (Animation)")
anim_df = work_df[["Date"] + contracts].copy().dropna(subset=contracts)

fig_anim = go.Figure(
    data=[go.Scatter(x=contracts, y=anim_df.loc[0, contracts], mode="lines+markers")],
    layout=go.Layout(
        title="Forward Curve Evolution",
        xaxis_title="Contract",
        yaxis_title="Price ($)",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.15, x=1.05,
            xanchor="right", yanchor="top",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, {"frame": {"duration": 1000, "redraw": True},
                                 "fromcurrent": True, "transition": {"duration": 300}}]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                )
            ]
        )],
        sliders=[dict(
            transition={"duration": 300},
            currentvalue={"prefix": "Date: ", "font": {"size": 14}},
            steps=[dict(
                method="animate",
                args=[[str(d.date())], {"mode": "immediate",
                                        "frame": {"duration": 0, "redraw": True},
                                        "transition": {"duration": 300}}],
                label=str(d.date())
            ) for d in anim_df["Date"]]
        )]
    ),
    frames=[go.Frame(data=[go.Scatter(x=contracts, y=anim_df.loc[i, contracts])],
                     name=str(anim_df.loc[i, "Date"].date()))
            for i in range(len(anim_df))]
)

st.plotly_chart(fig_anim, use_container_width=True)

# ---------------------------- Preview parsed data ----------------------------
with st.expander("Preview parsed data (first 25 rows)"):
    st.dataframe(df.head(25))
