
%%writefile app.py
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date

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
@st.cache_data(show_spinner=True)
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

# ---------------------------- Load Excel ----------------------------
FILE_PATH = "WTI_Outright.xlsx"  # <-- Use relative path (file must be in repo)

try:
    df, contracts = load_wti(FILE_PATH)
except Exception as e:
    st.error(f"Failed to read Excel file: {e}")
    st.stop()

# ---------------------------- Sidebar: Dates ----------------------------
st.sidebar.header("Select dates")
all_dates = sorted(df["Date"].dt.date.unique().tolist(), reverse=True)
max_d = all_dates[0]  # newest
min_d = all_dates[-1]  # oldest

single_date = st.sidebar.date_input(
    "Single date (calendar)", 
    value=max_d, 
    min_value=min_d, 
    max_value=max_d
)

multi_dates = st.sidebar.multiselect(
    "Multi-date overlay selection",
    options=all_dates,
    default=[all_dates[0], all_dates[1]],
    help="Add/remove dates for overlay comparison."
)

# ---------------------------- Sidebar: Options & Export ----------------------------
st.sidebar.header("Options")
normalize = st.sidebar.checkbox("Normalize each curve (z-score)")
st.sidebar.header("Export")
do_export = st.sidebar.checkbox("Enable CSV export")

# ---------------------------- Main title ----------------------------
st.title("WTI Outright Curve Viewer")
st.caption("Single-date curve • Multi-date overlay • Spread • Curve Evolution")

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

# Only create and display overlay chart if more than one curve exists
if len(curves) > 1:
    fig_overlay = overlay_figure(
        contracts, 
        curves, 
        y_label=("Z-score" if normalize else "Last Price ($)")
    )
    st.plotly_chart(fig_overlay, use_container_width=True, key=f"overlay_{len(curves)}")

# If only one curve is selected, skip overlay chart (already shown in single-date chart)


# ---------------------------- Section C: Custom Spread ----------------------------
st.subheader("Custom Spread (Select two contracts)")
c1 = st.selectbox("Select first contract", contracts, index=0)
c2 = st.selectbox("Select second contract", contracts, index=1)
spread_curve = work_df[c1] - work_df[c2]
spread_df = pd.DataFrame({"Date": work_df["Date"], f"Spread: {c1} - {c2}": spread_curve})

fig_spread = go.Figure()
fig_spread.add_trace(go.Scatter(x=spread_df["Date"], y=spread_df[f"Spread: {c1} - {c2}"], mode="lines+markers"))
fig_spread.update_layout(
    title=f"Spread: {c1} - {c2}",
    xaxis_title="Date",
    yaxis_title="Price Difference ($)",
    template="plotly_white",
    margin=dict(l=40, r=20, t=60, b=40),
)
st.plotly_chart(fig_spread, use_container_width=True)

if do_export:
    csv_spread = spread_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Spread CSV", csv_spread, file_name=f"spread_{c1}_{c2}.csv", mime="text/csv")

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
                    args=[None, {"frame": {"duration": 300, "redraw": True},
                                 "fromcurrent": True, "transition": {"duration": 0}}]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                )
            ]
        )],
        sliders=[dict(
            steps=[dict(
                method="animate",
                args=[[str(d.date())], {"mode": "immediate",
                                        "frame": {"duration": 0, "redraw": True},
                                        "transition": {"duration": 0}}],
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
