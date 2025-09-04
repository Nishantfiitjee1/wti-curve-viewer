import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta

# ---------------------------- Page config and CSS ----------------------------
st.set_page_config(page_title="Futures Curve Viewer", layout="wide")
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
/* Custom styling for placeholder text */
.placeholder-text {
    font-size: 1.5rem;
    font-weight: bold;
    color: #888;
    text-align: center;
    margin-top: 5rem;
    border: 2px dashed #ddd;
    padding: 2rem;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------- 1. CENTRAL FILE & PRODUCT CONFIGURATION ----------------------------
MASTER_EXCEL_FILE = "Futures_Data.xlsx"
NEWS_EXCEL_FILE = "Important_news_date.xlsx"

# This configuration uses your exact sheet names.
PRODUCT_CONFIG = {
    "CL": {"name": "WTI Crude Oil", "sheet": "WTI_Outright"},
    "BZ": {"name": "Brent Crude Oil", "sheet": "Brent_outright"},
    "DBI": {"name": "Dubai Crude Oil", "sheet": "Dubai_Outright"},
    "MRBN": {"name": "Murban Crude Oil", "sheet": "MURBAN_Outright"},
}


# ---------------------------- Data Loading & Utilities ----------------------------
@st.cache_data(show_spinner="Loading product data...", ttl=3600)
def load_product_data(file_path, sheet_name):
    """
    Loads and parses futures data by intelligently finding the header and data rows.
    """
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None, engine="openpyxl")
    
    # **FIX**: Dynamically find header and data start based on your Excel format
    # It finds the row with "Dates" in the first column to locate the structure.
    try:
        header_row_index = df_raw[df_raw[0] == 'Dates'].index[0] - 1
        data_start_row_index = header_row_index + 2
    except IndexError:
        st.error(f"Could not find the 'Dates' keyword in the first column of the '{sheet_name}' sheet. Please check the Excel file format.")
        st.stop()

    contracts = [str(x).strip() for x in df_raw.iloc[header_row_index].tolist()[1:] if pd.notna(x) and str(x).strip() != ""]
    col_names = ["Date"] + contracts
    
    df = df_raw.iloc[data_start_row_index:].copy()
    df.columns = col_names
    
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in contracts:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df, contracts

@st.cache_data(show_spinner="Loading news data...", ttl=3600)
def load_news_data(file_path):
    """Loads news data and handles 'Date' vs 'Dates' column names."""
    df_news = pd.read_excel(file_path, engine="openpyxl")
    
    date_col = None
    if 'Date' in df_news.columns:
        date_col = 'Date'
    elif 'Dates' in df_news.columns:
        date_col = 'Dates'

    if date_col:
        df_news.rename(columns={date_col: 'Date'}, inplace=True)
        df_news["Date"] = pd.to_datetime(df_news["Date"], errors="coerce")
        df_news = df_news.dropna(subset=["Date"])
        return df_news
    else:
        st.warning("The news file must contain a 'Date' or 'Dates' column.")
        return pd.DataFrame()

def curve_for_date(df: pd.DataFrame, contracts, d: date) -> pd.Series | None:
    row = df.loc[df["Date"].dt.date == d, contracts]
    return row.iloc[0] if not row.empty else None

def overlay_figure(contracts, curves: dict, y_label="Last Price ($)", title="Futures Curve") -> go.Figure:
    fig = go.Figure()
    for label, s in curves.items():
        fig.add_trace(go.Scatter(x=contracts, y=s.values, mode="lines+markers", name=str(label)))
    fig.update_layout(title=title, xaxis_title="Contract", yaxis_title=y_label, hovermode="x unified",
                      template="plotly_white", margin=dict(l=40, r=20, t=60, b=40))
    return fig

def filter_dates(df, selected_range):
    max_date = df["Date"].max()
    range_map = {
        "Last 1 Week": timedelta(weeks=1), "Last 2 Weeks": timedelta(weeks=2),
        "Last 1 Month": timedelta(days=30), "Last 6 Months": timedelta(days=180),
        "Last 1 Year": timedelta(days=365),
    }
    if selected_range == "Full History": return df
    min_date = max_date - range_map.get(selected_range, timedelta(0))
    return df[df["Date"] >= min_date]

# ---------------------------- Sidebar & Product Selection ----------------------------
st.sidebar.title("Global Controls")

selected_symbol = st.sidebar.selectbox(
    "Select Product",
    options=list(PRODUCT_CONFIG.keys()),
    format_func=lambda symbol: PRODUCT_CONFIG[symbol]["name"],
    key="product_selector"
)
selected_product_info = PRODUCT_CONFIG[selected_symbol]
target_sheet_name_from_config = selected_product_info["sheet"]

# ---------------------------- Main App Logic ----------------------------
st.title(f"{selected_product_info['name']} Curve Viewer")

if not os.path.exists(MASTER_EXCEL_FILE):
    st.error(f"Master data file not found: `{MASTER_EXCEL_FILE}`. Please ensure it is in the same directory.")
    st.stop()

if os.path.exists(NEWS_EXCEL_FILE):
    df_news = load_news_data(NEWS_EXCEL_FILE)
else:
    st.sidebar.warning(f"News file (`{NEWS_EXCEL_FILE}`) not found. Hover data will not be available.")
    df_news = pd.DataFrame()

try:
    excel_file_handler = pd.ExcelFile(MASTER_EXCEL_FILE)
    excel_sheets = excel_file_handler.sheet_names
    cleaned_target_sheet = target_sheet_name_from_config.strip().lower()
    actual_sheet_to_load = next((s for s in excel_sheets if s.strip().lower() == cleaned_target_sheet), None)
    
    if actual_sheet_to_load is None:
        st.caption("Analysis of futures curves, spreads, and historical evolution.")
        st.markdown(f'<div class="placeholder-text">Data for {selected_product_info["name"]} is not yet available.<br>Sheet `{target_sheet_name_from_config}` not found.</div>', unsafe_allow_html=True)
        st.stop()
        
    df, contracts = load_product_data(MASTER_EXCEL_FILE, actual_sheet_to_load)

except Exception as e:
    st.error(f"Could not read the data. Error: {e}")
    st.stop()

st.sidebar.header("Date Selection")
all_dates = sorted(df["Date"].dt.date.unique().tolist(), reverse=True)
max_d, min_d = all_dates[0], all_dates[-1]

single_date = st.sidebar.date_input("Single Date", value=max_d, min_value=min_d, max_value=max_d, key=f"date_input_{selected_symbol}")
multi_dates = st.sidebar.multiselect("Multi-Date Overlay", options=all_dates, default=[all_dates[0], all_dates[min(1, len(all_dates)-1)]], key=f"multiselect_{selected_symbol}")

st.sidebar.header("Display Options")
normalize = st.sidebar.checkbox("Normalize curves (z-score)", key=f"normalize_{selected_symbol}")
do_export = st.sidebar.checkbox("Enable CSV export", key=f"export_{selected_symbol}")

work_df = df.copy()
if normalize:
    vals = work_df[contracts].astype(float)
    work_df[contracts] = (vals - vals.mean(axis=1).values[:, None]) / vals.std(axis=1).values[:, None]

st.caption("Analysis of futures curves, spreads, and historical evolution.")
tab1, tab2, tab3 = st.tabs(["Outright", "Spread and Fly", "Curve Animation"])

with tab1:
    st.header(f"Curve Analysis for {single_date}")
    s1 = curve_for_date(work_df, contracts, single_date)
    if s1 is None:
        st.error("No data available for the chosen date.")
    else:
        # =============================
        # METRICS SECTION
        # =============================
        st.markdown("##### Key Curve Metrics")
        m_cols = st.columns(3)
        if len(contracts) > 0: 
            m_cols[0].metric(label=f"Prompt Price ({contracts[0]})", value=f"{s1.get(contracts[0], 0):.2f}")
        if len(contracts) > 1: 
            m_cols[1].metric(label=f"M1-M2 Spread ({contracts[0]}-{contracts[1]})", value=f"{s1[contracts[0]] - s1[contracts[1]]:.2f}")
        if len(contracts) > 11: 
            m_cols[2].metric(label=f"M1-M12 Spread ({contracts[0]}-{contracts[11]})", value=f"{s1[contracts[0]] - s1[contracts[11]]:.2f}")
        
        st.markdown("---")
        col1, col2 = st.columns(2)

        # =============================
        # SINGLE DATE CURVE
        # =============================
        with col1:
            st.markdown("##### Single Date Curve")
            fig_single = overlay_figure(
                contracts, 
                {single_date: s1}, 
                y_label=("Z-score" if normalize else "Last Price ($)")
            )
            st.plotly_chart(fig_single, use_container_width=True, key=f"single_chart_{selected_symbol}")

        # =============================
        # MULTI-DATE CURVE OVERLAY
        # =============================
        with col2:
            st.markdown("##### Multi-Date Overlay")
            valid_curves = {
                d: s for d, s in {d: curve_for_date(work_df, contracts, d) for d in multi_dates}.items() if s is not None
            }
            if not valid_curves: 
                st.warning("No data found for any overlay dates.")
            else:
                fig_overlay = overlay_figure(
                    contracts, 
                    valid_curves, 
                    y_label=("Z-score" if normalize else "Last Price ($)")
                )
                st.plotly_chart(fig_overlay, use_container_width=True, key=f"multi_chart_{selected_symbol}")

            # =============================
            # DYNAMIC SPREAD CURVE FROM DEDICATED SHEET
            # =============================
            st.markdown("---")
            st.markdown("##### Spread Curve Overlay (from Dedicated Sheet)")
            
            # To make this dynamic, we map each product symbol to its corresponding spread sheet name.
            # You can easily add more products here in the future.
            SPREAD_SHEET_MAP = {
                "CL": "Spread_CL",
                "BZ": "Spread_Brent",
                "DBI": "Spread_DBI",
                "MRBN": "Spread_MRBN"
            }
            
            # Get the correct sheet name for the currently selected product.
            target_spread_sheet = SPREAD_SHEET_MAP.get(selected_symbol)
            
            # The logic will now run for any product that has a mapping above.
            if target_spread_sheet:
                try:
                    # Step 1: Load the dedicated spread data sheet using the dynamic sheet name.
                    df_spreads, spread_contracts = load_product_data(MASTER_EXCEL_FILE, target_spread_sheet)
            
                    # Step 2: Get the curves for the dates selected in the sidebar. (This logic is unchanged)
                    valid_spread_curves = {}
                    for d in multi_dates:
                        s = curve_for_date(df_spreads, spread_contracts, d)
                        if s is not None:
                            valid_spread_curves[d] = s
            
                    # Step 3: Plot the data if any was found. (This logic is unchanged)
                    if not valid_spread_curves:
                        st.warning(f"No data found in the '{target_spread_sheet}' sheet for the selected overlay dates.")
                    else:
                        fig_spread_overlay = overlay_figure(
                            spread_contracts,
                            valid_spread_curves,
                            y_label="Spread ($)",
                            title=f"Spread Curve from '{target_spread_sheet}' Sheet"
                        )
                        st.plotly_chart(fig_spread_overlay, use_container_width=True, key=f"spread_overlay_{selected_symbol}")
            
                except Exception as e:
                    # The error message is now dynamic to help with debugging.
                    st.info(f"The '{target_spread_sheet}' sheet was not found or could not be loaded. This chart is unavailable.")
            else:
                # This message appears if the selected product doesn't have a spread sheet defined in our map.
                st.info(f"Spread curve analysis is not configured for {selected_product_info['name']} ({selected_symbol}).")



with tab2:
    st.header("Spread & Fly Time Series Analysis")
    selected_range = st.selectbox("Select date range for analysis", ["Full History", "Last 1 Year", "Last 6 Months", "Last 1 Month", "Last 2 Weeks", "Last 1 Week"], index=1, key=f"range_{selected_symbol}")
    
    filtered_df = filter_dates(work_df, selected_range)
    if not df_news.empty:
        merged_df = pd.merge(filtered_df, df_news, on="Date", how="left")
    else:
        merged_df = filtered_df.copy()

    sub_tab1, sub_tab2 = st.tabs(["Spread Analysis", "Fly Analysis"])

    with sub_tab1:
        st.markdown("**Compare Multiple Spreads Over Time**")
        default_spread = [f"{contracts[0]} - {contracts[1]}"] if len(contracts) > 1 else []
        spread_pairs = st.multiselect("Select contract pairs", options=[f"{c1} - {c2}" for i, c1 in enumerate(contracts) for c2 in contracts[i+1:]], default=default_spread, key=f"spread_pairs_{selected_symbol}")
        
        if spread_pairs:
            fig_spread = go.Figure()
            for pair in spread_pairs:
                c1, c2 = [x.strip() for x in pair.split("-")]
                
                price_hover_text = [
                    f"<b>Date:</b> {d.strftime('%Y-%m-%d')}<br>"
                    f"<b>{c1}:</b> {p1:.2f}<br>"
                    f"<b>{c2}:</b> {p2:.2f}<br>"
                    f"<b>Spread ({c1}-{c2}):</b> {s:.2f}"
                    for d, p1, p2, s in zip(merged_df['Date'], merged_df[c1], merged_df[c2], merged_df[c1] - merged_df[c2])
                ]
                fig_spread.add_trace(go.Scatter(
                    x=merged_df["Date"], y=merged_df[c1] - merged_df[c2], 
                    mode="lines", name=f"{c1}-{c2}",
                    hovertext=price_hover_text, hoverinfo="text"
                ))

                if not df_news.empty:
                    news_cols = df_news.columns.drop('Date')
                    news_df_in_view = merged_df.dropna(subset=news_cols, how='all')
                    
                    if not news_df_in_view.empty:
                        news_hover_text = news_df_in_view.apply(
                            lambda row: f"<b>Date:</b> {row['Date'].strftime('%Y-%m-%d')}<br><hr>" + 
                                        "<br>".join(f"<b>{col.replace('_', ' ')}:</b> {row[col]}" for col in news_cols if pd.notna(row[col])),
                            axis=1
                        )
                        fig_spread.add_trace(go.Scatter(
                            x=news_df_in_view['Date'], y=news_df_in_view[c1] - news_df_in_view[c2],
                            mode='markers', name='News Event',
                            marker=dict(size=10, color='rgba(255, 182, 193, .9)', symbol='circle'),
                            hovertext=news_hover_text, hoverinfo="text",
                            showlegend=False
                        ))

            fig_spread.update_layout(title="Historical Spread Comparison", xaxis_title="Date", yaxis_title="Price Difference ($)", template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig_spread, use_container_width=True, key=f"spread_chart_{selected_symbol}")

    with sub_tab2:
        st.markdown("**Compare Multiple Butterfly Spreads Over Time**")
        fly_type = st.radio("Fly construction method:", ["Auto (consecutive months)", "Manual selection"], index=0, horizontal=True, key=f"fly_type_{selected_symbol}")
        selected_flies = []
        if fly_type == "Manual selection":
            num_flies = st.number_input("Number of flies", min_value=1, max_value=5, value=1, step=1, key=f"num_flies_{selected_symbol}")
            for i in range(num_flies):
                cols = st.columns(3)
                f1 = cols[0].selectbox(f"Wing 1 (Fly {i+1})", contracts, index=0, key=f"fly_f1_{i}_{selected_symbol}")
                f2 = cols[1].selectbox(f"Body (Fly {i+1})", contracts, index=1, key=f"fly_f2_{i}_{selected_symbol}")
                f3 = cols[2].selectbox(f"Wing 2 (Fly {i+1})", contracts, index=2, key=f"fly_f3_{i}_{selected_symbol}")
                selected_flies.append((f1, f2, f3))
        else: # Auto
            default_fly = [contracts[0]] if len(contracts) > 2 else []
            base_contracts = st.multiselect("Select base contracts for Auto Fly", contracts, default=default_fly, key=f"fly_base_{selected_symbol}")
            for base in base_contracts:
                idx = contracts.index(base)
                if idx + 2 < len(contracts): selected_flies.append((contracts[idx], contracts[idx+1], contracts[idx+2]))
                else: st.warning(f"Not enough consecutive contracts for '{base}' auto fly. Skipping.")
        
        if selected_flies:
            fig_fly = go.Figure()
            for f1, f2, f3 in selected_flies:
                fly_values = merged_df[f1] - 2 * merged_df[f2] + merged_df[f3]
                price_hover_text_fly = [
                    f"<b>Date:</b> {d.strftime('%Y-%m-%d')}<br>"
                    f"<b>{f1}:</b> {p1:.2f}<br>"
                    f"<b>{f2}:</b> {p2:.2f}<br>"
                    f"<b>{f3}:</b> {p3:.2f}<br>"
                    f"<b>Fly Value:</b> {fv:.2f}"
                    for d, p1, p2, p3, fv in zip(merged_df['Date'], merged_df[f1], merged_df[f2], merged_df[f3], fly_values)
                ]
                fig_fly.add_trace(go.Scatter(
                    x=merged_df["Date"], y=fly_values, 
                    mode="lines", name=f"Fly {f1}-{f2}-{f3}",
                    hovertext=price_hover_text_fly, hoverinfo="text"
                ))

                if not df_news.empty:
                    news_cols = df_news.columns.drop('Date')
                    news_df_in_view = merged_df.dropna(subset=news_cols, how='all')
                    
                    if not news_df_in_view.empty:
                        news_hover_text = news_df_in_view.apply(
                            lambda row: f"<b>Date:</b> {row['Date'].strftime('%Y-%m-%d')}<br><hr>" + 
                                        "<br>".join(f"<b>{col.replace('_', ' ')}:</b> {row[col]}" for col in news_cols if pd.notna(row[col])),
                            axis=1
                        )
                        fly_values_news = news_df_in_view[f1] - 2 * news_df_in_view[f2] + news_df_in_view[f3]
                        fig_fly.add_trace(go.Scatter(
                            x=news_df_in_view['Date'], y=fly_values_news,
                            mode='markers', name='News Event',
                            marker=dict(size=10, color='rgba(255, 182, 193, .9)', symbol='circle'),
                            hovertext=news_hover_text, hoverinfo="text",
                            showlegend=False
                        ))
            fig_fly.update_layout(title="Historical Fly Comparison", xaxis_title="Date", yaxis_title="Price Difference ($)", template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig_fly, use_container_width=True, key=f"fly_chart_{selected_symbol}")

with tab3:
    st.header("Curve Evolution Animation")
    st.info("Use the slider or the 'Play' button to animate the daily changes in the forward curve.")
    anim_df = work_df[["Date"] + contracts].copy().dropna(subset=contracts).reset_index(drop=True)
    if not anim_df.empty:
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
                name=str( anim_df.loc[i, "Date"].date() )
            ) for i in range(len(anim_df))]
        )
        st.plotly_chart(fig_anim, use_container_width=True, key=f"anim_chart_{selected_symbol}")

with st.expander("Preview Raw Data"):
    st.dataframe(df.head(25))







