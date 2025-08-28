import pandas as pd
import plotly.graph_objects as go
import json

# --- CONFIGURATION ---
file_path = '/content/FLY_CHART.xlsx'  # Path to your Excel file
output_html_file = 'index.html'       # Name of the output file

try:
    xls = pd.ExcelFile(file_path)
except FileNotFoundError:
    print(f"Error: The file was not found at '{file_path}'.")
    exit()

# --- 1. DATA PREPARATION ---
all_sheets_data = {}
sheets_to_process = xls.sheet_names

for i, sheet in enumerate(sheets_to_process):
    df = pd.read_excel(xls, sheet_name=sheet, header=0)
    df.columns = [str(c).strip().lower() for c in df.columns]

    date_col = next((c for c in df.columns if 'date' in c), None)
    close_col = next((c for c in df.columns if 'close' in c), None)

    if not date_col or not close_col:
        print(f"Skipping sheet '{sheet}' as it does not contain 'date' and 'close' columns.")
        continue

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    year_start = df[date_col].dt.year.min()
    start_date = pd.Timestamp(year_start, 4, 1)
    end_date = pd.Timestamp(year_start + 1, 3, 31)
    full_dates = pd.DataFrame({'date': pd.date_range(start=start_date, end=end_date)})

    df_full = pd.merge(full_dates, df[[date_col, close_col]].rename(columns={date_col: 'date', close_col: 'close'}),
                       on='date', how='left')

    df_full['close'] = df_full['close'].interpolate(method='linear')
    df_full['x_axis_label'] = df_full['date'].dt.strftime('%b-%d')
    df_full['normalized_close'] = (df_full['close'] / df_full['close'].iloc[0] - 1) * 100

    all_sheets_data[sheet] = df_full.to_dict('list')

# Convert datetime objects to string for JSON serialization
for sheet, data in all_sheets_data.items():
    data['date'] = [d.isoformat() for d in data['date']]

# --- 2. HTML AND JAVASCRIPT GENERATION ---

# Embed the data directly into the HTML file as a JSON object
data_json = json.dumps(all_sheets_data)

html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Fly Prices Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; background-color: #f8f9fa; }}
        #container {{ display: flex; flex-direction: column; height: 100vh; }}
        #controls {{ padding: 10px 20px; background-color: #fff; border-bottom: 1px solid #ddd; display: flex; align-items: center; gap: 20px; }}
        #chart-div {{ flex-grow: 1; }}
        .dropdown-check-list {{ display: inline-block; position: relative; }}
        .dropdown-check-list .anchor {{ position: relative; cursor: pointer; display: inline-block; padding: 5px 50px 5px 10px; border: 1px solid #ccc; border-radius: 4px; background-color: #fff; }}
        .dropdown-check-list .anchor:after {{ position: absolute; content: ""; border-left: 2px solid black; border-top: 2px solid black; padding: 5px; right: 10px; top: 25%; transform: rotate(-135deg); }}
        .dropdown-check-list ul.items {{ display: none; position: absolute; background-color: #fff; border: 1px solid #ccc; border-top: none; list-style: none; margin: 0; padding: 0; z-index: 1000; }}
        .dropdown-check-list.visible .items {{ display: block; }}
        .dropdown-check-list .items li {{ padding: 5px 10px; }}
        .dropdown-check-list .items li:hover {{ background-color: #f0f0f0; }}
        button {{ padding: 6px 12px; border: 1px solid #ccc; border-radius: 4px; background-color: #f0f0f0; cursor: pointer; }}
    </style>
</head>
<body>
    <div id="container">
        <div id="controls">
            <div id="list1" class="dropdown-check-list" tabindex="100">
                <span class="anchor">Select Years</span>
                <ul id="year-selector" class="items"></ul>
            </div>
            <button id="toggle-all">Show/Hide All</button>
            <button id="normalize-btn">Normalize Data (%)</button>
        </div>
        <div id="chart-div"></div>
    </div>

    <script>
        const allData = {json.dumps(all_sheets_data, indent=4)};
        const chartDiv = document.getElementById('chart-div');
        const yearSelector = document.getElementById('year-selector');
        const normalizeBtn = document.getElementById('normalize-btn');
        const toggleAllBtn = document.getElementById('toggle-all');
        const anchor = document.querySelector('.dropdown-check-list .anchor');
        
        let isNormalized = false;
        const sheetNames = Object.keys(allData);
        const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];

        // Populate dropdown
        sheetNames.forEach((sheet, i) => {{
            const listItem = document.createElement('li');
            listItem.innerHTML = `<input type="checkbox" value="${'{sheet}'}" checked /> ${'{sheet}'}`;
            yearSelector.appendChild(listItem);
        }});

        function createTraces() {{
            const traces = [];
            const selectedYears = Array.from(yearSelector.querySelectorAll('input:checked')).map(input => input.value);

            sheetNames.forEach((sheet, i) => {{
                const data = allData[sheet];
                const yData = isNormalized ? data.normalized_close : data.close;
                
                traces.push({{
                    x: data.x_axis_label,
                    y: yData,
                    mode: 'lines',
                    name: sheet,
                    line: {{ color: colors[i % colors.length], width: 2.5 }},
                    visible: selectedYears.includes(sheet),
                    hovertemplate: isNormalized ? `<b>${'{sheet}'}</b><br>% Change: %{{y:.2f}}%<extra></extra>` : `<b>${'{sheet}'}</b><br>Price: %{{y:.2f}}<extra></extra>`
                }});
            }});
            return traces;
        }}

        function updateChart() {{
            const layout = {{
                title: '<b>Interactive Fly Prices Dashboard</b>',
                template: 'plotly_white',
                hovermode: 'x unified',
                xaxis: {{
                    title: 'Month',
                    tickformat: '%b',
                    rangeslider: {{ visible: true }}
                }},
                yaxis: {{
                    title: isNormalized ? 'Performance (%)' : 'Closing Price',
                    zeroline: isNormalized,
                }},
                legend: {{ title: {{ text: '<b>Financial Year</b>' }} }}
            }};
            Plotly.newPlot(chartDiv, createTraces(), layout);
        }}

        // Event Listeners
        yearSelector.addEventListener('change', updateChart);

        normalizeBtn.addEventListener('click', () => {{
            isNormalized = !isNormalized;
            normalizeBtn.textContent = isNormalized ? 'View Prices' : 'Normalize Data (%)';
            updateChart();
        }});

        let allVisible = true;
        toggleAllBtn.addEventListener('click', () => {{
            allVisible = !allVisible;
            yearSelector.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {{
                checkbox.checked = allVisible;
            }});
            updateChart();
        }});

        anchor.onclick = function() {{
            this.parentElement.classList.toggle('visible');
        }};

        // Initial plot
        updateChart();
    </script>
</body>
</html>
"""

# --- 3. WRITE TO FILE ---
try:
    with open(output_html_file, 'w') as f:
        f.write(html_template)
    print(f"Successfully generated '{output_html_file}'.")
    print("You can now open this file in your browser or deploy it to GitHub Pages.")
except Exception as e:
    print(f"An error occurred while writing the file: {e}")

