
# Fly Curve Comparator (Streamlit)

Compare **Close** curves across multiple sheets in any Excel. The app:
- Auto-detects **Date** and **Close** columns (any order, any column letter)
- Aligns curves to a **synthetic year** (start month → next months) so you can compare **April Fly**, **June Fly**, etc.
- Lets you **rebase** to compare absolute, change, or % change
- Works with a **built-in sample** and **user uploads**

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push `app.py` and `requirements.txt` to a GitHub repo
2. Create a new app on Streamlit Cloud and select your repo
3. Add your Excel to the repo (optional). Otherwise users can upload their own

## Data expectations
- Excel with one or more sheets
- Each sheet must have **Date** and **Close** columns (names are flexible; the app will search for them)
- Dates can be actual dates (YYYY-MM-DD) or `MM/DD` (yearless). Yearless dates are assigned a year inferred from the sheet name or fallback to current year.

