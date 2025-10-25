
# 🌾 International Rice & Basmati Company Forecasts — with News & Weather

Single Streamlit app with **3 tabs**:
1) **Rice Benchmarks** — Yahoo Rough Rice (ZR=F) & World Bank Thai 5% (+ optional exogenous news+weather)
2) **Company Stocks** — enter tickers (Yahoo symbols) and forecast
3) **🗞 News & Weather** — latest headlines and feature builder (sentiment + weather)

**No API keys required.**

## Deploy (Streamlit Cloud)
- Main file path: `streamlit_app.py`
- Python: 3.11
- After deploy, click **Fetch** buttons to download data.
- Toggle **Include News + Weather** to use exogenous features in rice forecasts.

## Automation
`.github/workflows/daily.yml` runs nightly and commits CSVs to `data/`.

## Edit presets
Update `config.json` to change ticker groups.
