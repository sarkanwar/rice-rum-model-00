
import os, json, math
import streamlit as st, pandas as pd
from fetchers import fetch_yahoo_rough_rice, fetch_worldbank_pinksheet_rice, fetch_stocks_to_csv
from model import multi_forecast_ci, HORIZONS
from news_tab import news_tab
from news_weather import assemble_exog
from model_exog import forecast_with_exog

st.set_page_config(page_title="International Rice & Basmati Company Forecasts", page_icon="🌾", layout="wide")
st.title("🌾 International Rice & Basmati Company Forecasts")

def fmt(x, nd=2):
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:
        return "-"

def kpi_block(latest_actual, first_fore, last_fore, unit=""):
    col1, col2, col3 = st.columns(3)
    col1.metric("Last actual", f"{fmt(latest_actual)}{unit}")
    if latest_actual == latest_actual:  # not NaN
        delta = first_fore - latest_actual if first_fore == first_fore else None
    else:
        delta = None
    col2.metric("Next day forecast", f"{fmt(first_fore)}{unit}", delta=fmt(delta,2) if delta is not None else None)
    chg = (last_fore/latest_actual - 1.0)*100 if (latest_actual and latest_actual==latest_actual and last_fore==last_fore) else None
    col3.metric("Change over horizon", (f"{fmt(last_fore)}{unit}"), (f"{fmt(chg,2)}%") if chg is not None else None)

# Load presets
cfg = json.load(open("config.json")) if os.path.exists("config.json") else {"company_groups":{}}
groups = cfg.get("company_groups", {})

tab1, tab2, tab3 = st.tabs(["Rice Benchmarks", "Company Stocks", "🗞 News & Weather"])

# ---------- Rice Benchmarks ----------
with tab1:
    st.subheader("Sources")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Fetch: Yahoo Rough Rice (daily)"):
            p = fetch_yahoo_rough_rice("data/rough_rice_yahoo.csv"); st.success(f"Saved {p}")
    with colB:
        if st.button("Fetch: World Bank Thai 5% (monthly)"):
            p = fetch_worldbank_pinksheet_rice("data/rice_wb_thai5.csv"); st.success(f"Saved {p}")

    st.divider()
    choice = st.radio("Choose dataset", ["Yahoo (ZR=F)", "World Bank (Thai 5%)"], horizontal=True)
    use_exog = st.toggle("Include News + Weather (exogenous) in forecast", value=False, help="Uses sentiment & weather as drivers (up to 16 days reliably).")

    path = "data/rough_rice_yahoo.csv" if choice.startswith("Yahoo") else "data/rice_wb_thai5.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        st.markdown("### Latest data")
        st.dataframe(df.tail(30), use_container_width=True)
        labels = {7:"1 Week", 30:"1 Month", 180:"6 Months", 365:"1 Year"}

        if not use_exog:
            outs = multi_forecast_ci(df, horizons=HORIZONS)
            latest_actual = pd.to_numeric(df["Price"], errors="coerce").dropna().iloc[-1] if not df.empty else float('nan')
            first_fore = outs[7]["mean"].iloc[0] if not outs[7].empty else float('nan')
            last_fore  = outs[365]["mean"].iloc[-1] if not outs[365].empty else float('nan')
            kpi_block(latest_actual, first_fore, last_fore)

            st.markdown("### Forecasts")
            cols = st.columns(4)
            for i, h in enumerate(HORIZONS):
                with cols[i]:
                    st.markdown(f"**{labels[h]}**")
                    plot_df = outs[h].copy()
                    plot_df["date"] = pd.to_datetime(plot_df["date"])
                    st.line_chart(data=plot_df.set_index("date")[["mean","lower80","upper80"]])
                    with st.expander("Show table"):
                        st.dataframe(plot_df.head(), use_container_width=True)
                    st.download_button(f"Download {labels[h]}", data=plot_df.to_csv(index=False).encode("utf-8"),
                                       file_name=f"{'yahoo' if choice.startswith('Yahoo') else 'worldbank'}_forecast_{h}d.csv", mime="text/csv", key=f"dlf_rice_{h}")
        else:
            st.info("Building exogenous features (news sentiment + weather)…")
            past, future = assemble_exog(days_back=120, days_forward=16)
            cols = st.columns(4)
            for i, h in enumerate(HORIZONS):
                with cols[i]:
                    st.markdown(f"**{labels[h]} (exog)**")
                    out = forecast_with_exog(df, past, future, horizon_days=h)
                    if out.empty:
                        st.warning("No exogenous forecast available.")
                        continue
                    plot_df = out.copy(); plot_df["date"] = pd.to_datetime(plot_df["date"])
                    st.line_chart(data=plot_df.set_index("date")[["mean","lower95","upper95"]])
                    with st.expander("Show table"):
                        st.dataframe(plot_df.head(), use_container_width=True)
                    st.download_button(f"Download {labels[h]}", data=plot_df.to_csv(index=False).encode("utf-8"),
                                       file_name=f"{'yahoo' if choice.startswith('Yahoo') else 'worldbank'}_forecast_exog_{h}d.csv", mime="text/csv", key=f"dlf_rice_exog_{h}")
    else:
        st.info("Click a fetch button above to download data.")

# ---------- Company Stocks ----------
with tab2:
    st.subheader("Ticker groups")
    presets = ["(none)"] + list(groups.keys())
    preset = st.selectbox("Preset groups", options=presets)
    default_list = ",".join(groups.get(preset, [])) if preset in groups else "ADM,BG,KRBL.NS,DAAWAT.NS"
    tickers = st.text_input("Tickers (comma-separated, Yahoo symbols)", value=default_list)
    if st.button("Fetch stock data"):
        tk = [t.strip() for t in tickers.split(",") if t.strip()]
        res = fetch_stocks_to_csv(tk, out_dir="data/stocks")
        st.success(f"Saved {len(res)} files") if res else st.error("No tickers returned data.")

    import glob
    files = sorted(glob.glob("data/stocks/*.csv"))
    labels = {7:"1 Week", 30:"1 Month", 180:"6 Months", 365:"1 Year"}
    if not files:
        st.info("No stock files yet. Enter tickers and click 'Fetch stock data'.")
    else:
        for path in files:
            name = os.path.basename(path).replace(".csv","")
            st.markdown(f"### {name}")
            df = pd.read_csv(path)
            st.dataframe(df.tail(10), use_container_width=True)
            outs = multi_forecast_ci(df, horizons=HORIZONS)

            latest_actual = pd.to_numeric(df.iloc[:, -1], errors="coerce").dropna().iloc[-1] if not df.empty else float('nan')
            first_fore = outs[7]["mean"].iloc[0] if not outs[7].empty else float('nan')
            last_fore  = outs[365]["mean"].iloc[-1] if not outs[365].empty else float('nan')
            kpi_block(latest_actual, first_fore, last_fore, unit="")

            cols = st.columns(4)
            for i, h in enumerate(HORIZONS):
                with cols[i]:
                    st.markdown(f"**{labels[h]}**")
                    plot_df = outs[h].copy()
                    plot_df["date"] = pd.to_datetime(plot_df["date"])
                    st.line_chart(data=plot_df.set_index("date")[["mean","lower80","upper80"]])
                    with st.expander("Show table"):
                        st.dataframe(plot_df.head(), use_container_width=True)
                    st.download_button(f"Download {labels[h]}", data=plot_df.to_csv(index=False).encode("utf-8"),
                                       file_name=f"{name}_forecast_{h}d.csv", mime="text/csv", key=f"dlf_{name}_{h}")

# ---------- News & Weather ----------
with tab3:
    news_tab()
