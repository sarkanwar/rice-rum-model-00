
import streamlit as st, pandas as pd
from news_weather import fetch_rice_news, assemble_exog

def news_tab():
    st.header("🗞 Latest Rice News & Weather Features")
    days = st.slider("Lookback (days)", 3, 30, 7)
    q = st.text_input("Query", "rice price OR basmati price OR FAO rice OR rough rice")
    if st.button("Fetch headlines"):
        items = fetch_rice_news(days=days, query=q, max_items=50)
        if not items:
            st.info("No results.")
        else:
            for it in items:
                title = it.get("title","")
                link = it.get("link","")
                src  = it.get("source","")
                pub  = it.get("published","")
                st.markdown(f"- [{title}]({link})  \n  <small>{src} — {pub}</small>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Build News + Weather Features")
    if st.button("Build features (no keys)"):
        past, future = assemble_exog(days_back=120, days_forward=16)
        st.success("Built features.")
        st.dataframe(past.tail(), use_container_width=True)
        st.download_button("Download past features CSV", data=past.to_csv(index=False).encode("utf-8"), file_name="exog_past.csv")
        st.download_button("Download future features CSV", data=future.to_csv(index=False).encode("utf-8"), file_name="exog_future.csv")
