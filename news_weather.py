
import datetime as dt
from datetime import date, timedelta
import pandas as pd, requests, feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ANALYZER = SentimentIntensityAnalyzer()

def _dates_to_date_series(obj):
    """
    Robustly convert an iterable of date-like values into a pandas Series of python date objects.
    Handles list/array/DatetimeIndex without using .dt on a DatetimeIndex.
    """
    if obj is None:
        return pd.Series([], dtype="object")
    # First, to pandas datetime (returns DatetimeIndex/Series)
    dti = pd.to_datetime(list(obj), errors="coerce")
    # Convert to pure python date objects
    try:
        # Works for DatetimeIndex
        py_dates = [d.date() if pd.notna(d) else None for d in dti]
    except Exception:
        # Fallback via Series
        s = pd.Series(dti)
        py_dates = list(s.dt.date)
    return pd.Series(py_dates)

# ---- NEWS ----

def fetch_rice_news(days=7, max_items=40, query="rice price OR basmati price OR rough rice OR FAO rice"):
    q = requests.utils.quote(f'{query} when:{days}d')
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[:max_items]:
        pub = None
        for k in ("published", "updated", "pubDate"):
            if k in e:
                try:
                    pub = pd.to_datetime(getattr(e, k))
                except Exception:
                    pub = None
        items.append({
            "title": e.title,
            "link": e.link,
            "published": pub,
            "source": getattr(getattr(e, "source", None), "title", ""),
            "summary": getattr(e, "summary", ""),
        })
    return items

def build_news_sentiment(days_back=120, query="rice price OR basmati price OR rough rice OR FAO rice"):
    items = fetch_rice_news(days=min(days_back, 30), max_items=100, query=query)
    if not items:
        return pd.DataFrame(columns=["Date","news_sentiment"])
    df = pd.DataFrame(items)
    def comp(row):
        text = f"{row.get('title','')} {row.get('summary','')}"
        return ANALYZER.polarity_scores(text)["compound"]
    df["sent"] = df.apply(comp, axis=1)
    df["Date"] = pd.to_datetime(df["published"]).dt.date
    df = df.dropna(subset=["Date"])
    agg = df.groupby("Date")["sent"].mean().rename("news_sentiment").reset_index()
    return agg

# ---- WEATHER ----

def fetch_weather_daily(lat, lon, start_date, end_date):
    """Open-Meteo ERA5 archive daily temp/precip with robust date handling."""
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "daily": ["temperature_2m_mean","precipitation_sum"],
        "timezone": "auto"
    }
    r = requests.get(url, params=params, timeout=60); r.raise_for_status()
    j = r.json()
    if "daily" not in j or not j["daily"].get("time"):
        return pd.DataFrame(columns=["Date","temp","precip"])
    dates = _dates_to_date_series(j["daily"]["time"])
    d = pd.DataFrame({
        "Date": dates,
        "temp": j["daily"].get("temperature_2m_mean", [None]*len(dates)),
        "precip": j["daily"].get("precipitation_sum", [None]*len(dates)),
    })
    d = d.dropna(subset=["Date"])
    return d

def fetch_weather_forecast(lat, lon, days_forward=16):
    """Open-Meteo forecast up to 16 days with robust date handling."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": ["temperature_2m_mean","precipitation_sum"],
        "forecast_days": days_forward,
        "timezone": "auto"
    }
    r = requests.get(url, params=params, timeout=60); r.raise_for_status()
    j = r.json()
    if "daily" not in j or not j["daily"].get("time"):
        return pd.DataFrame(columns=["Date","temp","precip"])
    dates = _dates_to_date_series(j["daily"]["time"])
    d = pd.DataFrame({
        "Date": dates,
        "temp": j["daily"].get("temperature_2m_mean", [None]*len(dates)),
        "precip": j["daily"].get("precipitation_sum", [None]*len(dates)),
    })
    d = d.dropna(subset=["Date"])
    return d

RICE_REGIONS = {
    "Karnal, India": (29.6857, 76.9905),
    "Bangkok, Thailand": (13.7563, 100.5018),
    "Can Tho, Vietnam": (10.0452, 105.7469),
}

def build_weather_features(days_back=120, days_forward=16, regions=RICE_REGIONS):
    today = date.today()
    start = today - timedelta(days=days_back)
    past_frames = []
    future_frames = []
    for name, (lat, lon) in regions.items():
        p = fetch_weather_daily(lat, lon, start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))
        if not p.empty:
            p = p.rename(columns={"temp": f"temp_{name}", "precip": f"precip_{name}"})
            past_frames.append(p)
        fut = fetch_weather_forecast(lat, lon, days_forward=days_forward)
        if not fut.empty:
            fut = fut.rename(columns={"temp": f"temp_{name}", "precip": f"precip_{name}"})
            future_frames.append(fut)

    if past_frames:
        past = past_frames[0]
        for f in past_frames[1:]:
            past = pd.merge(past, f, on="Date", how="outer")
        past = past.sort_values("Date")
        temp_cols = [c for c in past.columns if c.startswith("temp_")]
        pr_cols = [c for c in past.columns if c.startswith("precip_")]
        past["temp_avg"] = pd.to_numeric(past[temp_cols], errors="coerce").mean(axis=1)
        past["precip_avg"] = pd.to_numeric(past[pr_cols], errors="coerce").mean(axis=1)
    else:
        past = pd.DataFrame(columns=["Date","temp_avg","precip_avg"])

    if future_frames:
        fut = future_frames[0]
        for f in future_frames[1:]:
            fut = pd.merge(fut, f, on="Date", how="outer")
        fut = fut.sort_values("Date")
        temp_cols = [c for c in fut.columns if c.startswith("temp_")]
        pr_cols = [c for c in fut.columns if c.startswith("precip_")]
        fut["temp_avg"] = pd.to_numeric(fut[temp_cols], errors="coerce").mean(axis=1)
        fut["precip_avg"] = pd.to_numeric(fut[pr_cols], errors="coerce").mean(axis=1)
    else:
        fut = pd.DataFrame(columns=["Date","temp_avg","precip_avg"])

    return past, fut

def assemble_exog(days_back=120, days_forward=16):
    news = build_news_sentiment(days_back=min(days_back, 30))
    w_past, w_future = build_weather_features(days_back=days_back, days_forward=days_forward)
    past = pd.merge(w_past, news, on="Date", how="outer").sort_values("Date")
    for col in [c for c in past.columns if c != "Date"]:
        past[col] = pd.to_numeric(past[col], errors="coerce")
    past = past.fillna(method="ffill").fillna(method="bfill")
    future = w_future.copy()
    future["news_sentiment"] = 0.0
    cols = ["Date"] + [c for c in past.columns if c != "Date"]
    past = past[cols]; future = future[cols]
    return past, future
