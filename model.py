
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

HORIZONS = [7, 30, 180, 365]
PRICE_CANDIDATES = ["price","close","adj close","adj_close","settle","value","last","rate"]
DATE_CANDIDATES  = ["date","timestamp","time"]

def _find_col(cols, candidates):
    cl = [c.lower() for c in cols]
    for cand in candidates:
        if cand in cl:
            return cols[cl.index(cand)]
    return None

def _prepare_series(df: pd.DataFrame):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    price_col = _find_col(df.columns, PRICE_CANDIDATES) or df.columns[-1]
    date_col  = _find_col(df.columns, DATE_CANDIDATES)  or df.columns[0]
    s = df.copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s[price_col] = pd.to_numeric(s[price_col], errors="coerce")
    s = s.dropna(subset=[date_col, price_col])
    if s.empty:
        return pd.Series(dtype=float)
    s = s.sort_values(date_col).set_index(date_col)[price_col].astype(float)
    return s.asfreq("D").ffill()

def _fit(series):
    return SARIMAX(series, order=(1,1,1), seasonal_order=(0,1,1,7),
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

def multi_forecast_ci(date_price_df: pd.DataFrame, horizons=HORIZONS):
    """Returns dict[h] -> DataFrame(date, mean, lower80, upper80, lower95, upper95)"""
    s = _prepare_series(date_price_df)
    out = {}
    if s.empty:
        for h in horizons:
            out[h] = pd.DataFrame(columns=["date","mean","lower80","upper80","lower95","upper95"])
        return out
    if len(s) < 20:
        last = float(s.iloc[-1])
        for h in horizons:
            idx = pd.date_range(s.index.max(), periods=h, freq="D")
            base = pd.DataFrame({"date": idx})
            base["mean"] = last
            base["lower80"] = last; base["upper80"] = last
            base["lower95"] = last; base["upper95"] = last
            out[h] = base
        return out

    max_h = max(horizons)
    m95 = _fit(s)
    f95 = m95.get_forecast(steps=max_h, alpha=0.05)
    mean = f95.predicted_mean; ci95 = f95.conf_int(alpha=0.05)

    f80 = m95.get_forecast(steps=max_h, alpha=0.20)
    ci80 = f80.conf_int(alpha=0.20)

    idx_all = pd.date_range(s.index.max() + pd.Timedelta(days=1), periods=max_h, freq="D")
    mean.index = idx_all; ci95.index = idx_all; ci80.index = idx_all

    for h in horizons:
        idx = idx_all[:h]
        out[h] = pd.DataFrame({
            "date": idx,
            "mean": mean.loc[idx].values,
            "lower80": ci80.loc[idx].iloc[:,0].values,
            "upper80": ci80.loc[idx].iloc[:,1].values,
            "lower95": ci95.loc[idx].iloc[:,0].values,
            "upper95": ci95.loc[idx].iloc[:,1].values,
        })
    return out
