
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def _align_exog(price_df, exog_df):
    df = price_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    y = df["Price"].astype(float).asfreq("D").ffill()

    X = exog_df.copy()
    X["Date"] = pd.to_datetime(X["Date"])
    X = X.sort_values("Date").set_index("Date")
    X = X.select_dtypes(include="number").asfreq("D").ffill().reindex(y.index).ffill()
    return y, X

def forecast_with_exog(price_df, exog_past, exog_future, horizon_days):
    y, X = _align_exog(price_df, exog_past)
    if y.empty or X.empty:
        return pd.DataFrame(columns=["date","mean","lower95","upper95"])
    m = SARIMAX(y, exog=X, order=(1,1,1), seasonal_order=(0,1,1,7),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

    F = exog_future.copy()
    F["Date"] = pd.to_datetime(F["Date"])
    F = F.sort_values("Date").set_index("Date")
    F = F.select_dtypes(include="number").asfreq("D").ffill()
    if len(F) < horizon_days:
        add = pd.date_range(start=F.index.max() if len(F) else pd.Timestamp.today(), periods=horizon_days, freq="D")
        F = F.reindex(add, method="ffill")
    F = F.iloc[:horizon_days]

    f = m.get_forecast(steps=horizon_days, exog=F)
    mean = f.predicted_mean; ci = f.conf_int(alpha=0.05)
    idx = pd.date_range(y.index.max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    mean.index = idx; ci.index = idx
    return pd.DataFrame({"date": idx, "mean": mean.values, "lower95": ci.iloc[:,0].values, "upper95": ci.iloc[:,1].values})
