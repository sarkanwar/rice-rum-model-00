
import io, os, re
import pandas as pd, requests
import yfinance as yf

WB_XLSX = "https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/related/CMO-Historical-Data-Monthly.xlsx"

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def fetch_yahoo_rough_rice(out_csv="data/rough_rice_yahoo.csv", period="max", interval="1d"):
    """Fetch daily Rough Rice futures from Yahoo Finance (ZR=F)."""
    ensure_dir(out_csv)
    t = yf.Ticker("ZR=F")
    df = t.history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        df = yf.download("ZR=F", period=period, interval=interval, progress=False)
    if df.empty:
        pd.DataFrame(columns=["Date","Price"]).to_csv(out_csv, index=False); return out_csv
    out = df[["Close"]].reset_index().rename(columns={"Date":"Date","Close":"Price"})
    out["Date"] = pd.to_datetime(out["Date"]).dt.date
    out.to_csv(out_csv, index=False)
    return out_csv

def fetch_worldbank_pinksheet_rice(out_csv="data/rice_wb_thai5.csv"):
    """Download World Bank Pink Sheet, extract Thai 5% broken rice monthly series."""
    ensure_dir(out_csv)
    r = requests.get(WB_XLSX, timeout=60); r.raise_for_status()
    xls = pd.ExcelFile(io.BytesIO(r.content))
    # pick a monthly sheet
    sheet = [s for s in xls.sheet_names if "Monthly" in s or "monthly" in s]
    sheet = sheet[0] if sheet else xls.sheet_names[0]
    df = xls.parse(sheet, header=None)

    # locate row containing rice descriptor
    row_idx = None
    for i in range(min(200, len(df))):
        row = " ".join(str(x) for x in df.iloc[i].tolist())
        if re.search(r"Rice\s*\(Thailand\).*(5%|5 %).*broken", row, flags=re.I):
            row_idx = i; break
    if row_idx is None:
        pd.DataFrame(columns=["Date","Price"]).to_csv(out_csv, index=False); return out_csv

    # locate header row with years
    header_row = None
    for j in range(row_idx-5, row_idx+5):
        if j < 0: continue
        vals = df.iloc[j].tolist()
        if any(str(v).strip().isdigit() and len(str(v))==4 for v in vals):
            header_row = j; break
    if header_row is None: header_row = max(row_idx-1, 0)

    wide = xls.parse(sheet, header=header_row)
    mask = wide.iloc[:,0].astype(str).str.contains("Rice", case=False, na=False) & \
           wide.iloc[:,0].astype(str).str.contains("Thailand", case=False, na=False) & \
           wide.iloc[:,0].astype(str).str.contains("5", na=False)
    series_row = wide[mask]
    if series_row.empty:
        pd.DataFrame(columns=["Date","Price"]).to_csv(out_csv, index=False); return out_csv

    series = series_row.squeeze()
    tidy = series.to_frame(name="Price").reset_index().rename(columns={"index":"Period"})
    def parse_period(x):
        s = str(x)
        import re as _re, datetime as _dt
        m = _re.match(r"(\d{4})[^\d]?(\d{1,2})$", s) or _re.match(r"(\d{4})M(\d{1,2})", s) or _re.match(r"(\d{4})-(\d{1,2})", s)
        if m:
            y, mth = int(m.group(1)), int(m.group(2)); return _dt.date(y, mth, 1)
        try:
            d = pd.to_datetime(s, errors="raise"); return d.date().replace(day=1)
        except Exception:
            return None
    tidy["Date"] = tidy["Period"].map(parse_period)
    tidy = tidy.dropna(subset=["Date"])
    tidy["Price"] = pd.to_numeric(tidy["Price"], errors="coerce")
    tidy = tidy.dropna(subset=["Price"]).sort_values("Date")[["Date","Price"]]
    tidy.to_csv(out_csv, index=False)
    return out_csv

def fetch_stocks_to_csv(tickers, out_dir="data/stocks", period="max", interval="1d"):
    """Fetch Adjusted Close for tickers; returns dict ticker->path. Skips tickers with no data."""
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for t in tickers:
        try:
            df = yf.download(t, period=period, interval=interval, auto_adjust=True, progress=False)
            if df.empty: continue
            dfo = df[["Close"]].reset_index().rename(columns={"Date":"Date","Close":"Price"})
            dfo["Date"] = pd.to_datetime(dfo["Date"]).dt.date
            path = os.path.join(out_dir, f"{t.replace('.','_')}.csv")
            dfo.to_csv(path, index=False)
            out[t] = path
        except Exception:
            continue
    return out
