import os
import time
import pickle
import pandas as pd
import yfinance as yf

def try_download(ticker, start="2000-01-01", end=None, max_retries=3, sleep_time=2):
    """Download a single ticker with retry + rate-limit handling."""
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if not df.empty:
                return df
        except Exception as e:
            print(f"Download error for {ticker}, attempt {attempt+1}: {e}")
        time.sleep(sleep_time * (attempt + 1))  # exponential backoff
    return None


def smart_download(tickers, start="2000-01-01", end=None, cache_file="data_cache.pkl"):
    """Download multiple tickers with caching + safe retry. Returns DataFrame of Close prices."""
    # Try loading from cache
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                df_all = pickle.load(f)
            print(f"Loaded cached data from {cache_file}")
            return df_all
        except Exception as e:
            print(f"Cache load failed, redownloading. Reason: {e}")

    all_data = []
    for t in tickers:
        df = try_download(t, start=start, end=end)
        if df is not None and not df.empty:
            s = df["Close"].copy()
            s.name = t  # rename column to ticker
            all_data.append(s)
        else:
            print(f"Warning: No data for {t}")

    if all_data:
        df_all = pd.concat(all_data, axis=1)
        # Save to cache
        if cache_file:
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(df_all, f)
                print(f"Saved data to cache: {cache_file}")
            except Exception as e:
                print(f"Cache save failed: {e}")
        return df_all
    else:
        return pd.DataFrame()

ALL_TICKERS = ["SPY", "AAPL", "MSFT", "XLF", "XLK"]

data = smart_download(ALL_TICKERS, start="2020-01-01")
print(data.head())
