import os
import time
import pickle
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
from vbase_utils.stats.pit_robust_betas import pit_robust_betas
import warnings
import pmdarima as pm   # <-- Auto ARIMA

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Constants ---
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLY": "Consumer Discretionary",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}
MARKET = "SPY"
ALL_TICKERS = [MARKET] + list(SECTOR_ETFS.keys())
CACHE_DIR = "data/cache"
OUTPUT_DIR = "data/backtest"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_from_cache(ticker):
    path = os.path.join(CACHE_DIR, f"{ticker}.pkl")
    if os.path.exists(path):
        try:
            return pickle.load(open(path, "rb"))
        except Exception as e:
            print(f"[{ticker}] Cache load error: {e}")
    return None

def save_to_cache(ticker, df):
    path = os.path.join(CACHE_DIR, f"{ticker}.pkl")
    try:
        pickle.dump(df, open(path, "wb"))
    except Exception as e:
        print(f"[{ticker}] Cache save error: {e}")

def safe_download(ticker, start="2000-01-01", end=None, max_retries=5, pause=5):
    cached = load_from_cache(ticker)
    if cached is not None and not cached.empty:
        return cached

    attempt = 0
    while attempt < max_retries:
        try:
            df = yf.download(
                ticker, start=start, end=end,
                progress=False, auto_adjust=True
            )
            if not df.empty:
                save_to_cache(ticker, df)
                return df
        except Exception as e:
            if "Rate limited" in str(e) or "YFRateLimitError" in str(e):
                print(f"[{ticker}] Rate limited. Sleeping {pause}s...")
                time.sleep(pause)
                pause *= 2
            else:
                print(f"[{ticker}] Unexpected error: {e}")
                break
        attempt += 1
    print(f"[{ticker}] Failed after {max_retries} retries.")
    return pd.DataFrame()

def smart_download(tickers, start="2000-01-01", end=None):
    all_data = []
    for t in tickers:
        df = safe_download(t, start=start, end=end)
        if df.empty:
            print(f"[{t}] Warning: empty dataframe returned.")
            continue
        s = df["Close"].copy()
        s.name = t
        all_data.append(s)

    if not all_data:
        raise RuntimeError("No valid data downloaded. Try again later.")

    return pd.concat(all_data, axis=1).sort_index()


# --- Download all tickers ---
def download_all_tickers(tickers: List[str], start: str = "2020-01-01", end: str = None) -> pd.DataFrame:
    dfs = []
    for t in tickers:
        df = smart_download(t, start, end)
        if "Close" not in df.columns:
            raise ValueError(f"No 'Close' column for {t}")
        df = df[["Close"]].rename(columns={"Close": t})
        dfs.append(df)
    combined = pd.concat(dfs, axis=1)
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    return combined


# --- Compute residuals ---
def compute_residuals(df_prices: pd.DataFrame) -> pd.DataFrame:
    df_prices = df_prices[ALL_TICKERS].dropna()
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=df_prices.index.min(), end_date=df_prices.index.max())
    trading_days = pd.to_datetime(schedule.index)
    df_prices = df_prices[df_prices.index.isin(trading_days)]

    df_rets = df_prices.pct_change().dropna()

    Y = df_rets[[col for col in df_rets.columns if col in SECTOR_ETFS.keys()]]
    X = df_rets[[MARKET]]

    weekly_rebalance = pd.DatetimeIndex([d for d in Y.index if d.weekday() == 4])

    results = pit_robust_betas(
        df_asset_rets=Y,
        df_fact_rets=X,
        half_life=126,
        min_timestamps=63,
        rebalance_time_index=weekly_rebalance,
        progress=True
    )
    residuals = results["df_asset_resids"].dropna(how="all")
    return residuals


# --- Auto ARIMA-based strategy with debugging ---
def arima_strategy(residuals: pd.DataFrame, train_ratio=0.8) -> pd.Series:
    """
    ARIMA-based strategy on residuals.
    Fits Auto-ARIMA on training sample and forecasts test sample.
    Trading signal = sign(forecast).
    Strategy return = mean(signals * realized residuals).
    """

    n = len(residuals)
    train_size = int(train_ratio * n)

    train, test = residuals.iloc[:train_size], residuals.iloc[train_size:]
    forecasts = pd.DataFrame(index=test.index, columns=residuals.columns)

    print(f"\n[DEBUG] Residuals shape: {residuals.shape}")
    print(f"[DEBUG] Train size: {len(train)}, Test size: {len(test)}")
    if test.empty:
        print("[ERROR] Test set is empty! Returning empty series.")
        return pd.Series(dtype=float)

    for col in residuals.columns:
        y_train = train[col].dropna()
        if y_train.empty:
            print(f"[WARN] Skipping {col} (empty training series)")
            continue

        try:
            # Reset index for ARIMA to avoid ValueWarning
            y_train_reset = y_train.reset_index(drop=True)
            print(f"\n[DEBUG] Fitting ARIMA for {col}, train length={len(y_train_reset)}")

            model = pm.auto_arima(
                y_train_reset,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore"
            )

            fc = model.predict(n_periods=len(test))

            forecasts[col] = pd.Series(fc, index=test.index)

            print(f"[DEBUG] {col}: Forecast length={len(fc)}, head={fc[:5]}")

        except Exception as e:
            print(f"[ERROR] ARIMA failed for {col}: {e}")

    # Trading signals = sign of forecast
    signals = np.sign(forecasts)

    # Shift to avoid look-ahead bias
    signals = signals.shift(1).reindex(test.index)

    # Strategy returns = mean of signals * realized residuals
    strategy_returns = (signals * test).mean(axis=1).dropna()

    # Extra debug info
    print("\n[DEBUG] Forecasts (head):\n", forecasts.head())
    print("[DEBUG] Signals (head):\n", signals.head())
    print("[DEBUG] Test residuals (head):\n", test.head())
    print("[DEBUG] Strategy returns (head):\n", strategy_returns.head())
    print(f"[DEBUG] Final strategy length: {len(strategy_returns)}")

    if strategy_returns.empty:
        print("[ERROR] Strategy returns are empty. Check if forecasts/signals contain only NaNs.")

    return strategy_returns


# --- Backtest ---
def backtest(strategy_returns: pd.Series) -> dict:
    if strategy_returns.empty:
        return None
    ann_return = (1 + strategy_returns.mean()) ** 252 - 1
    ann_vol = strategy_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol else np.nan
    cum_returns = (1 + strategy_returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_dd = drawdown.min()

    cum_returns.plot(figsize=(12, 6), title="Auto-ARIMA Residual Strategy Cumulative Returns")
    plt.ylabel("Cumulative Return")
    plt.xlabel("Date")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/strategy_performance_autoarima.png")
    plt.close()

    return {
        "Sharpe": sharpe,
        "CAGR": ann_return,
        "MaxDrawdown": max_dd,
        "TotalReturn": cum_returns.iloc[-1] - 1,
    }


# --- Main ---
if __name__ == "__main__":
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("Downloading all tickers with smart caching...")
    data = smart_download(ALL_TICKERS, start="2020-01-01")

    print("Computing point-in-time robust residuals...")
    residuals = compute_residuals(data)

    print("Running Auto-ARIMA strategy...")
    strategy_returns = arima_strategy(residuals)
    print(strategy_returns)

    print("Backtesting...")
    metrics = backtest(strategy_returns)
    print(metrics)
